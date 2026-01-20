#!/usr/bin/env python3
"""Génération massive : Compressions avec valeurs w_error fixes (parallèle)."""

import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

from dota_analytics.controllers import process_full_match
from dota_analytics.views import export_match


def generate_visualization(df, match_id, w_error, results, output_path):
    """Génère une visualisation Original vs Compressé pour un match."""
    
    colors = plt.cm.tab10(np.arange(10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Match {match_id}\nOriginal vs Compressé (w_error={w_error:.2f})', 
                 fontsize=18, fontweight='bold')
    
    total_orig_points = 0
    total_segments = 0
    
    # Tracer chaque joueur
    for player_id in range(10):
        x_col, y_col = f'x{player_id}', f'y{player_id}'
        
        if x_col not in df.columns:
            continue
        
        mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
        x_orig = df[x_col][mask].values
        y_orig = df[y_col][mask].values
        
        if len(x_orig) == 0:
            continue
        
        total_orig_points += len(x_orig)
        color = colors[player_id]
        
        # CARTE 1 : ORIGINAL
        ax1.plot(x_orig, y_orig, color=color, linewidth=1.2, alpha=0.7)
        ax1.scatter(x_orig[::10], y_orig[::10], c=[color], s=10, alpha=0.4)
        
        # CARTE 2 : COMPRESSÉ
        if player_id in results:
            segments = results[player_id]
            total_segments += len(segments)
            
            for seg in segments:
                ax2.plot([seg.start.x, seg.end.x], 
                        [seg.start.y, seg.end.y], 
                        color=color, linewidth=2.5, alpha=0.8)
            
            for seg in segments:
                ax2.scatter([seg.start.x], [seg.start.y], 
                           c=[color], s=30, zorder=5, edgecolors='black', linewidth=0.5)
            if segments:
                ax2.scatter([segments[-1].end.x], [segments[-1].end.y], 
                           c=[color], s=30, zorder=5, edgecolors='black', linewidth=0.5)
    
    # Configuration
    reduction = (1 - total_segments / total_orig_points) * 100 if total_orig_points > 0 else 0
    
    ax1.set_title(f'ORIGINAL\n{total_orig_points} points', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    ax2.set_title(f'COMPRESSÉ\n{total_segments} segments ({reduction:.1f}% compression)', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return total_orig_points, total_segments, reduction


def process_single_compression(args):
    """Fonction worker pour traiter une compression en parallèle."""
    csv_path, w_error, json_base = args
    match_id = csv_path.stem.replace('coord_', '')
    
    try:
        # Charger CSV
        df = pd.read_csv(csv_path)
        
        # Compresser
        results = process_full_match(df, match_id, w_error=w_error, verbose=False)
        total_segs = sum(len(segs) for segs in results.values())
        
        # Compter points originaux
        original_points = {}
        for player_id in range(10):
            x_col = f'x{player_id}'
            if x_col in df.columns:
                non_zero = ((df[x_col] != 0.0) | (df[f'y{player_id}'] != 0.0)).sum()
                original_points[player_id] = int(non_zero)
        
        # Exporter JSON
        json_dir = json_base / f'w_error_{w_error}'
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f'{match_id}_compressed.json'
        export_match(results, match_id, json_path, w_error, original_points)
        
        # Calculer stats
        orig = sum(original_points.values())
        seg = sum(len(segs) for segs in results.values())
        red = (1 - seg / orig) * 100 if orig > 0 else 0
        size_kb = json_path.stat().st_size // 1024
        
        return True, match_id, w_error, size_kb, red
        
    except Exception as e:
        return False, match_id, w_error, 0, 0


def main():
    """Génération massive avec valeurs w_error fixes (PARALLÈLE)."""
    
    print('='*70)
    print('GÉNÉRATION MASSIVE: Compressions avec w_error fixes (PARALLÈLE)')
    print('='*70)
    
    # Charger 5 matchs
    data_dir = Path('data-dota')
    csv_files = sorted(data_dir.glob('coord_*.csv'))[:5]
    
    print(f'\nMatchs à traiter: {len(csv_files)}')
    for csv in csv_files:
        print(f'  - {csv.stem.replace("coord_", "")}')
    
    # Valeurs w_error granulaires
    w_error_values = [round(x * 0.1, 1) for x in range(1, 11)]  # 0.1-1.0
    w_error_values += list(range(2, 21))  # 2-20
    w_error_values += list(range(25, 101, 5))  # 25-100
    
    print(f'\nValeurs w_error: {len(w_error_values)} valeurs')
    print(f'Total: {len(csv_files)} matchs × {len(w_error_values)} w_error = {len(csv_files) * len(w_error_values)} compressions')
    
    # Créer le dossier de base
    json_base = Path('exported_data_mvc')
    
    # Préparer les tâches
    tasks = []
    for w_error in w_error_values:
        for csv_path in csv_files:
            tasks.append((csv_path, w_error, json_base))
    
    # Parallélisme
    num_workers = min(cpu_count(), 10)
    print(f'Parallélisme: {num_workers} workers (CPU: {cpu_count()} cœurs)\n')
    
    # Traiter en parallèle
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_compression, tasks)
    
    # Organiser les résultats par w_error
    results_by_w_error = {}
    for success, match_id, w_error, size_kb, red in results:
        if w_error not in results_by_w_error:
            results_by_w_error[w_error] = []
        results_by_w_error[w_error].append((success, match_id, size_kb, red))
    
    # Afficher les résultats groupés
    print(f'\n{"="*70}')
    print('RÉSULTATS')
    print(f'{"="*70}')
    
    successes = 0
    errors = 0
    
    for idx, w_error in enumerate(w_error_values):
        print(f'\n[{idx+1}/{len(w_error_values)}] w_error = {w_error}')
        
        if w_error in results_by_w_error:
            for success, match_id, size_kb, red in results_by_w_error[w_error]:
                if success:
                    print(f'  ✓ Match {match_id}: {size_kb}KB ({red:.1f}% compression)')
                    successes += 1
                else:
                    print(f'  ✗ Match {match_id}: ERREUR')
                    errors += 1
    
    print(f'\n{"="*70}')
    print('TERMINÉ !')
    print(f'  Succès: {successes} compressions')
    print(f'  Erreurs: {errors}')
    print(f'  Données: {json_base}/')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
