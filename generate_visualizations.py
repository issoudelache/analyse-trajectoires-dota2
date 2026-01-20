#!/usr/bin/env python3
"""Génération des visualisations à partir des JSON existants avec parallélisme."""

import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def load_compressed_data(json_path):
    """Charge les données compressées depuis un fichier JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for player_data in data['players']:
        player_id = player_data['player_id']
        segments = []
        
        for seg in player_data['segments']:
            # Créer un simple objet avec les coordonnées
            class Point:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            
            class Segment:
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
            
            start = Point(seg['start']['x'], seg['start']['y'])
            end = Point(seg['end']['x'], seg['end']['y'])
            segments.append(Segment(start, end))
        
        results[player_id] = segments
    
    return results, data


def generate_visualization(df, match_id, w_error, results, total_orig, output_path):
    """Génère une visualisation Original vs Compressé côte à côte."""
    
    colors = plt.cm.tab10(np.arange(10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Match {match_id} - Compression MDL (w_error={w_error})', 
                 fontsize=20, fontweight='bold')
    
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
        
        color = colors[player_id]
        
        # GRAPHE 1 : ORIGINAL (avec transparence)
        step = max(1, len(x_orig) // 500)  # Max 500 points par joueur
        ax1.plot(x_orig[::step], y_orig[::step], 
                 color=color, linewidth=1.0, alpha=0.45, 
                 label=f'Joueur {player_id}')
        ax1.scatter(x_orig[::step], y_orig[::step], 
                    c=[color]*len(x_orig[::step]), s=8, alpha=0.3)
        
        # GRAPHE 2 : COMPRESSÉ
        if player_id in results:
            segments = results[player_id]
            total_segments += len(segments)
            
            for seg in segments:
                ax2.plot([seg.start.x, seg.end.x], 
                        [seg.start.y, seg.end.y], 
                        color=color, linewidth=2.5, alpha=0.9)
            
            # Points de jonction
            if len(segments) > 0:
                xs = [seg.start.x for seg in segments] + [segments[-1].end.x]
                ys = [seg.start.y for seg in segments] + [segments[-1].end.y]
                ax2.scatter(xs, ys, c=[color]*len(xs), s=40, 
                           zorder=10, edgecolors='black', linewidth=1, alpha=0.9)
    
    # Configuration graphe 1 (Original)
    ax1.set_title(f'Original: {total_orig} points', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (coordonnées carte)', fontsize=12)
    ax1.set_ylabel('Y (coordonnées carte)', fontsize=12)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_aspect('equal')
    ax1.set_facecolor('#f8f8f8')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Configuration graphe 2 (Compressé)
    reduction = (1 - total_segments / total_orig) * 100 if total_orig > 0 else 0
    ax2.set_title(f'Compressé: {total_segments} segments ({reduction:.1f}% compression)', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax2.set_xlabel('X (coordonnées carte)', fontsize=12)
    ax2.set_ylabel('Y (coordonnées carte)', fontsize=12)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_aspect('equal')
    ax2.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return total_orig, total_segments, reduction


def process_single_visualization(args):
    """Fonction worker pour traiter une seule visualisation (pour le parallélisme)."""
    csv_path, w_error, json_base, viz_base = args
    
    match_id = csv_path.stem.replace('coord_', '')
    
    try:
        # Charger le CSV original
        df = pd.read_csv(csv_path)
        
        # Charger les données compressées depuis JSON
        json_dir = json_base / f'w_error_{w_error}'
        json_path = json_dir / f'{match_id}_compressed.json'
        results, data = load_compressed_data(json_path)
        
        # Calculer total_orig depuis les données des joueurs
        total_orig = sum(p['num_original_points'] for p in data['players'])
        
        # Générer la visualisation
        viz_dir = viz_base / f'w_error_{w_error}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f'{match_id}_comparison.png'
        
        orig, seg, red = generate_visualization(df, match_id, w_error, results, total_orig, viz_path)
        
        size_kb = viz_path.stat().st_size // 1024
        return True, match_id, w_error, size_kb, red
        
    except Exception as e:
        return False, match_id, w_error, str(e), 0


def main():
    """Génération des visualisations avec parallélisme multi-cœurs."""
    
    print('='*70)
    print('GÉNÉRATION DES VISUALISATIONS (PARALLÈLE)')
    print('='*70)
    
    # Charger les CSV
    data_dir = Path('data-dota')
    csv_files = sorted(data_dir.glob('coord_*.csv'))[:5]
    
    # Valeurs w_error - TOUTES les valeurs
    w_error_values = [round(x * 0.1, 1) for x in range(1, 11)]  # 0.1-1.0
    w_error_values += list(range(2, 21))  # 2-20
    w_error_values += list(range(25, 101, 5))  # 25-100
    
    # Nombre de cœurs à utiliser (Ryzen 5 5500U = 6 cœurs / 12 threads)
    num_workers = min(cpu_count(), 10)  # Max 10 workers pour ne pas saturer
    
    print(f'\nMatchs à traiter: {len(csv_files)}')
    print(f'Valeurs w_error: {len(w_error_values)} valeurs')
    print(f'  - 0.1 à 1.0 (pas 0.1): {[x for x in w_error_values if x <= 1.0]}')
    print(f'  - 2 à 20 (pas 1): {[x for x in w_error_values if 1 < x <= 20]}')
    print(f'  - 25 à 100 (pas 5): {[x for x in w_error_values if x > 20]}')
    print(f'\nTotal: {len(csv_files)} matchs × {len(w_error_values)} w_error = {len(csv_files) * len(w_error_values)} visualisations')
    print(f'Parallélisme: {num_workers} workers (CPU: {cpu_count()} cœurs)\n')
    
    json_base = Path('exported_data_mvc')
    viz_base = Path('visualizations_mvc')
    
    # Préparer toutes les tâches
    tasks = []
    for w_error in w_error_values:
        for csv_path in csv_files:
            tasks.append((csv_path, w_error, json_base, viz_base))
    
    # Traiter en parallèle
    total_success = 0
    total_error = 0
    
    print(f'{"="*70}')
    print('GÉNÉRATION EN COURS...')
    print(f'{"="*70}\n')
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_visualization, tasks)
        
        # Organiser les résultats par w_error
        results_by_werror = {}
        for result in results:
            success, match_id, w_error, info, red = result
            if w_error not in results_by_werror:
                results_by_werror[w_error] = []
            results_by_werror[w_error].append((success, match_id, info, red))
        
        # Afficher les résultats groupés
        for idx, w_error in enumerate(w_error_values):
            print(f'[{idx+1}/{len(w_error_values)}] w_error = {w_error}')
            
            if w_error in results_by_werror:
                for success, match_id, info, red in results_by_werror[w_error]:
                    if success:
                        print(f'  ✓ Match {match_id}: {info}KB ({red:.1f}% compression)')
                        total_success += 1
                    else:
                        print(f'  ✗ Match {match_id}: ERREUR - {info}')
                        total_error += 1
    
    print(f'\n{"="*70}')
    print(f'TERMINÉ !')
    print(f'  Succès: {total_success} visualisations')
    print(f'  Erreurs: {total_error}')
    print(f'  Dossier: {viz_base}/')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
