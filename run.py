#!/usr/bin/env python3
"""
Point d'entrée unique pour l'exécution des scripts d'analyse de trajectoires Dota 2.

Usage:
    python run.py compress --w_error 12 [--match_id XXXXX]
    python run.py compress-batch --w_errors 0.1,1,5,12,20
    python run.py visualize --w_error 12 --match_id 3841893562
    python run.py visualize-batch [--w_errors 12,13,15]
    python run.py zoom-proof [--match_id 3841665963] [--w_error 12]
    python run.py overlay --w_error 12 --match_id 3841893562 [--interactive]
    python run.py cluster --w_error 12 --max_files 10
"""

import argparse
import sys
from pathlib import Path

# Modules locaux
from dota_analytics.compression import MDLCompressor, process_full_match
from dota_analytics.structures import Trajectory, TrajectoryPoint, JSONExporter
from dota_analytics.plotting import (
    InteractiveOverlay, generate_static_overlay,
    get_available_w_errors, get_available_games
)
from dota_analytics.clustering import run_clustering

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif par défaut
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import json
import glob


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data-dota"
OUTPUT_DIR = BASE_DIR / "output"
COMPRESSED_DIR = OUTPUT_DIR / "compressed"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
CANVAS_PATH = BASE_DIR / "canvas.png"

# Dossiers de données (compatibilité)
EXPORTED_DATA_MVC = BASE_DIR / "exported_data_mvc"
if EXPORTED_DATA_MVC.exists():
    COMPRESSED_SOURCES = [COMPRESSED_DIR, EXPORTED_DATA_MVC]
else:
    COMPRESSED_SOURCES = [COMPRESSED_DIR]

# Créer les dossiers de sortie
for dir_path in [OUTPUT_DIR, COMPRESSED_DIR, VISUALIZATIONS_DIR, OVERLAYS_DIR, CLUSTERS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COMMANDE: COMPRESS
# =============================================================================

def compress_single_match(csv_path, w_error, output_base):
    """Compresse un seul match."""
    match_id = csv_path.stem.replace('coord_', '')
    
    try:
        # Charger CSV
        df = pd.read_csv(csv_path)
        
        # Compresser les 10 joueurs
        results = process_full_match(df, match_id, w_error=w_error)
        
        # Calculer statistiques
        total_orig = sum(len(df[f'x{i}']) for i in range(10) 
                        if f'x{i}' in df.columns)
        total_segments = sum(len(segs) for segs in results.values())
        
        # Export JSON
        output_dir = output_base / f"w_error_{w_error}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{match_id}_compressed.json"
        
        exporter = JSONExporter()
        output_path = exporter.export_match(results, match_id, output_path, w_error)
        
        size_kb = output_path.stat().st_size // 1024
        reduction = (1 - total_segments / total_orig) * 100 if total_orig > 0 else 0
        
        return True, match_id, total_orig, total_segments, reduction, size_kb
    
    except Exception as e:
        return False, match_id, 0, 0, 0, 0, str(e)


def cmd_compress(args):
    """Lance la compression."""
    w_error = args.w_error
    
    print("=" * 70)
    print("COMPRESSION MDL")
    print("=" * 70)
    print(f"w_error: {w_error}")
    
    # Fichiers à traiter
    if args.match_id:
        csv_files = [DATA_DIR / f"coord_{args.match_id}.csv"]
        if not csv_files[0].exists():
            print(f"❌ Fichier introuvable: {csv_files[0]}")
            return
    else:
        csv_files = sorted(DATA_DIR.glob("coord_*.csv"))
    
    print(f"Fichiers: {len(csv_files)}")
    
    # Parallélisation
    num_workers = min(cpu_count() - 2, 10)
    print(f"Workers: {num_workers}")
    print()
    
    tasks = [(csv_path, w_error, COMPRESSED_DIR) for csv_path in csv_files]
    
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(compress_single_match, tasks)
    
    # Résumé
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    
    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"✅ Succès: {len(successes)}/{len(results)}")
    
    if successes:
        avg_reduction = sum(r[4] for r in successes) / len(successes)
        total_size = sum(r[5] for r in successes)
        print(f"📊 Compression moyenne: {avg_reduction:.1f}%")
        print(f"💾 Taille totale: {total_size} KB")
    
    if failures:
        print(f"❌ Échecs: {len(failures)}")
        for fail in failures[:5]:
            print(f"   - {fail[1]}: {fail[6]}")
    
    print(f"\n📁 Output: {COMPRESSED_DIR / f'w_error_{w_error}'}")


def cmd_compress_batch(args):
    """Lance la compression batch pour plusieurs w_error."""
    w_errors = [float(x.strip()) for x in args.w_errors.split(',')]
    
    print("=" * 70)
    print("COMPRESSION BATCH")
    print("=" * 70)
    print(f"w_error values: {w_errors}")
    print()
    
    for w_error in w_errors:
        args.w_error = w_error
        args.match_id = None
        cmd_compress(args)
        print("\n")


# =============================================================================
# COMMANDE: VISUALIZE
# =============================================================================

def generate_comparison_image(csv_path, json_path, w_error, output_dir):
    """Génère une image de comparaison."""
    match_id = csv_path.stem.replace('coord_', '')
    
    try:
        # Charger données
        df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            compressed_data = json.load(f)
        
        # Créer figure
        colors = plt.cm.tab10(np.arange(10))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Match {match_id} - Compression MDL (w_error={w_error})', 
                     fontsize=20, fontweight='bold')
        
        total_orig = 0
        total_segments = 0
        
        # Pour chaque joueur
        for player_id in range(10):
            x_col, y_col = f'x{player_id}', f'y{player_id}'
            
            if x_col not in df.columns:
                continue
            
            # Original
            mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
            x_orig = df[x_col][mask].values
            y_orig = df[y_col][mask].values
            
            if len(x_orig) == 0:
                continue
            
            total_orig += len(x_orig)
            color = colors[player_id]
            
            # GRAPHE 1: Original
            step = max(1, len(x_orig) // 500)
            ax1.plot(x_orig[::step], y_orig[::step], 
                    color=color, linewidth=1.0, alpha=0.45, 
                    label=f'Joueur {player_id}')
            ax1.scatter(x_orig[::step], y_orig[::step], 
                       c=[color]*len(x_orig[::step]), s=8, alpha=0.3)
            
            # GRAPHE 2: Compressé
            player_data = next((p for p in compressed_data['players'] 
                              if p['player_id'] == player_id), None)
            
            if player_data:
                segments = player_data['segments']
                total_segments += len(segments)
                
                for seg in segments:
                    ax2.plot([seg['start']['x'], seg['end']['x']], 
                            [seg['start']['y'], seg['end']['y']], 
                            color=color, linewidth=1.0, alpha=0.45)
                
                if segments:
                    xs = [seg['start']['x'] for seg in segments] + [segments[-1]['end']['x']]
                    ys = [seg['start']['y'] for seg in segments] + [segments[-1]['end']['y']]
                    ax2.scatter(xs, ys, c=[color]*len(xs), s=8, 
                               zorder=10, edgecolors='black', linewidth=0.5, alpha=0.3)
        
        # Configuration graphes
        reduction = (1 - total_segments / total_orig) * 100 if total_orig > 0 else 0
        
        ax1.set_title(f'Original: {total_orig} points', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (coordonnées carte)', fontsize=12)
        ax1.set_ylabel('Y (coordonnées carte)', fontsize=12)
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.set_aspect('equal')
        ax1.set_facecolor('#f8f8f8')
        ax1.legend(loc='upper right', fontsize=8)
        
        ax2.set_title(f'Compressé: {total_segments} segments ({reduction:.1f}% compression)', 
                      fontsize=14, fontweight='bold', color='darkred')
        ax2.set_xlabel('X (coordonnées carte)', fontsize=12)
        ax2.set_ylabel('Y (coordonnées carte)', fontsize=12)
        ax2.grid(True, alpha=0.2, linestyle='--')
        ax2.set_aspect('equal')
        ax2.set_facecolor('#f8f8f8')
        
        # Sauvegarder
        output_path = output_dir / f'{match_id}_w{w_error}_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True, match_id, output_path.stat().st_size // 1024
    
    except Exception as e:
        return False, match_id, 0, str(e)


def cmd_visualize(args):
    """Génère les visualisations."""
    w_error = args.w_error
    match_id = args.match_id
    
    print("=" * 70)
    print("GÉNÉRATION VISUALISATIONS")
    print("=" * 70)
    print(f"w_error: {w_error}")
    print(f"match_id: {match_id}")
    
    csv_path = DATA_DIR / f"coord_{match_id}.csv"
    json_path = COMPRESSED_DIR / f"w_error_{w_error}" / f"{match_id}_compressed.json"
    
    if not csv_path.exists():
        print(f"❌ CSV introuvable: {csv_path}")
        return
    
    if not json_path.exists():
        print(f"❌ JSON introuvable: {json_path}")
        print(f"💡 Lancez d'abord: python run.py compress --w_error {w_error} --match_id {match_id}")
        return
    
    output_dir = VISUALIZATIONS_DIR / f"w_error_{w_error}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success, mid, size, *error = generate_comparison_image(csv_path, json_path, w_error, output_dir)
    
    if success:
        print(f"✅ Image générée: {size} KB")
        print(f"📁 Output: {output_dir / f'{match_id}_w{w_error}_comparison.png'}")
    else:
        print(f"❌ Erreur: {error[0]}")


def cmd_visualize_batch(args):
    """Génère les visualisations en batch."""
    if args.w_errors:
        w_errors = [float(x.strip()) for x in args.w_errors.split(',')]
    else:
        # Toutes les valeurs disponibles
        w_error_dirs = list(COMPRESSED_DIR.glob("w_error_*"))
        w_errors = sorted([float(d.name.replace('w_error_', '')) for d in w_error_dirs])
    
    print("=" * 70)
    print("GÉNÉRATION BATCH VISUALISATIONS")
    print("=" * 70)
    print(f"w_error values: {len(w_errors)}")
    print()
    
    # Collecter toutes les tâches
    tasks = []
    for w_error in w_errors:
        json_dir = COMPRESSED_DIR / f"w_error_{w_error}"
        if not json_dir.exists():
            continue
        
        for json_path in json_dir.glob("*_compressed.json"):
            match_id = json_path.stem.replace('_compressed', '')
            csv_path = DATA_DIR / f"coord_{match_id}.csv"
            
            if csv_path.exists():
                output_dir = VISUALIZATIONS_DIR / f"w_error_{w_error}"
                output_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((csv_path, json_path, w_error, output_dir))
    
    print(f"Total tâches: {len(tasks)}")
    
    # Parallélisation
    num_workers = min(cpu_count() - 2, 10)
    print(f"Workers: {num_workers}")
    print()
    
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(generate_comparison_image, tasks)
    
    # Résumé
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    
    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"✅ Succès: {len(successes)}/{len(results)}")
    
    if successes:
        total_size = sum(r[2] for r in successes)
        print(f"💾 Taille totale: {total_size} KB")
    
    if failures:
        print(f"❌ Échecs: {len(failures)}")
    
    print(f"\n📁 Output: {VISUALIZATIONS_DIR}")


# =============================================================================
# COMMANDE: ZOOM-PROOF
# =============================================================================

def cmd_zoom_proof(args):
    """Génère la preuve de concept d'élimination du bruit."""
    match_id = args.match_id or "3841665963"
    w_error = args.w_error
    player_id = 0
    
    print("=" * 70)
    print("PREUVE DE CONCEPT - ÉLIMINATION DU BRUIT")
    print("=" * 70)
    print(f"Match: {match_id}")
    print(f"Joueur: {player_id}")
    print(f"w_error: {w_error}")
    
    csv_path = DATA_DIR / f"coord_{match_id}.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV introuvable: {csv_path}")
        return
    
    # Charger données
    df = pd.read_csv(csv_path)
    
    # Trouver segment avec mouvement
    x0_nonzero = df[df[f'x{player_id}'] != 0.0]
    if len(x0_nonzero) == 0:
        print("❌ Aucune donnée pour ce joueur")
        return
    
    x0_nonzero = x0_nonzero.sort_values('tick')
    middle_idx = len(x0_nonzero) // 2
    sample = x0_nonzero.iloc[middle_idx-50:middle_idx+50]
    
    if len(sample) == 0:
        print("❌ Impossible de trouver un segment")
        return
    
    tick_start = sample['tick'].min()
    tick_end = sample['tick'].max()
    
    print(f"Fenêtre temporelle: ticks {tick_start}-{tick_end}")
    print(f"Points extraits: {len(sample)}")
    
    # Créer trajectoire
    points = []
    for _, row in sample.iterrows():
        point = TrajectoryPoint(
            x=float(row[f'x{player_id}']),
            y=float(row[f'y{player_id}']),
            tick=int(row['tick'])
        )
        points.append(point)
    
    trajectory = Trajectory(points=points)
    
    # Compresser
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments = compressor.compress_player_trajectory(trajectory)
    
    print(f"Segments générés: {len(segments)}")
    print(f"Taux compression: {100 - len(segments)/len(points)*100:.1f}%")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Points originaux
    x_orig = sample[f'x{player_id}'].values
    y_orig = sample[f'y{player_id}'].values
    
    ax.plot(x_orig, y_orig, 
           color='gray', alpha=0.5, linewidth=0.5,
           marker='x', markersize=4,
           label='Trajectoire originale (bruit)', zorder=1)
    
    # Segments compressés
    for segment in segments:
        x1, y1 = segment.start.x, segment.start.y
        x2, y2 = segment.end.x, segment.end.y
        
        ax.plot([x1, x2], [y1, y2], 
               color='red', linewidth=3, zorder=2)
        ax.plot([x1, x2], [y1, y2], 'o',
               color='red', markersize=8,
               markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    
    ax.plot([], [], color='red', linewidth=3, marker='o', markersize=8,
           label='Trajectoire compressée MDL (lissée)')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (coordonnées carte)', fontsize=12)
    ax.set_ylabel('Y (coordonnées carte)', fontsize=12)
    ax.set_title(
        f'Zoom sur l\'élimination du bruit (w_error={w_error})\n'
        f'Joueur {player_id} - Ticks {tick_start}-{tick_end}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'zoom_proof_w{w_error}.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Image sauvegardée: {output_path}")


# =============================================================================
# COMMANDE: OVERLAY
# =============================================================================

def find_compressed_file(w_error, match_id):
    """Trouve le fichier JSON compressé dans les sources disponibles."""
    w_error_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
    
    for source_dir in COMPRESSED_SOURCES:
        json_path = source_dir / f"w_error_{w_error_str}" / f"{match_id}_compressed.json"
        if json_path.exists():
            return json_path
        
        # Essayer format décimal
        json_path = source_dir / f"w_error_{float(w_error)}" / f"{match_id}_compressed.json"
        if json_path.exists():
            return json_path
    
    return None


def cmd_overlay(args):
    """Génère un overlay sur la carte Dota 2."""
    print("=" * 70)
    print("OVERLAY SUR CARTE DOTA 2")
    print("=" * 70)
    print(f"w_error: {args.w_error}")
    print(f"match_id: {args.match_id}")
    print(f"Mode: {'INTERACTIF' if args.interactive else 'STATIQUE'}")
    print()
    
    # Vérifier canvas
    if not CANVAS_PATH.exists():
        print(f"❌ Carte introuvable: {CANVAS_PATH}")
        return
    
    # Trouver le fichier compressé
    json_path = find_compressed_file(args.w_error, args.match_id)
    if not json_path:
        print(f"❌ Données compressées introuvables")
        print(f"💡 Lancez d'abord: python run.py compress --w_error {args.w_error} --match_id {args.match_id}")
        return
    
    data_dir = json_path.parent.parent
    
    if args.interactive:
        # Mode interactif
        print("🎮 Lancement mode interactif...")
        print("Contrôles:")
        print("  - Molette: Zoom")
        print("  - Clic gauche + glisser: Déplacement")
        print("  - Slider: Avancer dans le temps")
        print("  - Touche R: Reset vue")
        print("  - Touche S: Sauvegarder")
        print()
        
        # Changer backend pour interactif
        matplotlib.use('TkAgg')
        
        overlay = InteractiveOverlay(CANVAS_PATH, data_dir, args.w_error, args.match_id)
        overlay.show()
    else:
        # Mode statique
        print("🎨 Génération overlay statique...")
        output_path = OVERLAYS_DIR / f"{args.match_id}_w{args.w_error}_overlay.png"
        
        generate_static_overlay(CANVAS_PATH, data_dir, args.w_error, args.match_id, output_path)
        
        size_kb = output_path.stat().st_size // 1024
        print(f"✅ Image générée: {size_kb} KB")
        print(f"📁 Output: {output_path}")


def cmd_overlay_select(args):
    """Mode interactif avec sélection w_error et match."""
    print("=" * 70)
    print("SÉLECTION INTERACTIVE")
    print("=" * 70)
    
    # Lister w_errors disponibles
    w_errors_all = []
    for source_dir in COMPRESSED_SOURCES:
        if source_dir.exists():
            w_errors_all.extend(get_available_w_errors(source_dir))
    
    w_errors = sorted(set(w_errors_all))
    
    if not w_errors:
        print("❌ Aucune donnée compressée disponible")
        print("💡 Lancez d'abord: python run.py compress --w_error 12")
        return
    
    print(f"\n📊 Niveaux de compression disponibles ({len(w_errors)}):")
    
    # Afficher en colonnes
    for i in range(0, len(w_errors), 10):
        chunk = w_errors[i:i+10]
        print("  " + ", ".join(f"{w:.1f}" for w in chunk))
    
    # Demander w_error
    while True:
        w_error_input = input("\n🎯 Choisissez un w_error (ou 'q' pour quitter): ")
        if w_error_input.lower() == 'q':
            return
        
        try:
            w_error = float(w_error_input)
            if w_error in w_errors:
                break
            else:
                print(f"❌ w_error={w_error} non disponible")
        except ValueError:
            print("❌ Valeur invalide")
    
    # Lister matchs disponibles
    games_all = []
    for source_dir in COMPRESSED_SOURCES:
        if source_dir.exists():
            games_all.extend(get_available_games(source_dir, w_error))
    
    games = sorted(set(games_all))
    
    if not games:
        print(f"❌ Aucun match disponible pour w_error={w_error}")
        return
    
    print(f"\n🎮 Parties disponibles pour w_error={w_error} ({len(games)}):")
    for i, game_id in enumerate(games, 1):
        print(f"   {i}. {game_id}")
    
    # Demander match
    while True:
        game_input = input("\n🎯 Choisissez une partie (numéro ou ID, ou 'q'): ")
        if game_input.lower() == 'q':
            return
        
        # Par numéro
        if game_input.isdigit():
            idx = int(game_input) - 1
            if 0 <= idx < len(games):
                match_id = games[idx]
                break
        # Par ID complet
        elif game_input in games:
            match_id = game_input
            break
        
        print("❌ Choix invalide")
    
    # Lancer overlay interactif
    print()
    args.w_error = w_error
    args.match_id = match_id
    args.interactive = True
    cmd_overlay(args)


# =============================================================================
# COMMANDE: CLUSTER
# =============================================================================

def cmd_cluster(args):
    """Lance le clustering."""
    print("=" * 70)
    print("CLUSTERING (CUSTOM AFFINITY PROPAGATION)")
    print("=" * 70)
    
    # 1. On construit les deux noms de dossiers possibles
    # Cas A : w_error_12 (entier)
    name_int = f"w_error_{int(args.w_error)}"
    # Cas B : w_error_12.0 (float - c'est ton cas actuel)
    name_float = f"w_error_{float(args.w_error)}"
    
    path_int = COMPRESSED_DIR / name_int
    path_float = COMPRESSED_DIR / name_float
    
    # 2. On vérifie lequel existe
    if path_float.exists():
        target_folder = path_float
        print(f"✅ Dossier trouvé : {target_folder.name}")
    elif path_int.exists():
        target_folder = path_int
        print(f"✅ Dossier trouvé : {target_folder.name}")
    else:
        print(f"❌ Dossier introuvable.")
        print(f"   J'ai cherché : {path_int}")
        print(f"   ET aussi     : {path_float}")
        print(f"💡 Lancez d'abord: python run.py compress-batch --w_errors {args.w_error}")
        return

    if args.max_files:
        print(f"⚠️  Mode test : Limite de {args.max_files} fichiers.")
    
    # Lancement avec l'option de limitation
    run_clustering(target_folder, max_files=args.max_files)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Point d\'entrée unique pour l\'analyse de trajectoires Dota 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run.py compress --w_error 12
  python run.py compress --w_error 12 --match_id 3841893562
  python run.py compress-batch --w_errors 0.1,1,5,12,20
  
  python run.py visualize --w_error 12 --match_id 3841893562
  python run.py visualize-batch
  python run.py visualize-batch --w_errors 12,13,15
  
  python run.py zoom-proof --w_error 12
  python run.py overlay --w_error 12 --match_id 3841893562
  
  python run.py cluster --w_error 12 --max_files 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # COMPRESS
    parser_compress = subparsers.add_parser('compress', help='Compresser les trajectoires')
    parser_compress.add_argument('--w_error', type=float, required=True,
                                help='Paramètre de compression')
    parser_compress.add_argument('--match_id', type=str, default=None,
                                help='ID du match (optionnel, sinon tous)')
    
    # COMPRESS-BATCH
    parser_compress_batch = subparsers.add_parser('compress-batch', 
                                                  help='Compresser avec plusieurs w_error')
    parser_compress_batch.add_argument('--w_errors', type=str, required=True,
                                      help='Liste de w_error séparés par virgules (ex: 0.1,1,5,12)')
    
    # VISUALIZE
    parser_viz = subparsers.add_parser('visualize', help='Générer visualisation')
    parser_viz.add_argument('--w_error', type=float, required=True,
                           help='Paramètre de compression')
    parser_viz.add_argument('--match_id', type=str, required=True,
                           help='ID du match')
    
    # VISUALIZE-BATCH
    parser_viz_batch = subparsers.add_parser('visualize-batch', 
                                            help='Générer visualisations en batch')
    parser_viz_batch.add_argument('--w_errors', type=str, default=None,
                                 help='Liste de w_error (optionnel, sinon tous)')
    
    # ZOOM-PROOF
    parser_zoom = subparsers.add_parser('zoom-proof', 
                                       help='Générer preuve de concept')
    parser_zoom.add_argument('--match_id', type=str, default='3841665963',
                            help='ID du match (défaut: 3841665963)')
    parser_zoom.add_argument('--w_error', type=float, default=12.0,
                            help='Paramètre de compression (défaut: 12)')
    
    # OVERLAY
    parser_overlay = subparsers.add_parser('overlay', help='Générer overlay sur carte')
    parser_overlay.add_argument('--w_error', type=float, required=True,
                               help='Paramètre de compression')
    parser_overlay.add_argument('--match_id', type=str, required=True,
                               help='ID du match')
    parser_overlay.add_argument('--interactive', action='store_true',
                               help='Mode interactif avec slider temporel')
    
    # OVERLAY-SELECT
    parser_overlay_select = subparsers.add_parser('overlay-select', 
                                                 help='Sélection interactive w_error et match')
    
    # CLUSTER
    parser_cluster = subparsers.add_parser('cluster', help='Lancer clustering')
    parser_cluster.add_argument('--w_error', type=float, required=True,
                               help='Paramètre de compression')
    parser_cluster.add_argument('--max_files', type=int, default=None,
                               help='Nombre maximum de fichiers à traiter')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Router vers les commandes
    commands = {
        'compress': cmd_compress,
        'compress-batch': cmd_compress_batch,
        'visualize': cmd_visualize,
        'visualize-batch': cmd_visualize_batch,
        'zoom-proof': cmd_zoom_proof,
        'overlay': cmd_overlay,
        'overlay-select': cmd_overlay_select,
        'cluster': cmd_cluster,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()