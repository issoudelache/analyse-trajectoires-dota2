#!/usr/bin/env python3
"""
Benchmark Scientifique Exhaustif : Robustesse de l'algorithme MDL

Teste toutes les combinaisons de paramètres de compression (w_error) et de niveaux
de bruit (sigma) pour évaluer :
- La résistance au bruit (RMSE vs bruit croissant)
- L'efficacité de compression (nombre de segments)

Configuration :
- 7 valeurs de w_error : [1.0, 5.0, 10.0, 12.0, 15.0, 25.0, 50.0]
- 20 niveaux de sigma : 0 à 50 (linéaire)
- Total : 140 tests

Sorties :
- CSV : output/benchmark_matrix/stats/full_benchmark_results.csv
- Graphiques : output/benchmark_matrix/stats/resistance_au_bruit.png
               output/benchmark_matrix/stats/efficacite_compression.png
- Images : output/benchmark_matrix/images/w_{w}/ (sélection de sigmas clés)

Usage:
    python scripts/run_full_benchmark.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import json
from multiprocessing import Pool, cpu_count

# Ajouter le répertoire parent au path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.structures import Trajectory, TrajectoryPoint
from dota_analytics.compression import MDLCompressor
from dota_analytics.metrics import add_gaussian_noise


# =============================================================================
# CONFIGURATION
# =============================================================================

MATCH_ID = "3841665963"
PLAYER_ID = 0
TICK_START = 66000
TICK_END = 68000

# Paramètres du benchmark
# Progression très graduelle de w_error
W_ERROR_VALUES = (
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] +  # 0.5 → 3.0 : 0.5 en 0.5
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] +  # 3.0 → 10.0 : 1.0 en 1.0
    [12.0, 14.0, 16.0, 18.0, 20.0] +  # 10.0 → 20.0 : 2.0 en 2.0
    [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]  # 20.0 → 50.0 : 5.0 en 5.0
)

# Progression très graduelle de sigma
SIGMA_VALUES = (
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] +  # 0.0 → 1.0 : 0.1 en 0.1
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] +  # 1.0 → 5.0 : 0.5 en 0.5
    [6.0, 7.0, 8.0, 9.0, 10.0] +  # 5.0 → 10.0 : 1.0 en 1.0
    [12.0, 14.0, 16.0, 18.0, 20.0] +  # 10.0 → 20.0 : 2.0 en 2.0
    [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]  # 20.0 → 50.0 : 5.0 en 5.0
)

# Paramètres fixes pour les séries d'images
FIXED_SIGMA_FOR_W_VARIATION = 5.0  # On garde sigma=5.0 fixe et on varie w_error
FIXED_W_FOR_SIGMA_VARIATION = 12.0  # On garde w_error=12.0 fixe et on varie sigma

# Dossiers
DATA_DIR = BASE_DIR / "data-dota"
OUTPUT_DIR = BASE_DIR / "output" / "benchmark_matrix"
STATS_DIR = OUTPUT_DIR / "stats"
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_VAR_W_DIR = IMAGES_DIR / "variation_w_error"  # Sigma fixe, w_error varie
IMAGES_VAR_SIGMA_DIR = IMAGES_DIR / "variation_sigma"  # w_error fixe, sigma varie

STATS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_VAR_W_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_VAR_SIGMA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / f"coord_{MATCH_ID}.csv"
RESULTS_CSV = STATS_DIR / "full_benchmark_results.csv"


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_player_trajectory(csv_path, player_id, tick_start, tick_end):
    """
    Charge la trajectoire d'un joueur sur une fenêtre temporelle.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        player_id: ID du joueur (0-9)
        tick_start: Tick de début
        tick_end: Tick de fin
    
    Returns:
        Trajectory: Trajectoire du joueur dans la fenêtre
    """
    df = pd.read_csv(csv_path)
    
    x_col = f'x{player_id}'
    y_col = f'y{player_id}'
    
    # Filtrer par fenêtre temporelle et points non-nuls
    mask = (
        (df['tick'] >= tick_start) &
        (df['tick'] <= tick_end) &
        ((df[x_col] != 0.0) | (df[y_col] != 0.0))
    )
    
    player_df = df[mask].sort_values('tick')
    
    # Créer la trajectoire
    points = []
    for _, row in player_df.iterrows():
        point = TrajectoryPoint(
            x=float(row[x_col]),
            y=float(row[y_col]),
            tick=int(row['tick'])
        )
        points.append(point)
    
    return Trajectory(points=points)


def calculate_rmse_vs_original(original_points, compressed_segments):
    """
    Calcule le RMSE entre la trajectoire CLEAN originale et les segments compressés.
    
    Cette métrique mesure la fidélité de la reconstruction par rapport à la 
    trajectoire originale (sans bruit).
    
    Args:
        original_points: Liste de TrajectoryPoint (trajectoire clean)
        compressed_segments: Liste de Segment (résultat MDL)
    
    Returns:
        float: RMSE (Root Mean Square Error)
    """
    if not compressed_segments:
        return 0.0
    
    errors = []
    
    # Créer un dictionnaire de segments par tick pour accès rapide
    segment_map = {}
    for seg in compressed_segments:
        for tick in range(seg.start.tick, seg.end.tick + 1):
            segment_map[tick] = seg
    
    for point in original_points:
        # Trouver le segment correspondant temporellement (optimisé)
        segment = segment_map.get(point.tick)
        
        if segment is None:
            continue
        
        # Calculer distance perpendiculaire
        x1, y1 = segment.start.x, segment.start.y
        x2, y2 = segment.end.x, segment.end.y
        px, py = point.x, point.y
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            distance = np.sqrt((px - x1)**2 + (py - y1)**2)
        else:
            numerator = abs(dx * (y1 - py) - (x1 - px) * dy)
            denominator = np.sqrt(dx * dx + dy * dy)
            distance = numerator / denominator
        
        errors.append(distance)
    
    if not errors:
        return 0.0
    
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def generate_triptych_image(original_points, noisy_points, segments, w_error, sigma, output_path):
    """
    Génère une image triptyque pour une configuration donnée.
    
    Args:
        original_points: Liste de points originaux (clean)
        noisy_points: Liste de points bruités
        segments: Liste de segments compressés
        w_error: Paramètre de compression
        sigma: Niveau de bruit
        output_path: Chemin de sauvegarde
    """
    import matplotlib.patches as mpatches
    
    x_clean = [p.x for p in original_points]
    y_clean = [p.y for p in original_points]
    x_noisy = [p.x for p in noisy_points]
    y_noisy = [p.y for p in noisy_points]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Robustesse MDL : w_error={w_error}, σ={sigma:.1f}\n'
        f'Match {MATCH_ID} - Joueur {PLAYER_ID}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Panneau 1: Original
    ax1 = axes[0]
    ax1.plot(x_clean, y_clean, 
            color='blue', linewidth=1.5, alpha=0.8,
            marker='o', markersize=4, markerfacecolor='blue', 
            markeredgecolor='darkblue', markeredgewidth=0.5)
    ax1.set_title('1. Trajectoire Originale (Clean)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_facecolor('#f9f9ff')
    
    # Panneau 2: Bruit
    ax2 = axes[1]
    ax2.plot(x_noisy, y_noisy, 
            color='gray', linewidth=0.8, alpha=0.5, linestyle='-')
    ax2.plot(x_noisy, y_noisy, 
            color='gray', linewidth=0, alpha=0.7,
            marker='x', markersize=6, markeredgewidth=1.5)
    ax2.set_title(f'2. Injection de Bruit (σ={sigma:.1f})', 
                 fontsize=13, fontweight='bold', pad=10, color='darkred')
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_facecolor('#fff9f9')
    
    # Panneau 3: MDL
    ax3 = axes[2]
    ax3.plot(x_noisy, y_noisy, 
            color='lightgray', linewidth=0.6, alpha=0.4, linestyle='-', zorder=1)
    ax3.plot(x_noisy, y_noisy, 
            color='lightgray', linewidth=0, alpha=0.35,
            marker='x', markersize=4, markeredgewidth=0.8, zorder=2)
    
    for segment in segments:
        x_seg = [segment.start.x, segment.end.x]
        y_seg = [segment.start.y, segment.end.y]
        ax3.plot(x_seg, y_seg, color='red', linewidth=2.5, alpha=0.9, zorder=10)
        ax3.plot(x_seg, y_seg, 'o', color='red', markersize=8,
                markeredgecolor='darkred', markeredgewidth=1.5, zorder=11)
    
    ax3.set_title(f'3. Résultat MDL (w={w_error})', 
                 fontsize=13, fontweight='bold', pad=10, color='darkgreen')
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Y', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_facecolor('#f9fff9')
    
    # Alignement des axes
    all_x = x_clean + x_noisy
    all_y = y_clean + y_noisy
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# FONCTION WORKER POUR PARALLÉLISATION
# =============================================================================

def process_single_test(args):
    """
    Worker function pour traiter un seul test (w_error, sigma).
    
    Args:
        args: Tuple (w_error, sigma, original_points, image_type)
              image_type: None, 'var_w', ou 'var_sigma'
    
    Returns:
        dict: Résultats du test
    """
    w_error, sigma, original_points, image_type = args
    
    # 1. Générer trajectoire bruitée
    noisy_points = add_gaussian_noise(original_points, sigma=sigma)
    trajectory_noisy = Trajectory(points=noisy_points)
    
    # 2. Compresser avec MDL
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments = compressor.compress_player_trajectory(trajectory_noisy)
    
    # 3. Calculer métriques
    rmse = calculate_rmse_vs_original(original_points, segments)
    nb_segments = len(segments)
    compression_ratio = (1 - nb_segments / len(original_points)) * 100
    
    # 4. Générer image selon le type
    image_path = None
    if image_type == 'var_w':
        # Variation de w_error (sigma fixe)
        w_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
        image_path = IMAGES_VAR_W_DIR / f"w{w_str}_sigma{FIXED_SIGMA_FOR_W_VARIATION}.png"
        generate_triptych_image(
            original_points, noisy_points, segments,
            w_error, sigma, image_path
        )
    elif image_type == 'var_sigma':
        # Variation de sigma (w_error fixe)
        sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
        image_path = IMAGES_VAR_SIGMA_DIR / f"w{FIXED_W_FOR_SIGMA_VARIATION}_sigma{sigma_str}.png"
        generate_triptych_image(
            original_points, noisy_points, segments,
            w_error, sigma, image_path
        )
    
    return {
        'w_error': w_error,
        'sigma': sigma,
        'nb_segments': nb_segments,
        'rmse': rmse,
        'compression_ratio': compression_ratio,
        'nb_original_points': len(original_points),
        'image_generated': image_path is not None
    }


# =============================================================================
# BENCHMARK PRINCIPAL
# =============================================================================

def run_full_benchmark():
    """Exécute le benchmark exhaustif sur toutes les combinaisons de paramètres."""
    
    print("=" * 80)
    print("BENCHMARK SCIENTIFIQUE EXHAUSTIF : ROBUSTESSE MDL")
    print("=" * 80)
    print(f"Match: {MATCH_ID}")
    print(f"Joueur: {PLAYER_ID}")
    print(f"Fenêtre temporelle: ticks {TICK_START}-{TICK_END}")
    print(f"w_error values: {W_ERROR_VALUES}")
    print(f"sigma values ({len(SIGMA_VALUES)}): {SIGMA_VALUES}")
    print(f"Total tests: {len(W_ERROR_VALUES) * len(SIGMA_VALUES)}")
    print()
    
    # Charger trajectoire originale
    print("📁 Chargement de la trajectoire originale...")
    if not CSV_PATH.exists():
        print(f"❌ Fichier introuvable: {CSV_PATH}")
        return
    
    trajectory_clean = load_player_trajectory(CSV_PATH, PLAYER_ID, TICK_START, TICK_END)
    original_points = trajectory_clean.points
    
    print(f"   Points chargés: {len(original_points)}")
    print()
    
    # Préparer toutes les tâches
    print("🔧 Préparation des tâches...")
    tasks = []
    for w_error in W_ERROR_VALUES:
        for sigma in SIGMA_VALUES:
            # Déterminer si on génère une image
            image_type = None
            
            # Série 1: w_error varie, sigma fixe
            if abs(sigma - FIXED_SIGMA_FOR_W_VARIATION) < 0.01:
                image_type = 'var_w'
            
            # Série 2: sigma varie, w_error fixe
            elif abs(w_error - FIXED_W_FOR_SIGMA_VARIATION) < 0.01:
                image_type = 'var_sigma'
            
            tasks.append((w_error, sigma, original_points, image_type))
    
    print(f"   Total tâches: {len(tasks)}")
    
    # Compter les images
    nb_var_w = sum(1 for w in W_ERROR_VALUES)
    nb_var_sigma = sum(1 for s in SIGMA_VALUES)
    print(f"   Images variation w_error (σ={FIXED_SIGMA_FOR_W_VARIATION}): {nb_var_w}")
    print(f"   Images variation sigma (w={FIXED_W_FOR_SIGMA_VARIATION}): {nb_var_sigma}")
    
    # Déterminer le nombre de workers
    num_workers = max(1, cpu_count() - 2)
    print(f"   Workers: {num_workers}")
    print()
    
    # Exécution parallèle
    print("🚀 Exécution parallèle...")
    results = []
    
    with Pool(processes=num_workers) as pool:
        # Traiter par batch pour afficher la progression
        batch_size = num_workers * 2
        total_processed = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = pool.map(process_single_test, batch)
            results.extend(batch_results)
            
            total_processed += len(batch)
            progress = (total_processed / len(tasks)) * 100
            
            # Afficher stats du dernier résultat
            if batch_results:
                last = batch_results[-1]
                print(f"   [{total_processed}/{len(tasks)}] {progress:.1f}% | "
                      f"w={last['w_error']:.1f} σ={last['sigma']:.1f} | "
                      f"RMSE={last['rmse']:.2f} Seg={last['nb_segments']}")
    
    print()
    print(f"✅ {len(results)} tests terminés")
    
    print()
    print("=" * 80)
    print("📊 ANALYSE DES RÉSULTATS")
    print("=" * 80)
    
    # Créer DataFrame
    df_results = pd.DataFrame(results)
    
    # Sauvegarder CSV
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"✅ CSV sauvegardé: {RESULTS_CSV}")
    print(f"   Lignes: {len(df_results)}")
    print()
    
    # Statistiques globales
    print("📈 Statistiques Globales:")
    print(f"   RMSE moyen: {df_results['rmse'].mean():.2f}")
    print(f"   RMSE max: {df_results['rmse'].max():.2f}")
    print(f"   Segments moyen: {df_results['nb_segments'].mean():.1f}")
    print(f"   Compression moyenne: {df_results['compression_ratio'].mean():.1f}%")
    print()
    
    # Générer graphiques de synthèse
    print("🎨 Génération des graphiques de synthèse...")
    generate_synthesis_plots(df_results)
    
    print()
    print("=" * 80)
    print("✅ BENCHMARK TERMINÉ")
    print("=" * 80)
    print(f"📁 Dossier de sortie: {OUTPUT_DIR}")
    print(f"   - Statistiques: {STATS_DIR}")
    print(f"   - Images variation w_error: {IMAGES_VAR_W_DIR}")
    print(f"   - Images variation sigma: {IMAGES_VAR_SIGMA_DIR}")


# =============================================================================
# VISUALISATIONS FINALES
# =============================================================================

def generate_synthesis_plots(df):
    """
    Génère les 2 graphiques de synthèse finaux.
    
    Args:
        df: DataFrame avec les résultats du benchmark
    """
    # Configuration matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Palette de couleurs
    colors = plt.cm.tab10(np.linspace(0, 1, len(W_ERROR_VALUES)))
    
    # ---------------------------------------------------------------------------
    # GRAPHIQUE 1: Résistance au Bruit (RMSE vs Sigma)
    # ---------------------------------------------------------------------------
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    for idx, w_error in enumerate(W_ERROR_VALUES):
        df_w = df[df['w_error'] == w_error]
        ax1.plot(df_w['sigma'], df_w['rmse'], 
                marker='o', linewidth=2, markersize=6,
                label=f'w_error = {w_error}', alpha=0.8,
                color=colors[idx])
    
    ax1.set_xlabel('Niveau de Bruit (σ)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE (Erreur de Reconstruction)', fontsize=14, fontweight='bold')
    ax1.set_title('Résistance au Bruit : Impact du Bruit Gaussien sur la Fidélité MDL\n'
                  '(RMSE calculé vs Trajectoire Originale Clean)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Paramètre MDL', fontsize=11, title_fontsize=12, 
              loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Annotations
    ax1.axhline(y=df['rmse'].mean(), color='red', linestyle='--', 
               linewidth=1, alpha=0.5, label='RMSE moyen global')
    ax1.text(df['sigma'].max() * 0.95, df['rmse'].mean() * 1.05, 
            f'Moyenne: {df["rmse"].mean():.2f}',
            ha='right', va='bottom', fontsize=10, color='red')
    
    plt.tight_layout()
    resistance_path = STATS_DIR / "resistance_au_bruit.png"
    fig1.savefig(resistance_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    print(f"   ✅ {resistance_path.name}")
    
    # ---------------------------------------------------------------------------
    # GRAPHIQUE 2: Efficacité de Compression (Segments vs Sigma)
    # ---------------------------------------------------------------------------
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for idx, w_error in enumerate(W_ERROR_VALUES):
        df_w = df[df['w_error'] == w_error]
        ax2.plot(df_w['sigma'], df_w['nb_segments'], 
                marker='s', linewidth=2, markersize=6,
                label=f'w_error = {w_error}', alpha=0.8,
                color=colors[idx])
    
    ax2.set_xlabel('Niveau de Bruit (σ)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Nombre de Segments', fontsize=14, fontweight='bold')
    ax2.set_title('Efficacité de Compression : Stabilité du Nombre de Segments\n'
                  '(Vérification que le bruit ne fait pas exploser la complexité)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Paramètre MDL', fontsize=11, title_fontsize=12,
              loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Ligne de référence (nombre de points originaux)
    nb_original = df['nb_original_points'].iloc[0]
    ax2.axhline(y=nb_original, color='orange', linestyle=':', 
               linewidth=2, alpha=0.7, label='Nb points originaux')
    ax2.text(df['sigma'].max() * 0.95, nb_original * 1.05, 
            f'Original: {nb_original} points',
            ha='right', va='bottom', fontsize=10, color='orange')
    
    plt.tight_layout()
    efficacite_path = STATS_DIR / "efficacite_compression.png"
    fig2.savefig(efficacite_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"   ✅ {efficacite_path.name}")
    
    # ---------------------------------------------------------------------------
    # GRAPHIQUE BONUS: Heatmap 2D (w_error vs sigma pour RMSE)
    # ---------------------------------------------------------------------------
    
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    # Pivoter pour heatmap
    pivot_rmse = df.pivot(index='w_error', columns='sigma', values='rmse')
    
    # Créer heatmap avec imshow
    im = ax3.imshow(pivot_rmse.values, cmap='YlOrRd', aspect='auto', 
                    interpolation='nearest', origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('RMSE', fontsize=12, fontweight='bold')
    
    # Configurer les axes
    ax3.set_xticks(np.arange(len(pivot_rmse.columns)))
    ax3.set_yticks(np.arange(len(pivot_rmse.index)))
    ax3.set_xticklabels([f'{s:.1f}' for s in pivot_rmse.columns], rotation=45, ha='right')
    ax3.set_yticklabels([f'{w}' for w in pivot_rmse.index])
    
    ax3.set_xlabel('Niveau de Bruit (σ)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Paramètre MDL (w_error)', fontsize=14, fontweight='bold')
    ax3.set_title('Carte de Chaleur : RMSE en fonction de w_error et σ\n'
                  '(Plus foncé = Plus d\'erreur)',
                  fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    heatmap_path = STATS_DIR / "heatmap_rmse.png"
    fig3.savefig(heatmap_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    print(f"   ✅ {heatmap_path.name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    run_full_benchmark()
