#!/usr/bin/env python3
"""
Benchmark Scientifique Avancé : Robustesse MDL avec Data Augmentation

Amélioration du benchmark initial avec :
1. Data Augmentation : 5 fenêtres aléatoires de 1000 ticks
2. Métriques robustes : RMSE Clean (débruitage) et Stabilité des segments
3. Visualisation Heatmaps pour analyse 2D des paramètres

Auteurs: Équipe Analyse de Trajectoires Dota 2
Date: Février 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass

# Ajout du dossier parent au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dota_analytics.structures import Trajectory, TrajectoryPoint
from dota_analytics.compression import MDLCompressor
from dota_analytics.metrics import add_gaussian_noise


# ============================================================================
# CONFIGURATION
# ============================================================================

MATCH_ID = "3841665963"
PLAYER_ID = 0
CSV_PATH = PROJECT_ROOT / "data-dota" / f"coord_{MATCH_ID}.csv"

# Paramètres de test (réduits pour convergence scientifique)
W_ERROR_VALUES = [1.0, 5.0, 10.0, 12.0, 15.0, 25.0, 50.0]
SIGMA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Stratégie d'échantillonnage
N_SAMPLES = 5  # Nombre de fenêtres aléatoires
SAMPLE_DURATION_TICKS = 1000  # Taille de chaque fenêtre

# Dossiers de sortie
OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmark_matrix"
STATS_DIR = OUTPUT_DIR / "stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Résultat d'un test individuel"""
    w_error: float
    sigma: float
    sample_id: int
    nb_segments_clean: int
    nb_segments_noisy: int
    rmse_clean: float
    segment_stability: float
    nb_original_points: int


# ============================================================================
# CHARGEMENT ET ÉCHANTILLONNAGE
# ============================================================================

def load_full_trajectory(csv_path: Path, player_id: int) -> Trajectory:
    """
    Charge la trajectoire complète d'un joueur
    
    Args:
        csv_path: Chemin vers le fichier CSV
        player_id: ID du joueur (0-9)
    
    Returns:
        Trajectory object avec tous les points
    """
    df = pd.read_csv(csv_path)
    
    # Les colonnes sont x0, y0, x1, y1, ..., x9, y9
    x_col = f'x{player_id}'
    y_col = f'y{player_id}'
    
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Colonnes {x_col} ou {y_col} non trouvées")
    
    # Filtrer les points non-nuls
    mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
    player_data = df[mask].copy()
    
    if player_data.empty:
        raise ValueError(f"Aucune donnée pour le joueur {player_id}")
    
    player_data = player_data.sort_values('tick')
    
    points = [
        TrajectoryPoint(x=float(row[x_col]), y=float(row[y_col]), tick=int(row['tick']))
        for _, row in player_data.iterrows()
    ]
    
    return Trajectory(points=points)


def get_random_samples(
    trajectory: Trajectory, 
    n_samples: int = 5, 
    duration_ticks: int = 1000
) -> List[Trajectory]:
    """
    Extrait n_samples fenêtres aléatoires de duration_ticks ticks
    
    Cette fonction implémente une stratégie de Data Augmentation pour
    éviter le biais d'un seul échantillon et capturer la variabilité
    des situations de jeu (Laning, Jungle, Teamfight, etc.)
    
    Args:
        trajectory: Trajectoire complète
        n_samples: Nombre d'échantillons à extraire
        duration_ticks: Durée en ticks de chaque échantillon
    
    Returns:
        Liste de n_samples trajectoires
    """
    if len(trajectory.points) < duration_ticks:
        raise ValueError(f"Trajectoire trop courte: {len(trajectory.points)} < {duration_ticks}")
    
    samples = []
    tick_min = trajectory.points[0].tick
    tick_max = trajectory.points[-1].tick
    
    # Sélection aléatoire de points de départ
    random.seed(42)  # Pour reproductibilité
    
    for i in range(n_samples):
        # Trouver un tick de départ valide
        max_start_tick = tick_max - duration_ticks
        start_tick = random.randint(tick_min, max_start_tick)
        end_tick = start_tick + duration_ticks
        
        # Extraire les points dans cette fenêtre
        sample_points = [
            p for p in trajectory.points
            if start_tick <= p.tick < end_tick
        ]
        
        if len(sample_points) >= 10:  # Minimum 10 points pour être valide
            samples.append(Trajectory(points=sample_points))
        else:
            # Réessayer avec une autre fenêtre
            print(f"⚠️ Échantillon {i+1} invalide ({len(sample_points)} points), réessai...")
            # Augmenter la durée ou chercher une autre zone
            for _ in range(10):
                start_tick = random.randint(tick_min, max_start_tick)
                end_tick = start_tick + duration_ticks
                sample_points = [
                    p for p in trajectory.points
                    if start_tick <= p.tick < end_tick
                ]
                if len(sample_points) >= 10:
                    samples.append(Trajectory(points=sample_points))
                    break
    
    if len(samples) < n_samples:
        print(f"⚠️ Seulement {len(samples)}/{n_samples} échantillons valides trouvés")
    
    return samples


# ============================================================================
# MÉTRIQUES AMÉLIORÉES
# ============================================================================

def calculate_rmse_clean(
    original_points: List[TrajectoryPoint],
    segments: List[Segment]
) -> float:
    """
    Calcule le RMSE entre les segments compressés (issus du bruit)
    et les points ORIGINAUX (sans bruit).
    
    C'est la métrique de "Débruitage" : mesure la capacité de MDL
    à reconstruire fidèlement la trajectoire originale malgré le bruit.
    
    Args:
        original_points: Points originaux (sans bruit)
        segments: Segments MDL compressés (issus de données bruitées)
    
    Returns:
        RMSE entre segments et original
    """
    # Créer un mapping tick -> segment pour recherche rapide
    # Chaque segment couvre les ticks de start à end
    segment_map = {}
    for segment in segments:
        start_tick = segment.start.tick
        end_tick = segment.end.tick
        # Assigner tous les ticks de ce range au segment
        for tick in range(start_tick, end_tick + 1):
            segment_map[tick] = segment
    
    squared_errors = []
    
    for orig_point in original_points:
        if orig_point.tick in segment_map:
            segment = segment_map[orig_point.tick]
            
            # Interpolation linéaire sur le segment
            t1 = segment.start.tick
            t2 = segment.end.tick
            
            if t2 == t1:
                # Segment ponctuel
                reconstructed_x = segment.start.x
                reconstructed_y = segment.start.y
            else:
                # Interpolation linéaire
                alpha = (orig_point.tick - t1) / (t2 - t1)
                reconstructed_x = segment.start.x + alpha * (segment.end.x - segment.start.x)
                reconstructed_y = segment.start.y + alpha * (segment.end.y - segment.start.y)
            
            error = np.sqrt((orig_point.x - reconstructed_x)**2 + (orig_point.y - reconstructed_y)**2)
            squared_errors.append(error**2)
    
    if not squared_errors:
        return 0.0
    
    return np.sqrt(np.mean(squared_errors))


def calculate_segment_stability(nb_clean: int, nb_noisy: int) -> float:
    """
    Calcule la variation du nombre de segments par rapport à la version sans bruit.
    
    Stabilité = |nb_noisy - nb_clean| / nb_clean * 100
    
    Une faible stabilité (proche de 0%) signifie que le bruit n'affecte pas
    significativement la structure de compression.
    
    Args:
        nb_clean: Nombre de segments sans bruit
        nb_noisy: Nombre de segments avec bruit
    
    Returns:
        Variation en %
    """
    if nb_clean == 0:
        return 0.0
    
    return abs(nb_noisy - nb_clean) / nb_clean * 100.0


# ============================================================================
# EXÉCUTION DES TESTS
# ============================================================================

def run_single_test(
    original_points: List[TrajectoryPoint],
    w_error: float,
    sigma: float,
    sample_id: int
) -> BenchmarkResult:
    """
    Exécute un test pour une configuration donnée
    
    Args:
        original_points: Points originaux (sans bruit)
        w_error: Paramètre MDL
        sigma: Niveau de bruit gaussien
        sample_id: Identifiant de l'échantillon
    
    Returns:
        BenchmarkResult avec toutes les métriques
    """
    # 1. Compression de la version CLEAN (référence)
    trajectory_clean = Trajectory(points=original_points)
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments_clean = compressor.compress_player_trajectory(trajectory_clean)
    nb_segments_clean = len(segments_clean)
    
    # 2. Ajout du bruit
    noisy_points = add_gaussian_noise(original_points, sigma)
    trajectory_noisy = Trajectory(points=noisy_points)
    
    # 3. Compression de la version BRUITÉE
    segments_noisy = compressor.compress_player_trajectory(trajectory_noisy)
    nb_segments_noisy = len(segments_noisy)
    
    # 4. Calcul des métriques
    rmse_clean = calculate_rmse_clean(original_points, segments_noisy)
    segment_stability = calculate_segment_stability(nb_segments_clean, nb_segments_noisy)
    
    return BenchmarkResult(
        w_error=w_error,
        sigma=sigma,
        sample_id=sample_id,
        nb_segments_clean=nb_segments_clean,
        nb_segments_noisy=nb_segments_noisy,
        rmse_clean=rmse_clean,
        segment_stability=segment_stability,
        nb_original_points=len(original_points)
    )


def run_benchmark(samples: List[Trajectory]) -> pd.DataFrame:
    """
    Exécute le benchmark complet sur tous les échantillons
    
    Args:
        samples: Liste des trajectoires échantillonnées
    
    Returns:
        DataFrame avec tous les résultats
    """
    results = []
    total_tests = len(W_ERROR_VALUES) * len(SIGMA_VALUES) * len(samples)
    completed = 0
    
    print(f"\n🚀 Exécution du benchmark avancé...")
    print(f"   Total tests: {total_tests}")
    print(f"   Configurations: {len(W_ERROR_VALUES)} w_error × {len(SIGMA_VALUES)} sigma × {len(samples)} échantillons")
    print()
    
    for w in W_ERROR_VALUES:
        for sigma in SIGMA_VALUES:
            for sample_id, sample in enumerate(samples):
                result = run_single_test(sample.points, w, sigma, sample_id)
                results.append(result)
                
                completed += 1
                if completed % 10 == 0 or completed == total_tests:
                    progress = completed / total_tests * 100
                    print(f"   [{completed}/{total_tests}] {progress:.1f}% | w={w} σ={sigma} sample={sample_id} | RMSE={result.rmse_clean:.2f} Stab={result.segment_stability:.1f}%")
    
    # Convertir en DataFrame
    df = pd.DataFrame([
        {
            'w_error': r.w_error,
            'sigma': r.sigma,
            'sample_id': r.sample_id,
            'nb_segments_clean': r.nb_segments_clean,
            'nb_segments_noisy': r.nb_segments_noisy,
            'rmse_clean': r.rmse_clean,
            'segment_stability': r.segment_stability,
            'nb_original_points': r.nb_original_points
        }
        for r in results
    ])
    
    return df


# ============================================================================
# AGRÉGATION DES RÉSULTATS
# ============================================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les résultats en moyennant sur les échantillons
    
    Pour chaque configuration (w_error, sigma), calcule :
    - Moyenne et écart-type de toutes les métriques
    
    Args:
        df: DataFrame avec résultats bruts
    
    Returns:
        DataFrame agrégé avec moyennes et écarts-types
    """
    agg_df = df.groupby(['w_error', 'sigma']).agg({
        'nb_segments_clean': ['mean', 'std'],
        'nb_segments_noisy': ['mean', 'std'],
        'rmse_clean': ['mean', 'std'],
        'segment_stability': ['mean', 'std'],
        'nb_original_points': 'mean'
    }).reset_index()
    
    # Aplatir les colonnes multi-index
    agg_df.columns = [
        'w_error', 'sigma',
        'nb_segments_clean_mean', 'nb_segments_clean_std',
        'nb_segments_noisy_mean', 'nb_segments_noisy_std',
        'rmse_clean_mean', 'rmse_clean_std',
        'segment_stability_mean', 'segment_stability_std',
        'nb_original_points'
    ]
    
    return agg_df


# ============================================================================
# VISUALISATION : HEATMAPS
# ============================================================================

def generate_heatmaps(df_agg: pd.DataFrame):
    """
    Génère les 2 heatmaps scientifiques
    
    Heatmap 1 (Stabilité) : Nombre de segments moyen (w_error × sigma)
    Heatmap 2 (Fidélité) : RMSE Clean moyen (w_error × sigma)
    
    Args:
        df_agg: DataFrame agrégé
    """
    # Préparer les données pour heatmap (pivot)
    pivot_segments = df_agg.pivot(
        index='w_error', 
        columns='sigma', 
        values='nb_segments_noisy_mean'
    )
    
    pivot_rmse = df_agg.pivot(
        index='w_error', 
        columns='sigma', 
        values='rmse_clean_mean'
    )
    
    # === HEATMAP 1 : STABILITÉ (Nombre de segments) ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im1 = ax.imshow(
        pivot_segments.values,
        aspect='auto',
        cmap='YlOrRd',  # Jaune → Orange → Rouge
        interpolation='nearest'
    )
    
    # Configuration des axes
    ax.set_xticks(range(len(pivot_segments.columns)))
    ax.set_yticks(range(len(pivot_segments.index)))
    ax.set_xticklabels([f'{s:.1f}' for s in pivot_segments.columns])
    ax.set_yticklabels([f'{w:.1f}' for w in pivot_segments.index])
    
    ax.set_xlabel('Sigma (Niveau de bruit)', fontsize=12, fontweight='bold')
    ax.set_ylabel('w_error (Sensibilité MDL)', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap 1 : Stabilité Structurelle\n(Nombre de segments moyen)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('Nb segments', fontsize=11, fontweight='bold')
    
    # Annotations des valeurs
    for i in range(len(pivot_segments.index)):
        for j in range(len(pivot_segments.columns)):
            value = pivot_segments.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > pivot_segments.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{value:.0f}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path_1 = STATS_DIR / 'heatmap_stabilite.png'
    plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {output_path_1.name}")
    
    # === HEATMAP 2 : FIDÉLITÉ (RMSE Clean) ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im2 = ax.imshow(
        pivot_rmse.values,
        aspect='auto',
        cmap='RdYlGn_r',  # Rouge (mauvais) → Jaune → Vert (bon)
        interpolation='nearest'
    )
    
    # Configuration des axes
    ax.set_xticks(range(len(pivot_rmse.columns)))
    ax.set_yticks(range(len(pivot_rmse.index)))
    ax.set_xticklabels([f'{s:.1f}' for s in pivot_rmse.columns])
    ax.set_yticklabels([f'{w:.1f}' for w in pivot_rmse.index])
    
    ax.set_xlabel('Sigma (Niveau de bruit)', fontsize=12, fontweight='bold')
    ax.set_ylabel('w_error (Sensibilité MDL)', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap 2 : Fidélité au Signal Original\n(RMSE Clean - Métrique de Débruitage)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('RMSE Clean', fontsize=11, fontweight='bold')
    
    # Annotations des valeurs
    for i in range(len(pivot_rmse.index)):
        for j in range(len(pivot_rmse.columns)):
            value = pivot_rmse.values[i, j]
            if not np.isnan(value):
                # Colorier le texte selon l'intensité
                text_color = 'white' if value > pivot_rmse.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{value:.1f}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Marquer la zone optimale (w=12)
    w_12_idx = list(pivot_rmse.index).index(12.0) if 12.0 in pivot_rmse.index else None
    if w_12_idx is not None:
        ax.axhline(y=w_12_idx, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='w=12 (optimal)')
        ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    output_path_2 = STATS_DIR / 'heatmap_fidelite.png'
    plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {output_path_2.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("BENCHMARK SCIENTIFIQUE AVANCÉ : ROBUSTESSE MDL + DATA AUGMENTATION")
    print("=" * 80)
    print(f"Match: {MATCH_ID}")
    print(f"Joueur: {PLAYER_ID}")
    print(f"Échantillonnage: {N_SAMPLES} fenêtres de {SAMPLE_DURATION_TICKS} ticks")
    print(f"w_error values: {W_ERROR_VALUES}")
    print(f"sigma values: {SIGMA_VALUES}")
    print()
    
    # 1. Chargement de la trajectoire complète
    print("📁 Chargement de la trajectoire complète...")
    full_trajectory = load_full_trajectory(CSV_PATH, PLAYER_ID)
    print(f"   Points totaux: {len(full_trajectory.points)}")
    print(f"   Tick range: {full_trajectory.points[0].tick} → {full_trajectory.points[-1].tick}")
    print()
    
    # 2. Échantillonnage aléatoire (Data Augmentation)
    print("🎲 Échantillonnage aléatoire (Data Augmentation)...")
    samples = get_random_samples(full_trajectory, N_SAMPLES, SAMPLE_DURATION_TICKS)
    print(f"   Échantillons créés: {len(samples)}")
    for i, sample in enumerate(samples):
        tick_range = f"{sample.points[0].tick}-{sample.points[-1].tick}"
        print(f"   Sample {i}: {len(sample.points)} points (ticks {tick_range})")
    print()
    
    # 3. Exécution du benchmark
    df_raw = run_benchmark(samples)
    
    # 4. Agrégation des résultats
    print("\n📊 Agrégation des résultats (moyenne sur échantillons)...")
    df_agg = aggregate_results(df_raw)
    print(f"   Configurations testées: {len(df_agg)}")
    print()
    
    # 5. Sauvegarde CSV
    output_csv = STATS_DIR / 'advanced_stats.csv'
    df_agg.to_csv(output_csv, index=False)
    print(f"✅ CSV sauvegardé: {output_csv}")
    print(f"   Colonnes: {list(df_agg.columns)}")
    print()
    
    # 6. Statistiques globales
    print("=" * 80)
    print("📈 STATISTIQUES GLOBALES")
    print("=" * 80)
    print(f"RMSE Clean moyen: {df_agg['rmse_clean_mean'].mean():.2f} ± {df_agg['rmse_clean_std'].mean():.2f}")
    print(f"Stabilité moyenne: {df_agg['segment_stability_mean'].mean():.1f}% ± {df_agg['segment_stability_std'].mean():.1f}%")
    print(f"Segments (noisy) moyen: {df_agg['nb_segments_noisy_mean'].mean():.1f} ± {df_agg['nb_segments_noisy_std'].mean():.1f}")
    print()
    
    # Zone optimale (w=12)
    if 12.0 in df_agg['w_error'].values:
        df_w12 = df_agg[df_agg['w_error'] == 12.0]
        print("🎯 ZONE OPTIMALE (w=12):")
        print(f"   RMSE Clean moyen: {df_w12['rmse_clean_mean'].mean():.2f}")
        print(f"   Stabilité moyenne: {df_w12['segment_stability_mean'].mean():.1f}%")
        print(f"   Segments moyen: {df_w12['nb_segments_noisy_mean'].mean():.1f}")
        print()
    
    # 7. Génération des heatmaps
    print("🎨 Génération des heatmaps...")
    generate_heatmaps(df_agg)
    print()
    
    # 8. Résumé final
    print("=" * 80)
    print("✅ BENCHMARK AVANCÉ TERMINÉ")
    print("=" * 80)
    print(f"📁 Dossier de sortie: {STATS_DIR}")
    print(f"   - advanced_stats.csv : Résultats agrégés")
    print(f"   - heatmap_stabilite.png : Nombre de segments (w × sigma)")
    print(f"   - heatmap_fidelite.png : RMSE Clean (w × sigma)")
    print()
    print("💡 Interprétation:")
    print("   - Zones BLEUES/VERTES sur heatmap fidélité = Bon débruitage")
    print("   - Zones JAUNES sur heatmap stabilité = Structure stable")
    print("   - La zone w=12 doit montrer une combinaison optimale")
    print()


if __name__ == "__main__":
    main()
