#!/usr/bin/env python3
"""Benchmark de qualité massive pour validation scientifique.

Ce script évalue la précision de la compression MDL sur un large échantillon
de trajectoires réelles, en mesurant l'erreur de reconstruction.

Objectif:
    Prouver que la compression MDL maintient une haute précision même sur
    un volume important de données hétérogènes (différents matchs, styles de jeu).

Méthodologie:
    1. Sélectionner 50 matchs représentatifs
    2. Compresser le Joueur 0 de chaque match avec w_error = 12.0
    3. Mesurer RMSE et Erreur Max pour chaque trajectoire
    4. Agréger les résultats dans un CSV
    5. Visualiser la distribution des erreurs

Output:
    - output/benchmark_metrics.csv : Résultats détaillés
    - output/benchmark_quality.png : Histogramme des erreurs
"""

import sys
from pathlib import Path

# Ajouter le parent au path pour imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dota_analytics.structures import Trajectory, TrajectoryPoint
from dota_analytics.compression import MDLCompressor
from dota_analytics.metrics import calculate_reconstruction_error, calculate_compression_rate


def load_player_trajectory(csv_path: Path, player_id: int = 0) -> Trajectory:
    """Charge la trajectoire d'un joueur depuis un CSV."""
    df = pd.read_csv(csv_path)
    
    x_col, y_col = f'x{player_id}', f'y{player_id}'
    
    if x_col not in df.columns:
        raise ValueError(f"Colonnes {x_col}/{y_col} introuvables")
    
    # Filtrer points valides (non nuls)
    mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
    valid_rows = df[mask]
    
    points = []
    for _, row in valid_rows.iterrows():
        point = TrajectoryPoint(
            x=float(row[x_col]),
            y=float(row[y_col]),
            tick=int(row['tick'])
        )
        points.append(point)
    
    return Trajectory(points=points, player_id=player_id)


def benchmark_quality(data_dir: Path, output_dir: Path, num_matches: int = 50, w_error: float = 12.0):
    """Exécute le benchmark de qualité sur N matchs.
    
    Args:
        data_dir: Dossier contenant les CSV (data-dota/)
        output_dir: Dossier de sortie pour résultats
        num_matches: Nombre de matchs à analyser (défaut: 50)
        w_error: Paramètre de compression (défaut: 12.0)
    """
    print("=" * 70)
    print("🔬 BENCHMARK DE QUALITÉ - VALIDATION SCIENTIFIQUE")
    print("=" * 70)
    print(f"Paramètres:")
    print(f"  • Nombre de matchs: {num_matches}")
    print(f"  • w_error: {w_error}")
    print(f"  • Joueur analysé: Joueur 0")
    print()
    
    # Lister les CSV disponibles
    csv_files = sorted(data_dir.glob("coord_*.csv"))[:num_matches]
    
    if len(csv_files) < num_matches:
        print(f"⚠️  Seulement {len(csv_files)} fichiers disponibles (demandé: {num_matches})")
    
    print(f"📊 Traitement de {len(csv_files)} matchs...")
    print()
    
    # Compresseur MDL
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    
    # Résultats
    results = []
    
    # Barre de progression
    for csv_path in tqdm(csv_files, desc="Compression", unit="match"):
        match_id = csv_path.stem.replace('coord_', '')
        
        try:
            # Charger trajectoire
            trajectory = load_player_trajectory(csv_path, player_id=0)
            
            if len(trajectory) < 10:
                # Trajectoire trop courte
                continue
            
            # Compresser
            segments = compressor.compress_player_trajectory(trajectory)
            
            # Mesurer qualité
            metrics = calculate_reconstruction_error(trajectory, segments)
            
            # Calculer taux compression
            compression_rate = calculate_compression_rate(len(trajectory), len(segments))
            
            # Stocker résultats
            results.append({
                'match_id': match_id,
                'num_original_points': len(trajectory),
                'num_segments': len(segments),
                'compression_rate': compression_rate,
                'rmse': metrics['rmse'],
                'max_error': metrics['max_error'],
                'mean_error': metrics['mean_error'],
                'w_error': w_error
            })
            
        except Exception as e:
            print(f"\n❌ Erreur sur {match_id}: {e}")
            continue
    
    # Créer DataFrame
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("❌ Aucun résultat valide")
        return
    
    # Sauvegarder CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "benchmark_metrics.csv"
    df_results.to_csv(csv_output, index=False)
    
    print()
    print("=" * 70)
    print("📈 RÉSULTATS AGRÉGÉS")
    print("=" * 70)
    print(f"Matchs analysés: {len(df_results)}")
    print()
    print("Compression:")
    print(f"  • Taux moyen: {df_results['compression_rate'].mean():.1f}%")
    print(f"  • Segments moyens: {df_results['num_segments'].mean():.0f}")
    print()
    print("Qualité (Erreur de Reconstruction):")
    print(f"  • RMSE moyen: {df_results['rmse'].mean():.3f}")
    print(f"  • RMSE médian: {df_results['rmse'].median():.3f}")
    print(f"  • RMSE max: {df_results['rmse'].max():.3f}")
    print(f"  • Erreur max globale: {df_results['max_error'].max():.3f}")
    print()
    print(f"✅ Résultats sauvegardés: {csv_output}")
    
    # Générer visualisation
    generate_quality_plot(df_results, output_dir, w_error)


def generate_quality_plot(df: pd.DataFrame, output_dir: Path, w_error: float):
    """Génère le graphique de distribution des erreurs.
    
    Args:
        df: DataFrame avec les résultats
        output_dir: Dossier de sortie
        w_error: Valeur w_error utilisée
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Benchmark Qualité - Compression MDL (w_error={w_error})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Histogramme RMSE
    ax1 = axes[0, 0]
    ax1.hist(df['rmse'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['rmse'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {df["rmse"].mean():.3f}')
    ax1.axvline(df['rmse'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Médiane: {df["rmse"].median():.3f}')
    ax1.set_xlabel('RMSE (Root Mean Square Error)', fontsize=11)
    ax1.set_ylabel('Nombre de matchs', fontsize=11)
    ax1.set_title('Distribution de l\'erreur RMSE', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Histogramme Erreur Max
    ax2 = axes[0, 1]
    ax2.hist(df['max_error'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(df['max_error'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {df["max_error"].mean():.2f}')
    ax2.set_xlabel('Erreur Maximale', fontsize=11)
    ax2.set_ylabel('Nombre de matchs', fontsize=11)
    ax2.set_title('Distribution de l\'erreur maximale', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Scatter: RMSE vs Taux de compression
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['compression_rate'], df['rmse'], 
                         c=df['num_original_points'], cmap='viridis',
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Taux de compression (%)', fontsize=11)
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Trade-off: Compression vs Précision', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Nb points originaux', fontsize=9)
    
    # 4. Box plot des erreurs
    ax4 = axes[1, 1]
    box_data = [df['rmse'], df['mean_error'], df['max_error']]
    bp = ax4.boxplot(box_data, labels=['RMSE', 'Erreur Moyenne', 'Erreur Max'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'lightgreen', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Valeur d\'erreur', fontsize=11)
    ax4.set_title('Comparaison des métriques d\'erreur', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = output_dir / "benchmark_quality.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Graphique sauvegardé: {output_path}")


if __name__ == '__main__':
    DATA_DIR = BASE_DIR / "data-dota"
    OUTPUT_DIR = BASE_DIR / "output"
    
    if not DATA_DIR.exists():
        print(f"❌ Dossier data-dota introuvable: {DATA_DIR}")
        sys.exit(1)
    
    benchmark_quality(DATA_DIR, OUTPUT_DIR, num_matches=50, w_error=12.0)
