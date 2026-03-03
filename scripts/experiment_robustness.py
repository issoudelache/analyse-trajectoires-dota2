#!/usr/bin/env python3
"""Expérience de robustesse au bruit (Jitter Test).

Ce script évalue la stabilité de la compression MDL face au bruit de mesure,
simulant le "tremblement de souris" ou les imprécisions de capture.

Objectif:
    Démontrer que l'algorithme MDL extrait la trajectoire "réelle" même quand
    les données brutes contiennent du bruit aléatoire.

Méthodologie:
    1. Charger une trajectoire de référence (Joueur 0, Match exemple)
    2. Générer 4 versions bruitées avec sigma croissant: [0.0, 0.5, 1.0, 3.0]
    3. Compresser chaque version avec w_error = 12.0
    4. Comparer visuellement:
       - Nuage de points gris (trajectoire bruitée)
       - Ligne rouge (segments compressés)
    5. Observer que la ligne rouge reste stable malgré le bruit

Output:
    - output/robustness_test.png : 4 sous-graphiques montrant l'effet du bruit
"""

import sys
from pathlib import Path

# Ajouter le parent au path pour imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import matplotlib.pyplot as plt

from dota_analytics.structures import Trajectory, TrajectoryPoint
from dota_analytics.compression import MDLCompressor
from dota_analytics.metrics import add_gaussian_noise


def load_player_trajectory(csv_path: Path, player_id: int = 0) -> Trajectory:
    """Charge la trajectoire d'un joueur depuis un CSV."""
    df = pd.read_csv(csv_path)

    x_col, y_col = f"x{player_id}", f"y{player_id}"

    if x_col not in df.columns:
        raise ValueError(f"Colonnes {x_col}/{y_col} introuvables")

    # Filtrer points valides (non nuls)
    mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
    valid_rows = df[mask]

    points = []
    for _, row in valid_rows.iterrows():
        point = TrajectoryPoint(
            x=float(row[x_col]), y=float(row[y_col]), tick=int(row["tick"])
        )
        points.append(point)

    return Trajectory(points=points, player_id=player_id)


def experiment_robustness(
    csv_path: Path, output_dir: Path, player_id: int = 0, w_error: float = 12.0
):
    """Exécute l'expérience de robustesse au bruit.

    Args:
        csv_path: Chemin vers le CSV du match
        output_dir: Dossier de sortie
        player_id: ID du joueur à analyser (défaut: 0)
        w_error: Paramètre de compression (défaut: 12.0)
    """
    print("=" * 70)
    print("🧪 EXPÉRIENCE DE ROBUSTESSE - TEST DE STABILITÉ AU BRUIT")
    print("=" * 70)
    print("Paramètres:")
    print(f"  • Match: {csv_path.stem}")
    print(f"  • Joueur: {player_id}")
    print(f"  • w_error: {w_error}")
    print()

    # Charger trajectoire de référence
    print("📂 Chargement de la trajectoire...")
    trajectory = load_player_trajectory(csv_path, player_id)
    print(f"   Points chargés: {len(trajectory)}")
    print()

    # Définir niveaux de bruit
    sigmas = [0.0, 0.5, 1.0, 3.0]

    print("🔬 Génération des variantes bruitées...")
    print(f"   Niveaux de bruit (sigma): {sigmas}")
    print()

    # Compresseur
    compressor = MDLCompressor(w_error=w_error, verbose=False)

    # Résultats
    results = []

    for sigma in sigmas:
        print(f"  • Sigma = {sigma:.1f}...", end=" ")

        # Générer trajectoire bruitée
        noisy_points = add_gaussian_noise(trajectory.points, sigma)
        noisy_trajectory = Trajectory(points=noisy_points, player_id=player_id)

        # Compresser
        segments = compressor.compress_player_trajectory(noisy_trajectory)

        print(f"→ {len(segments)} segments")

        results.append(
            {
                "sigma": sigma,
                "noisy_trajectory": noisy_trajectory,
                "segments": segments,
                "num_segments": len(segments),
            }
        )

    print()
    print("=" * 70)
    print("📊 RÉSULTATS")
    print("=" * 70)

    # Tableau récapitulatif
    print(f"{'Sigma':<10} {'Segments':<12} {'Variation':<15}")
    print("-" * 40)

    baseline_segments = results[0]["num_segments"]

    for res in results:
        sigma = res["sigma"]
        num_seg = res["num_segments"]
        variation = num_seg - baseline_segments
        variation_pct = (
            (variation / baseline_segments) * 100 if baseline_segments > 0 else 0
        )

        print(f"{sigma:<10.1f} {num_seg:<12} {variation:+4d} ({variation_pct:+.1f}%)")

    print()
    print("💡 Interprétation:")
    print("   Une faible variation du nombre de segments indique que l'algorithme")
    print("   est robuste au bruit et extrait la structure sous-jacente stable.")
    print()

    # Générer visualisation
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_robustness_plot(results, output_dir, w_error, csv_path.stem)


def generate_robustness_plot(
    results: list, output_dir: Path, w_error: float, match_id: str
):
    """Génère le graphique de comparaison visuelle du bruit.

    Args:
        results: Liste des résultats pour chaque sigma
        output_dir: Dossier de sortie
        w_error: Valeur w_error utilisée
        match_id: ID du match
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Test de Robustesse au Bruit - Match {match_id} (w_error={w_error})",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten()

    for idx, res in enumerate(results):
        ax = axes[idx]
        sigma = res["sigma"]
        noisy_traj = res["noisy_trajectory"]
        segments = res["segments"]

        # Points bruités (nuage gris)
        x_noisy = [p.x for p in noisy_traj.points]
        y_noisy = [p.y for p in noisy_traj.points]

        # Downsampling pour lisibilité (1 point sur 5)
        step = max(1, len(x_noisy) // 500)

        ax.scatter(
            x_noisy[::step],
            y_noisy[::step],
            c="gray",
            s=3,
            alpha=0.4,
            label="Trajectoire bruitée",
        )

        # Segments compressés (ligne rouge stable)
        for seg in segments:
            ax.plot(
                [seg.start.x, seg.end.x],
                [seg.start.y, seg.end.y],
                color="red",
                linewidth=2.5,
                alpha=0.8,
            )

        # Points de jonction (rouges)
        xs = [seg.start.x for seg in segments] + (
            [segments[-1].end.x] if segments else []
        )
        ys = [seg.start.y for seg in segments] + (
            [segments[-1].end.y] if segments else []
        )
        ax.scatter(
            xs,
            ys,
            c="red",
            s=40,
            zorder=10,
            edgecolors="darkred",
            linewidth=1.5,
            label="Segments compressés",
        )

        # Configuration
        noise_label = "Aucun bruit" if sigma == 0.0 else f"Bruit σ={sigma}"
        ax.set_title(
            f"{noise_label} → {len(segments)} segments", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("X (coordonnées carte)", fontsize=10)
        ax.set_ylabel("Y (coordonnées carte)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.legend(loc="upper right", fontsize=9)

        # Ajouter annotation
        textstr = f"Segments: {len(segments)}\nBruit: σ={sigma}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

    plt.tight_layout()

    # Sauvegarder
    output_path = output_dir / "robustness_test.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"📊 Graphique sauvegardé: {output_path}")


if __name__ == "__main__":
    DATA_DIR = BASE_DIR / "data-dota"
    OUTPUT_DIR = BASE_DIR / "output"

    # Match de référence
    MATCH_ID = "3841665963"
    csv_path = DATA_DIR / f"coord_{MATCH_ID}.csv"

    if not csv_path.exists():
        print(f"❌ Fichier introuvable: {csv_path}")
        print(f"💡 Assurez-vous que le match {MATCH_ID} existe dans data-dota/")
        sys.exit(1)

    experiment_robustness(csv_path, OUTPUT_DIR, player_id=0, w_error=12.0)
