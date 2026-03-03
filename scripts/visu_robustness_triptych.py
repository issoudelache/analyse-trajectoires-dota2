#!/usr/bin/env python3
"""
Script de visualisation pour soutenance : Triptych de robustesse MDL

Génère une figure avec 3 panneaux horizontaux illustrant le processus de
filtrage du bruit par l'algorithme MDL :
1. Trajectoire Originale (Clean)
2. Injection de Bruit (σ=3.0)
3. Résultat MDL (Filtrage)

Usage:
    python scripts/visu_robustness_triptych.py [--w_error 12.0] [--sigma 3.0]
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

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

DATA_DIR = BASE_DIR / "data-dota"
OUTPUT_DIR = BASE_DIR / "output" / "bruit_gaussien"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / f"coord_{MATCH_ID}.csv"


# =============================================================================
# FONCTIONS
# =============================================================================


def load_trajectory_window(csv_path, player_id, tick_start, tick_end):
    """
    Charge une trajectoire pour un joueur sur une fenêtre temporelle.

    Args:
        csv_path: Chemin vers le fichier CSV
        player_id: ID du joueur (0-9)
        tick_start: Tick de début
        tick_end: Tick de fin

    Returns:
        Trajectory: Trajectoire extraite
    """
    print(f"📁 Chargement: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filtrer par joueur et fenêtre temporelle
    x_col = f"x{player_id}"
    y_col = f"y{player_id}"

    # Filtrer les points non-nuls dans la fenêtre
    mask = (
        (df["tick"] >= tick_start)
        & (df["tick"] <= tick_end)
        & ((df[x_col] != 0.0) | (df[y_col] != 0.0))
    )

    window_df = df[mask].sort_values("tick")

    print(f"   Joueur {player_id}, ticks {tick_start}-{tick_end}")
    print(f"   Points extraits: {len(window_df)}")

    # Créer la trajectoire
    points = []
    for _, row in window_df.iterrows():
        point = TrajectoryPoint(
            x=float(row[x_col]), y=float(row[y_col]), tick=int(row["tick"])
        )
        points.append(point)

    return Trajectory(points=points)


def compress_trajectory(trajectory, w_error):
    """
    Compresse une trajectoire avec MDL.

    Args:
        trajectory: Trajectory à compresser
        w_error: Paramètre de compression

    Returns:
        list[Segment]: Liste de segments compressés
    """
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments = compressor.compress_player_trajectory(trajectory)

    print(f"   Compression: {len(trajectory.points)} points → {len(segments)} segments")
    print(f"   Taux: {100 - len(segments) / len(trajectory.points) * 100:.1f}%")

    return segments


# =============================================================================
# GÉNÉRATION DU TRIPTYCH
# =============================================================================


def generate_triptych(w_error, sigma):
    """Génère la figure triptyque illustrant le processus de robustesse."""

    # Nom du fichier avec paramètres
    w_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
    output_path = OUTPUT_DIR / f"triptych_w{w_str}_sigma{sigma_str}.png"

    print("=" * 70)
    print("GÉNÉRATION TRIPTYCH DE ROBUSTESSE")
    print("=" * 70)
    print(f"Match: {MATCH_ID}")
    print(f"Joueur: {PLAYER_ID}")
    print(f"Fenêtre: ticks {TICK_START}-{TICK_END}")
    print(f"Paramètres: w_error={w_error}, σ={sigma}")
    print()

    # Vérifier le fichier
    if not CSV_PATH.exists():
        print(f"❌ Fichier introuvable: {CSV_PATH}")
        return

    # ÉTAPE 1: Charger trajectoire originale
    print("ÉTAPE 1: Chargement trajectoire originale")
    trajectory_clean = load_trajectory_window(CSV_PATH, PLAYER_ID, TICK_START, TICK_END)

    if len(trajectory_clean.points) == 0:
        print("❌ Aucun point dans la fenêtre temporelle")
        return

    # Extraire coordonnées
    x_clean = [p.x for p in trajectory_clean.points]
    y_clean = [p.y for p in trajectory_clean.points]

    print()

    # ÉTAPE 2: Ajouter bruit
    print("ÉTAPE 2: Injection de bruit gaussien")
    noisy_points = add_gaussian_noise(trajectory_clean.points, sigma=sigma)
    trajectory_noisy = Trajectory(points=noisy_points)

    x_noisy = [p.x for p in noisy_points]
    y_noisy = [p.y for p in noisy_points]

    print()

    # ÉTAPE 3: Compresser trajectoire bruitée
    print("ÉTAPE 3: Compression MDL sur données bruitées")
    segments = compress_trajectory(trajectory_noisy, w_error)

    print()

    # =============================================================================
    # CRÉATION DE LA FIGURE
    # =============================================================================

    print("🎨 Création de la figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Robustesse de l'algorithme MDL face au bruit\n"
        f"Match {MATCH_ID} - Joueur {PLAYER_ID} - Ticks {TICK_START}-{TICK_END}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ---------------------------------------------------------------------------
    # PANNEAU 1: Trajectoire Originale (Clean)
    # ---------------------------------------------------------------------------

    ax1 = axes[0]

    ax1.plot(
        x_clean,
        y_clean,
        color="blue",
        linewidth=1.5,
        alpha=0.8,
        marker="o",
        markersize=4,
        markerfacecolor="blue",
        markeredgecolor="darkblue",
        markeredgewidth=0.5,
    )

    ax1.set_title(
        "1. Trajectoire Originale (Clean)", fontsize=13, fontweight="bold", pad=10
    )
    ax1.set_xlabel("X (coordonnées carte)", fontsize=11)
    ax1.set_ylabel("Y (coordonnées carte)", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_facecolor("#f9f9ff")

    # Légende
    clean_patch = mpatches.Patch(color="blue", label=f"{len(x_clean)} points")
    ax1.legend(handles=[clean_patch], loc="upper right", fontsize=9)

    # ---------------------------------------------------------------------------
    # PANNEAU 2: Injection de Bruit
    # ---------------------------------------------------------------------------

    ax2 = axes[1]

    # Ligne continue reliant les points bruités (gribouillage)
    ax2.plot(
        x_noisy,
        y_noisy,
        color="gray",
        linewidth=0.8,
        alpha=0.5,
        linestyle="-",
        label="Trajectoire Tremblante",
    )

    # Marqueurs par-dessus pour montrer le bruit
    ax2.plot(
        x_noisy,
        y_noisy,
        color="gray",
        linewidth=0,
        alpha=0.7,
        marker="x",
        markersize=6,
        markeredgewidth=1.5,
    )

    ax2.set_title(
        f"2. Injection de Bruit (σ={sigma})",
        fontsize=13,
        fontweight="bold",
        pad=10,
        color="darkred",
    )
    ax2.set_xlabel("X (coordonnées carte)", fontsize=11)
    ax2.set_ylabel("Y (coordonnées carte)", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_facecolor("#fff9f9")

    # Légende
    noisy_patch = mpatches.Patch(color="gray", label=f"Bruit gaussien σ={sigma}")
    ax2.legend(handles=[noisy_patch], loc="upper right", fontsize=9)

    # ---------------------------------------------------------------------------
    # PANNEAU 3: Résultat MDL (Reconstruction)
    # ---------------------------------------------------------------------------

    ax3 = axes[2]

    # Fond: Points bruités (gris visible)
    ax3.plot(
        x_noisy,
        y_noisy,
        color="lightgray",
        linewidth=0.6,
        alpha=0.4,
        linestyle="-",
        zorder=1,
    )
    ax3.plot(
        x_noisy,
        y_noisy,
        color="lightgray",
        linewidth=0,
        alpha=0.35,
        marker="x",
        markersize=4,
        markeredgewidth=0.8,
        label="Données bruitées",
        zorder=2,
    )

    # Premier plan: Segments MDL (rouge)
    for i, segment in enumerate(segments):
        x_seg = [segment.start.x, segment.end.x]
        y_seg = [segment.start.y, segment.end.y]

        # Ligne
        ax3.plot(x_seg, y_seg, color="red", linewidth=2.5, alpha=0.9, zorder=10)

        # Points aux jonctions
        ax3.plot(
            x_seg,
            y_seg,
            "o",
            color="red",
            markersize=8,
            markeredgecolor="darkred",
            markeredgewidth=1.5,
            zorder=11,
        )

    ax3.set_title(
        f"3. Résultat MDL (Filtrage w={w_error})",
        fontsize=13,
        fontweight="bold",
        pad=10,
        color="darkgreen",
    )
    ax3.set_xlabel("X (coordonnées carte)", fontsize=11)
    ax3.set_ylabel("Y (coordonnées carte)", fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_facecolor("#f9fff9")

    # Légende
    mdl_patch = mpatches.Patch(color="red", label=f"{len(segments)} segments MDL")
    noise_patch = mpatches.Patch(color="lightgray", label="Bruit de fond")
    ax3.legend(handles=[mdl_patch, noise_patch], loc="upper right", fontsize=9)

    # ---------------------------------------------------------------------------
    # ALIGNEMENT DES AXES
    # ---------------------------------------------------------------------------

    # Trouver les limites communes pour tous les graphes
    all_x = x_clean + x_noisy
    all_y = y_clean + y_noisy

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Ajouter une marge de 5%
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # ---------------------------------------------------------------------------
    # ANNOTATIONS (Flèches entre panneaux)
    # ---------------------------------------------------------------------------

    # Ajouter des annotations textuelles entre les panneaux
    fig.text(0.33, 0.92, "→", fontsize=30, ha="center", va="center", color="gray")
    fig.text(0.66, 0.92, "→", fontsize=30, ha="center", va="center", color="gray")

    # ---------------------------------------------------------------------------
    # SAUVEGARDE
    # ---------------------------------------------------------------------------

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    size_kb = output_path.stat().st_size // 1024

    print()
    print("=" * 70)
    print("✅ TRIPTYCH GÉNÉRÉ")
    print("=" * 70)
    print(f"📁 Fichier: {output_path}")
    print(f"💾 Taille: {size_kb} KB")
    print()
    print("Résumé:")
    print(f"  - Panneau 1: {len(x_clean)} points originaux (bleu)")
    print(f"  - Panneau 2: {len(x_noisy)} points bruités (gris, σ={sigma})")
    print(f"  - Panneau 3: {len(segments)} segments MDL (rouge, w={w_error})")
    print(f"  - Taux de compression: {100 - len(segments) / len(x_clean) * 100:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génération du triptych de robustesse pour soutenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--w_error",
        type=float,
        default=12.0,
        help="Paramètre de compression MDL (défaut: 12.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Écart-type du bruit gaussien (défaut: 3.0)",
    )

    args = parser.parse_args()

    generate_triptych(w_error=args.w_error, sigma=args.sigma)
