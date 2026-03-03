"""Expérimentation 1 : Comparaison MDL vs Méthodes Classiques.

Compare MDL avec Échantillonnage Uniforme et Douglas-Peucker à taux de
compression égal pour prouver la supériorité sémantique de MDL.

Usage (depuis run.py ou un notebook):
    from scripts.experiment_baseline import run_baseline_comparison
    run_baseline_comparison()
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.compression import DouglasPeuckerCompressor, MDLCompressor
from dota_analytics.structures import Segment, Trajectory, TrajectoryPoint


# =============================================================================
# CONFIGURATION
# =============================================================================

MATCH_ID = "3841665963"
PLAYER_ID = 0
TICK_START = 40000
TICK_END = 43000
CSV_PATH = BASE_DIR / "data-dota" / f"coord_{MATCH_ID}.csv"
OUTPUT_PATH = BASE_DIR / "output" / "benchmark_matrix" / "baseline_comparison.png"
W_ERROR_MDL = 12.0


# =============================================================================
# CHARGEMENT
# =============================================================================


def _load_window(tick_start: int, tick_end: int) -> Trajectory:
    """Charge la trajectoire du joueur sur la fenêtre temporelle."""
    df = pd.read_csv(CSV_PATH)
    x_col, y_col = f"x{PLAYER_ID}", f"y{PLAYER_ID}"
    mask = (
        (df["tick"] >= tick_start)
        & (df["tick"] <= tick_end)
        & ((df[x_col] != 0.0) | (df[y_col] != 0.0))
    )
    sub = df[mask].sort_values("tick")
    points = [
        TrajectoryPoint(x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"]))
        for _, r in sub.iterrows()
    ]
    return Trajectory(points=points)


def _find_good_window() -> Trajectory:
    """
    Cherche automatiquement une fenêtre de 3000 ticks avec ≥ 30 points
    et un déplacement significatif (amplitude ≥ 50 unités).
    Commence par essayer TICK_START/TICK_END, puis élargit si besoin.
    """
    for start in range(TICK_START, 100000, 1000):
        end = start + 3000
        try:
            traj = _load_window(start, end)
        except Exception:
            continue
        if len(traj.points) < 30:
            continue
        xs = [p.x for p in traj.points]
        ys = [p.y for p in traj.points]
        amplitude = max(max(xs) - min(xs), max(ys) - min(ys))
        if amplitude >= 50:
            print(f"   Fenêtre sélectionnée : ticks {start}–{end} ({len(traj.points)} points, amplitude={amplitude:.0f})")
            return traj
    raise ValueError("Aucune fenêtre avec suffisamment de déplacement trouvée")


# =============================================================================
# UNIFORM SAMPLING
# =============================================================================


def _uniform_sampling(points: list, n_segments: int) -> list[Segment]:
    """
    Garde 1 point sur K pour produire exactement n_segments segments.

    Args:
        points: Liste de TrajectoryPoint
        n_segments: Nombre de segments cible

    Returns:
        Liste de Segment
    """
    n_pts = len(points)
    # On veut n_segments+1 points clés → K = pas entre chaque
    step = max(1, (n_pts - 1) // n_segments)
    indices = list(range(0, n_pts, step))

    # Forcer le dernier point
    if indices[-1] != n_pts - 1:
        indices.append(n_pts - 1)

    # Ajuster pour coller exactement à n_segments
    while len(indices) - 1 > n_segments and len(indices) > 2:
        indices.pop(-2)
    while len(indices) - 1 < n_segments and len(indices) < n_pts:
        mid = (indices[-2] + indices[-1]) // 2
        if mid not in indices:
            indices.insert(-1, mid)
        else:
            break
    indices = sorted(set(indices))

    return [
        Segment(start=points[indices[i]], end=points[indices[i + 1]])
        for i in range(len(indices) - 1)
    ]


# =============================================================================
# DOUGLAS-PEUCKER (AJUSTEMENT EPSILON AUTOMATIQUE)
# =============================================================================


def _find_dp_epsilon(trajectory: Trajectory, target_n: int, tolerance: int = 2) -> list[Segment]:
    """
    Cherche empiriquement epsilon pour que Douglas-Peucker produise
    environ target_n segments (à ± tolerance près).

    Args:
        trajectory: Trajectoire à compresser
        target_n: Nombre de segments cible
        tolerance: Écart acceptable

    Returns:
        Liste de Segment la plus proche de target_n
    """
    lo, hi = 0.01, 500.0
    best_segs = None
    best_diff = float("inf")

    for _ in range(40):  # Dichotomie sur 40 itérations
        mid = (lo + hi) / 2
        compressor = DouglasPeuckerCompressor(epsilon=mid)
        segs = compressor.compress_player_trajectory(trajectory)
        diff = abs(len(segs) - target_n)

        if diff < best_diff:
            best_diff = diff
            best_segs = segs

        if diff <= tolerance:
            break
        elif len(segs) > target_n:
            lo = mid  # Trop de segments → epsilon plus grand
        else:
            hi = mid  # Pas assez → epsilon plus petit

    return best_segs


# =============================================================================
# VISUALISATION
# =============================================================================


def _draw_segments(ax, segments: list[Segment], color: str, label: str, zorder: int = 5):
    """Dessine les segments sur un axe matplotlib."""
    for i, seg in enumerate(segments):
        ax.plot(
            [seg.start.x, seg.end.x],
            [seg.start.y, seg.end.y],
            color=color, linewidth=2.0, alpha=0.9, zorder=zorder,
            label=label if i == 0 else None,
        )
        ax.plot(
            [seg.start.x, seg.end.x],
            [seg.start.y, seg.end.y],
            "o", color=color, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=zorder + 1,
        )


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_baseline_comparison():
    """
    Expérimentation 1 : Comparaison MDL vs Échantillonnage Uniforme vs Douglas-Peucker.

    À taux de compression égal, prouve la supériorité sémantique de MDL.
    Sauvegarde la figure dans output/benchmark_matrix/baseline_comparison.png.
    """
    print("=" * 70)
    print("EXPÉRIMENTATION 1 : COMPARAISON MDL vs BASELINES")
    print("=" * 70)
    print(f"Match: {MATCH_ID} | Joueur: {PLAYER_ID} | w_error MDL: {W_ERROR_MDL}")
    print()

    # 1. Chargement
    print("📁 Chargement de la trajectoire...")
    trajectory = _find_good_window()
    points = trajectory.points
    n_pts = len(points)
    xs_raw = [p.x for p in points]
    ys_raw = [p.y for p in points]
    print()

    # 2. MDL (référence)
    print("⚙️  Compression MDL (référence)...")
    mdl = MDLCompressor(w_error=W_ERROR_MDL, verbose=False)
    segs_mdl = mdl.compress_player_trajectory(trajectory)
    N = len(segs_mdl)
    ratio = (1 - N / n_pts) * 100
    print(f"   → {N} segments ({ratio:.1f}% compression)")
    print()

    # 3. Uniform Sampling
    print(f"⚙️  Échantillonnage Uniforme (cible: {N} segments)...")
    segs_uniform = _uniform_sampling(points, N)
    print(f"   → {len(segs_uniform)} segments")
    print()

    # 4. Douglas-Peucker
    print(f"⚙️  Douglas-Peucker (recherche epsilon pour {N} segments)...")
    segs_dp = _find_dp_epsilon(trajectory, N)
    print(f"   → {len(segs_dp)} segments")
    print()

    # 5. Figure 1×4
    print("🎨 Génération de la figure comparative...")

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 6), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.05}
    )
    fig.suptitle(
        f"Expérimentation 1 : Comparaison à Taux de Compression Égal (~{ratio:.0f}%)\n"
        f"Match {MATCH_ID} – Joueur {PLAYER_ID}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    bg_kwargs = dict(color="lightgray", linewidth=0.6, alpha=0.5, zorder=1)
    pt_kwargs = dict(color="lightgray", marker=".", linewidth=0, markersize=4, alpha=0.4, zorder=2)

    # --- Panneau 1 : Vérité terrain ---
    ax = axes[0]
    ax.plot(xs_raw, ys_raw, color="steelblue", linewidth=1.4, alpha=0.85, zorder=3)
    ax.plot(xs_raw, ys_raw, "o", color="steelblue", markersize=3.5,
            markeredgecolor="navy", markeredgewidth=0.4, alpha=0.7, zorder=4)
    ax.set_title(f"① Original\n({n_pts} points)", fontsize=12, fontweight="bold", pad=8)
    ax.set_facecolor("#f0f4ff")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)

    # --- Panneau 2 : Uniforme ---
    ax = axes[1]
    ax.plot(xs_raw, ys_raw, **bg_kwargs)
    ax.plot(xs_raw, ys_raw, **pt_kwargs)
    _draw_segments(ax, segs_uniform, color="darkorange", label="Uniforme")
    ax.set_title(f"② Échantillonnage Uniforme\n({len(segs_uniform)} segments)", fontsize=12, fontweight="bold", pad=8)
    ax.set_facecolor("#fff8f0")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)

    # --- Panneau 3 : Douglas-Peucker ---
    ax = axes[2]
    ax.plot(xs_raw, ys_raw, **bg_kwargs)
    ax.plot(xs_raw, ys_raw, **pt_kwargs)
    _draw_segments(ax, segs_dp, color="forestgreen", label="Douglas-Peucker")
    ax.set_title(f"③ Douglas-Peucker\n({len(segs_dp)} segments)", fontsize=12, fontweight="bold", pad=8)
    ax.set_facecolor("#f0fff4")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)

    # --- Panneau 4 : MDL ---
    ax = axes[3]
    ax.plot(xs_raw, ys_raw, **bg_kwargs)
    ax.plot(xs_raw, ys_raw, **pt_kwargs)
    _draw_segments(ax, segs_mdl, color="crimson", label="MDL")
    ax.set_title(f"④ MDL (w={W_ERROR_MDL})\n({N} segments)", fontsize=12, fontweight="bold", pad=8)
    ax.set_facecolor("#fff0f0")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)

    # Marges uniformes
    margin_x = (max(xs_raw) - min(xs_raw)) * 0.07
    margin_y = (max(ys_raw) - min(ys_raw)) * 0.07
    axes[0].set_xlim(min(xs_raw) - margin_x, max(xs_raw) + margin_x)
    axes[0].set_ylim(min(ys_raw) - margin_y, max(ys_raw) + margin_y)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Figure sauvegardée → {OUTPUT_PATH}")
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"  Points originaux    : {n_pts}")
    print(f"  Segments MDL        : {N}   (w_error={W_ERROR_MDL})")
    print(f"  Segments Uniforme   : {len(segs_uniform)}")
    print(f"  Segments DP         : {len(segs_dp)}")
    print(f"  Taux de compression : ~{ratio:.1f}%")
    print()
    print("💡 Interprétation attendue :")
    print("   - Uniforme : segments réguliers, ignore la géométrie réelle")
    print("   - Douglas-Peucker : préserve les virages, supprime les droites")
    print("   - MDL : segments sémantiquement cohérents (phases de déplacement)")
