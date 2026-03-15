"""Expérimentation 1 (Quantitative) : Benchmark mathématique MDL vs Baselines.

Version parallèle — exploite tous les cœurs CPU disponibles.

Métriques calculées :
- RMSE (Root Mean Square Error) : erreur quadratique moyenne de reconstruction
- Distance de Hausdorff : erreur maximale garantie
- Taux de préservation des arrêts (%) : fidélité sémantique

Usage :
    from scripts.compression.experiment_baseline_maths import run_baseline_maths
    run_baseline_maths()
"""

import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.compression import DouglasPeuckerCompressor, MDLCompressor
from dota_analytics.metrics import (
    hausdorff_distance,
    rmse_segments_to_points,
    stop_preservation_rate,
)
from dota_analytics.structures import Segment, Trajectory, TrajectoryPoint

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = BASE_DIR / "data-dota"
PLAYER_ID = 0
OUTPUT_DIR = BASE_DIR / "output" / "benchmark_matrix"
OUTPUT_CSV = OUTPUT_DIR / "baseline_maths_stats.csv"
OUTPUT_PNG = OUTPUT_DIR / "baseline_maths_boxplots.png"

N_SAMPLES = 1000       # Nombre d'échantillons valides cibles
WINDOW_SIZE = 2000     # Fenêtre temporelle en ticks
W_ERROR_MDL = 12.0
MIN_POINTS = 20
MIN_SEGMENTS = 2
RANDOM_SEED = 42
N_WORKERS = max(1, os.cpu_count() - 1)

ALGO_COLORS = {
    "Uniforme": "darkorange",
    "Douglas-Peucker": "forestgreen",
    "MDL": "crimson",
}
ALGO_ORDER = ["Uniforme", "Douglas-Peucker", "MDL"]

# =============================================================================
# DONNÉES GLOBALES — accessibles dans les workers via fork (Linux COW)
# =============================================================================

_GLOBAL_DATA: dict = {}


def _init_worker(shared_data: dict) -> None:
    """Initialise chaque worker avec les DataFrames partagés.

    Appelé une fois par worker à la création du Pool. Injecte les données
    dans le global du worker, compatible avec fork ET spawn.
    """
    global _GLOBAL_DATA
    _GLOBAL_DATA = shared_data


# =============================================================================
# FONCTIONS WORKERS (niveau module → picklables)
# =============================================================================


def _extract_trajectory(df: pd.DataFrame, tick_start: int, tick_end: int) -> Optional[Trajectory]:
    x_col, y_col = f"x{PLAYER_ID}", f"y{PLAYER_ID}"
    if x_col not in df.columns or y_col not in df.columns:
        return None
    mask = (
        (df["tick"] >= tick_start)
        & (df["tick"] <= tick_end)
        & ((df[x_col] != 0.0) | (df[y_col] != 0.0))
    )
    sub = df[mask].sort_values("tick")
    if len(sub) < MIN_POINTS:
        return None
    points = [
        TrajectoryPoint(x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"]))
        for _, r in sub.iterrows()
    ]
    return Trajectory(points=points)


def _uniform_sampling(points: list, n_segments: int) -> list:
    n_pts = len(points)
    step = max(1, (n_pts - 1) // n_segments)
    indices = list(range(0, n_pts, step))
    if indices[-1] != n_pts - 1:
        indices.append(n_pts - 1)
    while len(indices) - 1 > n_segments and len(indices) > 2:
        indices.pop(-2)
    indices = sorted(set(indices))
    return [
        Segment(start=points[indices[i]], end=points[indices[i + 1]])
        for i in range(len(indices) - 1)
    ]


def _find_dp_epsilon(trajectory: Trajectory, target_n: int, tolerance: int = 2) -> list:
    lo, hi = 0.01, 5000.0
    best_segs: list = []
    best_diff = float("inf")
    for _ in range(40):
        mid = (lo + hi) / 2
        segs = DouglasPeuckerCompressor(epsilon=mid).compress_player_trajectory(trajectory)
        diff = abs(len(segs) - target_n)
        if diff < best_diff:
            best_diff = diff
            best_segs = segs
        if diff <= tolerance:
            break
        elif len(segs) > target_n:
            lo = mid
        else:
            hi = mid
    return best_segs


def _process_task(task: tuple) -> Optional[list]:
    """Worker : traite une fenêtre et retourne les métriques pour les 3 algos.

    Accède aux DataFrames via _GLOBAL_DATA (fork COW sur Linux).

    Args:
        task: (sample_id, match_id, tick_start)

    Returns:
        Liste de dicts (1 par algo) ou None si fenêtre invalide
    """
    sample_id, match_id, tick_start = task
    tick_end = tick_start + WINDOW_SIZE

    df = _GLOBAL_DATA.get(match_id)
    if df is None:
        return None

    traj = _extract_trajectory(df, tick_start, tick_end)
    if traj is None:
        return None

    pts = traj.points

    mdl = MDLCompressor(w_error=W_ERROR_MDL, verbose=False)
    segs_mdl = mdl.compress_player_trajectory(traj)
    N = len(segs_mdl)
    if N < MIN_SEGMENTS:
        return None

    segs_uniform = _uniform_sampling(pts, N)
    segs_dp = _find_dp_epsilon(traj, N)

    records = []
    for algo_name, segs in [
        ("MDL", segs_mdl),
        ("Uniforme", segs_uniform),
        ("Douglas-Peucker", segs_dp),
    ]:
        if not segs:
            continue
        records.append({
            "Sample_ID": sample_id,
            "Match_ID": match_id,
            "Tick_Start": tick_start,
            "Tick_End": tick_end,
            "N_Points": len(pts),
            "N_Segments": len(segs),
            "Algorithm": algo_name,
            "RMSE": rmse_segments_to_points(pts, segs),
            "Hausdorff": hausdorff_distance(pts, segs),
            "Stop_Preservation": stop_preservation_rate(pts, segs),
        })

    return records if records else None


# =============================================================================
# VISUALISATION
# =============================================================================


def _generate_boxplots(df_results: pd.DataFrame, n_valid: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Expérimentation 1 (Quantitative) : MDL vs Baselines — {n_valid} échantillons valides\n"
        f"Joueur {PLAYER_ID} | Fenêtres {WINDOW_SIZE} ticks | "
        f"w_error={W_ERROR_MDL} | {N_WORKERS} workers parallèles",
        fontsize=12, fontweight="bold", y=1.02,
    )

    metrics_cfg = [
        {
            "col": "RMSE",
            "title": "① RMSE\n(Plus bas = meilleur)",
            "ylabel": "RMSE (unités de carte)",
            "ax": axes[0],
            "note": "↓ Fidélité quadratique moyenne",
        },
        {
            "col": "Hausdorff",
            "title": "② Distance de Hausdorff\n(Plus bas = meilleur)",
            "ylabel": "Hausdorff (unités de carte)",
            "ax": axes[1],
            "note": "↓ Erreur maximale garantie",
        },
        {
            "col": "Stop_Preservation",
            "title": "③ Préservation des Arrêts\n(Plus haut = meilleur)",
            "ylabel": "Taux de préservation (%)",
            "ax": axes[2],
            "note": "↑ Fidélité sémantique",
        },
    ]

    for cfg in metrics_cfg:
        ax = cfg["ax"]
        col = cfg["col"]

        data_by_algo = []
        labels = []
        colors = []
        for algo in ALGO_ORDER:
            sub = df_results[df_results["Algorithm"] == algo][col].dropna()
            if len(sub) > 0:
                data_by_algo.append(sub.values)
                labels.append(algo)
                colors.append(ALGO_COLORS[algo])

        if not data_by_algo:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data_by_algo,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 2.5},
            whiskerprops={"linewidth": 1.4},
            capprops={"linewidth": 1.4},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.2, "linestyle": "none"},
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

        # Jitter limité à 400 points pour la lisibilité
        for i, (vals, color) in enumerate(zip(data_by_algo, colors)):
            sample = vals if len(vals) <= 400 else np.random.choice(vals, 400, replace=False)
            jitter = np.random.uniform(-0.2, 0.2, size=len(sample))
            ax.scatter(
                np.full(len(sample), i + 1) + jitter, sample,
                color=color, alpha=0.15, s=8, zorder=3, edgecolors="none",
            )

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        ax.text(
            0.5, -0.12, cfg["note"],
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, style="italic", color="gray",
        )

        for i, vals in enumerate(data_by_algo):
            med = np.median(vals)
            ax.text(
                i + 1.28, med, f"{med:.2f}",
                ha="left", va="center",
                fontsize=8.5, color="black", fontweight="bold",
            )
            ax.text(
                i + 1, ax.get_ylim()[0], f"n={len(vals)}",
                ha="center", va="bottom", fontsize=7.5, color="gray",
            )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"📊 Boxplots sauvegardés → {OUTPUT_PNG}")


# =============================================================================
# CHARGEMENT
# =============================================================================


def _load_all_csv() -> dict:
    """Charge TOUS les fichiers CSV du dossier data-dota."""
    files = sorted(DATA_DIR.glob("coord_*.csv"))
    result = {}
    for f in files:
        match_id = f.stem.replace("coord_", "")
        try:
            result[match_id] = pd.read_csv(f)
        except Exception:
            pass
    return result


def _build_candidates(all_data: dict) -> list:
    """Construit le pool de fenêtres candidates (match_id, tick_start).

    Stride = WINDOW_SIZE // 2 pour des fenêtres semi-chevauchantes.
    """
    stride = max(100, WINDOW_SIZE // 2)
    candidates = []
    for match_id, df in all_data.items():
        ticks_all = sorted(df["tick"].unique())
        tick_min_val = int(ticks_all[0])
        tick_max_val = int(ticks_all[-1])
        if tick_max_val - tick_min_val < WINDOW_SIZE:
            continue
        for t in range(tick_min_val, tick_max_val - WINDOW_SIZE + 1, stride):
            candidates.append((match_id, t))
    return candidates


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_baseline_maths() -> pd.DataFrame:
    """Benchmark mathématique MDL vs Uniforme vs Douglas-Peucker (parallèle).

    Charge tous les CSV, génère N_SAMPLES fenêtres valides en parallèle avec
    multiprocessing.Pool, calcule les 3 métriques, exporte CSV + boxplots.

    Returns:
        DataFrame pandas contenant tous les résultats bruts
    """
    global _GLOBAL_DATA

    t0 = time.time()
    print("=" * 70)
    print("EXPÉRIMENTATION 1 (QUANTITATIVE) : BENCHMARK MATHÉMATIQUE PARALLÈLE")
    print("=" * 70)
    print(f"Source       : {DATA_DIR.name}/ (tous les matchs)")
    print(f"Référence    : MDL w_error={W_ERROR_MDL}")
    print(f"Échantillons : {N_SAMPLES} fenêtres × {WINDOW_SIZE} ticks")
    print(f"Parallélisme : {N_WORKERS} workers / {os.cpu_count()} cœurs CPU")
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    print("📁 Chargement de tous les fichiers CSV...")
    _GLOBAL_DATA = _load_all_csv()
    n_files = len(_GLOBAL_DATA)
    total_rows = sum(len(df) for df in _GLOBAL_DATA.values())
    print(f"   {n_files} matchs | {total_rows:,} lignes au total")

    candidates = _build_candidates(_GLOBAL_DATA)
    random.shuffle(candidates)
    print(f"   {len(candidates):,} fenêtres candidates (stride={WINDOW_SIZE // 2} ticks)")

    tasks = [
        (i, match_id, tick_start)
        for i, (match_id, tick_start) in enumerate(candidates)
    ]

    # ── Traitement parallèle ──────────────────────────────────────────────────
    print(f"\n⚙️  Traitement en parallèle ({N_WORKERS} workers)...")

    all_records: list = []
    n_valid = 0
    n_skipped = 0
    last_pct = -1

    with mp.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(_GLOBAL_DATA,),
    ) as pool:
        for result in pool.imap_unordered(_process_task, tasks, chunksize=8):
            if result is None:
                n_skipped += 1
            else:
                for row in result:
                    row["Sample_ID"] = n_valid
                all_records.extend(result)
                n_valid += 1

                pct = n_valid * 100 // N_SAMPLES
                if pct // 5 > last_pct // 5:
                    elapsed = time.time() - t0
                    rate = n_valid / elapsed if elapsed > 0 else 0
                    eta = (N_SAMPLES - n_valid) / rate if rate > 0 else 0
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    print(
                        f"   [{n_valid:4d}/{N_SAMPLES}] {bar} {pct:3d}% "
                        f"| {rate:.1f} éch/s | ETA {eta:.0f}s"
                    )
                    last_pct = pct

            if n_valid >= N_SAMPLES:
                pool.terminate()
                break

    elapsed_total = time.time() - t0
    print(f"\n   ✅ {n_valid} valides | {n_skipped} rejetés | "
          f"durée {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")

    # ── Export CSV ────────────────────────────────────────────────────────────
    if not all_records:
        print("\n❌ Aucun échantillon valide collecté. Vérifier MIN_POINTS, WINDOW_SIZE et les données.")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_records)
    df_results.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
    print(f"\n💾 CSV sauvegardé → {OUTPUT_CSV}")
    print(f"   {len(df_results):,} lignes ({n_valid} × 3 algorithmes)")

    # ── Résumé statistique ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ STATISTIQUE")
    print("=" * 70)
    fmt = "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("Algorithme", "RMSE med", "RMSE std", "Hd med", "Hd std", "Stop%"))
    print("  " + "-" * 68)
    for algo in ALGO_ORDER:
        sub = df_results[df_results["Algorithm"] == algo]
        rm = sub["RMSE"].median()
        rs = sub["RMSE"].std()
        hm = sub["Hausdorff"].median()
        hs = sub["Hausdorff"].std()
        sp = sub["Stop_Preservation"].dropna()
        spr = f"{sp.median():.1f}%" if len(sp) > 0 else "N/A"
        print(fmt.format(algo, f"{rm:.3f}", f"±{rs:.3f}", f"{hm:.3f}", f"±{hs:.3f}", spr))

    print()
    print("💡 Interprétation :")
    print("   RMSE / Hausdorff : Plus bas = meilleur")
    print("   Stop%            : Plus haut = meilleur (fidélité sémantique)")

    # ── Boxplots ──────────────────────────────────────────────────────────────
    print()
    print("🎨 Génération des boxplots...")
    _generate_boxplots(df_results, n_valid)

    print()
    print("=" * 70)
    print(f"✅ Benchmark terminé en {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    print(f"   Sorties : {OUTPUT_DIR}/")
    print("=" * 70)

    return df_results
