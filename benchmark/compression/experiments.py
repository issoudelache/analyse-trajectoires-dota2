#!/usr/bin/env python3
"""
Expérimentations de compression : MDL vs Uniforme vs Douglas-Peucker

Modes :
  baseline  Exp. 1 visuelle — figure 4 panneaux comparative à taux égal
  maths     Exp. 1 quantitative — benchmark parallèle 1000 échantillons
              (RMSE, Distance de Hausdorff, Préservation des arrêts)
  time      Exp. 2 — complexité temporelle empirique sur 6 tailles

Usage :
  python scripts/compression/experiments.py --mode baseline
  python scripts/compression/experiments.py --mode maths
  python scripts/compression/experiments.py --mode time
"""

import argparse
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
# CONFIGURATION COMMUNE
# =============================================================================

DATA_DIR = BASE_DIR / "data-dota"
W_ERROR_MDL = 12.0
RANDOM_SEED = 42

ALGO_COLORS = {
    "Uniforme": "darkorange",
    "Douglas-Peucker": "forestgreen",
    "MDL": "crimson",
}
ALGO_ORDER = ["Uniforme", "Douglas-Peucker", "MDL"]

# --- Exp. 1 visuelle ---
BASELINE_MATCH_ID = "3841665963"
BASELINE_PLAYER_ID = 0
BASELINE_TICK_START = 40000
BASELINE_OUTPUT = BASE_DIR / "output" / "benchmark_matrix" / "baseline_comparison.png"

# --- Exp. 1 quantitative ---
MATHS_OUTPUT_DIR = BASE_DIR / "output" / "benchmark_matrix"
MATHS_OUTPUT_CSV = MATHS_OUTPUT_DIR / "baseline_maths_stats.csv"
MATHS_OUTPUT_PNG = MATHS_OUTPUT_DIR / "baseline_maths_boxplots.png"
MATHS_PLAYER_ID = 0
N_SAMPLES = 1000
WINDOW_SIZE = 2000
MIN_POINTS = 20
MIN_SEGMENTS = 2
N_WORKERS = max(1, os.cpu_count() - 1)

# --- Exp. 2 temps ---
TIME_OUTPUT_DIR = BASE_DIR / "output" / "benchmark_time"
TIME_OUTPUT_CSV = TIME_OUTPUT_DIR / "execution_times.csv"
TIME_OUTPUT_PNG = TIME_OUTPUT_DIR / "time_complexity_plot.png"
TIME_PRIMARY_MATCH = "3841665963"
TIME_PLAYER_ID = 0
SIZES = [100, 250, 500, 1000, 2500, 5000]
N_ITERATIONS = 20


# =============================================================================
# UTILITAIRES COMMUNS
# =============================================================================

def _uniform_sampling(points: list, n_segments: int) -> list:
    """Echantillonnage uniforme : garde 1 point sur K pour n_segments segments."""
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


def _calibrate_dp(trajectory: Trajectory, target_n: int, tolerance: int = 2):
    """Dichotomie pour trouver l'epsilon DP produisant ~target_n segments.

    Returns:
        (epsilon: float, segments: list)
    """
    lo, hi = 0.01, 5000.0
    best_eps = (lo + hi) / 2
    best_segs: list = []
    best_diff = float("inf")

    for _ in range(40):
        mid = (lo + hi) / 2
        segs = DouglasPeuckerCompressor(epsilon=mid).compress_player_trajectory(trajectory)
        diff = abs(len(segs) - target_n)
        if diff < best_diff:
            best_diff = diff
            best_segs = segs
            best_eps = mid
        if diff <= tolerance:
            break
        elif len(segs) > target_n:
            lo = mid
        else:
            hi = mid

    return best_eps, best_segs


def _load_trajectory_window(
    csv_path: Path, player_id: int, tick_start: int, tick_end: int
) -> Trajectory:
    """Charge la trajectoire d'un joueur sur une fenetre temporelle."""
    df = pd.read_csv(csv_path)
    x_col, y_col = f"x{player_id}", f"y{player_id}"
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


def _draw_segments(ax, segments: list, color: str, label: str, zorder: int = 5):
    """Dessine des segments sur un axe matplotlib."""
    for i, seg in enumerate(segments):
        ax.plot(
            [seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
            color=color, linewidth=2.0, alpha=0.9, zorder=zorder,
            label=label if i == 0 else None,
        )
        ax.plot(
            [seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
            "o", color=color, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=zorder + 1,
        )


# =============================================================================
# MODE BASELINE — Expérimentation 1 visuelle
# =============================================================================

def _find_good_window() -> Trajectory:
    """Cherche une fenetre de ~3000 ticks avec amplitude spatiale >= 50 unites."""
    csv_path = DATA_DIR / f"coord_{BASELINE_MATCH_ID}.csv"
    for start in range(BASELINE_TICK_START, 100_000, 1000):
        end = start + 3000
        try:
            traj = _load_trajectory_window(csv_path, BASELINE_PLAYER_ID, start, end)
        except Exception:
            continue
        if len(traj.points) < 30:
            continue
        xs = [p.x for p in traj.points]
        ys = [p.y for p in traj.points]
        if max(max(xs) - min(xs), max(ys) - min(ys)) >= 50:
            print(
                f"   Fenetre selectionnee : ticks {start}-{end} "
                f"({len(traj.points)} points)"
            )
            return traj
    raise ValueError("Aucune fenetre avec suffisamment de deplacement trouvee")


def run_baseline() -> None:
    """Exp. 1 visuelle : figure 4 panneaux (Original / Uniforme / DP / MDL)."""
    print("=" * 70)
    print("EXPERIMENTATION 1 : COMPARAISON MDL vs BASELINES (VISUEL)")
    print("=" * 70)
    print(f"Match: {BASELINE_MATCH_ID} | Joueur: {BASELINE_PLAYER_ID} | w_error: {W_ERROR_MDL}")
    print()

    print("Chargement...")
    trajectory = _find_good_window()
    points = trajectory.points
    n_pts = len(points)
    xs_raw = [p.x for p in points]
    ys_raw = [p.y for p in points]
    print()

    print("Compression MDL (reference)...")
    segs_mdl = MDLCompressor(w_error=W_ERROR_MDL).compress_player_trajectory(trajectory)
    N = len(segs_mdl)
    ratio = (1 - N / n_pts) * 100
    print(f"   -> {N} segments ({ratio:.1f}% compression)")

    print(f"Echantillonnage Uniforme (cible {N} segments)...")
    segs_uniform = _uniform_sampling(points, N)
    print(f"   -> {len(segs_uniform)} segments")

    print(f"Douglas-Peucker (cible {N} segments)...")
    _, segs_dp = _calibrate_dp(trajectory, N)
    print(f"   -> {len(segs_dp)} segments")
    print()

    print("Generation figure 4 panneaux...")
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 6), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    fig.suptitle(
        f"Experimentation 1 : Comparaison a Taux de Compression Egal (~{ratio:.0f}%)\n"
        f"Match {BASELINE_MATCH_ID} - Joueur {BASELINE_PLAYER_ID}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    bg = dict(color="lightgray", linewidth=0.6, alpha=0.5, zorder=1)
    pt = dict(color="lightgray", marker=".", linewidth=0, markersize=4, alpha=0.4, zorder=2)

    # Panneau 1 : Original
    ax = axes[0]
    ax.plot(xs_raw, ys_raw, color="steelblue", linewidth=1.4, alpha=0.85, zorder=3)
    ax.plot(xs_raw, ys_raw, "o", color="steelblue", markersize=3.5,
            markeredgecolor="navy", markeredgewidth=0.4, alpha=0.7, zorder=4)
    ax.set_title(f"Original\n({n_pts} points)", fontsize=12, fontweight="bold", pad=8)
    ax.set_facecolor("#f0f4ff")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)

    # Panneaux 2-4 : algos
    panels = [
        (axes[1], segs_uniform, "darkorange", "Echantillonnage Uniforme", "#fff8f0"),
        (axes[2], segs_dp,      "forestgreen", "Douglas-Peucker",         "#f0fff4"),
        (axes[3], segs_mdl,     "crimson",     f"MDL (w={W_ERROR_MDL})",  "#fff0f0"),
    ]
    for ax, segs, color, label, bg_col in panels:
        ax.plot(xs_raw, ys_raw, **bg)
        ax.plot(xs_raw, ys_raw, **pt)
        _draw_segments(ax, segs, color=color, label=label)
        ax.set_title(f"{label}\n({len(segs)} segments)", fontsize=12, fontweight="bold", pad=8)
        ax.set_facecolor(bg_col)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=10)

    mx = (max(xs_raw) - min(xs_raw)) * 0.07
    my = (max(ys_raw) - min(ys_raw)) * 0.07
    axes[0].set_xlim(min(xs_raw) - mx, max(xs_raw) + mx)
    axes[0].set_ylim(min(ys_raw) - my, max(ys_raw) + my)

    BASELINE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASELINE_OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure -> {BASELINE_OUTPUT}")


# =============================================================================
# MODE MATHS — Expérimentation 1 quantitative (parallèle)
# =============================================================================

_GLOBAL_DATA: dict = {}


def _init_worker(shared_data: dict) -> None:
    """Initialise chaque worker avec les DataFrames partages."""
    global _GLOBAL_DATA
    _GLOBAL_DATA = shared_data


def _extract_trajectory_maths(
    df: pd.DataFrame, tick_start: int, tick_end: int
) -> Optional[Trajectory]:
    x_col, y_col = f"x{MATHS_PLAYER_ID}", f"y{MATHS_PLAYER_ID}"
    if x_col not in df.columns:
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


def _process_task_maths(task: tuple) -> Optional[list]:
    """Worker : calcule RMSE / Hausdorff / Stop pour une fenetre (3 algos)."""
    sample_id, match_id, tick_start = task
    df = _GLOBAL_DATA.get(match_id)
    if df is None:
        return None

    traj = _extract_trajectory_maths(df, tick_start, tick_start + WINDOW_SIZE)
    if traj is None:
        return None
    pts = traj.points

    segs_mdl = MDLCompressor(w_error=W_ERROR_MDL).compress_player_trajectory(traj)
    N = len(segs_mdl)
    if N < MIN_SEGMENTS:
        return None

    _, segs_dp = _calibrate_dp(traj, N)
    segs_uni = _uniform_sampling(pts, N)

    records = []
    for algo, segs in [("MDL", segs_mdl), ("Uniforme", segs_uni), ("Douglas-Peucker", segs_dp)]:
        if not segs:
            continue
        records.append({
            "Sample_ID":        sample_id,
            "Match_ID":         match_id,
            "Tick_Start":       tick_start,
            "Tick_End":         tick_start + WINDOW_SIZE,
            "N_Points":         len(pts),
            "N_Segments":       len(segs),
            "Algorithm":        algo,
            "RMSE":             rmse_segments_to_points(pts, segs),
            "Hausdorff":        hausdorff_distance(pts, segs),
            "Stop_Preservation": stop_preservation_rate(pts, segs),
        })
    return records if records else None


def _generate_boxplots(df_results: pd.DataFrame, n_valid: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Experimentation 1 (Quantitative) : MDL vs Baselines - {n_valid} echantillons\n"
        f"Joueur {MATHS_PLAYER_ID} | Fenetres {WINDOW_SIZE} ticks | "
        f"w_error={W_ERROR_MDL} | {N_WORKERS} workers paralleles",
        fontsize=12, fontweight="bold", y=1.02,
    )
    metrics_cfg = [
        {"col": "RMSE",
         "title": "RMSE\n(Plus bas = meilleur)",
         "ylabel": "RMSE (unites de carte)",
         "ax": axes[0],
         "note": "Fidelite quadratique moyenne"},
        {"col": "Hausdorff",
         "title": "Distance de Hausdorff\n(Plus bas = meilleur)",
         "ylabel": "Hausdorff (unites de carte)",
         "ax": axes[1],
         "note": "Erreur maximale garantie"},
        {"col": "Stop_Preservation",
         "title": "Preservation des Arrets\n(Plus haut = meilleur)",
         "ylabel": "Taux de preservation (%)",
         "ax": axes[2],
         "note": "Fidelite semantique"},
    ]
    for cfg in metrics_cfg:
        ax, col = cfg["ax"], cfg["col"]
        data_by_algo, labels, colors = [], [], []
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
            data_by_algo, patch_artist=True, notch=False, widths=0.5,
            medianprops={"color": "black", "linewidth": 2.5},
            whiskerprops={"linewidth": 1.4}, capprops={"linewidth": 1.4},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.2, "linestyle": "none"},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

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
        ax.text(0.5, -0.12, cfg["note"], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, style="italic", color="gray")

        for i, vals in enumerate(data_by_algo):
            med = np.median(vals)
            ax.text(i + 1.28, med, f"{med:.2f}",
                    ha="left", va="center", fontsize=8.5, fontweight="bold")
            ax.text(i + 1, ax.get_ylim()[0], f"n={len(vals)}",
                    ha="center", va="bottom", fontsize=7.5, color="gray")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(MATHS_OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Boxplots -> {MATHS_OUTPUT_PNG}")


def run_maths() -> pd.DataFrame:
    """Exp. 1 quantitative : benchmark parallele sur 1000 echantillons."""
    global _GLOBAL_DATA
    t0 = time.time()
    print("=" * 70)
    print("EXPERIMENTATION 1 (QUANTITATIVE) : BENCHMARK MATHEMATIQUE PARALLELE")
    print("=" * 70)
    print(f"Echantillons : {N_SAMPLES} fenetres x {WINDOW_SIZE} ticks | w_error={W_ERROR_MDL}")
    print(f"Parallelisme : {N_WORKERS} workers / {os.cpu_count()} coeurs")
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    MATHS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des CSV...")
    _GLOBAL_DATA = {}
    for f in sorted(DATA_DIR.glob("coord_*.csv")):
        try:
            _GLOBAL_DATA[f.stem.replace("coord_", "")] = pd.read_csv(f)
        except Exception:
            pass
    n_files = len(_GLOBAL_DATA)
    total_rows = sum(len(df) for df in _GLOBAL_DATA.values())
    print(f"   {n_files} matchs | {total_rows:,} lignes")

    stride = max(100, WINDOW_SIZE // 2)
    candidates = []
    for match_id, df in _GLOBAL_DATA.items():
        ticks = sorted(df["tick"].unique())
        t_min, t_max = int(ticks[0]), int(ticks[-1])
        if t_max - t_min < WINDOW_SIZE:
            continue
        for t in range(t_min, t_max - WINDOW_SIZE + 1, stride):
            candidates.append((match_id, t))
    random.shuffle(candidates)
    print(f"   {len(candidates):,} fenetres candidates (stride={stride} ticks)")

    tasks = [(i, mid, ts) for i, (mid, ts) in enumerate(candidates)]

    print(f"\nTraitement ({N_WORKERS} workers)...")
    all_records: list = []
    n_valid = n_skipped = 0
    last_pct = -1

    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(_GLOBAL_DATA,)) as pool:
        for result in pool.imap_unordered(_process_task_maths, tasks, chunksize=8):
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
                        f"| {rate:.1f} ech/s | ETA {eta:.0f}s"
                    )
                    last_pct = pct

            if n_valid >= N_SAMPLES:
                pool.terminate()
                break

    elapsed_total = time.time() - t0
    print(f"\n   {n_valid} valides | {n_skipped} rejetes | {elapsed_total:.1f}s")

    if not all_records:
        print("\nAucun echantillon valide. Verifier MIN_POINTS, WINDOW_SIZE et les donnees.")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_records)
    df_results.to_csv(MATHS_OUTPUT_CSV, index=False, float_format="%.4f")
    print(f"\nCSV -> {MATHS_OUTPUT_CSV}  ({len(df_results):,} lignes)")

    print("\n" + "=" * 70)
    print("RESUME STATISTIQUE")
    print("=" * 70)
    fmt = "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("Algorithme", "RMSE med", "RMSE std", "Hd med", "Hd std", "Stop%"))
    print("  " + "-" * 68)
    for algo in ALGO_ORDER:
        sub = df_results[df_results["Algorithm"] == algo]
        sp = sub["Stop_Preservation"].dropna()
        print(fmt.format(
            algo,
            f"{sub['RMSE'].median():.3f}", f"+-{sub['RMSE'].std():.3f}",
            f"{sub['Hausdorff'].median():.3f}", f"+-{sub['Hausdorff'].std():.3f}",
            f"{sp.median():.1f}%" if len(sp) > 0 else "N/A",
        ))
    print("\n  RMSE / Hausdorff : Plus bas = meilleur")
    print("  Stop%            : Plus haut = meilleur (fidelite semantique)")

    print("\nGeneration boxplots...")
    _generate_boxplots(df_results, n_valid)

    print(f"\nTermine en {elapsed_total:.1f}s -> {MATHS_OUTPUT_DIR}/")
    return df_results


# =============================================================================
# MODE TIME — Expérimentation 2 complexité temporelle
# =============================================================================

def _load_points_pool(min_size: int) -> list:
    """Charge un pool de points suffisant pour les tailles testees."""
    x_col, y_col = f"x{TIME_PLAYER_ID}", f"y{TIME_PLAYER_ID}"
    pool: list = []

    def _extract(path: Path) -> list:
        df = pd.read_csv(path)
        if x_col not in df.columns:
            return []
        mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
        sub = df[mask].sort_values("tick")
        return [
            TrajectoryPoint(x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"]))
            for _, r in sub.iterrows()
        ]

    primary = DATA_DIR / f"coord_{TIME_PRIMARY_MATCH}.csv"
    pool.extend(_extract(primary))
    print(f"   {TIME_PRIMARY_MATCH} : {len(pool)} points")

    if len(pool) < min_size:
        for f in sorted(DATA_DIR.glob("coord_*.csv")):
            if f.name == primary.name:
                continue
            before = len(pool)
            pool.extend(_extract(f))
            added = len(pool) - before
            if added > 0:
                print(f"   {f.stem.replace('coord_', '')} : +{added} points (total: {len(pool)})")
            if len(pool) >= min_size:
                break
    return pool


def _generate_time_plot(df_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for algo in ALGO_ORDER:
        sub = df_results[df_results["Algorithm"] == algo].sort_values("N_points")
        xs = sub["N_points"].values
        ys = sub["Mean_Time_ms"].values
        stds = sub["Std_Time_ms"].values
        color = ALGO_COLORS[algo]
        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, label=algo, zorder=5)
        ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.15, zorder=3)

    n_max = df_results["N_points"].max()
    for algo in ALGO_ORDER:
        row = df_results[
            (df_results["Algorithm"] == algo) & (df_results["N_points"] == n_max)
        ]
        if len(row) == 0:
            continue
        y_val = row["Mean_Time_ms"].values[0]
        ax.annotate(
            f"{y_val:.1f} ms", xy=(n_max, y_val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color=ALGO_COLORS[algo], fontweight="bold",
        )

    ax.set_xlabel("Nombre de points N", fontsize=13, fontweight="bold")
    ax.set_ylabel("Temps d'execution moyen (ms)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Complexite Temporelle Empirique : MDL vs Baselines\n"
        f"({N_ITERATIONS} iterations par taille, w_error MDL={W_ERROR_MDL})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=12, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(df_results["N_points"].unique())
    ax.set_xticklabels(
        [f"{n:,}" for n in sorted(df_results["N_points"].unique())],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(TIME_OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Graphique -> {TIME_OUTPUT_PNG}")


def run_time() -> pd.DataFrame:
    """Exp. 2 : complexite temporelle empirique (6 tailles, 20 iterations)."""
    print("=" * 70)
    print("EXPERIMENTATION 2 : COMPLEXITE TEMPORELLE EMPIRIQUE")
    print("=" * 70)
    print(f"Tailles : {SIZES} | Iterations : {N_ITERATIONS} | w_error={W_ERROR_MDL}")
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    TIME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Construction du pool de points (cible >= {max(SIZES):,} points)...")
    pool = _load_points_pool(max(SIZES))
    n_available = len(pool)
    print(f"   Pool final : {n_available:,} points disponibles\n")

    sizes_to_run = [s for s in SIZES if s <= n_available]
    if len(sizes_to_run) < len(SIZES):
        print(f"   Tailles ignorees (pool insuffisant) : {[s for s in SIZES if s > n_available]}\n")

    mdl_compressor = MDLCompressor(w_error=W_ERROR_MDL, verbose=False)
    records: list = []

    for N in sizes_to_run:
        print(f"N = {N:,} points")
        start_idx = random.randint(0, n_available - N)
        window_pts = pool[start_idx: start_idx + N]
        trajectory = Trajectory(points=window_pts)

        segs_ref = mdl_compressor.compress_player_trajectory(trajectory)
        C = max(1, len(segs_ref))
        print(f"   MDL reference : {C} segments (compression {(1 - C/N)*100:.1f}%)")

        best_eps, _ = _calibrate_dp(trajectory, C)
        dp_compressor = DouglasPeuckerCompressor(epsilon=best_eps)

        times: dict = {"MDL": [], "Uniforme": [], "Douglas-Peucker": []}
        for _ in range(N_ITERATIONS):
            t = time.perf_counter()
            mdl_compressor.compress_player_trajectory(trajectory)
            times["MDL"].append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            _uniform_sampling(window_pts, C)
            times["Uniforme"].append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            dp_compressor.compress_player_trajectory(trajectory)
            times["Douglas-Peucker"].append((time.perf_counter() - t) * 1000)

        for algo, t_list in times.items():
            arr = np.array(t_list)
            print(f"   {algo:<20} : {arr.mean():7.3f} ms  (+-{arr.std():.3f})")
            records.append({
                "N_points":      N,
                "Algorithm":     algo,
                "Mean_Time_ms":  round(float(arr.mean()), 4),
                "Std_Time_ms":   round(float(arr.std()), 4),
                "N_segments_ref": C,
            })
        print()

    df_results = pd.DataFrame(records)
    df_results.to_csv(TIME_OUTPUT_CSV, index=False)
    print(f"CSV -> {TIME_OUTPUT_CSV}")

    print("\n" + "=" * 70 + "\nRESUME DES TEMPS MOYENS (ms)\n" + "=" * 70)
    pivot = df_results.pivot(index="N_points", columns="Algorithm", values="Mean_Time_ms")
    pivot = pivot[[a for a in ALGO_ORDER if a in pivot.columns]]
    print(pivot.to_string(float_format=lambda x: f"{x:8.3f}"))

    print("\nGeneration graphique...")
    _generate_time_plot(df_results)
    print(f"\nTermine -> {TIME_OUTPUT_DIR}/")
    return df_results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimentations MDL vs Baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes disponibles :
  baseline  Exp. 1 visuelle   : figure 4 panneaux comparative
  maths     Exp. 1 quantitative : benchmark parallele (RMSE, Hausdorff, Arrets)
  time      Exp. 2             : complexite temporelle empirique

exemples :
  python scripts/compression/experiments.py --mode baseline
  python scripts/compression/experiments.py --mode maths
  python scripts/compression/experiments.py --mode time
        """,
    )
    parser.add_argument(
        "--mode", choices=["baseline", "maths", "time"], required=True,
        help="Mode d'execution",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline()
    elif args.mode == "maths":
        run_maths()
    else:
        run_time()


if __name__ == "__main__":
    main()
