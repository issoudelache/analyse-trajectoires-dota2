#!/usr/bin/env python3
"""
MDL Robustesse - Analyse de robustesse et visualisation de l'algorithme MDL

Modes :
  full      Benchmark exhaustif (840 tests, parallele, series d'images triptyques)
  advanced  Benchmark statistique avec Data Augmentation et Heatmaps
  quality   Validation qualite sur N matchs reels (RMSE, compression, graphiques)
  triptych  Figure de soutenance : triptych Clean -> Bruit -> Resultat MDL

Usage :
  python scripts/compression/MDL_robustness.py --mode full
  python scripts/compression/MDL_robustness.py --mode advanced
  python scripts/compression/MDL_robustness.py --mode quality
  python scripts/compression/MDL_robustness.py --mode triptych
  python scripts/compression/MDL_robustness.py --mode triptych --w_error 12.0 --sigma 3.0
  python scripts/compression/MDL_robustness.py --mode quality --num_matches 30 --w_error 10.0
"""

import argparse
import random
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.compression import MDLCompressor
from dota_analytics.metrics import (
    add_gaussian_noise,
    calculate_compression_rate,
    calculate_reconstruction_error,
)
from dota_analytics.structures import Segment, Trajectory, TrajectoryPoint


# =============================================================================
# CONFIGURATION
# =============================================================================

MATCH_ID = "3841665963"
PLAYER_ID = 0
CSV_PATH = BASE_DIR / "data-dota" / f"coord_{MATCH_ID}.csv"

# --- Mode FULL ---
TICK_START = 66000
TICK_END = 68000

W_ERROR_VALUES_FULL = (
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    + [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    + [12.0, 14.0, 16.0, 18.0, 20.0]
    + [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
)
SIGMA_VALUES_FULL = (
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    + [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    + [6.0, 7.0, 8.0, 9.0, 10.0]
    + [12.0, 14.0, 16.0, 18.0, 20.0]
    + [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
)
FIXED_SIGMA_FOR_W_VARIATION = 5.0
FIXED_W_FOR_SIGMA_VARIATION = 12.0

# --- Mode ADVANCED ---
W_ERROR_VALUES_ADV = [1.0, 5.0, 10.0, 12.0, 15.0, 25.0, 50.0]
SIGMA_VALUES_ADV = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_SAMPLES = 5
SAMPLE_DURATION_TICKS = 1000

# --- Mode TRIPTYCH ---
TRIPTYCH_TICK_START = 66000
TRIPTYCH_TICK_END = 68000

# --- Dossiers de sortie ---
OUTPUT_DIR = BASE_DIR / "output" / "benchmark_matrix"
STATS_DIR = OUTPUT_DIR / "stats"
IMAGES_VAR_W_DIR = OUTPUT_DIR / "images" / "variation_w_error"
IMAGES_VAR_SIGMA_DIR = OUTPUT_DIR / "images" / "variation_sigma"
TRIPTYCH_DIR = BASE_DIR / "output" / "bruit_gaussien"


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Resultat d'un test individuel (mode advanced)."""
    w_error: float
    sigma: float
    sample_id: int
    nb_segments_clean: int
    nb_segments_noisy: int
    rmse_clean: float
    segment_stability: float
    nb_original_points: int


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_trajectory_window(
    csv_path: Path, player_id: int, tick_start: int, tick_end: int
) -> Trajectory:
    """Charge une trajectoire sur une fenetre temporelle."""
    df = pd.read_csv(csv_path)
    x_col, y_col = f"x{player_id}", f"y{player_id}"
    mask = (
        (df["tick"] >= tick_start)
        & (df["tick"] <= tick_end)
        & ((df[x_col] != 0.0) | (df[y_col] != 0.0))
    )
    player_df = df[mask].sort_values("tick")
    points = [
        TrajectoryPoint(x=float(row[x_col]), y=float(row[y_col]), tick=int(row["tick"]))
        for _, row in player_df.iterrows()
    ]
    return Trajectory(points=points)


def load_full_trajectory(csv_path: Path, player_id: int) -> Trajectory:
    """Charge la trajectoire complete d'un joueur."""
    df = pd.read_csv(csv_path)
    x_col, y_col = f"x{player_id}", f"y{player_id}"
    mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
    player_df = df[mask].sort_values("tick")
    points = [
        TrajectoryPoint(x=float(row[x_col]), y=float(row[y_col]), tick=int(row["tick"]))
        for _, row in player_df.iterrows()
    ]
    return Trajectory(points=points)


def get_random_samples(
    trajectory: Trajectory, n_samples: int = 5, duration_ticks: int = 1000
) -> List[Trajectory]:
    """Extrait n_samples fenetres aleatoires (Data Augmentation)."""
    samples = []
    tick_min = trajectory.points[0].tick
    tick_max = trajectory.points[-1].tick
    random.seed(42)

    for _ in range(n_samples):
        max_start = tick_max - duration_ticks
        for _ in range(10):
            start = random.randint(tick_min, max_start)
            pts = [p for p in trajectory.points if start <= p.tick < start + duration_ticks]
            if len(pts) >= 10:
                samples.append(Trajectory(points=pts))
                break

    if len(samples) < n_samples:
        print(f"  Seulement {len(samples)}/{n_samples} echantillons valides")

    return samples


# =============================================================================
# METRIQUES (locales au script de robustesse)
# =============================================================================

def calculate_rmse(
    original_points: List[TrajectoryPoint], segments: List[Segment]
) -> float:
    """RMSE entre les points originaux et les segments comprimes."""
    if not segments:
        return 0.0
    segment_map = {}
    for seg in segments:
        for tick in range(seg.start.tick, seg.end.tick + 1):
            segment_map[tick] = seg
    errors = []
    for p in original_points:
        seg = segment_map.get(p.tick)
        if seg is None:
            continue
        x1, y1 = seg.start.x, seg.start.y
        dx, dy = seg.end.x - x1, seg.end.y - y1
        if dx == 0 and dy == 0:
            errors.append(np.sqrt((p.x - x1) ** 2 + (p.y - y1) ** 2))
        else:
            errors.append(
                abs(dx * (y1 - p.y) - (x1 - p.x) * dy) / np.sqrt(dx**2 + dy**2)
            )
    return float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0.0


def calculate_rmse_clean(
    original_points: List[TrajectoryPoint], segments: List[Segment]
) -> float:
    """RMSE entre les segments (issus de donnees bruitees) et les points originaux sans bruit."""
    segment_map = {}
    for seg in segments:
        for tick in range(seg.start.tick, seg.end.tick + 1):
            segment_map[tick] = seg
    squared_errors = []
    for p in original_points:
        seg = segment_map.get(p.tick)
        if seg is None:
            continue
        t1, t2 = seg.start.tick, seg.end.tick
        if t2 == t1:
            rx, ry = seg.start.x, seg.start.y
        else:
            alpha = (p.tick - t1) / (t2 - t1)
            rx = seg.start.x + alpha * (seg.end.x - seg.start.x)
            ry = seg.start.y + alpha * (seg.end.y - seg.start.y)
        squared_errors.append((p.x - rx) ** 2 + (p.y - ry) ** 2)
    return float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0


def calculate_segment_stability(nb_clean: int, nb_noisy: int) -> float:
    """Variation (%) du nombre de segments par rapport a la version sans bruit."""
    if nb_clean == 0:
        return 0.0
    return abs(nb_noisy - nb_clean) / nb_clean * 100.0


# =============================================================================
# VISUALISATION — TRIPTYQUE (helper batch pour run_full)
# =============================================================================

def _generate_triptych_image(
    original_points, noisy_points, segments, w_error, sigma, output_path
):
    """Genere une image triptyque : Clean -> Bruit -> Resultat MDL."""
    x_c = [p.x for p in original_points]
    y_c = [p.y for p in original_points]
    x_n = [p.x for p in noisy_points]
    y_n = [p.y for p in noisy_points]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Robustesse MDL : w_error={w_error}, sigma={sigma:.1f}  |  "
        f"Match {MATCH_ID} - Joueur {PLAYER_ID}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ax = axes[0]
    ax.plot(x_c, y_c, color="blue", linewidth=1.5, alpha=0.8,
            marker="o", markersize=4, markerfacecolor="blue",
            markeredgecolor="darkblue", markeredgewidth=0.5)
    ax.set_title("1. Trajectoire Originale (Clean)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal"); ax.set_facecolor("#f9f9ff")

    ax = axes[1]
    ax.plot(x_n, y_n, color="gray", linewidth=0.8, alpha=0.5)
    ax.plot(x_n, y_n, color="gray", marker="x", linewidth=0,
            markersize=6, markeredgewidth=1.5, alpha=0.7)
    ax.set_title(f"2. Injection de Bruit (sigma={sigma:.1f})", fontsize=13,
                 fontweight="bold", pad=10, color="darkred")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal"); ax.set_facecolor("#fff9f9")

    ax = axes[2]
    ax.plot(x_n, y_n, color="lightgray", linewidth=0.6, alpha=0.4, zorder=1)
    for seg in segments:
        xs = [seg.start.x, seg.end.x]
        ys = [seg.start.y, seg.end.y]
        ax.plot(xs, ys, color="red", linewidth=2.5, alpha=0.9, zorder=10)
        ax.plot(xs, ys, "o", color="red", markersize=8,
                markeredgecolor="darkred", markeredgewidth=1.5, zorder=11)
    ax.set_title(f"3. Resultat MDL (w={w_error})", fontsize=13,
                 fontweight="bold", pad=10, color="darkgreen")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal"); ax.set_facecolor("#f9fff9")

    all_x, all_y = x_c + x_n, y_c + y_n
    xmn, xmx = min(all_x), max(all_x)
    ymn, ymx = min(all_y), max(all_y)
    mx, my = (xmx - xmn) * 0.05, (ymx - ymn) * 0.05
    for ax in axes:
        ax.set_xlim(xmn - mx, xmx + mx)
        ax.set_ylim(ymn - my, ymx + my)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# MODE FULL — Benchmark exhaustif parallèle
# =============================================================================

def _worker_full(args):
    """Worker multiprocessing pour un test (w_error, sigma)."""
    w_error, sigma, original_points, image_type = args

    noisy_points = add_gaussian_noise(original_points, sigma=sigma)
    segments = MDLCompressor(w_error=w_error, verbose=False).compress_player_trajectory(
        Trajectory(points=noisy_points)
    )
    rmse = calculate_rmse(original_points, segments)
    nb_seg = len(segments)

    if image_type == "var_w":
        w_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
        path = IMAGES_VAR_W_DIR / f"w{w_str}_sigma{FIXED_SIGMA_FOR_W_VARIATION}.png"
        _generate_triptych_image(original_points, noisy_points, segments, w_error, sigma, path)
    elif image_type == "var_sigma":
        s_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
        path = IMAGES_VAR_SIGMA_DIR / f"w{FIXED_W_FOR_SIGMA_VARIATION}_sigma{s_str}.png"
        _generate_triptych_image(original_points, noisy_points, segments, w_error, sigma, path)

    return {
        "w_error": w_error, "sigma": sigma, "nb_segments": nb_seg,
        "rmse": rmse,
        "compression_ratio": (1 - nb_seg / len(original_points)) * 100,
        "nb_original_points": len(original_points),
    }


def _generate_synthesis_plots(df: pd.DataFrame):
    """3 graphiques de synthese : resistance, efficacite, heatmap RMSE."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(W_ERROR_VALUES_FULL)))

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, w in enumerate(W_ERROR_VALUES_FULL):
        dw = df[df["w_error"] == w]
        ax.plot(dw["sigma"], dw["rmse"], marker="o", linewidth=2, markersize=5,
                label=f"w={w}", alpha=0.8, color=colors[idx])
    ax.set_xlabel("Niveau de Bruit (sigma)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=13, fontweight="bold")
    ax.set_title("Resistance au Bruit : RMSE vs Sigma", fontsize=15, fontweight="bold", pad=15)
    ax.legend(title="w_error", fontsize=9, title_fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = STATS_DIR / "resistance_au_bruit.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   {p.name}")

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, w in enumerate(W_ERROR_VALUES_FULL):
        dw = df[df["w_error"] == w]
        ax.plot(dw["sigma"], dw["nb_segments"], marker="s", linewidth=2, markersize=5,
                label=f"w={w}", alpha=0.8, color=colors[idx])
    ax.set_xlabel("Niveau de Bruit (sigma)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Nombre de Segments", fontsize=13, fontweight="bold")
    ax.set_title("Efficacite de Compression : Segments vs Sigma", fontsize=15, fontweight="bold", pad=15)
    ax.legend(title="w_error", fontsize=9, title_fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    nb_orig = df["nb_original_points"].iloc[0]
    ax.axhline(y=nb_orig, color="orange", linestyle=":", linewidth=2, alpha=0.7)
    ax.text(df["sigma"].max() * 0.95, nb_orig * 1.02,
            f"Original: {nb_orig} pts", ha="right", va="bottom", fontsize=10, color="orange")
    plt.tight_layout()
    p = STATS_DIR / "efficacite_compression.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   {p.name}")

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot = df.pivot(index="w_error", columns="sigma", values="rmse")
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest", origin="lower")
    plt.colorbar(im, ax=ax).set_label("RMSE", fontsize=12)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{s:.1f}" for s in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{w}" for w in pivot.index])
    ax.set_xlabel("Niveau de Bruit (sigma)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Parametre MDL (w_error)", fontsize=13, fontweight="bold")
    ax.set_title("Carte de Chaleur : RMSE (w_error x sigma)", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    p = STATS_DIR / "heatmap_rmse.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   {p.name}")


def run_full():
    """Mode FULL : Benchmark exhaustif 840 tests en parallele."""
    print("=" * 80)
    print("MODE FULL : BENCHMARK EXHAUSTIF (PARALLELE)")
    print("=" * 80)
    total = len(W_ERROR_VALUES_FULL) * len(SIGMA_VALUES_FULL)
    print(f"Match: {MATCH_ID} | Joueur: {PLAYER_ID} | Ticks: {TICK_START}-{TICK_END}")
    print(f"w_error: {len(W_ERROR_VALUES_FULL)} valeurs | sigma: {len(SIGMA_VALUES_FULL)} valeurs")
    print(f"Total tests: {total}")
    print()

    for d in [STATS_DIR, IMAGES_VAR_W_DIR, IMAGES_VAR_SIGMA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Chargement de la trajectoire...")
    traj = load_trajectory_window(CSV_PATH, PLAYER_ID, TICK_START, TICK_END)
    orig = traj.points
    print(f"   Points charges: {len(orig)}")
    print()

    tasks = []
    for w in W_ERROR_VALUES_FULL:
        for sigma in SIGMA_VALUES_FULL:
            image_type = None
            if abs(sigma - FIXED_SIGMA_FOR_W_VARIATION) < 0.01:
                image_type = "var_w"
            elif abs(w - FIXED_W_FOR_SIGMA_VARIATION) < 0.01:
                image_type = "var_sigma"
            tasks.append((w, sigma, orig, image_type))

    num_workers = max(1, cpu_count() - 2)
    print(f"   Total taches: {len(tasks)} | Workers: {num_workers}")
    print()

    print("Execution parallele...")
    results = []
    batch_size = num_workers * 2
    with Pool(processes=num_workers) as pool:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i: i + batch_size]
            batch_results = pool.map(_worker_full, batch)
            results.extend(batch_results)
            done = i + len(batch)
            last = batch_results[-1]
            print(
                f"   [{done}/{len(tasks)}] {done/len(tasks)*100:.1f}% | "
                f"w={last['w_error']:.1f} sigma={last['sigma']:.1f} | "
                f"RMSE={last['rmse']:.2f} Seg={last['nb_segments']}"
            )

    print(f"\n{len(results)} tests termines")
    df = pd.DataFrame(results)
    csv_out = STATS_DIR / "full_benchmark_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"CSV -> {csv_out}")
    print(f"RMSE={df['rmse'].mean():.2f} | Segments={df['nb_segments'].mean():.1f} | "
          f"Compression={df['compression_ratio'].mean():.1f}%")

    print("\nGeneration graphiques de synthese...")
    _generate_synthesis_plots(df)
    print(f"\nTermine -> {OUTPUT_DIR}")


# =============================================================================
# MODE ADVANCED — Benchmark statistique (Data Augmentation + Heatmaps)
# =============================================================================

def _run_single_test_adv(
    original_points: List[TrajectoryPoint], w_error: float, sigma: float, sample_id: int
) -> BenchmarkResult:
    """Compression clean + noisy + metriques etendues."""
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segs_clean = compressor.compress_player_trajectory(Trajectory(points=original_points))
    noisy_pts = add_gaussian_noise(original_points, sigma)
    segs_noisy = compressor.compress_player_trajectory(Trajectory(points=noisy_pts))
    return BenchmarkResult(
        w_error=w_error, sigma=sigma, sample_id=sample_id,
        nb_segments_clean=len(segs_clean), nb_segments_noisy=len(segs_noisy),
        rmse_clean=calculate_rmse_clean(original_points, segs_noisy),
        segment_stability=calculate_segment_stability(len(segs_clean), len(segs_noisy)),
        nb_original_points=len(original_points),
    )


def _generate_heatmaps(df_agg: pd.DataFrame):
    """Genere les 2 heatmaps : Stabilite et Fidelite."""
    specs = [
        ("nb_segments_noisy_mean",
         "Heatmap 1 : Stabilite Structurelle\n(Nombre de segments moyen)",
         "Nb segments", "YlOrRd", "heatmap_stabilite.png"),
        ("rmse_clean_mean",
         "Heatmap 2 : Fidelite au Signal Original\n(RMSE Clean - Metrique de Debruitage)",
         "RMSE Clean", "RdYlGn_r", "heatmap_fidelite.png"),
    ]
    for col, title, label, cmap, filename in specs:
        pivot = df_agg.pivot(index="w_error", columns="sigma", values=col)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{s:.1f}" for s in pivot.columns])
        ax.set_yticklabels([f"{w:.1f}" for w in pivot.index])
        ax.set_xlabel("Sigma (Niveau de bruit)", fontsize=12, fontweight="bold")
        ax.set_ylabel("w_error (Sensibilite MDL)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        plt.colorbar(im, ax=ax).set_label(label, fontsize=11, fontweight="bold")

        vmax = pivot.values.max()
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            color=color, fontsize=9, fontweight="bold")

        if "fidelite" in filename and 12.0 in list(pivot.index):
            w12_idx = list(pivot.index).index(12.0)
            ax.axhline(y=w12_idx, color="blue", linestyle="--", linewidth=2,
                       alpha=0.7, label="w=12 (optimal)")
            ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        p = STATS_DIR / filename
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   {filename}")


def run_advanced():
    """Mode ADVANCED : Benchmark statistique avec Data Augmentation et Heatmaps."""
    print("=" * 80)
    print("MODE ADVANCED : BENCHMARK STATISTIQUE + DATA AUGMENTATION")
    print("=" * 80)
    total = len(W_ERROR_VALUES_ADV) * len(SIGMA_VALUES_ADV) * N_SAMPLES
    print(f"Match: {MATCH_ID} | Joueur: {PLAYER_ID}")
    print(f"Echantillonnage: {N_SAMPLES} fenetres x {SAMPLE_DURATION_TICKS} ticks")
    print(f"w_error: {W_ERROR_VALUES_ADV}")
    print(f"sigma:   {SIGMA_VALUES_ADV}")
    print(f"Total tests: {total}")
    print()

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement de la trajectoire complete...")
    full_traj = load_full_trajectory(CSV_PATH, PLAYER_ID)
    print(f"   Points totaux: {len(full_traj.points)}")
    print(f"   Tick range: {full_traj.points[0].tick} -> {full_traj.points[-1].tick}")
    print()

    print("Echantillonnage aleatoire (Data Augmentation)...")
    samples = get_random_samples(full_traj, N_SAMPLES, SAMPLE_DURATION_TICKS)
    for i, s in enumerate(samples):
        print(f"   Sample {i}: {len(s.points)} points "
              f"(ticks {s.points[0].tick}-{s.points[-1].tick})")
    print()

    print("Execution du benchmark...")
    results = []
    completed = 0
    for w in W_ERROR_VALUES_ADV:
        for sigma in SIGMA_VALUES_ADV:
            for sample_id, sample in enumerate(samples):
                r = _run_single_test_adv(sample.points, w, sigma, sample_id)
                results.append(r)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        f"   [{completed}/{total}] {completed/total*100:.1f}% | "
                        f"w={w} sigma={sigma} | "
                        f"RMSE={r.rmse_clean:.2f} Stab={r.segment_stability:.1f}%"
                    )

    df_raw = pd.DataFrame([{
        "w_error": r.w_error, "sigma": r.sigma, "sample_id": r.sample_id,
        "nb_segments_clean": r.nb_segments_clean,
        "nb_segments_noisy": r.nb_segments_noisy,
        "rmse_clean": r.rmse_clean,
        "segment_stability": r.segment_stability,
        "nb_original_points": r.nb_original_points,
    } for r in results])

    print("\nAgregation des resultats...")
    agg = (
        df_raw.groupby(["w_error", "sigma"])
        .agg({
            "nb_segments_clean": ["mean", "std"],
            "nb_segments_noisy": ["mean", "std"],
            "rmse_clean": ["mean", "std"],
            "segment_stability": ["mean", "std"],
            "nb_original_points": "mean",
        })
        .reset_index()
    )
    agg.columns = [
        "w_error", "sigma",
        "nb_segments_clean_mean", "nb_segments_clean_std",
        "nb_segments_noisy_mean", "nb_segments_noisy_std",
        "rmse_clean_mean", "rmse_clean_std",
        "segment_stability_mean", "segment_stability_std",
        "nb_original_points",
    ]
    csv_out = STATS_DIR / "advanced_stats.csv"
    agg.to_csv(csv_out, index=False)
    print(f"CSV -> {csv_out}")
    print(f"RMSE Clean : {agg['rmse_clean_mean'].mean():.2f} +- {agg['rmse_clean_std'].mean():.2f}")
    print(f"Stabilite  : {agg['segment_stability_mean'].mean():.1f}% +- {agg['segment_stability_std'].mean():.1f}%")

    if 12.0 in agg["w_error"].values:
        df_w12 = agg[agg["w_error"] == 12.0]
        print(f"\nZone Optimale (w=12) :")
        print(f"   RMSE Clean : {df_w12['rmse_clean_mean'].mean():.2f}")
        print(f"   Stabilite  : {df_w12['segment_stability_mean'].mean():.1f}%")
        print(f"   Segments   : {df_w12['nb_segments_noisy_mean'].mean():.1f}")

    print("\nGeneration heatmaps...")
    _generate_heatmaps(agg)
    print(f"\nTermine -> {STATS_DIR}")


# =============================================================================
# MODE QUALITY — Validation sur N matchs réels
# =============================================================================

def run_quality(num_matches: int = 50, w_error: float = 12.0):
    """Mode QUALITY : mesure la precision de compression MDL sur N matchs reels."""
    print("=" * 70)
    print("MODE QUALITY : VALIDATION SCIENTIFIQUE SUR MATCHS REELS")
    print("=" * 70)
    print(f"  Matchs: {num_matches} | w_error: {w_error} | Joueur: 0")
    print()

    data_dir = BASE_DIR / "data-dota"
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("coord_*.csv"))[:num_matches]
    print(f"{len(csv_files)} matchs trouves...")

    compressor = MDLCompressor(w_error=w_error, verbose=False)
    results = []

    for i, csv_path in enumerate(csv_files, 1):
        match_id = csv_path.stem.replace("coord_", "")
        try:
            traj = load_full_trajectory(csv_path, player_id=0)
            if len(traj) < 10:
                continue
            segs = compressor.compress_player_trajectory(traj)
            metrics = calculate_reconstruction_error(traj, segs)
            rate = calculate_compression_rate(len(traj), len(segs))
            results.append({
                "match_id": match_id,
                "num_original_points": len(traj),
                "num_segments": len(segs),
                "compression_rate": rate,
                "rmse": metrics["rmse"],
                "max_error": metrics["max_error"],
                "mean_error": metrics["mean_error"],
                "w_error": w_error,
            })
        except Exception as e:
            print(f"  {match_id}: {e}")
            continue
        if i % 10 == 0 or i == len(csv_files):
            print(f"   [{i}/{len(csv_files)}] {i/len(csv_files)*100:.0f}%")

    df = pd.DataFrame(results)
    if df.empty:
        print("Aucun resultat valide")
        return

    csv_out = output_dir / "benchmark_metrics.csv"
    df.to_csv(csv_out, index=False)

    print()
    print("=" * 70)
    print("RESULTATS")
    print("=" * 70)
    print(f"  Matchs analyses  : {len(df)}")
    print(f"  Compression moy. : {df['compression_rate'].mean():.1f}%")
    print(f"  Segments moyens  : {df['num_segments'].mean():.0f}")
    print(f"  RMSE moyen       : {df['rmse'].mean():.3f}")
    print(f"  RMSE median      : {df['rmse'].median():.3f}")
    print(f"  RMSE max         : {df['rmse'].max():.3f}")
    print(f"CSV -> {csv_out}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark Qualite - Compression MDL (w_error={w_error})",
                 fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(df["rmse"], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(df["rmse"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Moyenne: {df['rmse'].mean():.3f}")
    ax.axvline(df["rmse"].median(), color="orange", linestyle="--", linewidth=2,
               label=f"Mediane: {df['rmse'].median():.3f}")
    ax.set_xlabel("RMSE"); ax.set_ylabel("Nb matchs")
    ax.set_title("Distribution RMSE", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(df["max_error"], bins=30, color="coral", edgecolor="black", alpha=0.7)
    ax.axvline(df["max_error"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Moyenne: {df['max_error'].mean():.2f}")
    ax.set_xlabel("Erreur Maximale"); ax.set_ylabel("Nb matchs")
    ax.set_title("Distribution Erreur Max", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sc = ax.scatter(df["compression_rate"], df["rmse"],
                    c=df["num_original_points"], cmap="viridis",
                    s=50, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Taux de compression (%)"); ax.set_ylabel("RMSE")
    ax.set_title("Trade-off : Compression vs Precision", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax).set_label("Nb points originaux", fontsize=9)

    ax = axes[1, 1]
    bp = ax.boxplot([df["rmse"], df["mean_error"], df["max_error"]],
                    labels=["RMSE", "Erreur Moy.", "Erreur Max"],
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["steelblue", "lightgreen", "coral"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Valeur d'erreur")
    ax.set_title("Comparaison des metriques", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_out = output_dir / "benchmark_quality.png"
    fig.savefig(plot_out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Graphique -> {plot_out}")
    print(f"\nTermine -> {output_dir}")


# =============================================================================
# MODE TRIPTYCH — Figure de soutenance
# =============================================================================

def run_triptych(w_error: float = 12.0, sigma: float = 3.0) -> None:
    """Mode TRIPTYCH : genere la figure de soutenance Clean -> Bruit -> MDL."""
    w_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
    output_path = TRIPTYCH_DIR / f"triptych_w{w_str}_sigma{sigma_str}.png"

    print("=" * 70)
    print("GENERATION TRIPTYCH DE ROBUSTESSE")
    print("=" * 70)
    print(f"Match: {MATCH_ID} | Joueur: {PLAYER_ID}")
    print(f"Fenetre: ticks {TRIPTYCH_TICK_START}-{TRIPTYCH_TICK_END}")
    print(f"Parametres: w_error={w_error}, sigma={sigma}")
    print()

    if not CSV_PATH.exists():
        print(f"Fichier introuvable: {CSV_PATH}")
        return

    # Etape 1 : trajectoire originale
    print("Etape 1 : Chargement trajectoire originale...")
    trajectory_clean = load_trajectory_window(
        CSV_PATH, PLAYER_ID, TRIPTYCH_TICK_START, TRIPTYCH_TICK_END
    )
    if len(trajectory_clean.points) == 0:
        print("Aucun point dans la fenetre temporelle")
        return
    x_clean = [p.x for p in trajectory_clean.points]
    y_clean = [p.y for p in trajectory_clean.points]
    print(f"   {len(x_clean)} points extraits")

    # Etape 2 : bruit gaussien
    print("Etape 2 : Injection de bruit gaussien...")
    noisy_points = add_gaussian_noise(trajectory_clean.points, sigma=sigma)
    x_noisy = [p.x for p in noisy_points]
    y_noisy = [p.y for p in noisy_points]

    # Etape 3 : compression MDL
    print("Etape 3 : Compression MDL sur donnees bruitees...")
    segments = MDLCompressor(w_error=w_error, verbose=False).compress_player_trajectory(
        Trajectory(points=noisy_points)
    )
    print(f"   {len(trajectory_clean.points)} points -> {len(segments)} segments "
          f"({100 - len(segments)/len(x_clean)*100:.1f}% compression)")

    # Generation de la figure
    print("\nCreation de la figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Robustesse de l'algorithme MDL face au bruit\n"
        f"Match {MATCH_ID} - Joueur {PLAYER_ID} - Ticks {TRIPTYCH_TICK_START}-{TRIPTYCH_TICK_END}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Panneau 1 : Original
    ax1 = axes[0]
    ax1.plot(x_clean, y_clean, color="blue", linewidth=1.5, alpha=0.8,
             marker="o", markersize=4, markerfacecolor="blue",
             markeredgecolor="darkblue", markeredgewidth=0.5)
    ax1.set_title("1. Trajectoire Originale (Clean)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("X (coordonnees carte)"); ax1.set_ylabel("Y (coordonnees carte)")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_aspect("equal"); ax1.set_facecolor("#f9f9ff")
    ax1.legend(handles=[mpatches.Patch(color="blue", label=f"{len(x_clean)} points")],
               loc="upper right", fontsize=9)

    # Panneau 2 : Bruit
    ax2 = axes[1]
    ax2.plot(x_noisy, y_noisy, color="gray", linewidth=0.8, alpha=0.5)
    ax2.plot(x_noisy, y_noisy, color="gray", linewidth=0, alpha=0.7,
             marker="x", markersize=6, markeredgewidth=1.5)
    ax2.set_title(f"2. Injection de Bruit (sigma={sigma})",
                  fontsize=13, fontweight="bold", pad=10, color="darkred")
    ax2.set_xlabel("X (coordonnees carte)"); ax2.set_ylabel("Y (coordonnees carte)")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect("equal"); ax2.set_facecolor("#fff9f9")
    ax2.legend(handles=[mpatches.Patch(color="gray", label=f"Bruit gaussien sigma={sigma}")],
               loc="upper right", fontsize=9)

    # Panneau 3 : MDL
    ax3 = axes[2]
    ax3.plot(x_noisy, y_noisy, color="lightgray", linewidth=0.6, alpha=0.4, zorder=1)
    ax3.plot(x_noisy, y_noisy, color="lightgray", linewidth=0, alpha=0.35,
             marker="x", markersize=4, markeredgewidth=0.8, zorder=2)
    for segment in segments:
        x_seg = [segment.start.x, segment.end.x]
        y_seg = [segment.start.y, segment.end.y]
        ax3.plot(x_seg, y_seg, color="red", linewidth=2.5, alpha=0.9, zorder=10)
        ax3.plot(x_seg, y_seg, "o", color="red", markersize=8,
                 markeredgecolor="darkred", markeredgewidth=1.5, zorder=11)
    ax3.set_title(f"3. Resultat MDL (Filtrage w={w_error})",
                  fontsize=13, fontweight="bold", pad=10, color="darkgreen")
    ax3.set_xlabel("X (coordonnees carte)"); ax3.set_ylabel("Y (coordonnees carte)")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_aspect("equal"); ax3.set_facecolor("#f9fff9")
    ax3.legend(
        handles=[
            mpatches.Patch(color="red", label=f"{len(segments)} segments MDL"),
            mpatches.Patch(color="lightgray", label="Bruit de fond"),
        ],
        loc="upper right", fontsize=9,
    )

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

    # Fleches entre panneaux
    fig.text(0.33, 0.92, "->", fontsize=30, ha="center", va="center", color="gray")
    fig.text(0.66, 0.92, "->", fontsize=30, ha="center", va="center", color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    TRIPTYCH_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    size_kb = output_path.stat().st_size // 1024
    print()
    print("=" * 70)
    print("TRIPTYCH GENERE")
    print("=" * 70)
    print(f"Fichier : {output_path}  ({size_kb} KB)")
    print(f"  Panneau 1 : {len(x_clean)} points originaux (bleu)")
    print(f"  Panneau 2 : {len(x_noisy)} points bruites (gris, sigma={sigma})")
    print(f"  Panneau 3 : {len(segments)} segments MDL (rouge, w={w_error})")
    print(f"  Taux de compression : {100 - len(segments)/len(x_clean)*100:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MDL Robustesse - Analyse de robustesse de l'algorithme MDL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes disponibles :
  full      Benchmark exhaustif (840 tests, parallele, series d'images triptyques)
  advanced  Benchmark statistique (data augmentation, heatmaps scientifiques)
  quality   Validation qualite sur N matchs reels
  triptych  Figure de soutenance : Clean -> Bruit -> Resultat MDL

exemples :
  python scripts/compression/MDL_robustness.py --mode full
  python scripts/compression/MDL_robustness.py --mode advanced
  python scripts/compression/MDL_robustness.py --mode quality --num_matches 30
  python scripts/compression/MDL_robustness.py --mode triptych --w_error 12.0 --sigma 3.0
        """,
    )
    parser.add_argument(
        "--mode", choices=["full", "advanced", "quality", "triptych"],
        required=True, help="Mode d'execution",
    )
    parser.add_argument(
        "--num_matches", type=int, default=50,
        help="(mode quality) Nombre de matchs a analyser (defaut: 50)",
    )
    parser.add_argument(
        "--w_error", type=float, default=12.0,
        help="Parametre w_error MDL (defaut: 12.0)",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0,
        help="(mode triptych) Ecart-type du bruit gaussien (defaut: 3.0)",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full()
    elif args.mode == "advanced":
        run_advanced()
    elif args.mode == "quality":
        run_quality(num_matches=args.num_matches, w_error=args.w_error)
    else:
        run_triptych(w_error=args.w_error, sigma=args.sigma)


if __name__ == "__main__":
    main()
