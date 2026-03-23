#!/usr/bin/env python3
"""
experiment_optimal_k.py
=======================
Recherche du k optimal pour KMeans et KMedoids sur les segments compressés.

La matrice de distance TRACLUS est calculée UNE seule fois, puis réutilisée
pour tous les k. Chaque k est testé sur plusieurs seeds aléatoires.

Métriques mesurées : silhouette, Davies-Bouldin, Calinski-Harabasz, inertie.

Sorties
-------
  output/benchmark_optimal_k/optimal_k_results.csv   (incrémental)
  output/benchmark_optimal_k/fig_elbow.png
  output/benchmark_optimal_k/fig_silhouette_vs_k.png
  output/benchmark_optimal_k/fig_combined.png

Usage
-----
  python benchmark/clustering/experiment_optimal_k.py
  python benchmark/clustering/experiment_optimal_k.py --n 5000 --seeds 5
  python benchmark/clustering/experiment_optimal_k.py --quick   (N=3000, 3 seeds, k=2..30)
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path

# Thread control — before numpy import
_THREADS = str(max(1, (os.cpu_count() or 1) // 2))
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from dota_analytics.clustering import load_data
from dota_analytics.custom_kmedoids import CustomKMedoids

# ── Paths ────────────────────────────────────────────────────────────────────
COMPRESSED_DIR = PROJECT_ROOT / "output" / "compressed" / "w_error_12.0"
OUT_DIR = PROJECT_ROOT / "output" / "benchmark_optimal_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "optimal_k_results.csv"

CSV_COLUMNS = [
    "k",
    "seed",
    "algorithm",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "inertia",
    "time_seconds",
]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
COLORS = {"KMeans": "#2196F3", "KMedoids": "#FF9800"}
LABELS = {"KMeans": "K-Means", "KMedoids": "K-Medoids (PAM)"}


# ── Distance matrix (same as experiment_clustering_scale.py) ─────────────────
def _compute_D(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    lengths = np.clip(lengths, 1e-9, None)
    directions = vectors / lengths[:, np.newaxis]

    cos_theta = np.clip(np.dot(directions, directions.T), -1.0, 1.0)
    d_angle = (1.0 - cos_theta) * (lengths[:, np.newaxis] + lengths[np.newaxis, :])

    vx, vy = directions[:, 0:1], directions[:, 1:2]
    vec_sx = starts[np.newaxis, :, 0] - starts[:, np.newaxis, 0]
    vec_sy = starts[np.newaxis, :, 1] - starts[:, np.newaxis, 1]
    vec_ex = ends[np.newaxis, :, 0] - starts[:, np.newaxis, 0]
    vec_ey = ends[np.newaxis, :, 1] - starts[:, np.newaxis, 1]

    cross_s = np.abs(vx * vec_sy - vy * vec_sx)
    cross_e = np.abs(vx * vec_ey - vy * vec_ex)
    sum_cross = cross_s + cross_e
    d_perp = np.zeros_like(sum_cross)
    mask = sum_cross > 0
    d_perp[mask] = (cross_s[mask] ** 2 + cross_e[mask] ** 2) / sum_cross[mask]

    proj_s = vec_sx * vx + vec_sy * vy
    proj_e = vec_ex * vx + vec_ey * vy
    base_l = lengths[:, np.newaxis]
    d_par = np.minimum(np.abs(proj_s), np.abs(proj_s - base_l)) + np.minimum(
        np.abs(proj_e), np.abs(proj_e - base_l)
    )

    D_asym = (d_perp + d_angle + d_par).astype(np.float32)
    len_mask = lengths[:, np.newaxis] > lengths[np.newaxis, :]
    D = np.where(len_mask, D_asym, D_asym.T)
    np.fill_diagonal(D, 0.0)
    return D


def _silhouette(D, labels):
    valid = labels >= 0
    if valid.sum() < 2:
        return float("nan")
    D_v, l_v = D[np.ix_(valid, valid)], labels[valid]
    if len(np.unique(l_v)) < 2:
        return float("nan")
    try:
        return float(
            silhouette_score(
                D_v.astype(np.float64), l_v, metric="precomputed", n_jobs=-1
            )
        )
    except Exception:
        return float("nan")


def _db_ch(features, labels):
    """Davies-Bouldin & Calinski-Harabasz (feature space)."""
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return float("nan"), float("nan")
    try:
        db = davies_bouldin_score(features[valid], labels[valid])
        ch = calinski_harabasz_score(features[valid], labels[valid])
        return db, ch
    except Exception:
        return float("nan"), float("nan")


# ── CSV helpers ──────────────────────────────────────────────────────────────
def _init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def _append_csv(row: dict):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in CSV_COLUMNS])


# ── Main benchmark ───────────────────────────────────────────────────────────
def run_benchmark(n_segments: int, k_range: range, n_seeds: int):
    print("═══ Recherche du k optimal ═══", flush=True)
    print(f"  N segments : {n_segments}", flush=True)
    print(f"  k range    : {k_range.start}..{k_range.stop - 1}", flush=True)
    print(f"  Seeds      : {n_seeds}", flush=True)
    print(flush=True)

    # 1. Load segments
    print("Chargement des segments compressés...", flush=True)
    segments, _ = load_data(str(COMPRESSED_DIR))
    total = len(segments)
    print(f"  {total} segments chargés", flush=True)

    if total < n_segments:
        print(f"  ⚠ Seulement {total} segments dispos, on prend tout", flush=True)
        n_segments = total

    # 2. Random subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=n_segments, replace=False)
    sub_segments = [segments[i] for i in indices]

    starts = np.array([[s.start.x, s.start.y] for s in sub_segments], dtype=np.float32)
    ends = np.array([[s.end.x, s.end.y] for s in sub_segments], dtype=np.float32)

    # Feature matrix for KMeans
    mids = (starts + ends) / 2.0
    vecs = ends - starts
    lens = np.linalg.norm(vecs, axis=1, keepdims=True)
    features = np.hstack([mids, vecs, lens])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 3. Compute distance matrix ONCE
    print("Calcul de la matrice de distance TRACLUS...", flush=True)
    t0 = time.perf_counter()
    D = _compute_D(starts, ends)
    t_matrix = time.perf_counter() - t0
    print(
        f"  Matrice {n_segments}×{n_segments} calculée en {t_matrix:.1f}s", flush=True
    )
    print(flush=True)

    # 4. Init CSV
    _init_csv()

    # 5. Sweep k
    total_runs = len(k_range) * n_seeds * 2  # 2 algos
    done = 0
    for k in k_range:
        for seed in range(n_seeds):
            # --- KMeans ---
            t0 = time.perf_counter()
            km = MiniBatchKMeans(
                n_clusters=k, random_state=seed, batch_size=4096, n_init=5, verbose=0
            )
            labels_km = km.fit_predict(X_scaled)
            t_km = time.perf_counter() - t0
            sil_km = _silhouette(D, labels_km)
            db_km, ch_km = _db_ch(X_scaled, labels_km)
            inertia_km = float(km.inertia_)

            _append_csv(
                {
                    "k": k,
                    "seed": seed,
                    "algorithm": "KMeans",
                    "silhouette": f"{sil_km:.6f}",
                    "davies_bouldin": f"{db_km:.6f}",
                    "calinski_harabasz": f"{ch_km:.6f}",
                    "inertia": f"{inertia_km:.6f}",
                    "time_seconds": f"{t_km:.6f}",
                }
            )
            done += 1

            # --- KMedoids ---
            t0 = time.perf_counter()
            kmed = CustomKMedoids(n_clusters=k, max_iter=300, random_state=seed)
            kmed.fit(D)
            labels_kmed = kmed.labels_
            t_kmed = time.perf_counter() - t0
            sil_kmed = _silhouette(D, labels_kmed)
            db_kmed, ch_kmed = _db_ch(X_scaled, labels_kmed)
            # KMedoids inertia: sum of distances to medoid
            inertia_kmed = sum(
                D[i, kmed.medoid_indices_[labels_kmed[i]]]
                for i in range(len(labels_kmed))
            )

            _append_csv(
                {
                    "k": k,
                    "seed": seed,
                    "algorithm": "KMedoids",
                    "silhouette": f"{sil_kmed:.6f}",
                    "davies_bouldin": f"{db_kmed:.6f}",
                    "calinski_harabasz": f"{ch_kmed:.6f}",
                    "inertia": f"{inertia_kmed:.6f}",
                    "time_seconds": f"{t_kmed:.6f}",
                }
            )
            done += 1

        pct = done / total_runs * 100
        print(
            f"  k={k:3d}  sil(KM)={sil_km:.4f}  sil(KMed)={sil_kmed:.4f}  "
            f"[{pct:5.1f}%]",
            flush=True,
        )

    print(
        f"\n✅ Résultats sauvegardés dans {CSV_PATH.relative_to(PROJECT_ROOT)}",
        flush=True,
    )
    return CSV_PATH


# ── Figures ──────────────────────────────────────────────────────────────────
def generate_figures(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"\nGénération des figures ({len(df)} lignes)...", flush=True)

    for algo in ["KMeans", "KMedoids"]:
        sub = df[df["algorithm"] == algo]
        med = sub.groupby("k").median(numeric_only=True)
        q1 = sub.groupby("k").quantile(0.25, numeric_only=True)
        q3 = sub.groupby("k").quantile(0.75, numeric_only=True)

        # --- Elbow (inertie) ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"Méthode du coude — {LABELS[algo]}", fontweight="bold")
        ks = med.index.values
        ax.plot(
            ks,
            med["inertia"].values,
            color=COLORS[algo],
            marker="o",
            markersize=4,
            lw=2,
        )
        ax.fill_between(
            ks,
            q1["inertia"].values,
            q3["inertia"].values,
            color=COLORS[algo],
            alpha=0.12,
        )
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertie")
        fig.tight_layout()
        name = f"fig_elbow_{algo.lower()}.png"
        fig.savefig(OUT_DIR / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ {name}", flush=True)

    # --- Silhouette vs k (both algos) ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Score silhouette en fonction de k", fontweight="bold")
    for algo in ["KMeans", "KMedoids"]:
        sub = df[df["algorithm"] == algo]
        med = sub.groupby("k")["silhouette"].median()
        q1 = sub.groupby("k")["silhouette"].quantile(0.25)
        q3 = sub.groupby("k")["silhouette"].quantile(0.75)
        ks = med.index.values
        ax.plot(
            ks,
            med.values,
            color=COLORS[algo],
            marker="o",
            markersize=4,
            lw=2,
            label=LABELS[algo],
        )
        ax.fill_between(ks, q1.values, q3.values, color=COLORS[algo], alpha=0.12)

    # Highlight optimal k per algo
    for algo in ["KMeans", "KMedoids"]:
        sub = df[df["algorithm"] == algo]
        med = sub.groupby("k")["silhouette"].median()
        k_best = int(med.idxmax())
        sil_best = med.max()
        ax.axvline(k_best, color=COLORS[algo], ls="--", lw=1, alpha=0.5)
        ax.annotate(
            f"k*={k_best} ({sil_best:.3f})",
            xy=(k_best, sil_best),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=COLORS[algo],
        )

    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Score silhouette (médiane)")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "fig_silhouette_vs_k.png", bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("  ✓ fig_silhouette_vs_k.png", flush=True)

    # --- Combined 4-panel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Recherche du k optimal — KMeans vs KMedoids", fontweight="bold", y=1.01
    )
    metrics = [
        ("silhouette", "Score silhouette (↑)"),
        ("inertia", "Inertie (↓)"),
        ("davies_bouldin", "Davies-Bouldin (↓)"),
        ("calinski_harabasz", "Calinski-Harabasz (↑)"),
    ]
    for ax, (col, ylabel) in zip(axes.flat, metrics):
        for algo in ["KMeans", "KMedoids"]:
            sub = df[df["algorithm"] == algo]
            med = sub.groupby("k")[col].median()
            q1 = sub.groupby("k")[col].quantile(0.25)
            q3 = sub.groupby("k")[col].quantile(0.75)
            ks = med.index.values
            ax.plot(
                ks,
                med.values,
                color=COLORS[algo],
                marker="o",
                markersize=3,
                lw=1.8,
                label=LABELS[algo],
            )
            ax.fill_between(ks, q1.values, q3.values, color=COLORS[algo], alpha=0.1)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "fig_combined_optimal_k.png", bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("  ✓ fig_combined_optimal_k.png", flush=True)

    print(f"\n✅ Figures dans {OUT_DIR.relative_to(PROJECT_ROOT)}/", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Recherche du k optimal")
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Nombre de segments à échantillonner (défaut: 5000)",
    )
    parser.add_argument("--k-min", type=int, default=2, help="k minimum (défaut: 2)")
    parser.add_argument("--k-max", type=int, default=50, help="k maximum (défaut: 50)")
    parser.add_argument(
        "--seeds", type=int, default=5, help="Nombre de seeds par k (défaut: 5)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Mode rapide : N=3000, seeds=3, k=2..30"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Ne pas relancer le benchmark, juste regénérer les figures",
    )
    args = parser.parse_args()

    if args.plot_only:
        if CSV_PATH.exists():
            generate_figures(CSV_PATH)
        else:
            print(f"❌ Pas de CSV trouvé : {CSV_PATH}")
        return

    if args.quick:
        args.n = 3000
        args.seeds = 3
        args.k_max = 30

    k_range = range(args.k_min, args.k_max + 1)
    csv_path = run_benchmark(args.n, k_range, args.seeds)
    generate_figures(csv_path)


if __name__ == "__main__":
    main()
