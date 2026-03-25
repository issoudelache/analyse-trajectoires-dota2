#!/usr/bin/env python3
"""
EXP. 0 — Impact de w_error (MDL) sur la compression et le clustering.

Question : Quel w_error offre le meilleur compromis fidélité/compression
           pour le clustering en aval ?

Fixé  : algo=kmeans, k=12, max_files=30, min_length=5.0, N=3000, seeds=7
Variable : w_error (26 valeurs)

Métriques collectées :
  nb_segments, ratio_compression, longueur_moyenne, silhouette,
  davies_bouldin, calinski_harabasz, temps_compression, temps_clustering

Écriture incrémentale : chaque ligne est appendée au CSV dès qu'elle est prête.
"""

import argparse
import csv
import gc
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.config import (
    DATA_DIR, OUTPUT_EXP0 as OUTPUT_DIR,
    K_DEFAULT as K, MAX_FILES, MIN_LENGTH,
    N_SUBSAMPLE_DEFAULT as N_SUBSAMPLE, SEEDS_MULTI as SEEDS,
    W_ERROR_GRID_26 as W_ERROR_GRID,
)
from dota_analytics.compression import MDLCompressor
from dota_analytics.structures import Trajectory, TrajectoryPoint

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

CSV_COLUMNS = [
    "w_error", "seed", "nb_segments_total", "nb_segments_sampled",
    "ratio_compression", "longueur_moyenne", "longueur_std",
    "nb_original_points", "nb_raw_segments",
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "inertia", "n_clusters_effective",
    "temps_compression_s", "temps_clustering_s", "temps_total_s",
]


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES TRAJECTOIRES (une seule fois)
# ═════════════════════════════════════════════════════════════════════════════

def preload_trajectories(max_files: int):
    """Charge les trajectoires brutes depuis les CSV."""
    import pandas as pd

    csv_files = sorted(DATA_DIR.glob("coord_*.csv"))[:max_files]
    if not csv_files:
        print(f"ERREUR : aucun CSV dans {DATA_DIR}")
        sys.exit(1)

    print(f"Chargement de {len(csv_files)} matchs…")
    trajectories = []
    n_original_points = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for player_id in range(10):
            x_col, y_col = f"x{player_id}", f"y{player_id}"
            if x_col not in df.columns:
                continue
            mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
            sub = df[mask].sort_values("tick")
            if len(sub) < 2:
                continue
            points = [
                TrajectoryPoint(x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"]))
                for _, r in sub.iterrows()
            ]
            trajectories.append(Trajectory(points=points, player_id=player_id))
            n_original_points += len(points)

    print(f"  → {len(trajectories)} trajectoires, {n_original_points:,} points bruts\n")
    return trajectories, n_original_points


# ═════════════════════════════════════════════════════════════════════════════
# COMPRESSION + FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def compress_all(trajectories, w_error, min_length):
    """Compresse toutes les trajectoires, retourne segments filtrés."""
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments = []
    n_raw = 0
    for traj in trajectories:
        segs = compressor.compress_player_trajectory(traj)
        n_raw += len(segs)
        segments.extend(s for s in segs if s.length() > min_length)
    return segments, n_raw


def segments_to_features(segments):
    """Extrait (mid_x, mid_y, dx, dy, length)."""
    feats = np.empty((len(segments), 5), dtype=np.float32)
    for i, s in enumerate(segments):
        feats[i, 0] = (s.start.x + s.end.x) * 0.5
        feats[i, 1] = (s.start.y + s.end.y) * 0.5
        feats[i, 2] = s.end.x - s.start.x
        feats[i, 3] = s.end.y - s.start.y
        feats[i, 4] = s.length()
    return feats


def segments_lengths(segments):
    """Retourne un array de longueurs."""
    return np.array([s.length() for s in segments], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# WORKER : 1 (w_error, seed) → 1 ligne CSV
# ═════════════════════════════════════════════════════════════════════════════

def run_single(w_error, seed, trajectories, n_original_points):
    """Exécute compression + clustering pour un (w_error, seed)."""
    t_start = time.perf_counter()

    # 1) Compression
    t0 = time.perf_counter()
    segments, n_raw = compress_all(trajectories, w_error, MIN_LENGTH)
    t_compress = time.perf_counter() - t0

    n_seg_total = len(segments)
    ratio_compression = n_original_points / max(n_raw, 1)
    lengths = segments_lengths(segments)
    longueur_moyenne = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    longueur_std = float(np.std(lengths)) if len(lengths) > 0 else 0.0

    # 2) Sous-échantillonnage
    rng = np.random.default_rng(seed)
    if n_seg_total > N_SUBSAMPLE:
        idx = rng.choice(n_seg_total, N_SUBSAMPLE, replace=False)
        idx.sort()
        segments_sample = [segments[i] for i in idx]
    else:
        segments_sample = segments
    n_seg_sampled = len(segments_sample)

    if n_seg_sampled < K + 1:
        return {
            "w_error": w_error, "seed": seed,
            "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
            "ratio_compression": ratio_compression,
            "longueur_moyenne": longueur_moyenne, "longueur_std": longueur_std,
            "nb_original_points": n_original_points, "nb_raw_segments": n_raw,
            "silhouette": np.nan, "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan, "inertia": np.nan,
            "n_clusters_effective": 0,
            "temps_compression_s": t_compress, "temps_clustering_s": 0.0,
            "temps_total_s": time.perf_counter() - t_start,
        }

    # 3) Features + normalisation
    X = segments_to_features(segments_sample)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) KMeans
    t0 = time.perf_counter()
    km = MiniBatchKMeans(
        n_clusters=K, random_state=seed, batch_size=min(4096, n_seg_sampled), n_init=3,
    )
    labels = km.fit_predict(X_scaled)
    t_cluster = time.perf_counter() - t0

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        sil = db = ch = np.nan
    else:
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, n_seg_sampled), random_state=seed)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)

    return {
        "w_error": w_error, "seed": seed,
        "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
        "ratio_compression": ratio_compression,
        "longueur_moyenne": longueur_moyenne, "longueur_std": longueur_std,
        "nb_original_points": n_original_points, "nb_raw_segments": n_raw,
        "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
        "inertia": float(km.inertia_), "n_clusters_effective": n_unique,
        "temps_compression_s": t_compress, "temps_clustering_s": t_cluster,
        "temps_total_s": time.perf_counter() - t_start,
    }


# ═════════════════════════════════════════════════════════════════════════════
# REPRISE : détecte les (w_error, seed) déjà calculés
# ═════════════════════════════════════════════════════════════════════════════

def load_done(csv_path):
    """Charge les (w_error, seed) déjà enregistrés."""
    done = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add((float(row["w_error"]), int(row["seed"])))
                except (KeyError, ValueError):
                    continue
    return done


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp. 0 — Impact de w_error")
    parser.add_argument("--max_files", type=int, default=MAX_FILES)
    parser.add_argument("--workers", type=int, default=None,
                        help="Nb de workers parallèles (défaut: cpu_count - 1)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "exp0_results.csv"

    # Reprise incrémentale
    done = load_done(csv_path)
    tasks = [
        (w, s) for w in W_ERROR_GRID for s in SEEDS
        if (w, s) not in done
    ]

    if not tasks:
        print("Toutes les combinaisons sont déjà calculées. Rien à faire.")
        return

    total_all = len(W_ERROR_GRID) * len(SEEDS)
    print(f"Exp. 0 — {len(tasks)} tâches restantes sur {total_all} "
          f"({total_all - len(tasks)} déjà faites)")

    # Charger les trajectoires
    trajectories, n_original_points = preload_trajectories(args.max_files)

    # Ouvrir le CSV (append)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # Nombre de workers
    import os
    n_workers = args.workers or max(1, os.cpu_count() - 1)

    # On parallélise groupé par w_error : les 7 seeds partagent la même compression
    # Pour optimiser, on compresse 1 fois par w_error, puis on clone pour les seeds
    t_global = time.perf_counter()
    completed = total_all - len(tasks)

    # Regrouper par w_error
    from collections import defaultdict
    tasks_by_w = defaultdict(list)
    for w, s in tasks:
        tasks_by_w[w].append(s)

    for w_idx, (w_error, seeds_todo) in enumerate(sorted(tasks_by_w.items())):
        # Compression une seule fois pour ce w_error
        t0 = time.perf_counter()
        segments, n_raw = compress_all(trajectories, w_error, MIN_LENGTH)
        t_compress = time.perf_counter() - t0

        n_seg_total = len(segments)
        ratio_compression = n_original_points / max(n_raw, 1)
        lengths = segments_lengths(segments)
        longueur_moyenne = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
        longueur_std = float(np.std(lengths)) if len(lengths) > 0 else 0.0

        for seed in seeds_todo:
            t_seed_start = time.perf_counter()

            # Sous-échantillonnage
            rng = np.random.default_rng(seed)
            if n_seg_total > N_SUBSAMPLE:
                idx = rng.choice(n_seg_total, N_SUBSAMPLE, replace=False)
                idx.sort()
                segments_sample = [segments[i] for i in idx]
            else:
                segments_sample = list(segments)
            n_seg_sampled = len(segments_sample)

            if n_seg_sampled < K + 1:
                row = {
                    "w_error": w_error, "seed": seed,
                    "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
                    "ratio_compression": ratio_compression,
                    "longueur_moyenne": longueur_moyenne, "longueur_std": longueur_std,
                    "nb_original_points": n_original_points, "nb_raw_segments": n_raw,
                    "silhouette": "", "davies_bouldin": "", "calinski_harabasz": "",
                    "inertia": "", "n_clusters_effective": 0,
                    "temps_compression_s": t_compress, "temps_clustering_s": 0.0,
                    "temps_total_s": time.perf_counter() - t_seed_start,
                }
                writer.writerow(row)
                csv_file.flush()
                completed += 1
                print(f"  [{completed}/{total_all}] w={w_error:7.2f} seed={seed}  "
                      f"→ {n_seg_sampled} seg  SKIP (trop peu)")
                continue

            # Features
            X = segments_to_features(segments_sample)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # KMeans
            t0 = time.perf_counter()
            km = MiniBatchKMeans(
                n_clusters=K, random_state=seed,
                batch_size=min(4096, n_seg_sampled), n_init=3,
            )
            labels = km.fit_predict(X_scaled)
            t_cluster = time.perf_counter() - t0

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                sil = db = ch = ""
            else:
                sil = silhouette_score(
                    X_scaled, labels,
                    sample_size=min(5000, n_seg_sampled), random_state=seed,
                )
                db = davies_bouldin_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)

            row = {
                "w_error": w_error, "seed": seed,
                "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
                "ratio_compression": ratio_compression,
                "longueur_moyenne": longueur_moyenne, "longueur_std": longueur_std,
                "nb_original_points": n_original_points, "nb_raw_segments": n_raw,
                "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                "inertia": float(km.inertia_), "n_clusters_effective": n_unique,
                "temps_compression_s": t_compress, "temps_clustering_s": t_cluster,
                "temps_total_s": time.perf_counter() - t_seed_start,
            }
            writer.writerow(row)
            csv_file.flush()

            completed += 1
            elapsed = time.perf_counter() - t_global
            print(f"  [{completed}/{total_all}] w={w_error:7.2f} seed={seed}  "
                  f"sil={sil if sil != '' else 'N/A':>6}  "
                  f"DB={db if db != '' else 'N/A':>6}  "
                  f"t_comp={t_compress:.1f}s  t_clust={t_cluster:.1f}s  "
                  f"({elapsed:.0f}s total)")

        # Libérer la mémoire
        del segments
        gc.collect()

    csv_file.close()
    print(f"\n✓ Exp. 0 terminée — résultats dans {csv_path}")


if __name__ == "__main__":
    main()
