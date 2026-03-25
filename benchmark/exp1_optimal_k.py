#!/usr/bin/env python3
"""
EXP. 1 — Recherche du k optimal (géométrie du clustering).

Question : Quel k maximise la qualité géométrique des clusters ?

Fixé  : w_error=12.0, algo=kmeans, max_files=30, min_length=5.0, N=5000, seeds=7
Variable : k (30 valeurs de 2 à 100)

Métriques :
  silhouette, davies_bouldin, calinski_harabasz, inertie, temps_kmeans

Écriture incrémentale : chaque (k, seed) → 1 ligne CSV appendée.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.config import (
    COMPRESSED_DIR, OUTPUT_EXP1 as OUTPUT_DIR,
    W_ERROR, MAX_FILES, MIN_LENGTH,
    N_SUBSAMPLE_EXP1 as N_SUBSAMPLE, SEEDS_MULTI as SEEDS,
    K_GRID_30 as K_GRID,
)
from dota_analytics.clustering import load_data

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

CSV_COLUMNS = [
    "k", "seed", "nb_segments_total", "nb_segments_sampled",
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "inertia", "n_clusters_effective",
    "temps_clustering_s",
]


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# REPRISE INCRÉMENTALE
# ═════════════════════════════════════════════════════════════════════════════

def load_done(csv_path):
    """Charge les (k, seed) déjà enregistrés."""
    done = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add((int(row["k"]), int(row["seed"])))
                except (KeyError, ValueError):
                    continue
    return done


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp. 1 — Recherche du k optimal")
    parser.add_argument("--max_files", type=int, default=MAX_FILES)
    parser.add_argument("--compressed_dir", type=str, default=str(COMPRESSED_DIR),
                        help="Dossier des JSON compressés (w_error=12.0)")
    args = parser.parse_args()

    compressed_dir = Path(args.compressed_dir)
    if not compressed_dir.exists():
        print(f"ERREUR : dossier compressé introuvable : {compressed_dir}")
        print("Lancez d'abord : python run.py compress --w_error 12.0 --max_files 30")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "exp1_results.csv"

    # Reprise
    done = load_done(csv_path)
    tasks = [(k, s) for k in K_GRID for s in SEEDS if (k, s) not in done]
    total_all = len(K_GRID) * len(SEEDS)

    if not tasks:
        print("Toutes les combinaisons sont déjà calculées. Rien à faire.")
        return

    print(f"Exp. 1 — {len(tasks)} tâches restantes sur {total_all} "
          f"({total_all - len(tasks)} déjà faites)")

    # Charger les segments
    print(f"Chargement des segments depuis {compressed_dir}…")
    segments, metadata = load_data(
        str(compressed_dir), max_files=args.max_files, min_length=MIN_LENGTH,
    )
    n_seg_total = len(segments)
    print(f"  → {n_seg_total} segments chargés")

    if n_seg_total < max(K_GRID) + 1:
        print(f"ERREUR : pas assez de segments ({n_seg_total}) pour k_max={max(K_GRID)}")
        sys.exit(1)

    # Sous-échantillonnage fixe (seed=42 pour la base)
    rng_base = np.random.default_rng(42)
    if n_seg_total > N_SUBSAMPLE:
        idx_base = rng_base.choice(n_seg_total, N_SUBSAMPLE, replace=False)
        idx_base.sort()
        segments_base = [segments[i] for i in idx_base]
    else:
        segments_base = segments
    n_seg_sampled = len(segments_base)
    print(f"  → {n_seg_sampled} segments après sous-échantillonnage")

    # Features + normalisation (une seule fois)
    X = segments_to_features(segments_base)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CSV (append)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    t_global = time.perf_counter()
    completed = total_all - len(tasks)

    # Trier les tâches par k pour un affichage propre
    tasks.sort(key=lambda x: (x[0], x[1]))

    for k, seed in tasks:
        if k >= n_seg_sampled:
            # Skip impossible k
            row = {
                "k": k, "seed": seed,
                "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
                "silhouette": "", "davies_bouldin": "", "calinski_harabasz": "",
                "inertia": "", "n_clusters_effective": 0,
                "temps_clustering_s": 0.0,
            }
            writer.writerow(row)
            csv_file.flush()
            completed += 1
            print(f"  [{completed}/{total_all}] k={k:3d} seed={seed}  SKIP (k >= n_samples)")
            continue

        t0 = time.perf_counter()
        km = MiniBatchKMeans(
            n_clusters=k, random_state=seed,
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
            "k": k, "seed": seed,
            "nb_segments_total": n_seg_total, "nb_segments_sampled": n_seg_sampled,
            "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
            "inertia": float(km.inertia_), "n_clusters_effective": n_unique,
            "temps_clustering_s": t_cluster,
        }
        writer.writerow(row)
        csv_file.flush()

        completed += 1
        elapsed = time.perf_counter() - t_global
        print(f"  [{completed}/{total_all}] k={k:3d} seed={seed}  "
              f"sil={sil if sil != '' else 'N/A':>7}  "
              f"inertia={float(km.inertia_):>10.0f}  "
              f"t={t_cluster:.2f}s  ({elapsed:.0f}s total)")

    csv_file.close()
    print(f"\n✓ Exp. 1 terminée — résultats dans {csv_path}")


if __name__ == "__main__":
    main()
