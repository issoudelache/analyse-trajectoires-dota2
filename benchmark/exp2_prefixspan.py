#!/usr/bin/env python3
"""
EXP. 2 — Sensibilité de PrefixSpan au k (sémantique).

Question : Dans quelle plage de k PrefixSpan extrait-il des motifs
           non-triviaux et exploitables ?

Fixé  : w_error=12.0, algo=kmeans, max_files=30, min_length=5.0,
        min_support=15, max_length=5, seed_kmeans=42
Variable : k (30 valeurs, même grille que Exp. 1)

Pipeline par k :
  1. KMeans(k, seed=42) → labels
  2. Recoder les trajectoires en séquences SPMF
  3. PrefixSpan(min_support=15, max_length=5)
  4. Mesurer

Métriques :
  nb_motifs_total, nb_motifs_len2, nb_motifs_len3,
  nb_aretes_markov, support_median,
  entropie_shannon, temps_prefixspan_s

Écriture incrémentale : chaque k → 1 ligne CSV appendée.
"""

import argparse
import csv
import json
import sys
import time
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.config import (
    COMPRESSED_DIR, OUTPUT_EXP2 as OUTPUT_DIR,
    W_ERROR, MAX_FILES, MIN_LENGTH,
    MIN_SUPPORT, MAX_LENGTH, SEED as SEED_KMEANS,
    K_GRID_30 as K_GRID,
)
from dota_analytics.clustering import load_data
from dota_analytics.mining import PrefixSpan
from dota_analytics.recoding import reconstruct_sequences, save_sequences_to_spmf

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

CSV_COLUMNS = [
    "k", "nb_segments", "nb_sequences",
    "nb_motifs_total", "nb_motifs_len2", "nb_motifs_len3",
    "nb_aretes_markov", "support_median", "support_mean", "support_max",
    "entropie_shannon",
    "temps_clustering_s", "temps_recoding_s", "temps_prefixspan_s", "temps_total_s",
    "n_clusters_effective",
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
# CLUSTERING INLINE (labels seulement, pas de sauvegarde JSON)
# ═════════════════════════════════════════════════════════════════════════════

def cluster_segments(segments, metadata, k, seed):
    """KMeans sur segments → labels + reconstruire la structure cluster JSON."""
    X = segments_to_features(segments)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed,
        batch_size=min(4096, len(segments)), n_init=3,
    )
    labels = km.fit_predict(X_scaled)

    # Reconstruire la structure {match_id: {seg_id: label}}
    match_clusters = {}
    for idx, label in enumerate(labels):
        m_id = metadata[idx]["match_id"]
        s_id = metadata[idx]["seg_id"]
        if m_id not in match_clusters:
            match_clusters[m_id] = {}
        match_clusters[m_id][s_id] = int(label)

    return labels, match_clusters


# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES PREFIXSPAN
# ═════════════════════════════════════════════════════════════════════════════

def compute_prefixspan_metrics(patterns, min_support):
    """Calcule toutes les métriques à partir des résultats PrefixSpan.

    Args:
        patterns: {(items...): support_count}
        min_support: seuil utilisé

    Returns:
        dict de métriques
    """
    if not patterns:
        return {
            "nb_motifs_total": 0, "nb_motifs_len2": 0, "nb_motifs_len3": 0,
            "nb_aretes_markov": 0, "support_median": 0, "support_mean": 0.0,
            "support_max": 0, "entropie_shannon": 0.0,
        }

    supports = np.array(list(patterns.values()), dtype=np.float64)
    lengths = np.array([len(p) for p in patterns.keys()])

    nb_total = len(patterns)
    nb_len2 = int(np.sum(lengths >= 2))
    nb_len3 = int(np.sum(lengths >= 3))

    # Arêtes Markov : transitions a→b distinctes parmi les motifs de longueur >= 2
    edges = set()
    for pattern, sup in patterns.items():
        if len(pattern) >= 2 and sup >= min_support:
            for i in range(len(pattern) - 1):
                edges.add((pattern[i], pattern[i + 1]))
    nb_aretes = len(edges)

    # Entropie de Shannon des supports
    p = supports / supports.sum()
    # Éviter log(0)
    p_nonzero = p[p > 0]
    entropie = -float(np.sum(p_nonzero * np.log2(p_nonzero)))

    return {
        "nb_motifs_total": nb_total,
        "nb_motifs_len2": nb_len2,
        "nb_motifs_len3": nb_len3,
        "nb_aretes_markov": nb_aretes,
        "support_median": float(np.median(supports)),
        "support_mean": float(np.mean(supports)),
        "support_max": int(np.max(supports)),
        "entropie_shannon": entropie,
    }


# ═════════════════════════════════════════════════════════════════════════════
# REPRISE INCRÉMENTALE
# ═════════════════════════════════════════════════════════════════════════════

def load_done(csv_path):
    """Charge les k déjà enregistrés."""
    done = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add(int(row["k"]))
                except (KeyError, ValueError):
                    continue
    return done


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp. 2 — Sensibilité PrefixSpan")
    parser.add_argument("--max_files", type=int, default=MAX_FILES)
    parser.add_argument("--compressed_dir", type=str, default=str(COMPRESSED_DIR))
    parser.add_argument("--min_support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    compressed_dir = Path(args.compressed_dir)
    if not compressed_dir.exists():
        print(f"ERREUR : dossier compressé introuvable : {compressed_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "exp2_results.csv"

    done = load_done(csv_path)
    tasks = [k for k in K_GRID if k not in done]
    total_all = len(K_GRID)

    if not tasks:
        print("Toutes les valeurs de k sont déjà calculées. Rien à faire.")
        return

    print(f"Exp. 2 — {len(tasks)} tâches restantes sur {total_all} "
          f"({total_all - len(tasks)} déjà faites)")
    print(f"  min_support={args.min_support}, max_length={args.max_length}")

    # Charger segments
    print(f"Chargement des segments depuis {compressed_dir}…")
    segments, metadata = load_data(
        str(compressed_dir), max_files=args.max_files, min_length=MIN_LENGTH,
    )
    n_seg = len(segments)
    print(f"  → {n_seg} segments chargés")

    # CSV (append)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # Dossier temporaire pour les SPMF
    tmp_dir = OUTPUT_DIR / "tmp_spmf"
    tmp_dir.mkdir(exist_ok=True)

    t_global = time.perf_counter()
    completed = total_all - len(tasks)

    for k in sorted(tasks):
        t_total_start = time.perf_counter()
        print(f"\n  ── k={k} ──")

        if k >= n_seg:
            row = {c: "" for c in CSV_COLUMNS}
            row["k"] = k
            row["nb_segments"] = n_seg
            row["nb_sequences"] = 0
            writer.writerow(row)
            csv_file.flush()
            completed += 1
            print(f"    SKIP (k >= n_segments)")
            continue

        # 1) Clustering
        t0 = time.perf_counter()
        labels, match_clusters = cluster_segments(segments, metadata, k, SEED_KMEANS)
        t_clustering = time.perf_counter() - t0
        n_clusters_eff = len(np.unique(labels))
        print(f"    Clustering: {n_clusters_eff} clusters, {t_clustering:.1f}s")

        # 2) Recodage → séquences SPMF
        t0 = time.perf_counter()
        sequences = reconstruct_sequences(match_clusters)
        spmf_path = tmp_dir / f"sequences_k{k}.spmf"
        save_sequences_to_spmf(sequences, str(spmf_path))
        t_recoding = time.perf_counter() - t0
        nb_sequences = len(sequences)
        print(f"    Recodage: {nb_sequences} séquences, {t_recoding:.1f}s")

        # 3) PrefixSpan
        t0 = time.perf_counter()
        miner = PrefixSpan(min_support=args.min_support, max_length=args.max_length)
        db = miner.load_spmf(str(spmf_path))
        patterns = miner.mine(db, parallel=True)
        t_prefixspan = time.perf_counter() - t0
        print(f"    PrefixSpan: {len(patterns)} motifs, {t_prefixspan:.1f}s")

        # Sauvegarder motifs pour analyse ultérieure
        patterns_path = OUTPUT_DIR / f"patterns_k{k}.spmf"
        miner.save_results_to_spmf(str(patterns_path))

        # 4) Métriques
        metrics = compute_prefixspan_metrics(patterns, args.min_support)
        t_total = time.perf_counter() - t_total_start

        row = {
            "k": k,
            "nb_segments": n_seg,
            "nb_sequences": nb_sequences,
            **metrics,
            "temps_clustering_s": t_clustering,
            "temps_recoding_s": t_recoding,
            "temps_prefixspan_s": t_prefixspan,
            "temps_total_s": t_total,
            "n_clusters_effective": n_clusters_eff,
        }
        writer.writerow(row)
        csv_file.flush()

        completed += 1
        elapsed = time.perf_counter() - t_global
        print(f"    [{completed}/{total_all}] motifs={metrics['nb_motifs_total']}  "
              f"arêtes={metrics['nb_aretes_markov']}  "
              f"H={metrics['entropie_shannon']:.2f}  "
              f"t_ps={t_prefixspan:.1f}s  ({elapsed:.0f}s total)")

    csv_file.close()

    # Nettoyage tmp
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"\n✓ Exp. 2 terminée — résultats dans {csv_path}")


if __name__ == "__main__":
    main()
