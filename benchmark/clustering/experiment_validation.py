#!/usr/bin/env python3
"""
experiment_validation.py
========================
Deux expériences de validation méthodologique :

  Exp A – Validation croisée (split 50/50)
  -----------------------------------------
  Sépare les fichiers compressés en deux moitiés (calibration / validation).
  Entraîne AP sur un sous-échantillon de la moitié calibration,
  puis réassigne les segments de la moitié validation aux exemplaires obtenus.
  Compare Silhouette, Davies-Bouldin et k obtenu entre les deux splits.

  Exp B – Stabilité bootstrap de l'échantillonnage AP
  ----------------------------------------------------
  Tire 20 sous-échantillons de 3000 segments (graines différentes),
  lance AP sur chacun, puis mesure :
    - Adjusted Rand Index (ARI) entre chaque paire de tirages
    - Distance de Hausdorff entre les ensembles d'exemplaires
    - Stabilité du nombre de clusters k

Sorties
-------
  output/benchmark_validation/cross_validation_results.csv
  output/benchmark_validation/bootstrap_stability_results.csv
  output/benchmark_validation/bootstrap_ari_matrix.csv
  output/benchmark_validation/validation_plots.png
"""

import sys
import os
import csv
import time
import itertools
from pathlib import Path

_THREADS = str(max(1, (os.cpu_count() or 1) // 3))
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import directed_hausdorff

from dota_analytics.clustering import load_data, compute_traclus_similarity
from dota_analytics.custom_ap import CustomAffinityPropagation
from benchmark.config import (
    COMPRESSED_DIR, AP_PREFERENCE, AP_DAMPING, AP_MAX_ITER, MIN_LENGTH, SEED,
)

# ── Paramètres ────────────────────────────────────────────────────────────
N_SUBSAMPLE = 3000        # Taille de l'échantillon AP
N_BOOTSTRAP = 10          # Nombre de tirages bootstrap
MAX_FILES_TOTAL = 70      # Corpus total (70 matchs)

OUT_DIR = PROJECT_ROOT / "output" / "benchmark_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Utilitaires ───────────────────────────────────────────────────────────

def _subsample(segments, metadata, n, rng):
    """Tire n segments sans remise."""
    idx = rng.choice(len(segments), size=min(n, len(segments)), replace=False)
    return [segments[i] for i in idx], [metadata[i] for i in idx], idx


def _run_ap(similarity_matrix, preference, damping, max_iter):
    """Lance AP et renvoie (labels, exemplar_indices, n_clusters)."""
    S = similarity_matrix.copy()
    np.fill_diagonal(S, preference)
    ap = CustomAffinityPropagation(damping=damping, max_iter=max_iter, verbose=False)
    ap.fit(S)
    return ap.labels_, ap.cluster_centers_indices_


def _assign_to_exemplars(all_segments, exemplar_segments):
    """Assigne chaque segment au plus proche exemplaire via TRACLUS."""
    # Calcul des features (milieu, direction, longueur)
    def _features(segs):
        feats = []
        for s in segs:
            mx = (s.start.x + s.end.x) / 2.0
            my = (s.start.y + s.end.y) / 2.0
            dx = s.end.x - s.start.x
            dy = s.end.y - s.start.y
            ln = np.sqrt(dx**2 + dy**2)
            feats.append([mx, my, dx, dy, ln])
        return np.array(feats, dtype=np.float32)

    F_all = _features(all_segments)
    F_ex = _features(exemplar_segments)
    # Distance euclidienne dans l'espace des features
    diff = F_all[:, np.newaxis, :] - F_ex[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    labels = np.argmin(dists, axis=1)
    return labels


def _hausdorff_exemplars(ex_a, ex_b, segments_a, segments_b):
    """Distance de Hausdorff entre deux ensembles d'exemplaires (coordonnées milieu)."""
    def _midpoints(segs, indices):
        pts = []
        for i in indices:
            s = segs[i]
            pts.append([(s.start.x + s.end.x) / 2, (s.start.y + s.end.y) / 2])
        return np.array(pts)

    A = _midpoints(segments_a, ex_a)
    B = _midpoints(segments_b, ex_b)
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return max(d1, d2)


# ══════════════════════════════════════════════════════════════════════════
# EXP A : VALIDATION CROISÉE 50/50
# ══════════════════════════════════════════════════════════════════════════

def run_cross_validation():
    print("=" * 70)
    print("EXP A : Validation croisée (split 50/50)")
    print("=" * 70)

    # Charger tous les fichiers compressés (70 matchs)
    folder = Path(COMPRESSED_DIR)
    all_files = sorted(folder.glob("*.json"))[:MAX_FILES_TOTAL]
    n_files = len(all_files)
    print(f"  {n_files} fichiers disponibles.")

    # Split déterministe
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_files)
    half = n_files // 2
    split_cal = sorted(perm[:half])
    split_val = sorted(perm[half:])
    print(f"  Calibration : {len(split_cal)} fichiers  |  Validation : {len(split_val)} fichiers")

    results = []

    for split_name, split_indices in [("calibration", split_cal), ("validation", split_val)]:
        print(f"\n  --- Split : {split_name} ({len(split_indices)} fichiers) ---")

        # Charger uniquement les fichiers du split
        split_files = [all_files[i] for i in split_indices]
        segments = []
        metadata = []
        for fp in split_files:
            import json
            with open(fp, "r") as f:
                data = json.load(f)
            match_id = str(data.get("match_id", "unknown"))
            if "players" not in data:
                continue
            from dota_analytics.structures import TrajectoryPoint, Segment
            for player in data["players"]:
                p_id = player["player_id"]
                for idx, s in enumerate(player["segments"]):
                    try:
                        p1 = TrajectoryPoint(s["start"]["x"], s["start"]["y"], s["start"]["tick"])
                        p2 = TrajectoryPoint(s["end"]["x"], s["end"]["y"], s["end"]["tick"])
                        seg = Segment(p1, p2)
                        if seg.length() > MIN_LENGTH:
                            segments.append(seg)
                            metadata.append({"match_id": match_id, "seg_id": f"P{p_id}_{idx}"})
                    except KeyError:
                        continue

        n_total = len(segments)
        print(f"  {n_total} segments chargés.")

        # Sous-échantillonner pour AP
        rng_sub = np.random.default_rng(SEED)
        sub_segs, sub_meta, sub_idx = _subsample(segments, metadata, N_SUBSAMPLE, rng_sub)
        n_sub = len(sub_segs)
        print(f"  Sous-échantillon AP : {n_sub} segments.")

        # Matrice TRACLUS
        t0 = time.perf_counter()
        S = compute_traclus_similarity(sub_segs)
        t_matrix = time.perf_counter() - t0
        print(f"  Matrice TRACLUS : {t_matrix:.1f}s", flush=True)

        # AP
        t0 = time.perf_counter()
        labels_sub, exemplar_idx = _run_ap(S, AP_PREFERENCE, AP_DAMPING, AP_MAX_ITER)
        t_ap = time.perf_counter() - t0
        k = len(exemplar_idx) if exemplar_idx is not None else 0
        print(f"  AP : {k} clusters en {t_ap:.1f}s", flush=True)

        if k < 2:
            print(f"  ⚠ Seulement {k} cluster(s), skip métriques.")
            results.append({"split": split_name, "n_files": len(split_indices),
                            "n_segments": n_total, "n_subsample": n_sub,
                            "k": k, "silhouette": None, "davies_bouldin": None})
            continue

        # Réassigner TOUS les segments du split aux exemplaires
        exemplar_segs = [sub_segs[i] for i in exemplar_idx]
        labels_all = _assign_to_exemplars(segments, exemplar_segs)

        # Métriques sur features (pour éviter une matrice N×N)
        feats = []
        for s in segments:
            mx = (s.start.x + s.end.x) / 2.0
            my = (s.start.y + s.end.y) / 2.0
            dx = s.end.x - s.start.x
            dy = s.end.y - s.start.y
            ln = np.sqrt(dx**2 + dy**2)
            feats.append([mx, my, dx, dy, ln])
        X = np.array(feats, dtype=np.float32)

        sil = silhouette_score(X, labels_all)
        db = davies_bouldin_score(X, labels_all)
        print(f"  Silhouette (full split) = {sil:.4f}")
        print(f"  Davies-Bouldin          = {db:.4f}")

        results.append({
            "split": split_name, "n_files": len(split_indices),
            "n_segments": n_total, "n_subsample": n_sub,
            "k": k, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
        })

    # Sauvegarde CSV
    csv_path = OUT_DIR / "cross_validation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  Résultats → {csv_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# EXP B : STABILITÉ BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════

def run_bootstrap_stability():
    print("\n" + "=" * 70)
    print("EXP B : Stabilité bootstrap de l'échantillonnage AP")
    print("=" * 70)

    # Charger tous les segments (30 matchs comme dans le pipeline)
    segments, metadata = load_data(COMPRESSED_DIR, max_files=30, min_length=MIN_LENGTH)
    n_total = len(segments)
    print(f"  {n_total} segments chargés (30 matchs).")

    bootstrap_results = []
    all_labels = []      # labels sur l'échantillon (pour ARI)
    all_exemplars = []   # indices exemplaires
    all_sub_segs = []    # segments sous-échantillonnés
    all_sub_idx = []     # indices globaux du sous-échantillon

    for b in range(N_BOOTSTRAP):
        seed_b = SEED + b
        rng = np.random.default_rng(seed_b)
        sub_segs, sub_meta, sub_idx = _subsample(segments, metadata, N_SUBSAMPLE, rng)
        n_sub = len(sub_segs)

        t0 = time.perf_counter()
        S = compute_traclus_similarity(sub_segs)
        labels, exemplar_idx = _run_ap(S, AP_PREFERENCE, AP_DAMPING, AP_MAX_ITER)
        elapsed = time.perf_counter() - t0

        k = len(exemplar_idx) if exemplar_idx is not None else 0

        # Stocker pour ARI pairwise
        all_labels.append((sub_idx, labels))
        all_exemplars.append(exemplar_idx)
        all_sub_segs.append(sub_segs)
        all_sub_idx.append(sub_idx)

        # Silhouette sur l'échantillon (matrice déjà disponible)
        sil = None
        if k >= 2:
            D = -S
            np.fill_diagonal(D, 0)
            sil = silhouette_score(D, labels, metric="precomputed")

        bootstrap_results.append({
            "bootstrap_id": b, "seed": seed_b, "n_subsample": n_sub,
            "k": k, "silhouette": round(sil, 4) if sil is not None else None,
            "time_s": round(elapsed, 2),
        })
        sil_str = f"{sil:.4f}" if sil is not None else "  N/A  "
        print(f"  Bootstrap {b:2d} (seed={seed_b:3d}) : k={k:2d}  sil={sil_str}  {elapsed:.1f}s", flush=True)

    # ── ARI pairwise ──────────────────────────────────────────────────────
    print("\n  Calcul ARI pairwise...")
    n_b = len(all_labels)
    ari_matrix = np.full((n_b, n_b), np.nan)

    for i in range(n_b):
        for j in range(i, n_b):
            idx_i, lab_i = all_labels[i]
            idx_j, lab_j = all_labels[j]
            # Intersection des indices
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 50:
                continue
            # Mappage
            map_i = {v: pos for pos, v in enumerate(idx_i)}
            map_j = {v: pos for pos, v in enumerate(idx_j)}
            li = [lab_i[map_i[c]] for c in common]
            lj = [lab_j[map_j[c]] for c in common]
            ari = adjusted_rand_score(li, lj)
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari

    # ── Hausdorff entre exemplaires ───────────────────────────────────────
    print("  Calcul Hausdorff entre exemplaires...")
    hausdorff_values = []
    for i in range(n_b):
        for j in range(i + 1, n_b):
            if all_exemplars[i] is not None and all_exemplars[j] is not None:
                h = _hausdorff_exemplars(
                    all_exemplars[i], all_exemplars[j],
                    all_sub_segs[i], all_sub_segs[j]
                )
                hausdorff_values.append(h)

    # ── Statistiques ──────────────────────────────────────────────────────
    ks = [r["k"] for r in bootstrap_results]
    sils = [r["silhouette"] for r in bootstrap_results if r["silhouette"] is not None]
    ari_upper = ari_matrix[np.triu_indices(n_b, k=1)]
    ari_valid = ari_upper[~np.isnan(ari_upper)]

    print(f"\n  ── Résumé Bootstrap ({N_BOOTSTRAP} tirages) ──")
    print(f"  k :         mean={np.mean(ks):.1f}  std={np.std(ks):.2f}  range=[{np.min(ks)}, {np.max(ks)}]")
    print(f"  Silhouette : mean={np.mean(sils):.4f}  std={np.std(sils):.4f}")
    if len(ari_valid) > 0:
        print(f"  ARI :        mean={np.mean(ari_valid):.4f}  std={np.std(ari_valid):.4f}  min={np.min(ari_valid):.4f}")
    if len(hausdorff_values) > 0:
        print(f"  Hausdorff :  mean={np.mean(hausdorff_values):.2f}  std={np.std(hausdorff_values):.2f}")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────
    csv_path = OUT_DIR / "bootstrap_stability_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bootstrap_results[0].keys())
        w.writeheader()
        w.writerows(bootstrap_results)

    ari_csv = OUT_DIR / "bootstrap_ari_matrix.csv"
    np.savetxt(ari_csv, ari_matrix, delimiter=",", fmt="%.4f")

    print(f"\n  Résultats → {csv_path}")
    print(f"  Matrice ARI → {ari_csv}")

    return bootstrap_results, ari_matrix, hausdorff_values, ks, sils


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

def plot_results(cv_results, boot_results, ari_matrix, hausdorff_values, ks, sils):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Validation méthodologique du pipeline", fontsize=14, fontweight="bold")

    # ── A1 : Cross-validation barres ──────────────────────────────────────
    ax = axes[0, 0]
    splits = [r["split"] for r in cv_results]
    sils_cv = [r["silhouette"] if r["silhouette"] else 0 for r in cv_results]
    ax.bar(splits, sils_cv, color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Exp A : Silhouette Cal. vs Val.")
    for i, v in enumerate(sils_cv):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    # ── A2 : Cross-validation DB ──────────────────────────────────────────
    ax = axes[0, 1]
    dbs = [r["davies_bouldin"] if r["davies_bouldin"] else 0 for r in cv_results]
    ax.bar(splits, dbs, color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Davies-Bouldin (↓ meilleur)")
    ax.set_title("Exp A : Davies-Bouldin Cal. vs Val.")
    for i, v in enumerate(dbs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    # ── A3 : Cross-validation k ───────────────────────────────────────────
    ax = axes[0, 2]
    ks_cv = [r["k"] for r in cv_results]
    ax.bar(splits, ks_cv, color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Nombre de clusters k")
    ax.set_title("Exp A : k Cal. vs Val.")
    for i, v in enumerate(ks_cv):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)

    # ── B1 : Distribution k bootstrap ─────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(ks, bins=range(min(ks), max(ks) + 2), color="#4CAF50", edgecolor="black", align="left")
    ax.axvline(np.mean(ks), color="red", linestyle="--", label=f"μ={np.mean(ks):.1f}")
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Exp B : Distribution de k ({N_BOOTSTRAP} tirages)")
    ax.legend()

    # ── B2 : Heatmap ARI ─────────────────────────────────────────────────
    ax = axes[1, 1]
    im = ax.imshow(ari_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Bootstrap ID")
    ax.set_ylabel("Bootstrap ID")
    ax.set_title(f"Exp B : Matrice ARI (μ={np.nanmean(ari_matrix[np.triu_indices(len(ari_matrix), k=1)]):.3f})")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ── B3 : Histogramme Hausdorff ────────────────────────────────────────
    ax = axes[1, 2]
    if len(hausdorff_values) > 0:
        ax.hist(hausdorff_values, bins=15, color="#9C27B0", edgecolor="black")
        ax.axvline(np.mean(hausdorff_values), color="red", linestyle="--",
                   label=f"μ={np.mean(hausdorff_values):.1f}")
        ax.set_xlabel("Distance de Hausdorff")
        ax.set_ylabel("Fréquence (paires)")
        ax.set_title(f"Exp B : Hausdorff entre exemplaires")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plot_path = OUT_DIR / "validation_plots.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  Figure → {plot_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (Exp A)")
    parser.add_argument("--skip-boot", action="store_true", help="Skip bootstrap (Exp B)")
    args = parser.parse_args()

    print(f"Projet racine : {PROJECT_ROOT}")
    print(f"Dossier compressé : {COMPRESSED_DIR}")
    print(f"Sortie : {OUT_DIR}\n", flush=True)

    cv_results = None
    if not args.skip_cv:
        cv_results = run_cross_validation()
    else:
        # Charger résultats existants
        cv_csv = OUT_DIR / "cross_validation_results.csv"
        if cv_csv.exists():
            import csv as _csv
            with open(cv_csv) as _f:
                reader = _csv.DictReader(_f)
                cv_results = []
                for row in reader:
                    for key in ("k", "n_files", "n_segments", "n_subsample"):
                        row[key] = int(row[key])
                    for key in ("silhouette", "davies_bouldin"):
                        row[key] = float(row[key]) if row[key] else None
                    cv_results.append(row)
            print("  Exp A : résultats chargés depuis CSV.\n", flush=True)

    boot_results, ari_matrix, hausdorff_values, ks, sils = None, None, None, None, None
    if not args.skip_boot:
        boot_results, ari_matrix, hausdorff_values, ks, sils = run_bootstrap_stability()

    if cv_results and boot_results:
        plot_results(cv_results, boot_results, ari_matrix, hausdorff_values, ks, sils)

    print("\n✅ Terminé.")
