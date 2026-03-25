#!/usr/bin/env python3
"""Calibrate AP preference for different N_SUBSAMPLE values, then run full pipeline."""
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.config import (
    COMPRESSED_DIR, SEED, AP_DAMPING, AP_MAX_ITER,
)
from dota_analytics.clustering import load_data, compute_traclus_similarity
from sklearn.cluster import AffinityPropagation

TARGET_K = 10

def find_preference(S, target_k=10):
    """Linear scan then refine to find preference giving target_k clusters."""
    # Coarse scan
    candidates = []
    for pref in [-500, -1000, -2000, -3000, -4000, -5000, -6000, -7000, -8000, -10000]:
        ap = AffinityPropagation(
            affinity="precomputed", preference=pref,
            random_state=SEED, max_iter=AP_MAX_ITER, damping=AP_DAMPING,
        )
        labels = ap.fit_predict(S)
        k = len(np.unique(labels))
        # AP non-convergence returns k=N, skip those
        if k >= len(S) * 0.5:
            print(f"    pref={pref:.0f} -> k={k} (non-convergence, skip)")
            continue
        print(f"    pref={pref:.0f} -> k={k}")
        candidates.append((pref, k))
        if k <= target_k - 3:
            break

    if not candidates:
        return -5000, -1  # fallback

    # Find closest to target
    best = min(candidates, key=lambda x: abs(x[1] - target_k))
    if best[1] == target_k:
        return best

    # Refine between neighbors
    under = [c for c in candidates if c[1] <= target_k]
    over = [c for c in candidates if c[1] > target_k]
    if under and over:
        lo_pref = max(c[0] for c in under)   # less negative -> more clusters
        hi_pref = min(c[0] for c in over)     # more negative -> fewer clusters
        # Actually: lower pref = fewer clusters, higher pref = more clusters
        # So lo_pref gives k<=target, hi_pref gives k>target
        # Refine between them
        for _ in range(8):
            mid = (lo_pref + hi_pref) / 2
            ap = AffinityPropagation(
                affinity="precomputed", preference=mid,
                random_state=SEED, max_iter=AP_MAX_ITER, damping=AP_DAMPING,
            )
            labels = ap.fit_predict(S)
            k = len(np.unique(labels))
            if k >= len(S) * 0.5:
                hi_pref = mid
                continue
            print(f"    refine pref={mid:.0f} -> k={k}")
            if k == target_k:
                return mid, k
            elif k > target_k:
                hi_pref = mid
            else:
                lo_pref = mid
        # Return closest
        all_c = candidates + [(mid, k)]
        return min(all_c, key=lambda x: abs(x[1] - target_k))

    return best


def main():
    print("Loading all segments...")
    segs_all, meta_all = load_data(str(COMPRESSED_DIR), max_files=30, min_length=5.0)
    n_total = len(segs_all)
    print(f"  {n_total} segments total\n")

    results = []

    for n_sub in [3000, 5000, 8000, 10000, 15000, 18716]:
        n_sub = min(n_sub, n_total)
        print(f"{'='*60}")
        print(f"N_SUBSAMPLE = {n_sub}")
        print(f"{'='*60}")

        rng = np.random.default_rng(SEED)
        if n_total > n_sub:
            idx = rng.choice(n_total, n_sub, replace=False)
            idx.sort()
            segs = [segs_all[i] for i in idx]
        else:
            segs = segs_all

        print(f"  Computing TRACLUS matrix {n_sub}x{n_sub}...")
        t0 = time.perf_counter()
        S = compute_traclus_similarity(segs)
        t_mat = time.perf_counter() - t0
        print(f"  Matrix done in {t_mat:.1f}s")

        print(f"  Binary search for preference (target k={TARGET_K})...")
        t0 = time.perf_counter()
        pref, k = find_preference(S, target_k=TARGET_K)
        t_search = time.perf_counter() - t0
        print(f"  -> Best: pref={pref:.0f}, k={k}, search={t_search:.0f}s")

        # Final AP with best preference
        print(f"  Final AP fit...")
        t0 = time.perf_counter()
        ap = AffinityPropagation(
            affinity="precomputed", preference=pref,
            random_state=SEED, max_iter=500, damping=0.7,
        )
        labels = ap.fit_predict(S)
        t_ap = time.perf_counter() - t0
        k_final = len(np.unique(labels))

        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        feats = np.empty((len(segs), 5), dtype=np.float32)
        for i, s in enumerate(segs):
            feats[i, 0] = (s.start.x + s.end.x) * 0.5
            feats[i, 1] = (s.start.y + s.end.y) * 0.5
            feats[i, 2] = s.end.x - s.start.x
            feats[i, 3] = s.end.y - s.start.y
            feats[i, 4] = s.length()
        X = StandardScaler().fit_transform(feats)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(segs)), random_state=SEED)

        # Cluster sizes
        sizes = np.bincount(labels)
        balance = sizes.min() / sizes.max()

        results.append({
            "n_sub": n_sub, "pref": pref, "k": k_final,
            "silhouette": sil, "balance": balance,
            "t_matrix": t_mat, "t_ap": t_ap,
        })

        print(f"  k={k_final}, silhouette={sil:.4f}, balance={balance:.2f}")
        print(f"  Cluster sizes: {sorted(sizes)}")
        print()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'N':>7} | {'pref':>8} | {'k':>3} | {'silhouette':>10} | {'balance':>7} | {'t_mat':>6} | {'t_ap':>6}")
    print("-"*70)
    for r in results:
        print(f"{r['n_sub']:>7} | {r['pref']:>8.0f} | {r['k']:>3} | {r['silhouette']:>10.4f} | {r['balance']:>7.2f} | {r['t_matrix']:>5.1f}s | {r['t_ap']:>5.1f}s")


def plot_ap_sensitivity(output_dir=None):
    """Génère la figure de sensibilité AP (4 panels) à partir des données de calibration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmark.config import OUTPUT_EXP3

    if output_dir is None:
        output_dir = OUTPUT_EXP3
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Données issues des runs de calibration (tuned pour k≈10)
    N_values =      [3000,   5000,   8000,   10000]
    k_values =      [10,     10,     12,     14]
    silhouette =    [0.0845, 0.0823, 0.0817, 0.0707]
    balance =       [0.48,   0.44,   0.47,   0.34]
    preference =    [-5000,  -8000,  -10000, -10000]

    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.suptitle("Sensibilité d'Affinity Propagation au nombre de segments ($N$)",
                 fontsize=14, fontweight="bold", y=0.97)

    colors = ["#2ecc71" if n == 3000 else "#95a5a6" for n in N_values]

    # Panel 1: Silhouette
    ax = axes[0, 0]
    bars = ax.bar([str(n) for n in N_values], silhouette, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Score de silhouette"); ax.set_xlabel("$N$ (segments)")
    ax.set_title("Silhouette (↑ = mieux)")
    ax.axhline(y=silhouette[0], color="#2ecc71", linestyle="--", alpha=0.4)
    for i, (v, bar) in enumerate(zip(silhouette, bars)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
    ax.set_ylim(0, max(silhouette) * 1.25)

    # Panel 2: k obtenu
    ax = axes[0, 1]
    bars = ax.bar([str(n) for n in N_values], k_values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Nombre de clusters ($k$)"); ax.set_xlabel("$N$ (segments)")
    ax.set_title("$k$ obtenu (cible = 10)")
    ax.axhline(y=10, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5, label="Cible $k=10$")
    ax.legend(fontsize=9)
    for i, (v, bar) in enumerate(zip(k_values, bars)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold" if i == 0 else "normal")
    ax.set_ylim(0, max(k_values) * 1.3)

    # Panel 3: Balance
    ax = axes[1, 0]
    bars = ax.bar([str(n) for n in N_values], balance, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Équilibre (min/max)"); ax.set_xlabel("$N$ (segments)")
    ax.set_title("Équilibre des clusters (↑ = mieux)")
    for i, (v, bar) in enumerate(zip(balance, bars)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
    ax.set_ylim(0, 0.7)

    # Panel 4: Preference requise
    ax = axes[1, 1]
    pref_abs = [-p for p in preference]
    bars = ax.bar([str(n) for n in N_values], pref_abs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("|preference|"); ax.set_xlabel("$N$ (segments)")
    ax.set_title("Preference requise pour $k \\approx 10$")
    for i, (v, bar) in enumerate(zip(pref_abs, bars)):
        suffix = f"\n(k>{k_values[i]})" if i >= 2 else ""
        ax.text(bar.get_x() + bar.get_width()/2, v + 150, f"{v}" + suffix,
                ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
    ax.set_ylim(0, max(pref_abs) * 1.3)

    fig.text(0.5, 0.01,
             "Vert = configuration optimale retenue ($N$=3000, preference=−5000, $k$=10).  "
             "Pour $N$ ≥ 8000, AP ne peut plus atteindre $k$=10.",
             ha="center", fontsize=10, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = Path(output_dir) / "fig_ap_sensitivity_N.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="Calibrate AP preference / generate sensitivity plot")
    p.add_argument("--plot", action="store_true", help="Générer uniquement la figure de sensibilité (sans relancer la calibration)")
    a = p.parse_args()
    if a.plot:
        plot_ap_sensitivity()
    else:
        main()
