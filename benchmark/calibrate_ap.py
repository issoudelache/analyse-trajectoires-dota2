#!/usr/bin/env python3
"""Calibrate AP preference for different N_SUBSAMPLE values, then run full pipeline."""
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.clustering import load_data, compute_traclus_similarity
from sklearn.cluster import AffinityPropagation

COMPRESSED_DIR = BASE_DIR / "output" / "compressed" / "w_error_12.0"
SEED = 42
TARGET_K = 10

def find_preference(S, target_k=10):
    """Linear scan then refine to find preference giving target_k clusters."""
    # Coarse scan
    candidates = []
    for pref in [-500, -1000, -2000, -3000, -4000, -5000, -6000, -7000, -8000, -10000]:
        ap = AffinityPropagation(
            affinity="precomputed", preference=pref,
            random_state=SEED, max_iter=500, damping=0.7,
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
                random_state=SEED, max_iter=500, damping=0.7,
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


if __name__ == "__main__":
    main()
