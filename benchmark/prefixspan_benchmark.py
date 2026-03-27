#!/usr/bin/env python3
"""
Benchmark PrefixSpan - Version parallélisée multi-cœurs.

Utilise tous les cœurs CPU disponibles pour exécuter les tests en parallèle.
"""

import argparse
import gc
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from dota_analytics.mining import PrefixSpan


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = BASE_DIR / "output" / "benchmark_prefixspan"
DEFAULT_SPMF = BASE_DIR / "output" / "sequences.spmf"

# Valeurs de test
SUPPORT_PERCENT_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
MAX_LENGTH_VALUES = [2, 3, 4, 5]
DB_SIZE_FRACTIONS = [0.25, 0.50, 0.75, 1.0]

DEFAULT_SUPPORT_PERCENT = 0.10
DEFAULT_MAX_LENGTH = 4

N_WORKERS = max(1, cpu_count() - 1)  # Garde 1 cœur libre


# =============================================================================
# FONCTIONS DE TRAVAIL (pour multiprocessing)
# =============================================================================

def _worker_support(args):
    """Worker pour tester un support."""
    db_lists, support_pct, max_length = args

    # Recréer la DB numpy (pas sérialisable directement)
    import numpy as np
    database = [np.array(seq, dtype=np.int32) for seq in db_lists]

    miner = PrefixSpan(min_support=support_pct, max_length=max_length)
    t0 = time.perf_counter()
    patterns = miner.mine(database, parallel=False)
    runtime = time.perf_counter() - t0

    return {
        "support_percent": support_pct * 100,
        "min_support_abs": miner.min_support,
        "nb_patterns": len(patterns),
        "runtime_s": runtime,
    }


def _worker_length(args):
    """Worker pour tester une longueur."""
    db_lists, support_pct, max_len = args

    import numpy as np
    database = [np.array(seq, dtype=np.int32) for seq in db_lists]

    miner = PrefixSpan(min_support=support_pct, max_length=max_len)
    t0 = time.perf_counter()
    patterns = miner.mine(database, parallel=False)
    runtime = time.perf_counter() - t0

    return {
        "max_length": max_len,
        "nb_patterns": len(patterns),
        "runtime_s": runtime,
    }


def _worker_dbsize(args):
    """Worker pour tester une taille de DB."""
    db_lists, support_pct, max_length, fraction = args

    import numpy as np
    n_subset = max(2, int(len(db_lists) * fraction))
    database = [np.array(seq, dtype=np.int32) for seq in db_lists[:n_subset]]

    miner = PrefixSpan(min_support=support_pct, max_length=max_length)
    t0 = time.perf_counter()
    patterns = miner.mine(database, parallel=False)
    runtime = time.perf_counter() - t0

    return {
        "db_size": n_subset,
        "db_fraction": fraction,
        "nb_patterns": len(patterns),
        "runtime_s": runtime,
    }


# =============================================================================
# BENCHMARKS PARALLÉLISÉS
# =============================================================================

def benchmark_support_parallel(db_lists, max_length=DEFAULT_MAX_LENGTH):
    """Benchmark support en parallèle."""
    print(f"\n[1/4] Support vs Runtime ({N_WORKERS} workers)...")

    tasks = [(db_lists, pct, max_length) for pct in SUPPORT_PERCENT_VALUES]
    results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_support, t): t[1] for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            pct = futures[future]
            result = future.result()
            results.append(result)
            print(f"  [{done}/{len(tasks)}] {pct*100:5.1f}% -> {result['nb_patterns']:6d} motifs | {result['runtime_s']:.2f}s")

    # Trier par support
    results.sort(key=lambda x: x["support_percent"])
    return pd.DataFrame(results)


def benchmark_length_parallel(db_lists, support_pct=DEFAULT_SUPPORT_PERCENT):
    """Benchmark length en parallèle."""
    print(f"\n[2/4] Length vs Patterns ({N_WORKERS} workers)...")

    tasks = [(db_lists, support_pct, ml) for ml in MAX_LENGTH_VALUES]
    results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_length, t): t[2] for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            ml = futures[future]
            result = future.result()
            results.append(result)
            print(f"  [{done}/{len(tasks)}] len={ml} -> {result['nb_patterns']:6d} motifs | {result['runtime_s']:.2f}s")

    results.sort(key=lambda x: x["max_length"])
    return pd.DataFrame(results)


def benchmark_dbsize_parallel(db_lists, support_pct=DEFAULT_SUPPORT_PERCENT, max_length=DEFAULT_MAX_LENGTH):
    """Benchmark DB size en parallèle."""
    print(f"\n[3/4] DB Size vs Runtime ({N_WORKERS} workers)...")

    tasks = [(db_lists, support_pct, max_length, frac) for frac in DB_SIZE_FRACTIONS]
    results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_dbsize, t): t[3] for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            frac = futures[future]
            result = future.result()
            results.append(result)
            print(f"  [{done}/{len(tasks)}] {frac*100:3.0f}% -> {result['nb_patterns']:6d} motifs | {result['runtime_s']:.2f}s")

    results.sort(key=lambda x: x["db_size"])
    return pd.DataFrame(results)


def benchmark_memory_sequential(db_lists, max_length=DEFAULT_MAX_LENGTH):
    """Benchmark mémoire (séquentiel car tracemalloc ne fonctionne pas bien en parallèle)."""
    print(f"\n[4/4] Support vs Memory (séquentiel pour précision)...")

    import numpy as np
    database = [np.array(seq, dtype=np.int32) for seq in db_lists]

    memory_values = [0.10, 0.30, 0.50]
    results = []

    for i, support_pct in enumerate(memory_values):
        miner = PrefixSpan(min_support=support_pct, max_length=max_length)

        gc.collect()
        tracemalloc.start()
        patterns = miner.mine(database, parallel=False)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        results.append({
            "support_percent": support_pct * 100,
            "min_support_abs": miner.min_support,
            "nb_patterns": len(patterns),
            "memory_peak_mb": peak_mb,
        })
        print(f"  [{i+1}/{len(memory_values)}] {support_pct*100:5.1f}% -> {len(patterns):6d} motifs | {peak_mb:.1f} MB")

    return pd.DataFrame(results)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all(df1, df2, df3, df4):
    """Génère les graphiques."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Benchmark PrefixSpan (Parallélisé)", fontsize=14, fontweight="bold")

    axes[0, 0].plot(df1["support_percent"], df1["runtime_s"], "o-", color="steelblue", lw=2)
    axes[0, 0].set_xlabel("Support (%)"); axes[0, 0].set_ylabel("Runtime (s)")
    axes[0, 0].set_title("Support vs Runtime"); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df2["max_length"], df2["nb_patterns"], "s-", color="forestgreen", lw=2)
    axes[0, 1].set_xlabel("Max Length"); axes[0, 1].set_ylabel("Patterns")
    axes[0, 1].set_title("Length vs Patterns"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df3["db_size"], df3["runtime_s"], "^-", color="coral", lw=2)
    axes[1, 0].set_xlabel("DB Size"); axes[1, 0].set_ylabel("Runtime (s)")
    axes[1, 0].set_title("DB Size vs Runtime"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df4["support_percent"], df4["memory_peak_mb"], "d-", color="purple", lw=2)
    axes[1, 1].set_xlabel("Support (%)"); axes[1, 1].set_ylabel("Memory (MB)")
    axes[1, 1].set_title("Support vs Memory"); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "prefixspan_benchmark.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spmf_file", default=str(DEFAULT_SPMF))
    args = parser.parse_args()

    spmf_path = Path(args.spmf_file)

    print("=" * 60)
    print(f"BENCHMARK PREFIXSPAN - Parallélisé ({N_WORKERS} cœurs)")
    print("=" * 60)

    if not spmf_path.exists():
        print(f"ERREUR: {spmf_path} introuvable")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Charger et convertir en listes (pour sérialisation multiprocessing)
    print(f"\nChargement de {spmf_path}...")
    miner = PrefixSpan()
    database = miner.load_spmf(str(spmf_path))
    db_lists = [seq.tolist() for seq in database]  # Convertir en listes Python

    n_seq = len(db_lists)
    avg_len = sum(len(s) for s in db_lists) / n_seq
    print(f"  {n_seq} séquences (longueur moy: {avg_len:.0f})")

    # Benchmarks
    t_start = time.perf_counter()

    df1 = benchmark_support_parallel(db_lists)
    df2 = benchmark_length_parallel(db_lists)
    df3 = benchmark_dbsize_parallel(db_lists)
    df4 = benchmark_memory_sequential(db_lists)

    total_time = time.perf_counter() - t_start

    # Save
    df1.to_csv(OUTPUT_DIR / "bench1_support_runtime.csv", index=False)
    df2.to_csv(OUTPUT_DIR / "bench2_length_patterns.csv", index=False)
    df3.to_csv(OUTPUT_DIR / "bench3_dbsize_runtime.csv", index=False)
    df4.to_csv(OUTPUT_DIR / "bench4_support_memory.csv", index=False)

    print("\nGénération graphique...")
    fig_path = plot_all(df1, df2, df3, df4)

    print("\n" + "=" * 60)
    print(f"TERMINÉ en {total_time:.1f}s")
    print(f"  Graphique : {fig_path}")
    print(f"  CSV       : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
