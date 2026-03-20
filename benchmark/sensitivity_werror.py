#!/usr/bin/env python3
"""
Analyse de Sensibilité : Impact de w_error sur la qualité du Clustering

Mesure l'influence du paramètre de compression MDL (w_error) sur trois
métriques de clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz)
pour 3 algorithmes : KMeans (K=12), K-Médoïdes (K=12), Affinity Propagation.

Protocole expérimental
──────────────────────
  1. Pré-chargement des trajectoires brutes (une seule fois)
  2. Pour chaque valeur de w_error (80 points, échelle quasi-logarithmique) :
     a. Compression MDL de toutes les trajectoires
     b. Sous-échantillonnage à 5 000 segments (matrice TRACLUS)
     c. Extraction de features (milieu, direction, longueur)
     d. KMeans (K=12) × N_SEEDS graines
     e. K-Médoïdes (K=12) × N_SEEDS graines  (matrice TRACLUS)
     f. Affinity Propagation  (1 run, K auto)  (matrice TRACLUS)
     g. Calcul des métriques (Silhouette, Davies-Bouldin, Calinski-Harabasz)
  3. Agrégation (moyenne ± IC 95 %) et visualisation
  4. Détection automatique du Sweet Spot via score consensus

Grille w_error (80 valeurs) :
  0.1 → 2.0 (pas 0.1)  |  2.2 → 5.0 (pas 0.2)  |  5.5 → 10.0 (pas 0.5)
  11  → 20  (pas 1  )  |  22  → 50  (pas 2  )  |  55  → 100  (pas 5  )

Sorties :
  output/benchmark_sensitivity/raw_results.csv
  output/benchmark_sensitivity/fig1_pipeline_impact.png
  output/benchmark_sensitivity/fig2_sweet_spot.png
  output/benchmark_sensitivity/fig3_comparison_algo.png
  output/benchmark_sensitivity/fig4_segment_distributions.png

Usage :
  python benchmark/sensitivity_werror.py
  python benchmark/sensitivity_werror.py --max_files 30 --n_seeds 10
  python benchmark/sensitivity_werror.py --quick
"""

import argparse
import contextlib
import gc
import io
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Chemin projet ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.clustering import compute_traclus_similarity
from dota_analytics.compression import MDLCompressor
from dota_analytics.custom_ap import CustomAffinityPropagation
from dota_analytics.custom_kmedoids import CustomKMedoids
from dota_analytics.structures import Trajectory, TrajectoryPoint

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = BASE_DIR / "data-dota"
OUTPUT_DIR = BASE_DIR / "output" / "benchmark_sensitivity"

DEFAULT_MAX_FILES = 20
DEFAULT_N_SEEDS = 5
DEFAULT_K = 12                     # K fixe pour KMeans / KMedoids
DEFAULT_MIN_LENGTH = 5.0
SILHOUETTE_SAMPLE = 5000           # Échantillon pour silhouette (perf)
MAX_SEGMENTS_TRACLUS = 5000        # Limite N×N pour KMedoids / AP
DEFAULT_N_WORKERS = 10

# ── Mode --quick (léger, séquentiel, RAM safe) ──────────────────────────────
QUICK_MAX_FILES = 10
QUICK_N_SEEDS = 3
QUICK_MAX_SEGMENTS = 2000          # Matrice 2000² ≈ 15 Mo au lieu de 100 Mo
QUICK_SILHOUETTE_SAMPLE = 2000

ALGO_LABELS = {
    "kmeans":   "KMeans",
    "kmedoids": "K-Médoïdes",
    "ap":       "Affinity Propagation",
}
ALGO_COLORS = {
    "kmeans":   "#2196F3",
    "kmedoids": "#FF9800",
    "ap":       "#4CAF50",
}

DEFAULT_MAX_RAM_GB = 26            # Limite RAM (Go) pour auto-calcul workers

# ── Variables globales pour partage mémoire (Pool initializer) ───────────
_SHARED_TRAJECTORIES: list = []
_SHARED_N_POINTS: int = 0


def _init_worker(trajs, n_pts):
    """Initialiser les variables globales dans chaque worker du Pool.

    Avec fork() sous Linux, les trajectoires sont partagées en COW
    (Copy-On-Write) sans duplication mémoire.
    Cela évite de pickler les trajectoires dans chaque tâche.
    """
    global _SHARED_TRAJECTORIES, _SHARED_N_POINTS
    _SHARED_TRAJECTORIES = trajs
    _SHARED_N_POINTS = n_pts


def _estimate_worker_peak_gb(max_seg: int) -> float:
    """Estime la mémoire pic (Go) d'un worker pendant Affinity Propagation.

    AP est le plus gourmand : sim_matrix + S_copy + R + A + ~4 temporaires.
    """
    matrix_bytes = max_seg * max_seg * 8  # float64
    # sim_matrix (worker) + S copy + R + A + ~4 temporaires AP
    n_matrices = 8
    worker_arrays = n_matrices * matrix_bytes
    overhead = 250 * 1024 * 1024  # 250 Mo pour Python, segments, features…
    return (worker_arrays + overhead) / (1024 ** 3)


def _compute_safe_workers(max_seg: int, max_ram_gb: float) -> int:
    """Calcule le nombre max de workers parallèles qui tiennent en RAM."""
    os_reserved = 3.0        # Go réservés à l'OS + processus parent
    available = max_ram_gb - os_reserved
    per_worker = _estimate_worker_peak_gb(max_seg)
    safe = max(1, int(available / per_worker))
    return safe


# ═════════════════════════════════════════════════════════════════════════════
# GRILLE w_error
# ═════════════════════════════════════════════════════════════════════════════

def build_werror_grid(quick: bool = False) -> list[float]:
    """Construit la grille de w_error.

    Mode normal : ~80 valeurs de 0.1 à 100 (densité adaptative).
    Mode quick  : 11 valeurs pour test rapide.
    """
    if quick:
        return [0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0, 75.0, 100.0]

    grid: list[float] = []
    grid.extend(np.arange(0.1, 2.001, 0.1))       # 20 valeurs
    grid.extend(np.arange(2.2, 5.001, 0.2))        # 15 valeurs
    grid.extend(np.arange(5.5, 10.001, 0.5))       # 10 valeurs
    grid.extend(np.arange(11.0, 20.001, 1.0))      # 10 valeurs
    grid.extend(np.arange(22.0, 50.001, 2.0))      # 15 valeurs
    grid.extend(np.arange(55.0, 100.001, 5.0))     # 10 valeurs
    return sorted(set(round(w, 2) for w in grid))


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT (une seule fois)
# ═════════════════════════════════════════════════════════════════════════════

def preload_trajectories(max_files: int):
    """Charge toutes les trajectoires joueur depuis les CSV.

    Returns
    -------
    trajectories : list[Trajectory]
    n_original_points : int
    """
    csv_files = sorted(DATA_DIR.glob("coord_*.csv"))[:max_files]
    if not csv_files:
        print(f"ERREUR : aucun CSV trouvé dans {DATA_DIR}")
        sys.exit(1)

    print(f"Chargement de {len(csv_files)} matchs…")
    trajectories: list[Trajectory] = []
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
                TrajectoryPoint(
                    x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"])
                )
                for _, r in sub.iterrows()
            ]
            trajectories.append(Trajectory(points=points))
            n_original_points += len(points)

    print(f"  → {len(trajectories)} trajectoires, {n_original_points:,} points bruts\n")
    return trajectories, n_original_points


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE : Compression → Features → Clustering → Métriques
# ═════════════════════════════════════════════════════════════════════════════

def compress_all(trajectories, w_error, min_length):
    """Compresse toutes les trajectoires et collecte les segments filtrés."""
    compressor = MDLCompressor(w_error=w_error, verbose=False)
    segments = []
    n_raw = 0
    for traj in trajectories:
        segs = compressor.compress_player_trajectory(traj)
        n_raw += len(segs)
        segments.extend(s for s in segs if s.length() > min_length)
    return segments, n_raw


def segments_to_features(segments) -> np.ndarray:
    """Extrait les features : (mid_x, mid_y, dx, dy, length)."""
    feats = np.empty((len(segments), 5), dtype=np.float32)
    for i, s in enumerate(segments):
        feats[i, 0] = (s.start.x + s.end.x) * 0.5   # milieu X
        feats[i, 1] = (s.start.y + s.end.y) * 0.5   # milieu Y
        feats[i, 2] = s.end.x - s.start.x             # direction X
        feats[i, 3] = s.end.y - s.start.y             # direction Y
        feats[i, 4] = s.length()                       # norme
    return feats


# ── Métriques communes ───────────────────────────────────────────────────────

def _compute_metrics(X_scaled, labels, n_seg, seed=0, sil_sample=SILHOUETTE_SAMPLE):
    """Calcule Silhouette, Davies-Bouldin, Calinski-Harabasz."""
    if len(np.unique(labels)) < 2:
        return None
    sil = silhouette_score(
        X_scaled, labels,
        sample_size=min(sil_sample, n_seg),
        random_state=seed,
    )
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    return dict(silhouette=sil, davies_bouldin=db, calinski_harabasz=ch)


# ── Évaluation KMeans ────────────────────────────────────────────────────────

def evaluate_kmeans(X_scaled, k, seed, n_seg, sil_sample=SILHOUETTE_SAMPLE):
    """MiniBatchKMeans + métriques."""
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed,
        batch_size=min(4096, n_seg), n_init=3,
    )
    labels = km.fit_predict(X_scaled)
    metrics = _compute_metrics(X_scaled, labels, n_seg, seed, sil_sample)
    if metrics is None:
        return None
    metrics["inertia"] = km.inertia_
    metrics["k_found"] = k
    return metrics


# ── Évaluation K-Médoïdes ────────────────────────────────────────────────────

def evaluate_kmedoids(sim_matrix, k, seed, X_scaled, n_seg, sil_sample=SILHOUETTE_SAMPLE):
    """K-Médoïdes (PAM) sur matrice TRACLUS + métriques sur features."""
    distance_matrix = -sim_matrix.copy()
    np.fill_diagonal(distance_matrix, 0.0)
    kmed = CustomKMedoids(n_clusters=k, max_iter=300, random_state=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        kmed.fit(distance_matrix)
    labels = kmed.labels_
    metrics = _compute_metrics(X_scaled, labels, n_seg, seed, sil_sample)
    if metrics is None:
        return None
    metrics["inertia"] = np.nan
    metrics["k_found"] = k
    return metrics


# ── Évaluation Affinity Propagation ──────────────────────────────────────────

def evaluate_ap(sim_matrix, X_scaled, n_seg, sil_sample=SILHOUETTE_SAMPLE):
    """Affinity Propagation sur matrice TRACLUS + métriques sur features."""
    S = sim_matrix.copy()
    med = np.median(S)
    np.fill_diagonal(S, med)
    ap = CustomAffinityPropagation(damping=0.9, max_iter=200, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        ap.fit(S)
    labels = ap.labels_
    k_found = (len(ap.cluster_centers_indices_)
               if ap.cluster_centers_indices_ is not None else 0)
    if k_found < 2 or np.all(labels == -1):
        return None
    metrics = _compute_metrics(X_scaled, labels, n_seg, 0, sil_sample)
    if metrics is None:
        return None
    metrics["inertia"] = np.nan
    metrics["k_found"] = k_found
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# WORKER PARALLÈLE (1 worker = 1 valeur de w_error, 3 algorithmes)
# ═════════════════════════════════════════════════════════════════════════════

def _worker_single_werror(args_tuple):
    """Worker : compression + 3 algos pour UNE valeur de w_error.

    En mode parallèle, les trajectoires sont lues via _SHARED_TRAJECTORIES
    (initialisées par _init_worker) pour éviter le pickle.
    En mode séquentiel, elles sont passées directement dans le tuple.
    """
    (w_error, trajectories, n_original_points,
     k, n_seeds, min_length, max_seg, sil_sample,
     counter, lock, total) = args_tuple

    # En mode parallèle, trajectories est None → utiliser les globales
    if trajectories is None:
        trajectories = _SHARED_TRAJECTORIES
        n_original_points = _SHARED_N_POINTS

    t0 = time.perf_counter()
    rows: list[dict] = []

    # ── 1. Compression ────────────────────────────────────────────────────
    segments, n_raw = compress_all(trajectories, w_error, min_length)
    t_compress = time.perf_counter() - t0
    n_seg_total = len(segments)
    comp_rate = 1.0 - n_raw / n_original_points

    if n_seg_total < 20:
        with lock:
            counter.value += 1
            print(f"  [{counter.value:3d}/{total}] w={w_error:7.2f}  →  "
                  f"{n_seg_total:5d} seg   SKIP (< 20)")
        return [_empty_row(w_error, n_raw, n_seg_total, n_seg_total,
                           comp_rate, t_compress, algo)
                for algo in ("kmeans", "kmedoids", "ap")]

    # ── Sous-échantillonnage (commun aux 3 algos pour comparaison) ────
    rng = np.random.default_rng(42)
    if n_seg_total > max_seg:
        idx = rng.choice(n_seg_total, max_seg, replace=False)
        idx.sort()
        segments = [segments[i] for i in idx]
    n_seg = len(segments)

    # ── Statistiques segments ─────────────────────────────────────────────
    lengths = np.array([s.length() for s in segments])
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    med_len = float(np.median(lengths))

    # ── 2. Features + normalisation ───────────────────────────────────────
    X = segments_to_features(segments)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    del X  # libérer la version non-normalisée

    # ── 3. Matrice TRACLUS (pour KMedoids + AP, calculée une seule fois) ─
    sim_matrix = compute_traclus_similarity(segments)
    del segments  # plus besoin, libérer la liste de segments

    # ── 4. KMeans (K fixe) × n_seeds ─────────────────────────────────────
    sil_km: list[float] = []
    for seed in range(n_seeds):
        if k >= n_seg:
            continue
        t1 = time.perf_counter()
        result = evaluate_kmeans(X_scaled, k, seed, n_seg, sil_sample)
        t_cluster = time.perf_counter() - t1
        if result is None:
            continue
        rows.append(_make_row(
            w_error, n_raw, n_seg_total, n_seg, comp_rate,
            mean_len, std_len, med_len, "kmeans", k,
            result, seed, t_compress, t_cluster,
        ))
        sil_km.append(result["silhouette"])

    # ── 5. K-Médoïdes (K fixe) × n_seeds ────────────────────────────────
    sil_kmed: list[float] = []
    for seed in range(n_seeds):
        if k >= n_seg:
            continue
        t1 = time.perf_counter()
        result = evaluate_kmedoids(sim_matrix, k, seed, X_scaled, n_seg, sil_sample)
        t_cluster = time.perf_counter() - t1
        if result is None:
            continue
        rows.append(_make_row(
            w_error, n_raw, n_seg_total, n_seg, comp_rate,
            mean_len, std_len, med_len, "kmedoids", k,
            result, seed, t_compress, t_cluster,
        ))
        sil_kmed.append(result["silhouette"])

    # ── 6. Affinity Propagation (1 seul run, pas de seed) ────────────────
    sil_ap: list[float] = []
    t1 = time.perf_counter()
    result = evaluate_ap(sim_matrix, X_scaled, n_seg, sil_sample)
    t_cluster = time.perf_counter() - t1
    if result is not None:
        rows.append(_make_row(
            w_error, n_raw, n_seg_total, n_seg, comp_rate,
            mean_len, std_len, med_len, "ap", result["k_found"],
            result, 0, t_compress, t_cluster,
        ))
        sil_ap.append(result["silhouette"])

    # ── Libération mémoire (critique en parallèle) ───────────────────────
    del sim_matrix, X_scaled
    gc.collect()

    # ── Affichage progression ─────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    def _s(vals, tag):
        return f"{tag}={np.mean(vals):.3f}" if vals else f"{tag}=N/A"

    summary = f"{_s(sil_km, 'km')} | {_s(sil_kmed, 'kmed')} | {_s(sil_ap, 'ap')}"

    with lock:
        counter.value += 1
        print(f"  [{counter.value:3d}/{total}] w={w_error:7.2f}  →  {n_seg:5d} seg | "
              f"{summary} | {elapsed:5.1f}s")

    return rows


def _make_row(w_error, n_raw, n_seg_total, n_seg, comp_rate,
              mean_len, std_len, med_len, algorithm, n_clusters,
              result, seed, t_compress, t_cluster):
    """Construit un dict-row à partir des résultats."""
    return dict(
        w_error=w_error,
        algorithm=algorithm,
        n_segments_raw=n_raw,
        n_segments_total=n_seg_total,
        n_segments=n_seg,
        compression_rate=comp_rate,
        mean_length=mean_len,
        std_length=std_len,
        median_length=med_len,
        n_clusters=n_clusters,
        k_found=result["k_found"],
        seed=seed,
        silhouette=result["silhouette"],
        davies_bouldin=result["davies_bouldin"],
        calinski_harabasz=result["calinski_harabasz"],
        inertia=result["inertia"],
        t_compress=t_compress,
        t_cluster=t_cluster,
    )


# ═════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE (SÉQUENTIELLE ou PARALLÈLE)
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(trajectories, n_original_points, w_errors,
                  k, n_seeds, min_length, *,
                  n_workers=DEFAULT_N_WORKERS,
                  max_seg=MAX_SEGMENTS_TRACLUS,
                  sil_sample=SILHOUETTE_SAMPLE,
                  max_ram_gb=DEFAULT_MAX_RAM_GB):
    """Exécute le benchmark et retourne un DataFrame.

    n_workers=1 → mode séquentiel (pas de fork, RAM safe).
    n_workers>1 → mode parallèle (multiprocessing.Pool).
    Le nombre de workers est automatiquement réduit si la RAM estimée
    dépasse max_ram_gb.
    """
    total = len(w_errors)
    runs_per_w = 2 * n_seeds + 1   # KMeans×seeds + KMedoids×seeds + AP×1
    total_runs = total * runs_per_w
    n_workers = min(n_workers, total)

    # ── Estimation mémoire & auto-limitation ─────────────────────────────
    per_worker_gb = _estimate_worker_peak_gb(max_seg)
    safe_workers = _compute_safe_workers(max_seg, max_ram_gb)

    if n_workers > 1 and n_workers > safe_workers:
        print(f"⚠️  {n_workers} workers × {per_worker_gb:.1f} Go/worker "
              f"= {n_workers * per_worker_gb:.0f} Go pic > limite {max_ram_gb} Go")
        n_workers = safe_workers
        print(f"    → Réduit automatiquement à {n_workers} workers")

    estimated_total = n_workers * per_worker_gb + 3.0  # +3 Go OS/parent
    mode = "séquentiel" if n_workers <= 1 else f"{n_workers} processus parallèles"

    print(f"Benchmark : {total} w_error × (KMeans + KMedoids) × {n_seeds} graines + AP"
          f" = {total_runs} runs")
    print(f"K fixe    : {k}  |  Sous-éch. TRACLUS : {max_seg}  |  Sil. sample : {sil_sample}")
    print(f"Mode      : {mode}")
    print(f"RAM estimée : {per_worker_gb:.1f} Go/worker × {n_workers} "
          f"+ 3 Go OS ≈ {estimated_total:.0f} Go  (limite : {max_ram_gb} Go)")
    print("=" * 78)
    sys.stdout.flush()

    t_global = time.perf_counter()

    if n_workers <= 1:
        # ── Mode séquentiel (RAM safe) ────────────────────────────────────
        rows = _run_sequential(
            trajectories, n_original_points, w_errors,
            k, n_seeds, min_length, max_seg, sil_sample, t_global,
        )
    else:
        # ── Mode parallèle ────────────────────────────────────────────────
        rows = _run_parallel(
            trajectories, n_original_points, w_errors,
            k, n_seeds, min_length, max_seg, sil_sample, n_workers,
            max_ram_gb,
        )

    total_time = time.perf_counter() - t_global
    print("=" * 78)
    print(f"Terminé en {total_time:.0f}s ({total_time / 60:.1f} min)\n")
    sys.stdout.flush()

    return pd.DataFrame(rows)


def _run_sequential(trajectories, n_original_points, w_errors,
                    k, n_seeds, min_length, max_seg, sil_sample, t_global):
    """Boucle séquentielle avec progression temps réel."""
    total = len(w_errors)
    all_rows: list[dict] = []

    # Compteur simple (pas besoin de Manager)
    class _Counter:
        value = 0
    counter = _Counter()

    class _DummyLock:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    lock = _DummyLock()

    for idx, w_error in enumerate(w_errors):
        task = (w_error, trajectories, n_original_points,
                k, n_seeds, min_length, max_seg, sil_sample,
                counter, lock, total)

        # Affichage «en cours» en temps réel
        print(f"\r  ⏳ [{idx+1:3d}/{total}] w={w_error:7.2f}  en cours…", end="", flush=True)

        batch = _worker_single_werror(task)
        all_rows.extend(batch)

        # ETA
        eta = _eta(t_global, idx + 1, total)
        # Efface la ligne "en cours" (le worker a déjà imprimé sa ligne)
        print(f"  {eta}", flush=True)

    return all_rows


def _run_parallel(trajectories, n_original_points, w_errors,
                  k, n_seeds, min_length, max_seg, sil_sample, n_workers,
                  max_ram_gb=DEFAULT_MAX_RAM_GB):
    """Boucle parallèle (multiprocessing.Pool).

    Les trajectoires sont partagées via _init_worker (COW sous Linux)
    au lieu d'être picklées dans chaque tâche → économie massive de RAM.
    """
    total = len(w_errors)
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    # trajectories=None dans chaque tâche → le worker lit _SHARED_TRAJECTORIES
    tasks = [
        (w, None, None,
         k, n_seeds, min_length, max_seg, sil_sample,
         counter, lock, total)
        for w in w_errors
    ]

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(trajectories, n_original_points),
    ) as pool:
        results = pool.map(_worker_single_werror, tasks)

    rows: list[dict] = []
    for batch in results:
        rows.extend(batch)
    return rows


def _empty_row(w_error, n_raw, n_seg_total, n_seg, comp_rate,
               t_compress, algorithm):
    """Ligne vide pour les w_error sans assez de segments."""
    return dict(
        w_error=w_error, algorithm=algorithm,
        n_segments_raw=n_raw, n_segments_total=n_seg_total, n_segments=n_seg,
        compression_rate=comp_rate, mean_length=0, std_length=0,
        median_length=0, n_clusters=0, k_found=0, seed=0,
        silhouette=np.nan, davies_bouldin=np.nan,
        calinski_harabasz=np.nan, inertia=np.nan,
        t_compress=t_compress, t_cluster=0,
    )


def _eta(t_start, done, total):
    """Estimation du temps restant."""
    elapsed = time.perf_counter() - t_start
    if done == 0:
        return ""
    remaining = elapsed / done * (total - done)
    if remaining > 120:
        return f"ETA {remaining / 60:.0f} min"
    return f"ETA {remaining:.0f}s"


# ═════════════════════════════════════════════════════════════════════════════
# LISSAGE (Savitzky-Golay)
# ═════════════════════════════════════════════════════════════════════════════

def _smooth(y, window=11, polyorder=3):
    """Lissage Savitzky-Golay robuste (gère les petits tableaux)."""
    n = len(y)
    if n < 5:
        return y.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    p = min(polyorder, w - 1)
    return savgol_filter(y, window_length=w, polyorder=p)


def _minmax(arr, invert=False):
    """Normalisation min-max → [0, 1].  Si invert=True, 1 = min original."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-10:
        return np.full_like(arr, 0.5)
    normed = (arr - mn) / (mx - mn)
    return 1.0 - normed if invert else normed


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Vue d'ensemble (2 × 3) — KMeans
# ═════════════════════════════════════════════════════════════════════════════

def plot_pipeline_impact(df, output_dir):
    """Silhouette / DB / CH + n_seg / longueur / temps  (KMeans uniquement)."""
    sub = df[(df["algorithm"] == "kmeans") & df["silhouette"].notna()]
    if sub.empty:
        print("  ⚠️  Pas de données KMeans pour la figure 1")
        return

    agg = sub.groupby("w_error").agg(
        sil_mean=("silhouette", "mean"),   sil_std=("silhouette", "std"),
        db_mean=("davies_bouldin", "mean"),  db_std=("davies_bouldin", "std"),
        ch_mean=("calinski_harabasz", "mean"), ch_std=("calinski_harabasz", "std"),
        n_seg=("n_segments", "first"),
        mean_len=("mean_length", "first"),
        t_compress=("t_compress", "first"),
    ).reset_index()

    x = agg["w_error"].values
    n_seeds = max(sub.groupby("w_error").size().min(), 1)
    ci = 1.96 / np.sqrt(n_seeds)      # facteur IC 95 %

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Impact de w_error sur le Pipeline  (KMeans, K = {DEFAULT_K})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- Silhouette ---
    _panel(axes[0, 0], x, agg["sil_mean"], agg["sil_std"] * ci,
           "#2196F3", "#0D47A1", "Silhouette Score", "↑", peak=True)

    # --- Davies-Bouldin ---
    _panel(axes[0, 1], x, agg["db_mean"], agg["db_std"] * ci,
           "#FF9800", "#E65100", "Davies-Bouldin Index", "↓")

    # --- Calinski-Harabasz ---
    _panel(axes[0, 2], x, agg["ch_mean"], agg["ch_std"] * ci,
           "#4CAF50", "#1B5E20", "Calinski-Harabasz Index", "↑")

    # --- Nombre de segments ---
    ax = axes[1, 0]
    ax.plot(x, agg["n_seg"], "o-", color="#9C27B0", ms=3, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="w_error",
           ylabel="Segments (après filtrage)", title="Nombre de segments")
    ax.grid(True, alpha=0.3)

    # --- Longueur moyenne ---
    ax = axes[1, 1]
    ax.plot(x, agg["mean_len"], "o-", color="#F44336", ms=3, lw=1)
    ax.set(xscale="log", xlabel="w_error", ylabel="Longueur moyenne",
           title="Longueur moyenne des segments")
    ax.grid(True, alpha=0.3)

    # --- Temps de compression ---
    ax = axes[1, 2]
    ax.plot(x, agg["t_compress"], "o-", color="#607D8B", ms=3, lw=1)
    ax.set(xscale="log", xlabel="w_error", ylabel="Temps (s)",
           title="Temps de compression")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig1_pipeline_impact.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def _panel(ax, x, y_mean, y_err, c1, c2, title, direction, peak=False):
    """Dessine un sous-graphe métrique (points + IC + tendance)."""
    y = y_mean.values
    err = y_err.fillna(0).values if hasattr(y_err, "fillna") else np.nan_to_num(y_err)

    ax.plot(x, y, "o", color=c1, ms=3, alpha=0.5)
    ax.fill_between(x, y - err, y + err, alpha=0.15, color=c1)

    y_s = _smooth(y)
    ax.plot(x, y_s, "-", color=c2, lw=2.5, label="Tendance (Savitzky-Golay)")

    if peak:
        best = int(np.argmax(y_s))
        ax.axvline(x[best], color="red", ls=":", alpha=0.7,
                   label=f"Pic ≈ {x[best]:.1f}")

    ax.set(xscale="log", xlabel="w_error", ylabel=f"{title} {direction}",
           title=title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sweet Spot (Silhouette + Compression + Consensus)
# ═════════════════════════════════════════════════════════════════════════════

def plot_sweet_spot(df, output_dir):
    """Dual-axis : Silhouette vs Compression Rate + score consensus normalisé.

    Le consensus est calculé sur la moyenne des 3 algorithmes.
    """
    valid = df[df["silhouette"].notna()]
    if valid.empty:
        print("  ⚠️  Pas de données pour la figure 2")
        return

    agg = valid.groupby("w_error").agg(
        sil_mean=("silhouette", "mean"),
        db_mean=("davies_bouldin", "mean"),
        ch_mean=("calinski_harabasz", "mean"),
        comp_rate=("compression_rate", "first"),
    ).reset_index()

    x = agg["w_error"].values

    # ── Score consensus (3 métriques normalisées [0, 1]) ─────────────────
    sil_n = _minmax(agg["sil_mean"].values)
    db_n  = _minmax(agg["db_mean"].values, invert=True)   # ↓ = mieux
    ch_n  = _minmax(agg["ch_mean"].values)
    consensus = (sil_n + db_n + ch_n) / 3.0
    consensus_s = _smooth(consensus)

    sil_s = _smooth(agg["sil_mean"].values)
    comp_pct = agg["comp_rate"].values * 100

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # ── Silhouette (axe gauche) ──────────────────────────────────────────
    c_sil = "#2196F3"
    ax1.plot(x, agg["sil_mean"].values, "o", color=c_sil, ms=3, alpha=0.35)
    ax1.plot(x, sil_s, "-", color=c_sil, lw=2.5, label="Silhouette (lissé)")
    ax1.set_xlabel("w_error", fontsize=13)
    ax1.set_ylabel("Silhouette Score ↑", fontsize=13, color=c_sil)
    ax1.tick_params(axis="y", labelcolor=c_sil)
    ax1.set_xscale("log")

    # ── Taux de compression (axe droit) ──────────────────────────────────
    ax2 = ax1.twinx()
    c_comp = "#F44336"
    ax2.plot(x, comp_pct, "s-", color=c_comp, ms=3, lw=1.5, alpha=0.7,
             label="Taux de compression (%)")
    ax2.set_ylabel("Taux de compression (%) ↑", fontsize=13, color=c_comp)
    ax2.tick_params(axis="y", labelcolor=c_comp)

    # ── Consensus (sur l'axe gauche, redimensionné) ──────────────────────
    c_cons = "#4CAF50"
    scale = np.nanmax(sil_s) if np.any(np.isfinite(sil_s)) else 1.0
    ax1.plot(x, consensus_s * scale, "--", color=c_cons, lw=2.5, alpha=0.8,
             label="Score consensus (normalisé)")

    # ── Sweet spot : pic du consensus lissé ──────────────────────────────
    best_idx = int(np.argmax(consensus_s))
    best_w = x[best_idx]
    ax1.axvline(best_w, color="#E91E63", ls=":", lw=2,
                label=f"Sweet spot ≈ {best_w:.1f}")

    # Zone optimale (≥ 95 % du pic consensus)
    threshold = consensus_s[best_idx] * 0.95
    zone_mask = consensus_s >= threshold
    zone_x = x[zone_mask]
    if len(zone_x) >= 2:
        ax1.axvspan(zone_x[0], zone_x[-1], alpha=0.08, color="#E91E63",
                    label=f"Zone optimale [{zone_x[0]:.1f} – {zone_x[-1]:.1f}]")

    # ── Légendes combinées ───────────────────────────────────────────────
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=10, framealpha=0.95)

    ax1.set_title(
        "Sweet Spot : Compromis Compression / Qualité  (3 algos combinés)",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig2_sweet_spot.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Comparaison des 3 algorithmes
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison_algo(df, output_dir):
    """Silhouette / DB / CH vs w_error pour chaque algorithme."""
    valid = df[df["silhouette"].notna()]
    algos = [a for a in ("kmeans", "kmedoids", "ap")
             if a in valid["algorithm"].unique()]

    if not algos:
        print("  ⚠️  Pas assez de données pour la figure 3")
        return

    metrics_info = [
        ("silhouette",        "Silhouette Score",        "↑"),
        ("davies_bouldin",    "Davies-Bouldin Index",    "↓"),
        ("calinski_harabasz", "Calinski-Harabasz Index",  "↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "Comparaison KMeans / K-Médoïdes / Affinity Propagation",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for ax, (col, title, direction) in zip(axes, metrics_info):
        for algo in algos:
            sub = valid[valid["algorithm"] == algo]
            agg = sub.groupby("w_error").agg(
                y_mean=(col, "mean"),
                y_std=(col, "std"),
            ).reset_index()

            x = agg["w_error"].values
            y = agg["y_mean"].values
            y_s = _smooth(y)
            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]

            ax.plot(x, y_s, "-", color=color, lw=2.5, label=label)
            ax.fill_between(
                x,
                y - agg["y_std"].fillna(0).values,
                y + agg["y_std"].fillna(0).values,
                alpha=0.1, color=color,
            )

        ax.set(xscale="log", xlabel="w_error", ylabel=f"{title} {direction}")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig3_comparison_algo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Distribution des longueurs de segments (boxplots)
# ═════════════════════════════════════════════════════════════════════════════

def plot_segment_distributions(trajectories, output_dir, min_length,
                               w_samples=None):
    """Boxplots de longueur de segments pour quelques w_error clés."""
    if w_samples is None:
        w_samples = [0.5, 2.0, 5.0, 12.0, 25.0, 50.0, 100.0]

    all_lengths: dict[float, list[float]] = {}

    for w in w_samples:
        segs, _ = compress_all(trajectories, w, min_length)
        if segs:
            all_lengths[w] = [s.length() for s in segs]

    if not all_lengths:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [f"{w}" for w in all_lengths]
    data = [all_lengths[w] for w in all_lengths]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("w_error", fontsize=13)
    ax.set_ylabel("Longueur des segments", fontsize=13)
    ax.set_title("Distribution des longueurs de segments selon w_error",
                 fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotations : nombre de segments au-dessus de chaque boxplot
    for i, (w, lengths) in enumerate(all_lengths.items()):
        ax.text(i + 1, ax.get_ylim()[1] * 0.98, f"n={len(lengths):,}",
                ha="center", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    path = output_dir / "fig4_segment_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ STATISTIQUE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(df):
    """Affiche le tableau récapitulatif."""
    valid = df[df["silhouette"].notna()]
    if valid.empty:
        print("Aucune donnée valide.")
        return

    # ── Par algorithme ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RÉSUMÉ PAR ALGORITHME")
    print("=" * 64)
    by_algo = valid.groupby("algorithm").agg(
        sil=("silhouette", "mean"),
        db=("davies_bouldin", "mean"),
        ch=("calinski_harabasz", "mean"),
        k_mean=("k_found", "mean"),
    )
    for algo, row in by_algo.iterrows():
        name = ALGO_LABELS.get(algo, algo)
        print(f"  {name:<25s}  Sil={row['sil']:.4f}  "
              f"DB={row['db']:.3f}  CH={row['ch']:.0f}  "
              f"K_moyen={row['k_mean']:.1f}")

    # ── Top 5 w_error (toutes algos confondues) ──────────────────────────
    by_w = valid.groupby("w_error")["silhouette"].mean().sort_values(ascending=False)

    print(f"\n{'─' * 64}")
    print("  TOP 5 w_error (Silhouette moyenne, tous algos)")
    print(f"{'─' * 64}")
    for i, (w, sil) in enumerate(by_w.head(5).items()):
        marker = "  ◀ SWEET SPOT" if i == 0 else ""
        print(f"    w_error = {w:7.2f}  →  Silhouette = {sil:.4f}{marker}")

    # ── Sweet spot ────────────────────────────────────────────────────────
    best_w = by_w.idxmax()
    best_sil = by_w.max()
    print(f"\n{'=' * 64}")
    print(f"  SWEET SPOT :  w_error ≈ {best_w:.2f}  "
          f"(Silhouette moyen = {best_sil:.4f})")
    print(f"{'=' * 64}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse de Sensibilité : w_error → Qualité du Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python benchmark/sensitivity_werror.py --quick          # Léger, séquentiel
  python benchmark/sensitivity_werror.py                  # Complet, 10 workers
  python benchmark/sensitivity_werror.py --workers 4      # Complet, 4 workers
  python benchmark/sensitivity_werror.py --max_files 30 --n_seeds 10
        """,
    )
    parser.add_argument("--max_files", type=int, default=None,
                        help=f"Matchs CSV chargés (défaut: {DEFAULT_MAX_FILES}, quick: {QUICK_MAX_FILES})")
    parser.add_argument("--n_seeds", type=int, default=None,
                        help=f"Graines par (w_error, algo) (défaut: {DEFAULT_N_SEEDS}, quick: {QUICK_N_SEEDS})")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Nombre de clusters K (défaut: {DEFAULT_K})")
    parser.add_argument("--min_length", type=float, default=DEFAULT_MIN_LENGTH,
                        help=f"Longueur min segments (défaut: {DEFAULT_MIN_LENGTH})")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Nombre de workers (défaut: {DEFAULT_N_WORKERS}, quick: 1 séquentiel)")
    parser.add_argument("--max-ram", type=float, default=DEFAULT_MAX_RAM_GB,
                        help=f"Limite RAM en Go pour auto-calcul workers (défaut: {DEFAULT_MAX_RAM_GB})")
    parser.add_argument("--quick", action="store_true",
                        help="Mode léger : 10 fichiers, 3 seeds, 2000 seg, séquentiel")

    args = parser.parse_args()

    # ── Résolution des valeurs (quick override les défauts) ───────────────
    if args.quick:
        max_files   = args.max_files  or QUICK_MAX_FILES
        n_seeds     = args.n_seeds    or QUICK_N_SEEDS
        n_workers   = args.workers if args.workers is not None else 1
        max_seg     = QUICK_MAX_SEGMENTS
        sil_sample  = QUICK_SILHOUETTE_SAMPLE
    else:
        max_files   = args.max_files  or DEFAULT_MAX_FILES
        n_seeds     = args.n_seeds    or DEFAULT_N_SEEDS
        n_workers   = args.workers if args.workers is not None else DEFAULT_N_WORKERS
        max_seg     = MAX_SEGMENTS_TRACLUS
        sil_sample  = SILHOUETTE_SAMPLE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ANALYSE DE SENSIBILITÉ : w_error → Qualité du Clustering")
    print("  Algorithmes : KMeans · K-Médoïdes · Affinity Propagation")
    if args.quick:
        print("  ⚡ MODE QUICK : léger, séquentiel, RAM safe")
    print("=" * 78)

    # 1. Pré-chargement
    trajectories, n_original_points = preload_trajectories(max_files)

    # 2. Grille
    w_errors = build_werror_grid(args.quick)
    print(f"Grille : {len(w_errors)} valeurs  "
          f"({w_errors[0]:.1f} → {w_errors[-1]:.1f})")
    print(f"K = {args.k} (fixe),  {n_seeds} graines/combinaison")

    # 3. Benchmark
    df = run_benchmark(
        trajectories, n_original_points, w_errors,
        args.k, n_seeds, args.min_length,
        n_workers=n_workers, max_seg=max_seg, sil_sample=sil_sample,
        max_ram_gb=args.max_ram,
    )

    # 4. Sauvegarde CSV
    csv_path = OUTPUT_DIR / "raw_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Résultats bruts : {csv_path}")

    # 5. Figures
    print("\nGénération des figures…")
    plot_pipeline_impact(df, OUTPUT_DIR)
    plot_sweet_spot(df, OUTPUT_DIR)
    plot_comparison_algo(df, OUTPUT_DIR)
    plot_segment_distributions(trajectories, OUTPUT_DIR, args.min_length)

    # 6. Résumé
    print_summary(df)


if __name__ == "__main__":
    main()
