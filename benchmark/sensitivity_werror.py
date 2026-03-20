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
import csv
import gc
import io
import platform
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
import pandas as pd

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
DEFAULT_N_WORKERS = 4

# ── Mode --quick (léger, séquentiel, RAM safe) ──────────────────────────────
QUICK_MAX_FILES = 10
QUICK_N_SEEDS = 3
QUICK_MAX_SEGMENTS = 2000          # Matrice 2000² ≈ 15 Mo au lieu de 100 Mo
QUICK_SILHOUETTE_SAMPLE = 2000

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


def _estimate_worker_peak_gb(max_seg: int, n_traj_bytes: int = 0) -> float:
    """Estime la mémoire pic (Go) d'un worker pendant Affinity Propagation.

    AP est le plus gourmand : sim_matrix + S_copy + R + A + ~4 temporaires.
    compute_traclus_similarity alloue ~10 matrices float32 en interne,
    et le résultat est promu float64 (numpy broadcast).

    Sur Windows (spawn), chaque worker reçoit une copie pickle des
    trajectoires → n_traj_bytes supplémentaires par worker.
    """
    matrix_f64 = max_seg * max_seg * 8   # float64, 1 matrice
    matrix_f32 = max_seg * max_seg * 4   # float32, 1 matrice
    # AP : S_copy + R + A + AS + max_AS + R_new + Rp + A_new ≈ 9 float64
    ap_arrays = 9 * matrix_f64
    # compute_traclus_similarity : ~6 float32 intermédiaires + 1 float64 sortie
    sim_arrays = 6 * matrix_f32 + matrix_f64
    # pic = max(AP, similarity)  — ils ne tournent pas en même temps,
    # mais sim_matrix reste en mémoire pendant AP
    peak_arrays = ap_arrays + matrix_f64   # AP + sim_matrix retenue
    overhead = 350 * 1024 * 1024  # Python runtime, segments, features…
    # Sur Windows (spawn), ajouter la copie des trajectoires
    return (peak_arrays + overhead + n_traj_bytes) / (1024 ** 3)


def _estimate_traj_bytes(trajectories) -> int:
    """Estime la taille mémoire des trajectoires (pour Windows spawn)."""
    n_points = sum(len(t.points) for t in trajectories)
    # TrajectoryPoint : 3 floats + overhead objet Python ≈ 128 octets
    return n_points * 128


def _compute_safe_workers(max_seg: int, max_ram_gb: float,
                          n_traj_bytes: int = 0) -> int:
    """Calcule le nombre max de workers parallèles qui tiennent en RAM."""
    os_reserved = 4.0        # Go réservés à l'OS + processus parent
    available = max_ram_gb - os_reserved
    per_worker = _estimate_worker_peak_gb(max_seg, n_traj_bytes)
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
                  f"{n_seg_total:5d} seg   SKIP (< 20)", flush=True)
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
              f"{summary} | {elapsed:5.1f}s", flush=True)

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

# Colonnes du CSV (ordre garanti)
_CSV_COLUMNS = [
    "w_error", "algorithm", "n_segments_raw", "n_segments_total",
    "n_segments", "compression_rate", "mean_length", "std_length",
    "median_length", "n_clusters", "k_found", "seed",
    "silhouette", "davies_bouldin", "calinski_harabasz", "inertia",
    "t_compress", "t_cluster",
]


def _init_csv(csv_path: Path) -> None:
    """Écrit l'en-tête CSV (écrase le fichier existant)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()


def _append_csv(csv_path: Path, rows: list[dict]) -> None:
    """Ajoute des lignes au CSV (mode append)."""
    if not rows:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writerows(rows)


def run_benchmark(trajectories, n_original_points, w_errors,
                  k, n_seeds, min_length, *,
                  n_workers=DEFAULT_N_WORKERS,
                  max_seg=MAX_SEGMENTS_TRACLUS,
                  sil_sample=SILHOUETTE_SAMPLE,
                  max_ram_gb=DEFAULT_MAX_RAM_GB,
                  csv_path: Path | None = None):
    """Exécute le benchmark et retourne un DataFrame.

    n_workers=1 → mode séquentiel (pas de fork, RAM safe).
    n_workers>1 → mode parallèle (multiprocessing.Pool).
    Le nombre de workers est automatiquement réduit si la RAM estimée
    dépasse max_ram_gb.

    Si csv_path est fourni, les résultats sont écrits de manière
    incrémentale (append après chaque w_error terminé).
    """
    total = len(w_errors)
    runs_per_w = 2 * n_seeds + 1   # KMeans×seeds + KMedoids×seeds + AP×1
    total_runs = total * runs_per_w
    n_workers = min(n_workers, total)

    # ── Estimation mémoire & auto-limitation ─────────────────────────────
    n_traj_bytes = (_estimate_traj_bytes(trajectories)
                    if platform.system() == "Windows" else 0)
    per_worker_gb = _estimate_worker_peak_gb(max_seg, n_traj_bytes)
    safe_workers = _compute_safe_workers(max_seg, max_ram_gb, n_traj_bytes)

    if n_workers > 1 and n_workers > safe_workers:
        print(f"⚠️  {n_workers} workers × {per_worker_gb:.1f} Go/worker "
              f"= {n_workers * per_worker_gb:.0f} Go pic > limite {max_ram_gb} Go",
              flush=True)
        n_workers = safe_workers
        print(f"    → Réduit automatiquement à {n_workers} workers", flush=True)

    estimated_total = n_workers * per_worker_gb + 3.0  # +3 Go OS/parent
    mode = "séquentiel" if n_workers <= 1 else f"{n_workers} processus parallèles"

    print(f"Benchmark : {total} w_error × (KMeans + KMedoids) × {n_seeds} graines + AP"
          f" = {total_runs} runs", flush=True)
    print(f"K fixe    : {k}  |  Sous-éch. TRACLUS : {max_seg}  |  Sil. sample : {sil_sample}",
          flush=True)
    print(f"Mode      : {mode}", flush=True)
    print(f"RAM estimée : {per_worker_gb:.1f} Go/worker × {n_workers} "
          f"+ 3 Go OS ≈ {estimated_total:.0f} Go  (limite : {max_ram_gb} Go)",
          flush=True)
    if csv_path:
        print(f"CSV incrémental : {csv_path}", flush=True)
    print("=" * 78, flush=True)

    # ── Initialiser le CSV incrémental ────────────────────────────────────
    if csv_path:
        _init_csv(csv_path)

    t_global = time.perf_counter()

    if n_workers <= 1:
        # ── Mode séquentiel (RAM safe) ────────────────────────────────────
        rows = _run_sequential(
            trajectories, n_original_points, w_errors,
            k, n_seeds, min_length, max_seg, sil_sample, t_global,
            csv_path=csv_path,
        )
    else:
        # ── Mode parallèle ────────────────────────────────────────────────
        rows = _run_parallel(
            trajectories, n_original_points, w_errors,
            k, n_seeds, min_length, max_seg, sil_sample, n_workers,
            max_ram_gb, csv_path=csv_path,
        )

    total_time = time.perf_counter() - t_global
    print("=" * 78, flush=True)
    print(f"Terminé en {total_time:.0f}s ({total_time / 60:.1f} min)\n",
          flush=True)

    return pd.DataFrame(rows)


def _run_sequential(trajectories, n_original_points, w_errors,
                    k, n_seeds, min_length, max_seg, sil_sample, t_global,
                    *, csv_path=None):
    """Boucle séquentielle avec progression temps réel + écriture incrémentale."""
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

        # ── Écriture incrémentale ─────────────────────────────────────────
        if csv_path:
            _append_csv(csv_path, batch)

        # ETA
        eta = _eta(t_global, idx + 1, total)
        print(f"  {eta}", flush=True)

    return all_rows


def _run_parallel(trajectories, n_original_points, w_errors,
                  k, n_seeds, min_length, max_seg, sil_sample, n_workers,
                  max_ram_gb=DEFAULT_MAX_RAM_GB, *, csv_path=None):
    """Boucle parallèle (multiprocessing.Pool) avec écriture incrémentale.

    Utilise imap_unordered pour recevoir les résultats au fil de l'eau
    et les écrire immédiatement dans le CSV.
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

    rows: list[dict] = []
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(trajectories, n_original_points),
    ) as pool:
        for batch in pool.imap_unordered(_worker_single_werror, tasks):
            rows.extend(batch)
            if csv_path:
                _append_csv(csv_path, batch)

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

    # 3. Benchmark (écriture CSV incrémentale)
    csv_path = OUTPUT_DIR / "raw_results.csv"
    df = run_benchmark(
        trajectories, n_original_points, w_errors,
        args.k, n_seeds, args.min_length,
        n_workers=n_workers, max_seg=max_seg, sil_sample=sil_sample,
        max_ram_gb=args.max_ram, csv_path=csv_path,
    )
    print(f"📊 Résultats bruts : {csv_path}", flush=True)

    # 5. Figures
    from benchmark.sensitivity_werror_plots import (
        plot_pipeline_impact,
        plot_sweet_spot,
        plot_comparison_algo,
        plot_segment_distributions,
        print_summary,
    )

    print("\nGénération des figures…")
    plot_pipeline_impact(df, OUTPUT_DIR)
    plot_sweet_spot(df, OUTPUT_DIR)
    plot_comparison_algo(df, OUTPUT_DIR)

    # Pré-calcul des longueurs pour plot_segment_distributions
    w_samples = [0.5, 2.0, 5.0, 12.0, 25.0, 50.0, 100.0]
    all_lengths: dict[float, list[float]] = {}
    for w in w_samples:
        segs, _ = compress_all(trajectories, w, args.min_length)
        if segs:
            all_lengths[w] = [s.length() for s in segs]
    plot_segment_distributions(all_lengths, OUTPUT_DIR)

    # 6. Résumé
    print_summary(df)


if __name__ == "__main__":
    main()
