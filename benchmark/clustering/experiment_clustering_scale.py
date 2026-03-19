#!/usr/bin/env python3
"""
experiment_clustering_scale.py
================================
Stress Test & Benchmark de Scalabilité pour 3 algorithmes de clustering.

Compare KMeans (sklearn MiniBatchKMeans), CustomKMedoids (PAM numpy) et
CustomAffinityPropagation sur des sous-échantillons de taille croissante
de segments de trajectoires Dota 2 compressés par MDL.

Métriques mesurées
------------------
  - Temps de fit   : time.perf_counter() sur le .fit() brut
  - Silhouette     : sklearn.metrics.silhouette_score(D, labels, metric="precomputed")
                     La matrice D est toujours la distance TRACLUS custom
                     → comparaison équitable indépendante de l'espace de chaque algo.

Sorties
-------
  output/benchmark_clustering/heavy_benchmark_results.csv  (append en temps réel)
  output/benchmark_clustering/benchmark.log
  output/benchmark_clustering/clustering_stress_test.png

Usage
-----
  python scripts/clustering/experiment_clustering_scale.py
"""

# ---------------------------------------------------------------------------
# Imports standard
# ---------------------------------------------------------------------------
import sys
import os
import math
import csv
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Forcer l'utilisation de TOUS les threads CPU pour les libs numériques.
# Doit être fait AVANT l'import de numpy/sklearn (sinon ignoré).
# Sur Ryzen 5 3600 (6 cœurs / 12 threads) : OpenBLAS, MKL et OpenMP
# utiliseront les 12 threads pour les opérations matricielles internes.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS",      str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS",      str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(os.cpu_count() or 1))

# ---------------------------------------------------------------------------
# Résolution du chemin projet
# scripts/clustering/ → parents[2] → racine du projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports tiers
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Pas d'affichage GUI — export PNG uniquement
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Imports internes dota_analytics
# ---------------------------------------------------------------------------
from dota_analytics.clustering import load_data
from dota_analytics.custom_kmedoids import CustomKMedoids
from dota_analytics.custom_ap import CustomAffinityPropagation
from config import COMPRESSED_DIR

# ---------------------------------------------------------------------------
# PARAMÈTRES DU BENCHMARK
# ---------------------------------------------------------------------------
N_LIST: list[int] = [500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000]
N_ITERATIONS: int = 3          # Itérations par taille (moyenne + écart-type)
N_CLUSTERS_KMEANS: int = 50    # Clusters fixes pour KMeans
N_CLUSTERS_KMEDOIDS: int = 50  # Clusters fixes pour KMedoids
AP_MAX_N: int = 7500           # AP skippée au-delà — O(N²) message passing trop long
N_CALIB: int = 300             # Taille de calibration pour l'estimation du temps
RAM_LIMIT_GB: float = 26.0    # Limite critique RAM (32 Go - marge OS)

# ---------------------------------------------------------------------------
# CHEMINS DE SORTIE
# ---------------------------------------------------------------------------
BENCH_DIR = PROJECT_ROOT / "output" / "benchmark_clustering"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = BENCH_DIR / "heavy_benchmark_results.csv"
PLOT_PATH = BENCH_DIR / "clustering_stress_test.png"
LOG_PATH  = BENCH_DIR / "benchmark.log"

# Logger module-level — configuré dans main() pour éviter les double-handlers
# quand les workers réimportent ce module (start method 'spawn').
log = logging.getLogger(__name__)


# ===========================================================================
# UTILITAIRES
# ===========================================================================

def _hms(seconds: float) -> str:
    """Formate un nombre de secondes en '00h00m00s'."""
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}h{int(m):02d}m{int(s):02d}s"


def _fmt(v) -> str:
    """Formate une valeur pour écriture CSV (NaN-safe)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NaN"
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def _compute_D(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de distance TRACLUS symétrisée (N×N, float32).

    Accepte des tableaux numpy bruts (et non des objets Segment) →
    peut être appelée depuis un worker sans sérialiser de gros objets Python.
    Le pickling d'un array float32 via le buffer protocol est ~100× plus
    rapide que le pickling d'une liste d'objets Segment.
    """
    vectors    = ends - starts
    lengths    = np.linalg.norm(vectors, axis=1)
    lengths    = np.clip(lengths, 1e-9, None)
    directions = vectors / lengths[:, np.newaxis]

    cos_theta = np.clip(np.dot(directions, directions.T), -1.0, 1.0)
    d_angle   = (1.0 - cos_theta) * (lengths[:, np.newaxis] + lengths[np.newaxis, :])

    vx = directions[:, 0:1]
    vy = directions[:, 1:2]
    vec_sx = starts[np.newaxis, :, 0] - starts[:, np.newaxis, 0]
    vec_sy = starts[np.newaxis, :, 1] - starts[:, np.newaxis, 1]
    vec_ex = ends[np.newaxis,   :, 0] - starts[:, np.newaxis, 0]
    vec_ey = ends[np.newaxis,   :, 1] - starts[:, np.newaxis, 1]

    cross_s   = np.abs(vx * vec_sy - vy * vec_sx)
    cross_e   = np.abs(vx * vec_ey - vy * vec_ex)
    sum_cross = cross_s + cross_e
    d_perp    = np.zeros_like(sum_cross)
    mask      = sum_cross > 0
    d_perp[mask] = (cross_s[mask]**2 + cross_e[mask]**2) / sum_cross[mask]

    proj_s = vec_sx * vx + vec_sy * vy
    proj_e = vec_ex * vx + vec_ey * vy
    base_l = lengths[:, np.newaxis]
    d_par  = (
        np.minimum(np.abs(proj_s), np.abs(proj_s - base_l))
        + np.minimum(np.abs(proj_e), np.abs(proj_e - base_l))
    )

    D_asym   = (d_perp + d_angle + d_par).astype(np.float32)
    len_mask = lengths[:, np.newaxis] > lengths[np.newaxis, :]
    D        = np.where(len_mask, D_asym, D_asym.T)
    np.fill_diagonal(D, 0.0)
    return D


def _silhouette(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score avec metric='precomputed' — robuste aux labels -1 (AP).
    Retourne NaN si le calcul est impossible (cluster unique, etc.).
    NE loggue PAS directement : appelée depuis un worker qui retourne ses
    messages au main process pour éviter les écritures concurrentes sur le log.
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return float("nan")
    D_v, l_v = D[np.ix_(valid, valid)], labels[valid]
    if len(np.unique(l_v)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(D_v.astype(np.float64), l_v,
                                      metric="precomputed", n_jobs=-1))
    except Exception:
        return float("nan")


# ===========================================================================
# GESTION CSV — append en temps réel (survie aux crashs)
# ===========================================================================

CSV_HEADER = ["N", "Iteration", "Algorithm", "Time_Seconds",
              "Silhouette_Score", "N_Clusters_Found"]


def init_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        log.info(f"CSV initialisé        : {path}")
    else:
        log.info(f"CSV existant (append) : {path}  ← reprise sans écrasement")


def append_row(path: Path, N, it, algo, t, sil, k) -> None:
    """Écrit une ligne CSV immédiatement. Appelé depuis le main process uniquement."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([N, it, algo, _fmt(t), _fmt(sil), _fmt(k)])


# ===========================================================================
# WORKER — MODULE-LEVEL (obligatoire pour le pickling multiprocessing)
# ===========================================================================

def _iteration_worker(args: tuple):
    """
    Worker indépendant : une itération complète pour une taille N donnée.

    Reçoit : (N, it, seed, starts_all, ends_all) — tableaux numpy float32.
    Retourne : (list[dict], list[str]) = (résultats, messages de log).

    Pourquoi numpy arrays et non Segment objects ?
    → pickle protocol 5 (buffer protocol) sérialise un array float32 N×2
      en quelques millisecondes. Sérialiser N objets Python personnalisés
      serait ~100× plus lent et doublerait la RAM pendant le transfert.

    Pourquoi retourner des messages de log ?
    → Sur Linux (fork), le worker hérite des handlers du parent. Si plusieurs
      workers écrivent simultanément dans le fichier log, les lignes se
      corrompent. Les messages sont donc retournés et écrits séquentiellement
      par le main process (via as_completed).
    """
    N, it, seed, starts_all, ends_all = args
    msgs    = []
    results = []

    rng     = np.random.RandomState(seed)
    indices = rng.choice(len(starts_all), size=N, replace=False)
    starts  = starts_all[indices]
    ends    = ends_all[indices]

    # ── Matrice TRACLUS ────────────────────────────────────────────────────
    t0    = time.perf_counter()
    D     = _compute_D(starts, ends)
    t_mat = time.perf_counter() - t0
    msgs.append(f"  [N={N:>6} it={it}] Matrice   : {t_mat:>7.2f}s  "
                f"({D.nbytes / 1e6:.0f} MB)")

    # ── Features pour KMeans (espace euclidien) ────────────────────────────
    mids     = (starts + ends) / 2.0
    vectors  = ends - starts
    lengths  = np.linalg.norm(vectors, axis=1, keepdims=True)
    X_scaled = StandardScaler().fit_transform(
        np.hstack([mids, vectors, lengths]).astype(np.float32)
    )

    # ── ALGO 1 : KMeans ───────────────────────────────────────────────────
    t0        = time.perf_counter()
    km_labels = MiniBatchKMeans(
        n_clusters=N_CLUSTERS_KMEANS, random_state=42,
        batch_size=min(4096, N), n_init=5, verbose=0,
    ).fit_predict(X_scaled)
    t_km  = time.perf_counter() - t0
    s_km  = _silhouette(D, km_labels)
    msgs.append(f"  [N={N:>6} it={it}] KMeans    : {t_km:>7.3f}s  "
                f"sil={s_km:+.4f}")
    results.append(dict(N=N, it=it, algo="KMeans",
                        t=t_km, sil=s_km, k=N_CLUSTERS_KMEANS))

    # ── ALGO 2 : CustomKMedoids (PAM) ────────────────────────────────────
    t0   = time.perf_counter()
    kmed = CustomKMedoids(n_clusters=N_CLUSTERS_KMEDOIDS,
                          max_iter=300, random_state=42)
    kmed.fit(D)
    t_kmed = time.perf_counter() - t0
    n_kmed = len(np.unique(kmed.labels_))
    s_kmed = _silhouette(D, kmed.labels_)
    msgs.append(f"  [N={N:>6} it={it}] KMedoids  : {t_kmed:>7.3f}s  "
                f"sil={s_kmed:+.4f}  k={n_kmed}")
    results.append(dict(N=N, it=it, algo="KMedoids",
                        t=t_kmed, sil=s_kmed, k=n_kmed))

    # ── ALGO 3 : CustomAffinityPropagation — SKIP si N > AP_MAX_N ────────
    if N > AP_MAX_N:
        msgs.append(f"  [N={N:>6} it={it}] AP        : AP ignoré pour N={N} "
                    f"(Temps estimé trop long — seuil={AP_MAX_N})")
        results.append(dict(N=N, it=it, algo="AffinityPropagation",
                            t=float("nan"), sil=float("nan"), k=None))
    else:
        S = -(D.astype(np.float64))
        np.fill_diagonal(S, np.median(S))
        t0 = time.perf_counter()
        ap = CustomAffinityPropagation(damping=0.9, max_iter=400,
                                       convergence_iter=15, verbose=False)
        ap.fit(S)
        t_ap = time.perf_counter() - t0
        n_ap = (len(ap.cluster_centers_indices_)
                if ap.cluster_centers_indices_ is not None else 0)
        s_ap = _silhouette(D, ap.labels_)
        msgs.append(f"  [N={N:>6} it={it}] AP        : {t_ap:>7.3f}s  "
                    f"sil={s_ap:+.4f}  k={n_ap}")
        results.append(dict(N=N, it=it, algo="AffinityPropagation",
                            t=t_ap, sil=s_ap, k=n_ap))

    return results, msgs


# ===========================================================================
# PARALLÉLISME ADAPTATIF
# ===========================================================================

def _max_workers(N: int) -> int:
    """
    Nombre max de workers parallèles selon N — adapté pour 32 Go RAM.

    Chaque worker alloue ~3× la taille de la matrice D :
      - D elle-même (float32, N×N)
      - S = -D pour AP (float64, N×N)  — conservé même si AP est skippée
      - Intermédiaires numpy (argmin, np.ix_, silhouette_score)

    Seuils conservatifs pour 32 Go RAM (26 Go utilisables) :
      N ≤ 2500   →  D ≈   25 MB × 3 × W → W=N_ITER workers  (~225 MB) ✓
      N ≤ 5000   →  D ≈  100 MB × 3 × W → W=3 workers       (~900 MB) ✓
      N ≤ 10000  →  D ≈  400 MB × 3 × W → W=2 workers       (~2.4 GB) ✓
      N ≤ 20000  →  D ≈  1.6 GB × 3 × 1 → W=1 worker        (~4.8 GB) ✓
      N > 20000  →  D ≈  3.6+ GB × 3 × 1→ W=1 worker        (~10+ GB) ⚠
    """
    n_cpu = os.cpu_count() or 1
    if N <= 2500:
        return min(N_ITERATIONS, n_cpu)
    elif N <= 5000:
        return min(N_ITERATIONS, n_cpu, 3)
    elif N <= 10000:
        return min(N_ITERATIONS, 2)
    else:
        return 1  # séquentiel — matrice N×N trop volumineuse pour paralléliser


# ===========================================================================
# ESTIMATION DU TEMPS (calibration avant la boucle)
# ===========================================================================

def estimate_runtime(starts_all: np.ndarray, ends_all: np.ndarray) -> float:
    """
    Calibration rapide sur N=N_CALIB puis extrapolation par complexité.

    Complexités utilisées :
      Matrice TRACLUS  : O(N²)   → quadratique
      KMeans MiniBatch : O(N)    → linéaire (pas de matrice)
      KMedoids PAM     : O(N²)   → quadratique (converge vite en pratique)
      AP               : O(N²)   → quadratique (message passing × max_iter)
      Silhouette       : O(N²)   → quadratique (sklearn precomputed)

    Retourne l'estimation du temps total en secondes.
    """
    log.info("─" * 72)
    log.info(f"  CALIBRATION (N={N_CALIB}) — estimation durée totale")
    log.info("─" * 72)

    rng = np.random.RandomState(999)
    idx = rng.choice(len(starts_all), size=N_CALIB, replace=False)
    sc, ec = starts_all[idx], ends_all[idx]

    # Matrice
    t0 = time.perf_counter()
    D_c = _compute_D(sc, ec)
    t_mat = time.perf_counter() - t0

    # KMeans
    mids = (sc + ec) / 2.0
    vecs = ec - sc
    lgs  = np.linalg.norm(vecs, axis=1, keepdims=True)
    X_sc = StandardScaler().fit_transform(
        np.hstack([mids, vecs, lgs]).astype(np.float32))
    t0 = time.perf_counter()
    km_lab = MiniBatchKMeans(n_clusters=N_CLUSTERS_KMEANS, random_state=42,
                              batch_size=min(4096, N_CALIB), n_init=5,
                              verbose=0).fit_predict(X_sc)
    t_km = time.perf_counter() - t0

    # KMedoids
    t0 = time.perf_counter()
    CustomKMedoids(n_clusters=N_CLUSTERS_KMEDOIDS, max_iter=300,
                   random_state=42).fit(D_c)
    t_kmed = time.perf_counter() - t0

    # AP
    S_c = -(D_c.astype(np.float64))
    np.fill_diagonal(S_c, np.median(S_c))
    t0 = time.perf_counter()
    CustomAffinityPropagation(damping=0.9, max_iter=400,
                               convergence_iter=15, verbose=False).fit(S_c)
    t_ap = time.perf_counter() - t0

    # Silhouette × 1 algo
    t0 = time.perf_counter()
    try:
        silhouette_score(D_c.astype(np.float64), km_lab,
                         metric="precomputed", n_jobs=-1)
    except Exception:
        pass
    t_sil = time.perf_counter() - t0

    log.info(f"  Résultats N={N_CALIB} :")
    log.info(f"    Matrice TRACLUS :  {t_mat:.4f}s")
    log.info(f"    KMeans fit      :  {t_km:.4f}s")
    log.info(f"    KMedoids fit    :  {t_kmed:.4f}s")
    log.info(f"    AP fit          :  {t_ap:.4f}s")
    log.info(f"    Silhouette ×1   :  {t_sil:.4f}s")
    log.info("")

    header = (f"  {'N':>7}  {'Workers':>7}  {'t/iter':>10}  "
              f"{'t_bloc':>12}  {'Cumulé':>12}  {'Speedup':>8}  Notes")
    log.info(header)
    log.info(f"  {'─' * 75}")

    total_est = 0.0
    for N in N_LIST:
        q2 = (N / N_CALIB) ** 2
        q1 =  N / N_CALIB
        t_i = (
            t_mat  * q2
            + t_km   * q1                              # MiniBatch ~ linéaire
            + t_kmed * q2
            + (t_ap * q2 if N <= AP_MAX_N else 0.0)   # AP ou skippé
            + 3 * t_sil * q2                           # silhouette × 3 algos
        )
        w       = _max_workers(N)
        t_bloc  = math.ceil(N_ITERATIONS / w) * t_i
        total_est += t_bloc
        speedup = N_ITERATIONS / math.ceil(N_ITERATIONS / w)
        note    = "AP=skip" if N > AP_MAX_N else ""
        log.info(f"  {N:>7}  {w:>7}  {t_i:>9.0f}s  "
                 f"{_hms(t_bloc):>12s}  {_hms(total_est):>12s}  "
                 f"×{speedup:.1f}      {note}")

    log.info("")
    log.info(f"  ⏱  ESTIMATION TOTALE : {total_est:.0f}s  ≈  {_hms(total_est)}")
    log.info(f"     (sans parallélisme ce serait ~{_hms(total_est * N_ITERATIONS)}")
    log.info("─" * 72)
    return total_est


# ===========================================================================
# BOUCLE PRINCIPALE (parallèle)
# ===========================================================================

def run_benchmark(all_segments: list) -> None:
    """
    Boucle principale du benchmark.

    Stratégie de parallélisme
    ──────────────────────────
    1. Conversion Segment objects → numpy arrays (starts_all, ends_all) une
       seule fois. Le pickling numpy est ~100× plus rapide que les objets Python.
    2. Pour chaque N : _max_workers(N) iterations tournent simultanément via
       ProcessPoolExecutor. Chaque worker a sa propre seed reproductible.
    3. as_completed() : les résultats sont traités dès qu'ils arrivent
       (pas besoin d'attendre toutes les itérations d'un bloc).
    4. CSV écrit par le main process uniquement → pas de concurrence sur le fichier.
    """
    log.info("Conversion segments → numpy arrays pour les workers...")
    starts_all = np.array([(s.start.x, s.start.y) for s in all_segments],
                          dtype=np.float32)
    ends_all   = np.array([(s.end.x,   s.end.y  ) for s in all_segments],
                          dtype=np.float32)
    log.info(f"  starts_all : {starts_all.nbytes / 1e6:.1f} MB  "
             f"ends_all : {ends_all.nbytes / 1e6:.1f} MB")

    estimate_runtime(starts_all, ends_all)
    init_csv(CSV_PATH)

    total_runs    = len(N_LIST) * N_ITERATIONS * 3
    done_runs     = 0
    t_bench_start = time.perf_counter()

    for N in N_LIST:
        log.info("═" * 72)
        log.info(f"  BLOC  N = {N:>6}   ({N_ITERATIONS} itérations × 3 algos)")
        log.info("═" * 72)

        if N > len(all_segments):
            log.warning(f"  N={N} > corpus ({len(all_segments)}). SKIP → NaN.")
            for it in range(1, N_ITERATIONS + 1):
                for algo in ["KMeans", "KMedoids", "AffinityPropagation"]:
                    append_row(CSV_PATH, N, it, algo,
                               float("nan"), float("nan"), None)
            done_runs += N_ITERATIONS * 3
            continue

        workers  = _max_workers(N)
        # RAM peak : D float32 + S float64 + intermédiaires ≈ N*N*8*2 par worker
        ram_peak_gb = (N * N * 8) / (1024**3) * 2 * workers
        log.info(f"  Workers : {workers}  |  RAM estimée (peak) : {ram_peak_gb:.1f} Go")
        if ram_peak_gb > RAM_LIMIT_GB:
            log.warning(f"  ⚠  RAM estimée ({ram_peak_gb:.1f} Go) dépasse la limite "
                        f"de sécurité ({RAM_LIMIT_GB} Go). Risque d'OOM !")

        args_list = [
            (N, it, N * 1000 + it, starts_all, ends_all)
            for it in range(1, N_ITERATIONS + 1)
        ]
        t_bloc_start = time.perf_counter()

        if workers > 1:
            # ── Exécution parallèle ───────────────────────────────────────
            with ProcessPoolExecutor(max_workers=workers) as executor:
                fut_to_it = {
                    executor.submit(_iteration_worker, a): a[1]
                    for a in args_list
                }
                for fut in as_completed(fut_to_it):
                    it = fut_to_it[fut]
                    try:
                        rows, msgs = fut.result()
                        for msg in msgs:
                            log.info(msg)
                        for r in rows:
                            append_row(CSV_PATH, r["N"], r["it"], r["algo"],
                                       r["t"], r["sil"], r["k"])
                        done_runs += len(rows)
                    except Exception as exc:
                        log.error(f"  ✗ N={N} it={it} : {exc}")
                        for algo in ["KMeans", "KMedoids", "AffinityPropagation"]:
                            append_row(CSV_PATH, N, it, algo,
                                       float("nan"), float("nan"), None)
                        done_runs += 3
        else:
            # ── Exécution séquentielle pour les grands N ─────────────────
            for a in args_list:
                try:
                    rows, msgs = _iteration_worker(a)
                    for msg in msgs:
                        log.info(msg)
                    for r in rows:
                        append_row(CSV_PATH, r["N"], r["it"], r["algo"],
                                   r["t"], r["sil"], r["k"])
                    done_runs += len(rows)
                except Exception as exc:
                    it = a[1]
                    log.error(f"  ✗ N={N} it={it} : {exc}")
                    for algo in ["KMeans", "KMedoids", "AffinityPropagation"]:
                        append_row(CSV_PATH, N, it, algo,
                                   float("nan"), float("nan"), None)
                    done_runs += 3

        t_bloc   = time.perf_counter() - t_bloc_start
        elapsed  = time.perf_counter() - t_bench_start
        progress = done_runs / total_runs * 100
        log.info(f"  BLOC N={N} : {t_bloc:.1f}s  |  "
                 f"{done_runs}/{total_runs} runs ({progress:.1f}%)  |  "
                 f"Écoulé : {_hms(elapsed)}")

    log.info("═" * 72)
    log.info("  BENCHMARK COMPLET.")
    log.info(f"  CSV : {CSV_PATH}")
    log.info("═" * 72)


# ===========================================================================
# VISUALISATION — Double graphique
# ===========================================================================

def plot_heavy_benchmark(csv_path: Path) -> None:
    """
    Lit le CSV généré et produit une figure matplotlib 1×2 :

    Graphique 1 — Scalabilité (log/log)
        Axe X : N (log), Axe Y : Temps moyen (log)
        Montre l'explosion quadratique de AP vs linéarité de KMeans.

    Graphique 2 — Qualité sémantique
        Axe X : N, Axe Y : Silhouette moyen
        Prouve que KMedoids / AP offrent de meilleurs clusters TRACLUS que KMeans.
    """
    log.info("Génération du graphique double ...")

    df = pd.read_csv(csv_path)
    df["Time_Seconds"]    = pd.to_numeric(df["Time_Seconds"],    errors="coerce")
    df["Silhouette_Score"] = pd.to_numeric(df["Silhouette_Score"], errors="coerce")
    df["N"]               = pd.to_numeric(df["N"],               errors="coerce")

    ALGOS   = ["KMeans", "KMedoids", "AffinityPropagation"]
    COLORS  = {"KMeans": "#1f77b4", "KMedoids": "#ff7f0e", "AffinityPropagation": "#2ca02c"}
    MARKERS = {"KMeans": "o",       "KMedoids": "s",       "AffinityPropagation": "^"}
    LABELS  = {
        "KMeans":              "KMeans (MiniBatch, euclidien)",
        "KMedoids":            "KMedoids (PAM custom, TRACLUS)",
        "AffinityPropagation": f"AffinityPropagation (custom, TRACLUS) — skippé N>{AP_MAX_N}",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Benchmark de Scalabilité — Clustering Trajectoires Dota 2\n"
        "Métrique de distance : TRACLUS (perpendiculaire + parallèle + angulaire)",
        fontsize=14, fontweight="bold",
    )

    # -----------------------------------------------------------------------
    # Graphique 1 : Scalabilité — Temps (échelle log/log)
    # -----------------------------------------------------------------------
    ax1.set_title("Scalabilité : Temps d'exécution du .fit()", fontsize=12, fontweight="bold")
    ax1.set_xlabel("N — nombre de segments", fontsize=11)
    ax1.set_ylabel("Temps moyen (secondes)", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    for algo in ALGOS:
        sub = df[df["Algorithm"] == algo].copy()
        sub = sub.dropna(subset=["Time_Seconds"])
        if sub.empty:
            continue
        stats = (sub.groupby("N")["Time_Seconds"]
                   .agg(mean="mean", std="std")
                   .reset_index())
        stats["std"] = stats["std"].fillna(0.0)

        ax1.plot(
            stats["N"], stats["mean"],
            color=COLORS[algo], marker=MARKERS[algo],
            linewidth=2.2, markersize=8,
            label=LABELS[algo],
        )
        ax1.fill_between(
            stats["N"],
            (stats["mean"] - stats["std"]).clip(lower=1e-5),
            stats["mean"] + stats["std"],
            color=COLORS[algo], alpha=0.12,
        )

    # Repère vertical AP_MAX_N
    ax1.axvline(x=AP_MAX_N, color=COLORS["AffinityPropagation"],
                linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.text(AP_MAX_N * 1.05, ax1.get_ylim()[0] * 2,
             f"AP skippée\n(N > {AP_MAX_N})",
             color=COLORS["AffinityPropagation"], fontsize=8, va="bottom")

    ax1.legend(fontsize=9, loc="upper left")

    # -----------------------------------------------------------------------
    # Graphique 2 : Qualité — Silhouette Score
    # -----------------------------------------------------------------------
    ax2.set_title(
        "Qualité Sémantique : Silhouette Score\n"
        "(distance TRACLUS precomputed — identique pour les 3 algos)",
        fontsize=12, fontweight="bold",
    )
    ax2.set_xlabel("N — nombre de segments", fontsize=11)
    ax2.set_ylabel("Silhouette Score moyen  [-1 ; 1]", fontsize=11)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    for algo in ALGOS:
        sub = df[df["Algorithm"] == algo].copy()
        sub = sub.dropna(subset=["Silhouette_Score"])
        if sub.empty:
            continue
        stats = (sub.groupby("N")["Silhouette_Score"]
                   .agg(mean="mean", std="std")
                   .reset_index())
        stats["std"] = stats["std"].fillna(0.0)

        ax2.plot(
            stats["N"], stats["mean"],
            color=COLORS[algo], marker=MARKERS[algo],
            linewidth=2.2, markersize=8,
            label=LABELS[algo],
        )
        ax2.fill_between(
            stats["N"],
            (stats["mean"] - stats["std"]).clip(lower=-1.0),
            (stats["mean"] + stats["std"]).clip(upper=1.0),
            color=COLORS[algo], alpha=0.12,
        )

    ax2.legend(fontsize=9, loc="best")

    # Annotation interprétative
    ax2.annotate(
        "Score > 0 : structure cohérente\nScore < 0 : clusters mal définis",
        xy=(0.03, 0.05), xycoords="axes fraction",
        fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7),
    )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Graphique sauvegardé : {PLOT_PATH}")


# ===========================================================================
# POINT D'ENTRÉE
# ===========================================================================

def main() -> None:
    # Logging configuré ici (et non au niveau module) pour éviter les
    # double-handlers quand les workers réimportent ce module (spawn).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ],
    )

    n_cpu = os.cpu_count() or 1
    log.info("╔" + "═" * 70 + "╗")
    log.info("║  STRESS TEST — Benchmark Scalabilité Clustering Dota 2  v2 Parallèle  ║")
    log.info("╠" + "═" * 70 + "╣")
    log.info(f"║  Tailles N testées     : {N_LIST}")
    log.info(f"║  Itérations / taille   : {N_ITERATIONS}")
    log.info(f"║  k (KMeans / KMedoids) : {N_CLUSTERS_KMEANS}")
    log.info(f"║  AP skippée pour N >   : {AP_MAX_N}")
    log.info(f"║  CPUs disponibles      : {n_cpu}  (parallélisme adaptatif par N)")
    log.info(f"║  CSV                   : {CSV_PATH}")
    log.info(f"║  Log                   : {LOG_PATH}")
    log.info(f"║  Graphique             : {PLOT_PATH}")
    log.info("╚" + "═" * 70 + "╝")

    # Chargement du corpus
    w_error_folders = sorted(COMPRESSED_DIR.glob("w_error_*"))
    if not w_error_folders:
        log.error(f"Aucun dossier w_error_* dans {COMPRESSED_DIR}.")
        log.error("Lancez d'abord : python run.py compress --w_error 12")
        sys.exit(1)

    # Sélectionne le dossier avec le plus de fichiers JSON (corpus le plus grand)
    target_folder = max(w_error_folders, key=lambda p: len(list(p.glob("*.json"))))
    log.info(f"Dossier source sélectionné : {target_folder}")
    log.info(f"  ({len(list(target_folder.glob('*.json')))} fichiers JSON détectés)")

    log.info("Chargement de tous les segments (peut prendre 1-2 min selon le corpus)...")
    t0_load        = time.perf_counter()
    all_segments, _ = load_data(str(target_folder))
    t_load         = time.perf_counter() - t0_load

    if not all_segments:
        log.error("Aucun segment chargé. Vérifiez le dossier compressé.")
        sys.exit(1)

    corpus_size = len(all_segments)
    log.info(f"✅ {corpus_size:,} segments chargés en {t_load:.1f}s")

    # Avertissement si le corpus est plus petit que le N max demandé
    n_max_testable = max((n for n in N_LIST if n <= corpus_size), default=None)
    if n_max_testable is None:
        log.error(f"Corpus trop petit ({corpus_size}) pour tester N_LIST={N_LIST}.")
        sys.exit(1)
    if n_max_testable < max(N_LIST):
        log.warning(f"N max testable = {n_max_testable} (corpus = {corpus_size}). "
                    f"Les tailles > {corpus_size} seront loguées NaN.")

    # -----------------------------------------------------------------------
    # Benchmark principal
    # -----------------------------------------------------------------------
    t0_bench = time.perf_counter()
    run_benchmark(all_segments)
    t_bench = time.perf_counter() - t0_bench

    log.info(f"Durée totale du benchmark : {t_bench/3600:.2f} heure(s)  "
             f"({t_bench:.0f} secondes)")

    # -----------------------------------------------------------------------
    # Génération du graphique
    # -----------------------------------------------------------------------
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        try:
            plot_heavy_benchmark(CSV_PATH)
        except Exception as exc:
            log.error(f"Erreur lors de la génération du graphique : {exc}")
            log.info("Pour re-générer le graphique manuellement :")
            log.info("  from scripts.clustering.experiment_clustering_scale import plot_heavy_benchmark")
            log.info(f"  plot_heavy_benchmark(Path('{CSV_PATH}'))")
    else:
        log.warning("CSV vide ou inexistant — graphique non généré.")

    log.info("FIN DU SCRIPT.")


if __name__ == "__main__":
    main()
