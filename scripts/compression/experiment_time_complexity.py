"""Expérimentation 2 : Complexité Temporelle Empirique.

Compare les temps d'exécution de MDL, Douglas-Peucker et Échantillonnage
Uniforme en fonction du nombre de points N pour analyser empiriquement leur
complexité algorithmique.

Les tailles testées vont jusqu'à 5 000 points. Comme un seul match ne
fournit que ~2 500 points valides pour un joueur, le script construit un
pool en agrégeant plusieurs fichiers (player 0 de chaque match). Les
sous-fenêtres sont extraites aléatoirement depuis ce pool continu.

Usage :
    from scripts.compression.experiment_time_complexity import run_time_benchmark
    run_time_benchmark()
"""

import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dota_analytics.compression import DouglasPeuckerCompressor, MDLCompressor
from dota_analytics.structures import Segment, Trajectory, TrajectoryPoint

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = BASE_DIR / "data-dota"
PRIMARY_MATCH = "3841665963"          # Fichier de référence (chargé en priorité)
PLAYER_ID = 0

OUTPUT_DIR = BASE_DIR / "output" / "benchmark_time"
OUTPUT_CSV = OUTPUT_DIR / "execution_times.csv"
OUTPUT_PNG = OUTPUT_DIR / "time_complexity_plot.png"

SIZES = [100, 250, 500, 1000, 2500, 5000]   # Tailles N (nombre de points)
N_ITERATIONS = 20                             # Répétitions par taille
W_ERROR_MDL = 12.0
RANDOM_SEED = 42

ALGO_COLORS = {
    "Uniforme": "darkorange",
    "Douglas-Peucker": "forestgreen",
    "MDL": "crimson",
}
ALGO_ORDER = ["Uniforme", "Douglas-Peucker", "MDL"]


# =============================================================================
# CHARGEMENT DU POOL DE POINTS
# =============================================================================


def _load_points_pool(min_size: int) -> list[TrajectoryPoint]:
    """Charge un pool de points suffisant pour couvrir la taille maximale.

    Commence par coord_{PRIMARY_MATCH}.csv (joueur 0), puis agrège d'autres
    fichiers si le pool est trop petit.

    Args:
        min_size: Nombre minimal de points nécessaires dans le pool

    Returns:
        Liste de TrajectoryPoint (pool continu)
    """
    x_col, y_col = f"x{PLAYER_ID}", f"y{PLAYER_ID}"
    pool: list[TrajectoryPoint] = []

    def _extract(csv_path: Path) -> list[TrajectoryPoint]:
        df = pd.read_csv(csv_path)
        if x_col not in df.columns:
            return []
        mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
        sub = df[mask].sort_values("tick")
        return [
            TrajectoryPoint(x=float(r[x_col]), y=float(r[y_col]), tick=int(r["tick"]))
            for _, r in sub.iterrows()
        ]

    # Fichier de référence en priorité
    primary = DATA_DIR / f"coord_{PRIMARY_MATCH}.csv"
    pool.extend(_extract(primary))
    print(f"   {PRIMARY_MATCH} : {len(pool)} points")

    # Fichiers supplémentaires si besoin
    if len(pool) < min_size:
        files = sorted(DATA_DIR.glob("coord_*.csv"))
        for f in files:
            if f.name == primary.name:
                continue
            before = len(pool)
            pool.extend(_extract(f))
            added = len(pool) - before
            if added > 0:
                print(f"   {f.stem.replace('coord_', '')} : +{added} points (total: {len(pool)})")
            if len(pool) >= min_size:
                break

    return pool


# =============================================================================
# ALGORITHMES
# =============================================================================


def _uniform_sampling(points: list, n_segments: int) -> list[Segment]:
    """Échantillonnage uniforme ciblant n_segments segments."""
    n_pts = len(points)
    step = max(1, (n_pts - 1) // n_segments)
    indices = list(range(0, n_pts, step))
    if indices[-1] != n_pts - 1:
        indices.append(n_pts - 1)
    while len(indices) - 1 > n_segments and len(indices) > 2:
        indices.pop(-2)
    indices = sorted(set(indices))
    return [
        Segment(start=points[indices[i]], end=points[indices[i + 1]])
        for i in range(len(indices) - 1)
    ]


def _find_dp_epsilon(trajectory: Trajectory, target_n: int, tolerance: int = 2) -> float:
    """Trouve l'epsilon DP produisant ~target_n segments par dichotomie.

    La recherche d'epsilon est effectuée UNE SEULE FOIS hors de la boucle
    de chronométrage, pour mesurer uniquement la complexité de l'algorithme
    de compression, pas du paramétrage.

    Returns:
        Valeur d'epsilon optimale
    """
    lo, hi = 0.01, 5000.0
    best_eps = (lo + hi) / 2
    best_diff = float("inf")

    for _ in range(40):
        mid = (lo + hi) / 2
        segs = DouglasPeuckerCompressor(epsilon=mid).compress_player_trajectory(trajectory)
        diff = abs(len(segs) - target_n)
        if diff < best_diff:
            best_diff = diff
            best_eps = mid
        if diff <= 2:
            break
        elif len(segs) > target_n:
            lo = mid
        else:
            hi = mid

    return best_eps


# =============================================================================
# CHRONOMÉTRAGE
# =============================================================================


def _time_compression(func, *args) -> float:
    """Exécute func(*args) et retourne le temps en millisecondes.

    Utilise time.perf_counter() pour une mesure haute résolution.
    """
    t_start = time.perf_counter()
    func(*args)
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0  # ms


# =============================================================================
# VISUALISATION
# =============================================================================


def _generate_plot(df_results: pd.DataFrame) -> None:
    """Génère le graphique de complexité temporelle (courbes + zones d'écart-type)."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for algo in ALGO_ORDER:
        sub = df_results[df_results["Algorithm"] == algo].sort_values("N_points")
        xs = sub["N_points"].values
        ys = sub["Mean_Time_ms"].values
        stds = sub["Std_Time_ms"].values
        color = ALGO_COLORS[algo]

        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                label=algo, zorder=5)
        ax.fill_between(xs, ys - stds, ys + stds,
                        color=color, alpha=0.15, zorder=3)

    ax.set_xlabel("Nombre de points N", fontsize=13, fontweight="bold")
    ax.set_ylabel("Temps d'exécution moyen (ms)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Complexité Temporelle Empirique : MDL vs Baselines\n"
        f"({N_ITERATIONS} itérations par taille, w_error MDL={W_ERROR_MDL})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=12, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(df_results["N_points"].unique())
    ax.set_xticklabels(
        [f"{n:,}" for n in sorted(df_results["N_points"].unique())],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Annotation de la valeur finale (N le plus grand)
    n_max = df_results["N_points"].max()
    for algo in ALGO_ORDER:
        row = df_results[
            (df_results["Algorithm"] == algo) & (df_results["N_points"] == n_max)
        ]
        if len(row) == 0:
            continue
        y_val = row["Mean_Time_ms"].values[0]
        ax.annotate(
            f"{y_val:.1f} ms",
            xy=(n_max, y_val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color=ALGO_COLORS[algo], fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"📈 Graphique sauvegardé → {OUTPUT_PNG}")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_time_benchmark() -> pd.DataFrame:
    """Benchmark de complexité temporelle empirique des 3 algorithmes.

    Pour chaque taille N dans SIZES :
      1. Sélectionne aléatoirement une fenêtre de N points consécutifs.
      2. Compresse avec MDL (référence → C segments).
      3. Calibre l'epsilon DP hors boucle pour cibler C segments.
      4. Répète N_ITERATIONS fois la mesure des 3 algos.
      5. Calcule moyenne et écart-type des temps (en ms).

    Returns:
        DataFrame avec colonnes N_points, Algorithm, Mean_Time_ms, Std_Time_ms
    """
    print("=" * 70)
    print("EXPÉRIMENTATION 2 : COMPLEXITÉ TEMPORELLE EMPIRIQUE")
    print("=" * 70)
    print(f"Tailles testées  : {SIZES}")
    print(f"Itérations/taille: {N_ITERATIONS}")
    print(f"Référence MDL    : w_error={W_ERROR_MDL}")
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement du pool ───────────────────────────────────────────────────
    max_size = max(SIZES)
    print(f"📁 Construction du pool de points (cible ≥ {max_size:,} points)...")
    pool = _load_points_pool(max_size)
    n_available = len(pool)
    print(f"   Pool final : {n_available:,} points disponibles\n")

    sizes_to_run = [s for s in SIZES if s <= n_available]
    if len(sizes_to_run) < len(SIZES):
        skipped = [s for s in SIZES if s > n_available]
        print(f"   ⚠️  Tailles ignorées (pool insuffisant) : {skipped}\n")

    # ── Boucle principale ────────────────────────────────────────────────────
    mdl_compressor = MDLCompressor(w_error=W_ERROR_MDL, verbose=False)
    records: list[dict] = []

    for N in sizes_to_run:
        print(f"⏱️  N = {N:,} points")

        # Sélection aléatoire d'un sous-ensemble de N points consécutifs
        start_idx = random.randint(0, n_available - N)
        window_pts = pool[start_idx : start_idx + N]
        trajectory = Trajectory(points=window_pts)

        # Compression MDL (référence) → C segments
        segs_mdl_ref = mdl_compressor.compress_player_trajectory(trajectory)
        C = max(1, len(segs_mdl_ref))
        print(f"   MDL référence : {C} segments (compression {(1 - C/N)*100:.1f}%)")

        # Calibration epsilon DP hors boucle de chronométrage
        best_eps = _find_dp_epsilon(trajectory, C)
        dp_compressor = DouglasPeuckerCompressor(epsilon=best_eps)

        # Mesure des temps sur N_ITERATIONS
        times: dict[str, list[float]] = {
            "MDL": [],
            "Uniforme": [],
            "Douglas-Peucker": [],
        }

        for _ in range(N_ITERATIONS):
            times["MDL"].append(
                _time_compression(mdl_compressor.compress_player_trajectory, trajectory)
            )
            times["Uniforme"].append(
                _time_compression(_uniform_sampling, window_pts, C)
            )
            times["Douglas-Peucker"].append(
                _time_compression(dp_compressor.compress_player_trajectory, trajectory)
            )

        for algo, t_list in times.items():
            arr = np.array(t_list)
            mean_ms = float(arr.mean())
            std_ms = float(arr.std())
            print(f"   {algo:<20} : {mean_ms:7.3f} ms  (±{std_ms:.3f})")
            records.append({
                "N_points": N,
                "Algorithm": algo,
                "Mean_Time_ms": round(mean_ms, 4),
                "Std_Time_ms": round(std_ms, 4),
                "N_segments_ref": C,
            })

        print()

    # ── Export CSV ────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(records)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 CSV sauvegardé → {OUTPUT_CSV}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ DES TEMPS MOYENS (ms)")
    print("=" * 70)
    pivot = df_results.pivot(index="N_points", columns="Algorithm", values="Mean_Time_ms")
    pivot = pivot[[a for a in ALGO_ORDER if a in pivot.columns]]
    print(pivot.to_string(float_format=lambda x: f"{x:8.3f}"))

    print()
    print("💡 Interprétation attendue :")
    print("   Uniforme        → O(N)      (indexage simple)")
    print("   MDL             → O(N)      (parcours glouton linéaire)")
    print("   Douglas-Peucker → O(N log N) (récursion avec distances vectorisées)")

    # ── Graphique ─────────────────────────────────────────────────────────────
    print()
    print("🎨 Génération du graphique...")
    _generate_plot(df_results)

    print()
    print("=" * 70)
    print(f"✅ Benchmark terminé. Sorties → {OUTPUT_DIR}/")
    print("=" * 70)

    return df_results
