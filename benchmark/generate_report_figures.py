"""
Génération de figures pour le rapport — Benchmark de clustering sur trajectoires Dota 2

Sources de données (flexibles) :
  - benchmark_fast_k12.csv  → KMeans + KMedoids, N élevé (mode fast)
  - benchmark_ap_k12.csv    → AP seule, N modéré (mode ap)
  - mid_benchmark_results_k12.csv → ancien benchmark mixte k=12 (fallback)
  - mid_benchmark_results.csv     → ancien benchmark mixte k=50
  - heavy_benchmark_results.csv   → benchmark étendu k=50

Le script fusionne automatiquement les CSV disponibles.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent  # racine du projet
DATA_DIR = ROOT / "output" / "benchmark_clustering"
OUT_DIR = ROOT / "output" / "rapport_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Chargement intelligent ──────────────────────────────────────────────────
def _load_if_exists(name):
    p = DATA_DIR / name
    if p.exists():
        df = pd.read_csv(p)
        print(f"  Chargé : {name}  ({len(df)} lignes)")
        return df
    return None


def _build_k12():
    """
    Fusionne les données k=12 disponibles.
    Priorité : benchmark_fast_k12 + benchmark_ap_k12 > mid_benchmark_results_k12.
    """
    parts = []
    df_fast = _load_if_exists("benchmark_fast_k12.csv")
    df_ap = _load_if_exists("benchmark_ap_k12.csv")
    df_old = _load_if_exists("mid_benchmark_results_k12.csv")

    if df_fast is not None:
        parts.append(df_fast)
    if df_ap is not None:
        parts.append(df_ap)
    if not parts and df_old is not None:
        parts.append(df_old)
    elif df_old is not None:
        # Compléter les données manquantes (ex: AP manquante dans fast)
        if df_fast is not None and df_ap is None:
            # On a fast mais pas ap → prendre AP de l'ancien fichier
            ap_old = df_old[df_old["Algorithm"] == "AffinityPropagation"]
            if not ap_old.empty:
                parts.append(ap_old)
                print(
                    f"  Complété AP depuis mid_benchmark_results_k12.csv ({len(ap_old)} lignes)"
                )
        if df_ap is not None and df_fast is None:
            # On a ap mais pas fast → prendre KMeans/KMedoids de l'ancien
            km_old = df_old[df_old["Algorithm"].isin(["KMeans", "KMedoids"])]
            if not km_old.empty:
                parts.append(km_old)
                print(
                    f"  Complété KMeans/KMedoids depuis mid_benchmark_results_k12.csv ({len(km_old)} lignes)"
                )

    if not parts:
        raise FileNotFoundError("Aucun fichier CSV k=12 trouvé dans " + str(DATA_DIR))
    return pd.concat(parts, ignore_index=True)


print("Chargement des données...")
df_k12 = _build_k12()
df_k50 = _load_if_exists("mid_benchmark_results.csv")
df_heavy = _load_if_exists("heavy_benchmark_results.csv")

SENSITIVITY_PATH = ROOT / "output" / "benchmark_sensitivity" / "raw_results.csv"
df_sens = None
if SENSITIVITY_PATH.exists():
    df_sens = pd.read_csv(SENSITIVITY_PATH)
    print(f"  Chargé : raw_results.csv (sensitivity, {len(df_sens)} lignes)")

OPTIMAL_K_PATH = ROOT / "output" / "benchmark_optimal_k" / "optimal_k_results.csv"
df_optk = None
if OPTIMAL_K_PATH.exists():
    df_optk = pd.read_csv(OPTIMAL_K_PATH)
    print(f"  Chargé : optimal_k_results.csv ({len(df_optk)} lignes)")

# ── Style global ─────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ALGO_COLORS = {
    "KMeans": "#2196F3",
    "KMedoids": "#FF9800",
    "AffinityPropagation": "#4CAF50",
}
ALGO_LABELS = {
    "KMeans": "K-Means",
    "KMedoids": "K-Medoids (PAM)",
    "AffinityPropagation": "Propagation d'affinité",
}
ALGO_MARKERS = {
    "KMeans": "o",
    "KMedoids": "s",
    "AffinityPropagation": "^",
}

K_STYLES = {
    50: {"ls": "--", "alpha": 0.55, "lw": 1.4},
    12: {"ls": "-", "alpha": 0.95, "lw": 2.0},
}


def agg(df):
    """Agrège par (N, Algorithm) → médiane + Q1/Q3."""
    g = df.groupby(["N", "Algorithm"])
    med = g.median(numeric_only=True).reset_index()
    q1 = g.quantile(0.25, numeric_only=True).reset_index()
    q3 = g.quantile(0.75, numeric_only=True).reset_index()
    return med, q1, q3


def _plot_algo(ax, df, algo, metric, **kwargs):
    """Trace une courbe + bande IQR pour un algo donné."""
    sub = df[df["Algorithm"] == algo].dropna(subset=[metric])
    if sub.empty:
        return
    med, q1, q3 = agg(sub)
    m = med[med["Algorithm"] == algo].sort_values("N")
    lo = q1[q1["Algorithm"] == algo].sort_values("N")
    hi = q3[q3["Algorithm"] == algo].sort_values("N")
    ns = m["N"].values
    label = kwargs.pop("label", ALGO_LABELS[algo])
    ax.plot(
        ns,
        m[metric].values,
        label=label,
        color=ALGO_COLORS[algo],
        marker=ALGO_MARKERS[algo],
        markersize=4,
        lw=2,
        **kwargs,
    )
    ax.fill_between(
        ns, lo[metric].values, hi[metric].values, color=ALGO_COLORS[algo], alpha=0.12
    )


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 : Scalabilité KMeans + KMedoids (k=12) — temps et qualité
# ══════════════════════════════════════════════════════════════════════════════
def fig_scalability_fast():
    """KMeans et KMedoids poussés le plus loin possible (grands N)."""
    df = df_k12[df_k12["Algorithm"].isin(["KMeans", "KMedoids"])]
    if df.empty:
        print("  ⚠ Pas de données KMeans/KMedoids k=12, fig1 ignorée")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Scalabilité de K-Means et K-Medoids (k = 12)", fontweight="bold", y=1.02
    )

    for algo in ["KMeans", "KMedoids"]:
        _plot_algo(ax1, df, algo, "Time_Seconds")
        _plot_algo(ax2, df, algo, "Silhouette_Score")

    ax1.set_yscale("log")
    ax1.set_xlabel("Taille de l'échantillon (N)")
    ax1.set_ylabel("Temps (secondes, échelle log)")
    ax1.set_title("Temps d'exécution")
    ax1.legend(framealpha=0.8)

    ax2.set_xlabel("Taille de l'échantillon (N)")
    ax2.set_ylabel("Score silhouette (médiane)")
    ax2.set_title("Qualité du clustering")
    ax2.legend(framealpha=0.8)
    ax2.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)

    fig.tight_layout()
    savefig(fig, "fig1_scalabilite_kmeans_kmedoids.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 : Scalabilité AP seule (k=12) — temps, qualité, nb clusters
# ══════════════════════════════════════════════════════════════════════════════
def fig_scalability_ap():
    """Propagation d'affinité analysée séparément."""
    df = df_k12[df_k12["Algorithm"] == "AffinityPropagation"].dropna(
        subset=["Time_Seconds"]
    )
    if df.empty:
        print("  ⚠ Pas de données AP k=12, fig2 ignorée")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        "Propagation d'affinité — Analyse dédiée (k = 12)", fontweight="bold", y=1.02
    )

    med, q1, q3 = agg(df)
    m = med[med["Algorithm"] == "AffinityPropagation"].sort_values("N")
    lo = q1[q1["Algorithm"] == "AffinityPropagation"].sort_values("N")
    hi = q3[q3["Algorithm"] == "AffinityPropagation"].sort_values("N")
    ns = m["N"].values
    c = ALGO_COLORS["AffinityPropagation"]

    # Temps
    ax1.plot(ns, m["Time_Seconds"].values, color=c, marker="^", lw=2, markersize=5)
    ax1.fill_between(
        ns, lo["Time_Seconds"].values, hi["Time_Seconds"].values, color=c, alpha=0.15
    )
    ax1.set_xlabel("N")
    ax1.set_ylabel("Temps (secondes)")
    ax1.set_title("Temps d'exécution")

    # Silhouette
    ax2.plot(ns, m["Silhouette_Score"].values, color=c, marker="^", lw=2, markersize=5)
    ax2.fill_between(
        ns,
        lo["Silhouette_Score"].values,
        hi["Silhouette_Score"].values,
        color=c,
        alpha=0.15,
    )
    ax2.set_xlabel("N")
    ax2.set_ylabel("Score silhouette")
    ax2.set_title("Qualité")
    ax2.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)

    # Nombre de clusters
    ax3.plot(ns, m["N_Clusters_Found"].values, color=c, marker="^", lw=2, markersize=5)
    ax3.fill_between(
        ns,
        lo["N_Clusters_Found"].values,
        hi["N_Clusters_Found"].values,
        color=c,
        alpha=0.15,
    )
    ax3.set_xlabel("N")
    ax3.set_ylabel("Nombre de clusters")
    ax3.set_title("Clusters découverts")

    fig.tight_layout()
    savefig(fig, "fig2_scalabilite_ap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 : Vue d'ensemble — 3 algorithmes comparés (k=12)
# ══════════════════════════════════════════════════════════════════════════════
def fig_overview_k12():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Comparaison des 3 algorithmes de clustering (k = 12)",
        fontweight="bold",
        y=1.02,
    )

    for algo in ["KMeans", "KMedoids", "AffinityPropagation"]:
        _plot_algo(ax1, df_k12, algo, "Silhouette_Score")
        _plot_algo(ax2, df_k12, algo, "Time_Seconds")

    ax1.set_xlabel("Taille de l'échantillon (N)")
    ax1.set_ylabel("Score silhouette (médiane)")
    ax1.set_title("Qualité")
    ax1.legend(framealpha=0.8)
    ax1.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)

    ax2.set_yscale("log")
    ax2.set_xlabel("Taille de l'échantillon (N)")
    ax2.set_ylabel("Temps (secondes, échelle log)")
    ax2.set_title("Scalabilité")
    ax2.legend(framealpha=0.8)

    fig.tight_layout()
    savefig(fig, "fig3_overview_k12.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 : Temps de calcul de la matrice de distance
# ══════════════════════════════════════════════════════════════════════════════
def fig_matrix_time():
    # Prendre la matrice time d'un algo quelconque (identique pour tous)
    if "Matrix_Time_Seconds" not in df_k12.columns:
        print("  ⚠ Colonne Matrix_Time_Seconds absente, fig4 ignorée")
        return
    sub = df_k12.dropna(subset=["Matrix_Time_Seconds"])
    # Dédupliquer : même matrice pour tous les algos d'une même itération
    sub = sub.drop_duplicates(subset=["N", "Iteration"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Temps de calcul de la matrice de distance TRACLUS", fontweight="bold")

    g = sub.groupby("N")["Matrix_Time_Seconds"]
    ns = sorted(sub["N"].unique())
    med = [g.get_group(n).median() for n in ns]
    q1 = [g.get_group(n).quantile(0.25) for n in ns]
    q3 = [g.get_group(n).quantile(0.75) for n in ns]

    ax.plot(ns, med, color="#E91E63", marker="D", markersize=4, lw=2, label="Médiane")
    ax.fill_between(ns, q1, q3, color="#E91E63", alpha=0.15, label="IQR (Q1–Q3)")

    # Régression quadratique
    ns_arr, med_arr = np.array(ns, dtype=float), np.array(med, dtype=float)
    coeffs = np.polyfit(ns_arr, med_arr, 2)
    ns_smooth = np.linspace(ns_arr.min(), ns_arr.max(), 200)
    ax.plot(
        ns_smooth,
        np.polyval(coeffs, ns_smooth),
        color="#E91E63",
        ls=":",
        lw=1.2,
        alpha=0.6,
        label="Régression O(n²)",
    )

    ax.set_xlabel("Taille de l'échantillon (N)")
    ax.set_ylabel("Temps (secondes)")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    savefig(fig, "fig4_matrice_distance.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 : Barplot comparatif à un N donné (k=12 seulement)
# ══════════════════════════════════════════════════════════════════════════════
def fig_barplot_at_n():
    N_TARGET = 5000
    algos_order = ["KMeans", "KMedoids", "AffinityPropagation"]

    sub = df_k12[df_k12["N"] == N_TARGET]
    if sub.empty:
        closest = df_k12["N"].unique()
        N_TARGET = int(closest[np.argmin(np.abs(closest - N_TARGET))])
        sub = df_k12[df_k12["N"] == N_TARGET]

    available = [a for a in algos_order if not sub[sub["Algorithm"] == a].empty]
    if not available:
        print("  ⚠ Pas de données à N~5000, fig5 ignorée")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Comparaison des algorithmes à N = {N_TARGET} (k = 12)",
        fontweight="bold",
        y=1.02,
    )

    labels = [ALGO_LABELS[a] for a in available]
    x = np.arange(len(available))
    colors = [ALGO_COLORS[a] for a in available]

    sil_med, time_med, sil_err, time_err = [], [], [], []
    for algo in available:
        s = sub[sub["Algorithm"] == algo]
        sil_med.append(s["Silhouette_Score"].median())
        sil_err.append(s["Silhouette_Score"].std())
        time_med.append(s["Time_Seconds"].median())
        time_err.append(s["Time_Seconds"].std())

    ax1.bar(
        x,
        sil_med,
        0.55,
        yerr=sil_err,
        capsize=4,
        color=colors,
        alpha=0.82,
        edgecolor="white",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("Score silhouette")
    ax1.set_title("Qualité")

    ax2.bar(
        x,
        time_med,
        0.55,
        yerr=time_err,
        capsize=4,
        color=colors,
        alpha=0.82,
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("Temps (secondes)")
    ax2.set_yscale("log")
    ax2.set_title("Temps d'exécution")

    fig.tight_layout()
    savefig(fig, "fig5_barplot_N5000.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 : Boxplots score silhouette (k=12)
# ══════════════════════════════════════════════════════════════════════════════
def fig_boxplots():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Distribution des scores silhouette par algorithme (k = 12)", fontweight="bold"
    )

    algos_order = ["KMeans", "KMedoids", "AffinityPropagation"]
    available = [a for a in algos_order if not df_k12[df_k12["Algorithm"] == a].empty]
    data = [
        df_k12[df_k12["Algorithm"] == a]["Silhouette_Score"].dropna().values
        for a in available
    ]
    colors = [ALGO_COLORS[a] for a in available]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6
        ),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels([ALGO_LABELS[a] for a in available])
    ax.set_ylabel("Score silhouette")
    fig.tight_layout()
    savefig(fig, "fig6_boxplots_silhouette_k12.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 : Comparaison k=50 vs k=12 (uniquement si données k=50 disponibles)
# ══════════════════════════════════════════════════════════════════════════════
def fig_k_comparison():
    if df_k50 is None:
        print("  ⚠ Pas de données k=50, fig7 (comparaison k) ignorée")
        return

    # On compare sur les N communs
    common_ns = sorted(set(df_k50["N"].unique()) & set(df_k12["N"].unique()))
    if len(common_ns) < 3:
        print("  ⚠ Trop peu de N communs entre k=50 et k=12, fig7 ignorée")
        return

    d50 = df_k50[df_k50["N"].isin(common_ns)]
    d12 = df_k12[df_k12["N"].isin(common_ns)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Impact du choix de k sur la qualité (score silhouette)",
        fontweight="bold",
        y=1.02,
    )

    for ax, algo in zip(axes, ["KMeans", "KMedoids", "AffinityPropagation"]):
        for k_val, df in [(50, d50), (12, d12)]:
            sub = df[df["Algorithm"] == algo].dropna(subset=["Silhouette_Score"])
            if sub.empty:
                continue
            med, q1, q3 = agg(sub)
            m = med[med["Algorithm"] == algo].sort_values("N")
            lo = q1[q1["Algorithm"] == algo].sort_values("N")
            hi = q3[q3["Algorithm"] == algo].sort_values("N")
            ns = m["N"].values
            style = K_STYLES[k_val]
            ax.plot(
                ns,
                m["Silhouette_Score"].values,
                label=f"k = {k_val}",
                color=ALGO_COLORS[algo],
                marker=ALGO_MARKERS[algo],
                markersize=4,
                **style,
            )
            ax.fill_between(
                ns,
                lo["Silhouette_Score"].values,
                hi["Silhouette_Score"].values,
                color=ALGO_COLORS[algo],
                alpha=0.10,
            )
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Taille de l'échantillon (N)")
        ax.legend(framealpha=0.8)
    axes[0].set_ylabel("Score silhouette (médiane)")
    fig.tight_layout()
    savefig(fig, "fig7_comparaison_k50_vs_k12.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 : Ratio Speedup KMedoids / KMeans vs N
# ══════════════════════════════════════════════════════════════════════════════
def fig_speedup():
    """Montre combien de fois KMedoids est plus rapide que KMeans."""
    km = df_k12[df_k12["Algorithm"] == "KMeans"]
    kmed = df_k12[df_k12["Algorithm"] == "KMedoids"]
    if km.empty or kmed.empty:
        print("  ⚠ Données KMeans ou KMedoids manquantes, fig8 ignorée")
        return

    med_km = km.groupby("N")["Time_Seconds"].median()
    med_kmed = kmed.groupby("N")["Time_Seconds"].median()
    # Filtrer N >= 1000 pour éviter le bruit à petit N
    common = sorted(n for n in set(med_km.index) & set(med_kmed.index) if n >= 1000)
    if len(common) < 3:
        print("  ⚠ Trop peu de N communs, fig8 ignorée")
        return

    ns = np.array(common)
    ratio = np.array([med_km[n] / med_kmed[n] for n in common])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        "Accélération de K-Medoids par rapport à K-Means (k = 12, N ≥ 1 000)",
        fontweight="bold",
    )

    ax.plot(ns, ratio, color="#9C27B0", marker="o", lw=2.2, markersize=6, zorder=3)
    ax.fill_between(ns, 1, ratio, where=(ratio >= 1), color="#9C27B0", alpha=0.12)
    ax.axhline(1, color="black", ls=":", lw=0.8, alpha=0.5)

    # Annotations aux extrêmes
    i_max = int(np.argmax(ratio))
    ax.annotate(
        f"×{ratio[i_max]:.1f}",
        xy=(ns[i_max], ratio[i_max]),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#9C27B0",
        ha="center",
    )

    ax.set_xlabel("Taille de l'échantillon (N)")
    ax.set_ylabel("Ratio (temps K-Means / temps K-Medoids)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, "fig8_speedup_kmedoids_vs_kmeans.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 : Heatmap silhouette (algorithme × N)
# ══════════════════════════════════════════════════════════════════════════════
def fig_heatmap_silhouette():
    """Heatmap colorée : score silhouette médian par (algo, N)."""
    algos_order = ["KMeans", "KMedoids", "AffinityPropagation"]
    available = [a for a in algos_order if not df_k12[df_k12["Algorithm"] == a].empty]
    if len(available) < 2:
        print("  ⚠ Pas assez d'algos, fig9 ignorée")
        return

    med = df_k12.groupby(["N", "Algorithm"])["Silhouette_Score"].median().reset_index()
    pivot = med.pivot(index="Algorithm", columns="N", values="Silhouette_Score")
    pivot = pivot.reindex(available)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6), 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.25)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [str(n) for n in pivot.columns], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([ALGO_LABELS[a] for a in available])

    # Valeurs dans les cellules
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val < 0.05 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
            )

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cb.set_label("Score silhouette (médiane)")
    ax.set_title(
        "Heatmap des scores silhouette par algorithme et taille N (k = 12)",
        fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, "fig9_heatmap_silhouette.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 : Stacked bar — décomposition matrice + clustering
# ══════════════════════════════════════════════════════════════════════════════
def fig_stacked_time():
    """Temps matrice vs temps clustering — barres groupées côte à côte."""
    if "Matrix_Time_Seconds" not in df_k12.columns:
        print("  ⚠ Colonne Matrix_Time_Seconds absente, fig10 ignorée")
        return

    algos_order = ["KMeans", "KMedoids", "AffinityPropagation"]
    all_ns = sorted(df_k12["N"].unique())
    if len(all_ns) > 8:
        indices = np.linspace(0, len(all_ns) - 1, 8, dtype=int)
        selected_ns = [all_ns[i] for i in indices]
    else:
        selected_ns = all_ns

    available = [a for a in algos_order if not df_k12[df_k12["Algorithm"] == a].empty]
    # 2 barres par algo (matrice, clustering) → sous-groupes
    n_bars = len(available) * 2
    bar_width = 0.8 / n_bars
    x = np.arange(len(selected_ns))
    hatches = ["", "///"]  # plein = matrice, hachuré = clustering

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_title(
        "Décomposition du temps : matrice de distance vs clustering (k = 12)",
        fontweight="bold",
    )

    for i, algo in enumerate(available):
        sub = df_k12[(df_k12["Algorithm"] == algo) & (df_k12["N"].isin(selected_ns))]
        med = sub.groupby("N")[["Time_Seconds", "Matrix_Time_Seconds"]].median()
        mat_times = np.array(
            [
                med.loc[n, "Matrix_Time_Seconds"] if n in med.index else np.nan
                for n in selected_ns
            ]
        )
        clust_times = np.array(
            [
                med.loc[n, "Time_Seconds"] if n in med.index else np.nan
                for n in selected_ns
            ]
        )
        c = ALGO_COLORS[algo]

        off_mat = (2 * i - n_bars / 2 + 0.5) * bar_width
        off_clu = (2 * i + 1 - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            x + off_mat,
            mat_times,
            bar_width,
            color=c,
            alpha=0.45,
            edgecolor=c,
            linewidth=0.8,
            label=f"{ALGO_LABELS[algo]} — matrice",
        )
        ax.bar(
            x + off_clu,
            clust_times,
            bar_width,
            color=c,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            hatch="///",
            label=f"{ALGO_LABELS[algo]} — clustering",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in selected_ns])
    ax.set_xlabel("Taille de l'échantillon (N)")
    ax.set_ylabel("Temps (secondes, échelle log)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=3, framealpha=0.8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig10_decomposition_temps.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 : Coefficient de variation du silhouette — stabilité
# ══════════════════════════════════════════════════════════════════════════════
def fig_cv_silhouette():
    """CV = std/mean du score silhouette à chaque N, par algorithme."""
    algos_order = ["KMeans", "KMedoids", "AffinityPropagation"]
    available = [a for a in algos_order if not df_k12[df_k12["Algorithm"] == a].empty]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        "Stabilité du clustering : coefficient de variation du silhouette (k = 12)",
        fontweight="bold",
    )

    for algo in available:
        sub = df_k12[df_k12["Algorithm"] == algo].dropna(subset=["Silhouette_Score"])
        g = sub.groupby("N")["Silhouette_Score"]
        ns = sorted(sub["N"].unique())
        cv = []
        valid_ns = []
        for n in ns:
            vals = g.get_group(n)
            if len(vals) >= 2 and vals.mean() != 0:
                cv.append(vals.std() / abs(vals.mean()))
                valid_ns.append(n)
        if not valid_ns:
            continue
        ax.plot(
            valid_ns,
            cv,
            color=ALGO_COLORS[algo],
            marker=ALGO_MARKERS[algo],
            markersize=5,
            lw=1.8,
            label=ALGO_LABELS[algo],
        )

    ax.set_xlabel("Taille de l'échantillon (N)")
    ax.set_ylabel("CV (écart-type / moyenne)")
    ax.legend(framealpha=0.8)
    ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5, label="seuil 10 %")
    fig.tight_layout()
    savefig(fig, "fig11_stabilite_cv_silhouette.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12 : Davies-Bouldin & Calinski-Harabasz vs w_error (sensitivity)
# ══════════════════════════════════════════════════════════════════════════════
def fig_db_ch_sensitivity():
    """Métriques internes de qualité en fonction de w_error (depuis sensitivity)."""
    if df_sens is None:
        print("  ⚠ Pas de données sensitivity, fig12 ignorée")
        return

    ALGO_MAP = {"kmeans": "KMeans", "kmedoids": "KMedoids", "ap": "AffinityPropagation"}
    metrics = [
        ("silhouette", "Score silhouette (↑)", False),
        ("davies_bouldin", "Davies-Bouldin (↓)", True),
        ("calinski_harabasz", "Calinski-Harabasz (↑)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Métriques de qualité du clustering en fonction de w_error",
        fontweight="bold",
        y=1.02,
    )

    for ax, (col, ylabel, invert) in zip(axes, metrics):
        for algo_key, algo_name in ALGO_MAP.items():
            sub = df_sens[df_sens["algorithm"] == algo_key].dropna(subset=[col])
            if sub.empty:
                continue
            med = sub.groupby("w_error")[col].median().sort_index()
            q1 = sub.groupby("w_error")[col].quantile(0.25).sort_index()
            q3 = sub.groupby("w_error")[col].quantile(0.75).sort_index()
            ax.plot(
                med.index,
                med.values,
                color=ALGO_COLORS[algo_name],
                lw=1.8,
                label=ALGO_LABELS[algo_name],
            )
            ax.fill_between(
                med.index, q1.values, q3.values, color=ALGO_COLORS[algo_name], alpha=0.1
            )
        ax.set_xlabel("w_error")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend(fontsize=8, framealpha=0.8)
        if invert:
            ax.invert_yaxis()

    fig.tight_layout()
    savefig(fig, "fig12_metriques_qualite_vs_werror.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 13 : Taux de compression vs silhouette — trade-off
# ══════════════════════════════════════════════════════════════════════════════
def fig_compression_vs_silhouette():
    """Scatter : compression_rate (%) vs silhouette, coloré par algorithme."""
    if df_sens is None:
        print("  ⚠ Pas de données sensitivity, fig13 ignorée")
        return

    ALGO_MAP = {"kmeans": "KMeans", "kmedoids": "KMedoids", "ap": "AffinityPropagation"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        "Trade-off : taux de compression vs qualité du clustering", fontweight="bold"
    )

    for algo_key, algo_name in ALGO_MAP.items():
        sub = df_sens[df_sens["algorithm"] == algo_key].dropna(
            subset=["compression_rate", "silhouette"]
        )
        if sub.empty:
            continue
        med = sub.groupby("w_error")[["compression_rate", "silhouette"]].median()
        cr_pct = med["compression_rate"] * 100
        ax.scatter(
            cr_pct,
            med["silhouette"],
            color=ALGO_COLORS[algo_name],
            marker=ALGO_MARKERS[algo_name],
            s=40,
            alpha=0.75,
            label=ALGO_LABELS[algo_name],
            edgecolors="white",
            linewidths=0.5,
        )
        ax.plot(
            cr_pct.values,
            med["silhouette"].values,
            color=ALGO_COLORS[algo_name],
            lw=1.0,
            alpha=0.4,
        )

    # Annoter quelques w_error clés (uniquement KMeans pour lisibilité)
    ref_werrors = [1, 5, 12, 30, 60]
    sub_km = df_sens[df_sens["algorithm"] == "kmeans"]
    prev_y = None
    for we in ref_werrors:
        row = sub_km[sub_km["w_error"] == we]
        if row.empty:
            closest = sub_km.iloc[(sub_km["w_error"] - we).abs().argsort()[:1]]
            if closest.empty:
                continue
            row = closest
        cr = row["compression_rate"].median() * 100
        sil = row["silhouette"].median()
        # Alterner le décalage vertical pour éviter les superpositions
        dy = 12 if (prev_y is None or abs(sil - prev_y) > 0.03) else -14
        ax.annotate(
            f"w={we}",
            xy=(cr, sil),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=8,
            color="#555",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
        )
        prev_y = sil

    ax.set_xlabel("Taux de compression (%)")
    ax.set_ylabel("Score silhouette (médiane)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f %%"))
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    savefig(fig, "fig13_compression_vs_silhouette.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 14 : Silhouette vs k — recherche du k optimal
# ══════════════════════════════════════════════════════════════════════════════
def fig_silhouette_vs_k():
    if df_optk is None:
        print("  ⏩ fig14 ignorée (pas de données optimal_k)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, color, marker in [
        ("KMeans", ALGO_COLORS["KMeans"], "o"),
        ("KMedoids", ALGO_COLORS["KMedoids"], "s"),
    ]:
        sub = df_optk[df_optk["algorithm"] == algo]
        g = sub.groupby("k")["silhouette"]
        med = g.median()
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        ax.plot(
            med.index,
            med.values,
            color=color,
            marker=marker,
            markersize=4,
            lw=2,
            label=ALGO_LABELS[algo],
        )
        ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.12)
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Score silhouette")
    ax.set_title("Silhouette en fonction de k")
    ax.axvline(12, color="grey", ls="--", lw=1, alpha=0.6, label="k = 12")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    savefig(fig, "fig14_silhouette_vs_k.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 15 : Courbe du coude (inertie) — recherche du k optimal
# ══════════════════════════════════════════════════════════════════════════════
def fig_elbow():
    if df_optk is None:
        print("  ⏩ fig15 ignorée (pas de données optimal_k)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, algo, title in [
        (ax1, "KMeans", "K-Means"),
        (ax2, "KMedoids", "K-Medoids (PAM)"),
    ]:
        sub = df_optk[df_optk["algorithm"] == algo]
        g = sub.groupby("k")["inertia"]
        med = g.median()
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        color = ALGO_COLORS[algo]
        ax.plot(med.index, med.values, color=color, marker="o", markersize=3, lw=2)
        ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.12)
        ax.axvline(12, color="grey", ls="--", lw=1, alpha=0.6, label="k = 12")
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertie")
        ax.set_title(f"Méthode du coude — {title}")
        ax.legend(framealpha=0.8)
    fig.tight_layout()
    savefig(fig, "fig15_elbow.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 16 : Panneau combiné 4 métriques vs k
# ══════════════════════════════════════════════════════════════════════════════
def fig_combined_optimal_k():
    if df_optk is None:
        print("  ⏩ fig16 ignorée (pas de données optimal_k)")
        return
    metrics = [
        ("silhouette", "Score silhouette", False),
        ("davies_bouldin", "Indice Davies-Bouldin", True),
        ("calinski_harabasz", "Indice Calinski-Harabasz", False),
        ("inertia", "Inertie", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, ylabel, invert) in zip(axes.flat, metrics):
        for algo in ["KMeans", "KMedoids"]:
            sub = df_optk[df_optk["algorithm"] == algo]
            g = sub.groupby("k")[col]
            med = g.median()
            q1 = g.quantile(0.25)
            q3 = g.quantile(0.75)
            color = ALGO_COLORS[algo]
            ax.plot(
                med.index,
                med.values,
                color=color,
                marker=ALGO_MARKERS[algo],
                markersize=3,
                lw=1.8,
                label=ALGO_LABELS[algo],
            )
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.10)
        ax.axvline(12, color="grey", ls="--", lw=1, alpha=0.6, label="k = 12")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if invert:
            ax.invert_yaxis()
        ax.legend(fontsize=8, framealpha=0.8)
    fig.suptitle("Recherche du nombre optimal de clusters", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "fig16_combined_optimal_k.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGénération des figures du rapport...\n")
    fig_scalability_fast()  # Fig 1 : KMeans + KMedoids seuls, grands N
    fig_scalability_ap()  # Fig 2 : AP seule, analyse dédiée
    fig_overview_k12()  # Fig 3 : Vue d'ensemble 3 algos
    fig_matrix_time()  # Fig 4 : Matrice de distance O(n²)
    fig_barplot_at_n()  # Fig 5 : Barplot à N fixe
    fig_boxplots()  # Fig 6 : Boxplots silhouette
    fig_k_comparison()  # Fig 7 : k=50 vs k=12 (si données dispo)
    fig_speedup()  # Fig 8  : Speedup KMedoids / KMeans
    fig_heatmap_silhouette()  # Fig 9  : Heatmap silhouette (algo × N)
    fig_stacked_time()  # Fig 10 : Décomposition matrice + clustering
    fig_cv_silhouette()  # Fig 11 : Stabilité (CV silhouette)
    fig_db_ch_sensitivity()  # Fig 12 : DB / CH / Sil vs w_error
    fig_compression_vs_silhouette()  # Fig 13 : Compression vs silhouette trade-off
    fig_silhouette_vs_k()  # Fig 14 : Silhouette vs k
    fig_elbow()  # Fig 15 : Courbe du coude (inertie)
    fig_combined_optimal_k()  # Fig 16 : Panneau combiné 4 métriques vs k
    print(f"\n✅ Figures enregistrées dans {OUT_DIR.relative_to(ROOT)}/")
    print(
        "   Figures k=12 (fig1–fig11), sensitivity (fig12–fig13), optimal k (fig14–fig16)."
    )
