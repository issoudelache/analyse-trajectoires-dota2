#!/usr/bin/env python3
"""
Vue — Visualisation de l'analyse de sensibilité w_error.

Toutes les fonctions de tracé (matplotlib) et d'affichage résumé
sont regroupées ici, séparées de la logique de benchmark (contrôleur).

Fonctions publiques
───────────────────
  plot_pipeline_impact(df, output_dir)
  plot_sweet_spot(df, output_dir)
  plot_comparison_algo(df, output_dir)
  plot_segment_distributions(all_lengths, output_dir)
  print_summary(df)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES VISUELLES
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_K = 12

ALGO_LABELS = {
    "kmeans": "KMeans",
    "kmedoids": "K-Médoïdes",
    "ap": "Affinity Propagation",
}
ALGO_COLORS = {
    "kmeans": "#2196F3",
    "kmedoids": "#FF9800",
    "ap": "#4CAF50",
}

MIN_SEGMENTS_FILTER = 2500


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES DE TRACÉ
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
        ax.axvline(
            x[best], color="red", ls=":", alpha=0.7, label=f"Pic ≈ {x[best]:.1f}"
        )

    ax.set(xlabel="w_error", ylabel=f"{title} {direction}", title=title, xlim=(0, 50))
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Vue d'ensemble (2 × 3) — KMeans
# ═════════════════════════════════════════════════════════════════════════════


def plot_pipeline_impact(df, output_dir):
    """Silhouette / DB / CH + n_seg / longueur / temps  (KMeans uniquement)."""
    sub = df[(df["algorithm"] == "kmeans") & df["silhouette"].notna()]
    sub = sub[sub["n_segments_total"] >= MIN_SEGMENTS_FILTER]
    if sub.empty:
        print("  ⚠️  Pas de données KMeans pour la figure 1")
        return

    agg = (
        sub.groupby("w_error")
        .agg(
            sil_mean=("silhouette", "mean"),
            sil_std=("silhouette", "std"),
            db_mean=("davies_bouldin", "mean"),
            db_std=("davies_bouldin", "std"),
            ch_mean=("calinski_harabasz", "mean"),
            ch_std=("calinski_harabasz", "std"),
            n_seg=("n_segments", "first"),
            mean_len=("mean_length", "first"),
            t_compress=("t_compress", "first"),
        )
        .reset_index()
    )

    x = agg["w_error"].values
    n_seeds = max(sub.groupby("w_error").size().min(), 1)
    ci = 1.96 / np.sqrt(n_seeds)  # facteur IC 95 %

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Impact de w_error sur le Pipeline  (KMeans, K = {DEFAULT_K})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # --- Silhouette ---
    _panel(
        axes[0, 0],
        x,
        agg["sil_mean"],
        agg["sil_std"] * ci,
        "#2196F3",
        "#0D47A1",
        "Silhouette Score",
        "↑",
        peak=True,
    )

    # --- Davies-Bouldin ---
    _panel(
        axes[0, 1],
        x,
        agg["db_mean"],
        agg["db_std"] * ci,
        "#FF9800",
        "#E65100",
        "Davies-Bouldin Index",
        "↓",
    )

    # --- Calinski-Harabasz ---
    _panel(
        axes[0, 2],
        x,
        agg["ch_mean"],
        agg["ch_std"] * ci,
        "#4CAF50",
        "#1B5E20",
        "Calinski-Harabasz Index",
        "↑",
    )

    # --- Nombre de segments ---
    ax = axes[1, 0]
    ax.plot(x, agg["n_seg"], "o-", color="#9C27B0", ms=3, lw=1)
    ax.set(
        xlabel="w_error",
        ylabel="Segments (après filtrage)",
        title="Nombre de segments",
        xlim=(0, 50),
    )
    ax.grid(True, alpha=0.3)

    # --- Longueur moyenne ---
    ax = axes[1, 1]
    ax.plot(x, agg["mean_len"], "o-", color="#F44336", ms=3, lw=1)
    ax.set(
        xlabel="w_error",
        ylabel="Longueur moyenne",
        title="Longueur moyenne des segments",
        xlim=(0, 50),
    )
    ax.grid(True, alpha=0.3)

    # --- Temps de compression ---
    ax = axes[1, 2]
    ax.plot(x, agg["t_compress"], "o-", color="#607D8B", ms=3, lw=1)
    ax.set(
        xlabel="w_error", ylabel="Temps (s)", title="Temps de compression", xlim=(0, 50)
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig1_pipeline_impact.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sweet Spot (Silhouette + Compression + Trade-off)
# ═════════════════════════════════════════════════════════════════════════════


def plot_sweet_spot(df, output_dir):
    """Dual-axis : Silhouette vs Compression Rate + Trade-off Score.

    Score_Compromis = 0.6 × Silhouette_norm + 0.4 × Compression_norm
    Sweet spot = w_error maximisant ce score.
    """
    valid = df[df["silhouette"].notna()]
    valid = valid[valid["n_segments_total"] >= MIN_SEGMENTS_FILTER]
    if valid.empty:
        print("  ⚠️  Pas de données pour la figure 2")
        return

    agg = (
        valid.groupby("w_error")
        .agg(
            sil_mean=("silhouette", "mean"),
            comp_rate=("compression_rate", "first"),
        )
        .reset_index()
    )

    # Limiter à w_error <= 50
    agg = agg[agg["w_error"] <= 50].reset_index(drop=True)
    if agg.empty:
        print("  ⚠️  Pas de données pour la figure 2 après filtrage")
        return

    x = agg["w_error"].values

    sil_s = _smooth(agg["sil_mean"].values)
    comp_pct = agg["comp_rate"].values * 100

    # ── Trade-off Score (Score de compromis) ─────────────────────────────
    sil_norm = _minmax(sil_s)
    comp_norm = _minmax(comp_pct)
    tradeoff = 0.6 * sil_norm + 0.4 * comp_norm

    best_idx = int(np.argmax(tradeoff))
    best_w = x[best_idx]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # ── Silhouette (axe gauche) ──────────────────────────────────────────
    c_sil = "#2196F3"
    ax1.plot(x, agg["sil_mean"].values, "o", color=c_sil, ms=3, alpha=0.35)
    ax1.plot(x, sil_s, "-", color=c_sil, lw=2.5, label="Silhouette (lissé)")
    ax1.set_xlabel("w_error", fontsize=13)
    ax1.set_ylabel("Silhouette Score ↑", fontsize=13, color=c_sil)
    ax1.tick_params(axis="y", labelcolor=c_sil)
    ax1.set_xlim(0, 50)

    # ── Taux de compression (axe droit) ──────────────────────────────────
    ax2 = ax1.twinx()
    c_comp = "#F44336"
    ax2.plot(
        x,
        comp_pct,
        "s-",
        color=c_comp,
        ms=3,
        lw=1.5,
        alpha=0.7,
        label="Taux de compression (%)",
    )
    ax2.set_ylabel("Taux de compression (%) ↑", fontsize=13, color=c_comp)
    ax2.tick_params(axis="y", labelcolor=c_comp)

    # ── Trade-off Score (sur l'axe gauche, redimensionné) ────────────────
    c_trade = "#4CAF50"
    scale = np.nanmax(sil_s) if np.any(np.isfinite(sil_s)) else 1.0
    ax1.plot(
        x,
        tradeoff * scale,
        "--",
        color=c_trade,
        lw=2.5,
        alpha=0.8,
        label="Score compromis (0.6·Sil + 0.4·Comp)",
    )

    # ── Sweet spot : pic du Trade-off Score ──────────────────────────────
    ax1.axvline(best_w, color="red", ls=":", lw=2, label=f"Sweet spot = {best_w:.2f}")
    ax1.annotate(
        f"w_error = {best_w:.2f}",
        xy=(best_w, sil_s[best_idx]),
        xycoords="data",
        xytext=(15, 25),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # ── Légendes combinées ───────────────────────────────────────────────
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=10,
        framealpha=0.95,
        bbox_to_anchor=(0.98, 0.98),
    )

    ax1.set_title(
        "Sweet Spot : Compromis Compression / Qualité  (3 algos combinés)",
        fontsize=15,
        fontweight="bold",
        pad=15,
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
    valid = valid[valid["n_segments_total"] >= MIN_SEGMENTS_FILTER]
    algos = [
        a for a in ("kmeans", "kmedoids", "ap") if a in valid["algorithm"].unique()
    ]

    if not algos:
        print("  ⚠️  Pas assez de données pour la figure 3")
        return

    metrics_info = [
        ("silhouette", "Silhouette Score", "↑"),
        ("davies_bouldin", "Davies-Bouldin Index", "↓"),
        ("calinski_harabasz", "Calinski-Harabasz Index", "↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "Comparaison KMeans / K-Médoïdes / Affinity Propagation",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for ax, (col, title, direction) in zip(axes, metrics_info):
        for algo in algos:
            sub = valid[valid["algorithm"] == algo]
            agg = (
                sub.groupby("w_error")
                .agg(
                    y_mean=(col, "mean"),
                    y_std=(col, "std"),
                )
                .reset_index()
            )

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
                alpha=0.1,
                color=color,
            )

        ax.set(xlabel="w_error", ylabel=f"{title} {direction}", xlim=(0, 50))
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, framealpha=0.95, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig3_comparison_algo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Distribution des longueurs de segments (boxplots)
# ═════════════════════════════════════════════════════════════════════════════


def plot_segment_distributions(all_lengths, output_dir):
    """Boxplots de longueur de segments pour quelques w_error clés.

    Parameters
    ----------
    all_lengths : dict[float, list[float]]
        Clé = w_error, valeur = liste des longueurs de segments.
        Doit être pré-calculé par le contrôleur.
    output_dir : Path
        Répertoire de sortie.
    """
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
    ax.set_title(
        "Distribution des longueurs de segments selon w_error",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Annotations : nombre de segments au-dessus de chaque boxplot
    for i, (w, lengths) in enumerate(all_lengths.items()):
        ax.text(
            i + 1,
            ax.get_ylim()[1] * 0.98,
            f"n={len(lengths):,}",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
        )

    plt.tight_layout()
    path = output_dir / "fig4_segment_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Justification du choix w_error ≈ 12  (KMeans uniquement)
# ═════════════════════════════════════════════════════════════════════════════


def plot_optimal_werror(df, output_dir, chosen_w=12):
    """Graphique de justification du choix w_error pour un rapport d'analyse.

    Deux panneaux :
      (a) Trade-off Score KMeans-only avec zone optimale et annotation.
      (b) Silhouette et Compression normalisées [0,1] pour montrer le compromis.

    Parameters
    ----------
    df : DataFrame  — résultats bruts (raw_results.csv)
    output_dir : Path
    chosen_w : float — valeur de w_error à justifier (par défaut 12)
    """
    # ── Données KMeans filtrées ──────────────────────────────────────────
    km = df[(df["algorithm"] == "kmeans") & df["silhouette"].notna()]
    km = km[km["n_segments_total"] >= MIN_SEGMENTS_FILTER]
    if km.empty:
        print("  ⚠️  Pas de données KMeans pour la figure 5")
        return

    agg = (
        km.groupby("w_error")
        .agg(
            sil_mean=("silhouette", "mean"),
            sil_std=("silhouette", "std"),
            comp_rate=("compression_rate", "first"),
            n_seg=("n_segments_total", "first"),
        )
        .reset_index()
    )
    agg = agg[agg["w_error"] <= 50].reset_index(drop=True)

    x = agg["w_error"].values
    sil_raw = agg["sil_mean"].values
    sil_s = _smooth(sil_raw)
    comp_pct = agg["comp_rate"].values * 100

    # ── Scores normalisés ────────────────────────────────────────────────
    sil_norm = _minmax(sil_s)
    comp_norm = _minmax(comp_pct)
    tradeoff = 0.6 * sil_norm + 0.4 * comp_norm

    best_idx = int(np.argmax(tradeoff))
    best_w = x[best_idx]

    # Trouver l'index le plus proche de chosen_w
    chosen_idx = int(np.argmin(np.abs(x - chosen_w)))
    chosen_w_actual = x[chosen_idx]

    # ── Zone optimale [8, 20] ────────────────────────────────────────────
    zone_lo, zone_hi = 8, 20

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True,
    )
    fig.suptitle(
        f"Justification du choix  w_error = {chosen_w}  (KMeans, K = {DEFAULT_K})",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # ══════════════════════════════════════════════════════════════════════
    # (a) PANNEAU SUPÉRIEUR — Trade-off Score
    # ══════════════════════════════════════════════════════════════════════
    ax = ax_top

    # Zone optimale
    ax.axvspan(
        zone_lo,
        zone_hi,
        alpha=0.10,
        color="#4CAF50",
        label=f"Zone optimale [{zone_lo}–{zone_hi}]",
    )

    # Courbe du score
    ax.plot(
        x,
        tradeoff,
        "-",
        color="#4CAF50",
        lw=3,
        label="Trade-off Score  (0.6·Sil + 0.4·Comp)",
    )

    # Pic global
    ax.plot(
        best_w,
        tradeoff[best_idx],
        "D",
        color="red",
        ms=10,
        zorder=5,
        label=f"Maximum global : w = {best_w:.0f}  (score = {tradeoff[best_idx]:.3f})",
    )

    # Point choisi (w_error = chosen_w)
    ax.plot(
        chosen_w_actual,
        tradeoff[chosen_idx],
        "o",
        color="#FF6F00",
        ms=12,
        zorder=5,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"w_error = {chosen_w}  (score = {tradeoff[chosen_idx]:.3f})",
    )

    # Annotation du point choisi
    delta_pct = (tradeoff[chosen_idx] - tradeoff[best_idx]) / tradeoff[best_idx] * 100
    ax.annotate(
        f"w = {chosen_w}\n"
        f"Score = {tradeoff[chosen_idx]:.3f}\n"
        f"({delta_pct:+.1f}% vs max)",
        xy=(chosen_w_actual, tradeoff[chosen_idx]),
        xytext=(-80, 40),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#FF6F00",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#FF6F00", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="#FF6F00", lw=2),
    )

    # Ligne de seuil 95% du max
    threshold_95 = 0.95 * tradeoff[best_idx]
    ax.axhline(
        threshold_95,
        color="gray",
        ls="--",
        lw=1,
        alpha=0.6,
        label=f"95% du max ({threshold_95:.3f})",
    )

    ax.set_ylabel("Trade-off Score", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a)  Score de compromis Qualité / Compression", fontsize=13)

    # ══════════════════════════════════════════════════════════════════════
    # (b) PANNEAU INFÉRIEUR — Composantes normalisées
    # ══════════════════════════════════════════════════════════════════════
    ax = ax_bot

    ax.axvspan(zone_lo, zone_hi, alpha=0.10, color="#4CAF50")

    ax.plot(x, sil_norm, "-", color="#2196F3", lw=2.5, label="Silhouette normalisée")
    ax.plot(x, comp_norm, "-", color="#F44336", lw=2.5, label="Compression normalisée")

    # Points pour w choisi
    ax.plot(
        chosen_w_actual,
        sil_norm[chosen_idx],
        "o",
        color="#2196F3",
        ms=10,
        zorder=5,
        markeredgecolor="black",
    )
    ax.plot(
        chosen_w_actual,
        comp_norm[chosen_idx],
        "s",
        color="#F44336",
        ms=10,
        zorder=5,
        markeredgecolor="black",
    )

    ax.annotate(
        f"Sil = {sil_s[chosen_idx]:.4f}\n({sil_norm[chosen_idx]:.2f} normalisé)",
        xy=(chosen_w_actual, sil_norm[chosen_idx]),
        xytext=(40, 20),
        textcoords="offset points",
        fontsize=10,
        color="#2196F3",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2196F3", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#2196F3"),
    )
    ax.annotate(
        f"Comp = {comp_pct[chosen_idx]:.1f}%\n({comp_norm[chosen_idx]:.2f} normalisé)",
        xy=(chosen_w_actual, comp_norm[chosen_idx]),
        xytext=(40, -30),
        textcoords="offset points",
        fontsize=10,
        color="#F44336",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F44336", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#F44336"),
    )

    # Interprétation textuelle en bas
    ax.text(
        0.5,
        -0.22,
        f"À w_error = {chosen_w}, la Silhouette reste élevée ({sil_s[chosen_idx]:.4f}) "
        f"avec un taux de compression de {comp_pct[chosen_idx]:.1f}%.  "
        f"Le Trade-off Score ({tradeoff[chosen_idx]:.3f}) est à "
        f"{100 + delta_pct:.1f}% du maximum ({tradeoff[best_idx]:.3f} à w = {best_w:.0f}).",
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        va="top",
        style="italic",
        color="#333",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#ccc"),
    )

    ax.set_xlabel("w_error", fontsize=13)
    ax.set_ylabel("Score normalisé [0, 1]", fontsize=13)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10, loc="center left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "(b)  Composantes : Silhouette vs Compression (normalisées)", fontsize=13
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    path = Path(output_dir) / "fig5_optimal_werror.png"
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
        print(
            f"  {name:<25s}  Sil={row['sil']:.4f}  "
            f"DB={row['db']:.3f}  CH={row['ch']:.0f}  "
            f"K_moyen={row['k_mean']:.1f}"
        )

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
    print(
        f"  SWEET SPOT :  w_error ≈ {best_w:.2f}  (Silhouette moyen = {best_sil:.4f})"
    )
    print(f"{'=' * 64}")
