#!/usr/bin/env python3
"""
Génération de toutes les figures du benchmark (15 figures).

Lit les CSV produits par exp0, exp1, exp2, exp3 et génère les figures :
  Fig 0.1 : Nb segments + Ratio compression vs w_error (double axe Y, log-x)
  Fig 0.2 : Longueur moyenne des segments vs w_error
  Fig 0.3 : Silhouette vs w_error (bandes IC 95%, sweet spot annoté)
  Fig 0.4 : Panel 1×3 — Silhouette / Davies-Bouldin / Calinski-Harabasz vs w_error

  Fig 1.1 : Méthode du coude — Inertie vs k (bandes IC 95%)
  Fig 1.2 : Silhouette vs k (bandes IC 95%, k* annoté)
  Fig 1.3 : Panel 2×2 — Silhouette / Inertie / Davies-Bouldin / Calinski-Harabasz vs k

  Fig 2.1 : Nb motifs (total / >=2 / >=3) vs k
  Fig 2.2 : Temps PrefixSpan vs k (log-y)
  Fig 2.3 : Nb arêtes Markov vs k
  Fig 2.4 : Entropie de Shannon des supports vs k

  Fig S.1 : Silhouette (Exp 1) + Nb arêtes Markov (Exp 2) vs k — synthèse

  (Fig 3.x sont générées directement par exp3_final_pipeline.py)

Usage :
  python benchmark/plot_all.py
  python benchmark/plot_all.py --exp 0 1 2   # seulement ces exp
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_BASE = BASE_DIR / "output"

# ═════════════════════════════════════════════════════════════════════════════
# STYLE
# ═════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Désactiver la notation scientifique sur tous les axes
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

COLOR_PRIMARY = "#2196F3"
COLOR_SECONDARY = "#FF5722"
COLOR_TERTIARY = "#4CAF50"
COLOR_ACCENT = "#FFC107"
BAND_ALPHA = 0.2


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def ci95(values):
    """Retourne (mean, lower, upper) pour IC 95%."""
    n = len(values)
    m = np.mean(values)
    if n < 2:
        return m, m, m
    se = stats.sem(values)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h


def aggregate_by(df, group_col, value_col):
    """Agrège par group_col, retourne (x, mean, lo, hi)."""
    groups = df.groupby(group_col)[value_col].apply(list)
    xs = sorted(groups.index)
    means, los, his = [], [], []
    for x in xs:
        vals = [v for v in groups[x] if not np.isnan(v)]
        if not vals:
            means.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
        else:
            m, lo, hi = ci95(vals)
            means.append(m)
            los.append(lo)
            his.append(hi)
    return np.array(xs), np.array(means), np.array(los), np.array(his)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _nice_number(x, _pos=None):
    """Formate un nombre pour un axe log : supprime les zéros inutiles."""
    if x == 0:
        return "0"
    if x == int(x) and abs(x) >= 1:
        return f"{int(x)}"
    return f"{x:g}"

def plain_log_axis(ax, axis="x"):
    """Force les ticks d'un axe log à afficher des nombres sans puissances."""
    fmt = FuncFormatter(_nice_number)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(NullFormatter())


# ═════════════════════════════════════════════════════════════════════════════
# EXP 0 — FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def plot_exp0(output_dir):
    csv_path = OUTPUT_BASE / "benchmark_exp0" / "exp0_results.csv"
    if not csv_path.exists():
        print(f"  SKIP Exp 0 : {csv_path} introuvable")
        return

    df = pd.read_csv(csv_path)
    # Convertir colonnes numériques
    for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fig 0.1 : Nb segments + Ratio compression vs w_error ──────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Agrégation par w_error (moyenne sur seeds)
    grp = df.groupby("w_error").agg(
        nb_seg_mean=("nb_segments_total", "mean"),
        ratio_mean=("ratio_compression", "mean"),
    ).reset_index()

    ax1.plot(grp["w_error"], grp["nb_seg_mean"], "o-", color=COLOR_PRIMARY,
             label="Nb segments", markersize=4)
    ax2.plot(grp["w_error"], grp["ratio_mean"], "s-", color=COLOR_SECONDARY,
             label="Ratio compression", markersize=4)

    ax1.set_xscale("log")
    plain_log_axis(ax1)
    ax1.set_xlabel("w_error")
    ax1.set_ylabel("Nb segments", color=COLOR_PRIMARY)
    ax2.set_ylabel("Ratio compression (original / segments)", color=COLOR_SECONDARY)
    ax1.set_title("Fig 0.1 — Nb segments et ratio de compression vs w_error")
    ax1.ticklabel_format(axis='y', style='plain')
    ax2.ticklabel_format(axis='y', style='plain')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    savefig(fig, output_dir / "fig0_1_segments_compression.png")

    # ── Fig 0.2 : Longueur moyenne vs w_error ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    grp2 = df.groupby("w_error").agg(
        len_mean=("longueur_moyenne", "mean"),
        len_std=("longueur_std", "mean"),
    ).reset_index()

    ax.plot(grp2["w_error"], grp2["len_mean"], "o-", color=COLOR_PRIMARY, markersize=4)
    ax.fill_between(grp2["w_error"],
                    grp2["len_mean"] - grp2["len_std"],
                    grp2["len_mean"] + grp2["len_std"],
                    alpha=BAND_ALPHA, color=COLOR_PRIMARY)
    ax.set_xscale("log")
    plain_log_axis(ax)
    ax.set_xlabel("w_error")
    ax.set_ylabel("Longueur moyenne des segments")
    ax.set_title("Fig 0.2 — Longueur moyenne des segments vs w_error")
    savefig(fig, output_dir / "fig0_2_longueur_moyenne.png")

    # ── Fig 0.3 : Silhouette vs w_error (IC 95%, sweet spot) ─────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xs, means, los, his = aggregate_by(df, "w_error", "silhouette")
    ax.plot(xs, means, "o-", color=COLOR_PRIMARY, markersize=4, label="Silhouette")
    ax.fill_between(xs, los, his, alpha=BAND_ALPHA, color=COLOR_PRIMARY)

    # Sweet spot
    valid = ~np.isnan(means)
    if valid.any():
        best_idx = np.nanargmax(means)
        ax.axvline(xs[best_idx], color=COLOR_ACCENT, linestyle="--", alpha=0.7)
        ax.annotate(f"w*={xs[best_idx]:.1f}\nsil={means[best_idx]:.4f}",
                    xy=(xs[best_idx], means[best_idx]),
                    xytext=(15, 15), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_ACCENT, alpha=0.3),
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xscale("log")
    plain_log_axis(ax)
    ax.set_xlabel("w_error")
    ax.set_ylabel("Silhouette")
    ax.set_title("Fig 0.3 — Silhouette vs w_error (IC 95%)")
    ax.legend()
    savefig(fig, output_dir / "fig0_3_silhouette_werror.png")

    # ── Fig 0.4 : Panel 1×3 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title, color in zip(
        axes,
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        ["Silhouette (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)"],
        [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY],
    ):
        xs, means, los, his = aggregate_by(df, "w_error", metric)
        ax.plot(xs, means, "o-", color=color, markersize=3)
        ax.fill_between(xs, los, his, alpha=BAND_ALPHA, color=color)
        ax.set_xscale("log")
        plain_log_axis(ax)
        ax.set_xlabel("w_error")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.ticklabel_format(axis='y', style='plain')

    fig.suptitle("Fig 0.4 — Métriques de clustering vs w_error", fontsize=16, y=1.02)
    savefig(fig, output_dir / "fig0_4_panel_metrics_werror.png")


# ═════════════════════════════════════════════════════════════════════════════
# EXP 1 — FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def plot_exp1(output_dir):
    csv_path = OUTPUT_BASE / "benchmark_exp1" / "exp1_results.csv"
    if not csv_path.exists():
        print(f"  SKIP Exp 1 : {csv_path} introuvable")
        return

    df = pd.read_csv(csv_path)
    for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fig 1.1 : Méthode du coude — Inertie vs k ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xs, means, los, his = aggregate_by(df, "k", "inertia")
    ax.plot(xs, means, "o-", color=COLOR_PRIMARY, markersize=4)
    ax.fill_between(xs, los, his, alpha=BAND_ALPHA, color=COLOR_PRIMARY)
    ax.set_xlabel("k (nombre de clusters)")
    ax.set_ylabel("Inertie")
    ax.set_title("Fig 1.1 — Méthode du coude : Inertie vs k (IC 95%)")
    ax.ticklabel_format(axis='y', style='plain')
    savefig(fig, output_dir / "fig1_1_elbow_inertie.png")

    # ── Fig 1.2 : Silhouette vs k ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xs, means, los, his = aggregate_by(df, "k", "silhouette")
    ax.plot(xs, means, "o-", color=COLOR_PRIMARY, markersize=4, label="Silhouette")
    ax.fill_between(xs, los, his, alpha=BAND_ALPHA, color=COLOR_PRIMARY)

    # k* annoté
    valid = ~np.isnan(means)
    if valid.any():
        best_idx = np.nanargmax(means)
        ax.axvline(xs[best_idx], color=COLOR_ACCENT, linestyle="--", alpha=0.7)
        ax.annotate(f"k*={int(xs[best_idx])}\nsil={means[best_idx]:.4f}",
                    xy=(xs[best_idx], means[best_idx]),
                    xytext=(15, 15), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_ACCENT, alpha=0.3),
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("Fig 1.2 — Silhouette vs k (IC 95%)")
    ax.legend()
    savefig(fig, output_dir / "fig1_2_silhouette_k.png")

    # ── Fig 1.3 : Panel 2×2 ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_data = [
        ("silhouette", "Silhouette (↑)", COLOR_PRIMARY),
        ("inertia", "Inertie (↓)", COLOR_SECONDARY),
        ("davies_bouldin", "Davies-Bouldin (↓)", COLOR_TERTIARY),
        ("calinski_harabasz", "Calinski-Harabasz (↑)", COLOR_ACCENT),
    ]
    for ax, (metric, title, color) in zip(axes.flat, panel_data):
        xs, means, los, his = aggregate_by(df, "k", metric)
        ax.plot(xs, means, "o-", color=color, markersize=3)
        ax.fill_between(xs, los, his, alpha=BAND_ALPHA, color=color)
        ax.set_xlabel("k")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.ticklabel_format(axis='y', style='plain')

    fig.suptitle("Fig 1.3 — Métriques géométriques vs k", fontsize=16)
    savefig(fig, output_dir / "fig1_3_panel_metrics_k.png")


# ═════════════════════════════════════════════════════════════════════════════
# EXP 2 — FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def plot_exp2(output_dir):
    csv_path = OUTPUT_BASE / "benchmark_exp2" / "exp2_results.csv"
    if not csv_path.exists():
        print(f"  SKIP Exp 2 : {csv_path} introuvable")
        return

    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col != "k":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["nb_motifs_total"])

    ks = df["k"].values

    # ── Fig 2.1 : Nb motifs vs k ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, df["nb_motifs_total"], "o-", color=COLOR_PRIMARY,
            label="Total", markersize=4)
    ax.plot(ks, df["nb_motifs_len2"], "s-", color=COLOR_SECONDARY,
            label="Longueur ≥ 2", markersize=4)
    ax.plot(ks, df["nb_motifs_len3"], "^-", color=COLOR_TERTIARY,
            label="Longueur ≥ 3", markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Nombre de motifs")
    ax.set_title("Fig 2.1 — Nombre de motifs PrefixSpan vs k")
    ax.ticklabel_format(axis='y', style='plain')
    ax.legend()
    savefig(fig, output_dir / "fig2_1_nb_motifs_k.png")

    # ── Fig 2.2 : Temps PrefixSpan vs k (log-y) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, df["temps_prefixspan_s"], "o-", color=COLOR_SECONDARY, markersize=4)
    ax.set_yscale("log")
    plain_log_axis(ax, axis="y")
    ax.set_xlabel("k")
    ax.set_ylabel("Temps PrefixSpan (s)")
    ax.set_title("Fig 2.2 — Temps d'exécution PrefixSpan vs k (échelle log)")
    savefig(fig, output_dir / "fig2_2_temps_prefixspan_k.png")

    # ── Fig 2.3 : Nb arêtes Markov vs k ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, df["nb_aretes_markov"], "o-", color=COLOR_TERTIARY, markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Nb arêtes Markov")
    ax.set_title("Fig 2.3 — Nombre d'arêtes du graphe de Markov vs k")
    ax.ticklabel_format(axis='y', style='plain')
    savefig(fig, output_dir / "fig2_3_aretes_markov_k.png")

    # ── Fig 2.4 : Entropie de Shannon vs k ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, df["entropie_shannon"], "o-", color=COLOR_PRIMARY, markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Entropie de Shannon (bits)")
    ax.set_title("Fig 2.4 — Entropie de Shannon des supports vs k")
    savefig(fig, output_dir / "fig2_4_entropie_k.png")


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHÈSE EXP1 + EXP2
# ═════════════════════════════════════════════════════════════════════════════

def plot_synthesis(output_dir):
    """Fig S.1 : Silhouette (Exp 1) + Nb arêtes Markov (Exp 2) vs k."""
    csv1 = OUTPUT_BASE / "benchmark_exp1" / "exp1_results.csv"
    csv2 = OUTPUT_BASE / "benchmark_exp2" / "exp2_results.csv"
    if not csv1.exists() or not csv2.exists():
        print("  SKIP Synthèse : CSV Exp 1 ou Exp 2 introuvable")
        return

    df1 = pd.read_csv(csv1)
    df1["silhouette"] = pd.to_numeric(df1["silhouette"], errors="coerce")
    df2 = pd.read_csv(csv2)
    for col in df2.columns:
        if col != "k":
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=["nb_aretes_markov"])

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # Silhouette (Exp 1, agrégée)
    xs1, means1, los1, his1 = aggregate_by(df1, "k", "silhouette")
    l1 = ax1.plot(xs1, means1, "o-", color=COLOR_PRIMARY, markersize=4,
                  label="Silhouette (Exp 1)")
    ax1.fill_between(xs1, los1, his1, alpha=BAND_ALPHA, color=COLOR_PRIMARY)

    # Arêtes Markov (Exp 2)
    ks2 = df2["k"].values
    aretes2 = df2["nb_aretes_markov"].values
    l2 = ax2.plot(ks2, aretes2, "s-", color=COLOR_SECONDARY, markersize=4,
                  label="Arêtes Markov (Exp 2)")

    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette", color=COLOR_PRIMARY)
    ax2.set_ylabel("Nb arêtes Markov", color=COLOR_SECONDARY)
    ax1.set_title("Fig S.1 — Synthèse : Qualité géométrique × richesse sémantique vs k")
    ax2.ticklabel_format(axis='y', style='plain')

    # Zone de chevauchement
    # Trouver k où silhouette est "bonne" ET arêtes sont "suffisantes"
    if len(means1) > 0 and len(aretes2) > 0:
        # Normaliser les deux courbes
        sil_norm = (means1 - np.nanmin(means1)) / (np.nanmax(means1) - np.nanmin(means1) + 1e-9)
        k_common = np.intersect1d(xs1, ks2)
        if len(k_common) > 0:
            sil_at_common = np.interp(k_common, xs1, means1)
            aretes_at_common = np.interp(k_common, ks2, aretes2)
            sil_n = (sil_at_common - np.nanmin(sil_at_common)) / (np.nanmax(sil_at_common) - np.nanmin(sil_at_common) + 1e-9)
            are_n = (aretes_at_common - np.nanmin(aretes_at_common)) / (np.nanmax(aretes_at_common) - np.nanmin(aretes_at_common) + 1e-9)
            # Score composite = silhouette_norm + aretes_norm
            composite = sil_n + are_n
            best_k_idx = np.argmax(composite)
            best_k = k_common[best_k_idx]
            ax1.axvline(best_k, color=COLOR_ACCENT, linestyle="--", linewidth=2, alpha=0.7)
            ax1.annotate(f"k*={int(best_k)}",
                         xy=(best_k, means1[np.searchsorted(xs1, best_k)]),
                         xytext=(20, 20), textcoords="offset points",
                         fontsize=12, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_ACCENT, alpha=0.4),
                         arrowprops=dict(arrowstyle="->", color="black"))

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    savefig(fig, output_dir / "figS_1_synthese_sil_aretes.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Génération des figures benchmark")
    parser.add_argument("--exp", nargs="*", type=int, default=None,
                        help="Expériences à tracer (0 1 2). Défaut: toutes")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_BASE / "benchmark_figures"))
    parser.add_argument("--report", action="store_true",
                        help="Générer aussi les figures de rapport (clustering scalabilité, heatmaps…)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exps = args.exp if args.exp is not None else [0, 1, 2]

    print(f"Génération des figures dans {output_dir}\n")

    if 0 in exps:
        print("── Exp 0 ──")
        plot_exp0(output_dir)

    if 1 in exps:
        print("── Exp 1 ──")
        plot_exp1(output_dir)

    if 2 in exps:
        print("── Exp 2 ──")
        plot_exp2(output_dir)

    # Synthèse (nécessite Exp 1 + Exp 2)
    if 1 in exps and 2 in exps:
        print("── Synthèse ──")
        plot_synthesis(output_dir)

    print(f"\n✓ Toutes les figures générées dans {output_dir}")

    # Figures rapport clustering (scalabilité, heatmaps, etc.)
    if args.report:
        print("\n── Figures rapport ──")
        from benchmark.generate_report_figures import (
            fig_scalability_fast, fig_scalability_ap, fig_overview_k12,
            fig_matrix_time, fig_barplot_at_n, fig_boxplots,
            fig_k_comparison, fig_speedup, fig_heatmap_silhouette,
            fig_stacked_time, fig_cv_silhouette, fig_db_ch_sensitivity,
            fig_compression_vs_silhouette, fig_silhouette_vs_k,
            fig_elbow, fig_combined_optimal_k,
        )
        fig_scalability_fast()
        fig_scalability_ap()
        fig_overview_k12()
        fig_matrix_time()
        fig_barplot_at_n()
        fig_boxplots()
        fig_k_comparison()
        fig_speedup()
        fig_heatmap_silhouette()
        fig_stacked_time()
        fig_cv_silhouette()
        fig_db_ch_sensitivity()
        fig_compression_vs_silhouette()
        fig_silhouette_vs_k()
        fig_elbow()
        fig_combined_optimal_k()
        print("  ✓ Figures rapport générées")


if __name__ == "__main__":
    main()
