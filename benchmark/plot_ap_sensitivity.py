#!/usr/bin/env python3
"""Generate a report figure comparing AP performance across N_SUBSAMPLE values."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "benchmark_exp3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from calibration runs (all tuned to k≈10)
N_values =      [3000,   5000,   8000,   10000]
k_values =      [10,     10,     12,     14]
silhouette =    [0.0845, 0.0823, 0.0817, 0.0707]
balance =       [0.48,   0.44,   0.47,   0.34]
preference =    [-5000,  -8000,  -10000, -10000]

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
fig.suptitle("Sensibilité d'Affinity Propagation au nombre de segments ($N$)",
             fontsize=14, fontweight="bold", y=0.97)

colors = ["#2ecc71" if n == 3000 else "#95a5a6" for n in N_values]
best_color = "#2ecc71"
other_color = "#7f8c8d"

# ── Panel 1: Silhouette ──────────────────────────────────────────────────
ax = axes[0, 0]
bars = ax.bar([str(n) for n in N_values], silhouette, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Score de silhouette")
ax.set_xlabel("$N$ (segments)")
ax.set_title("Silhouette (↑ = mieux)")
ax.axhline(y=silhouette[0], color=best_color, linestyle="--", alpha=0.4, linewidth=1)
for i, (v, bar) in enumerate(zip(silhouette, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
ax.set_ylim(0, max(silhouette) * 1.25)

# ── Panel 2: k obtenu ───────────────────────────────────────────────────
ax = axes[0, 1]
bars = ax.bar([str(n) for n in N_values], k_values, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Nombre de clusters ($k$)")
ax.set_xlabel("$N$ (segments)")
ax.set_title("$k$ obtenu (cible = 10)")
ax.axhline(y=10, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5, label="Cible $k=10$")
ax.legend(fontsize=9)
for i, (v, bar) in enumerate(zip(k_values, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v),
            ha="center", va="bottom", fontsize=10, fontweight="bold" if i == 0 else "normal")
ax.set_ylim(0, max(k_values) * 1.3)

# ── Panel 3: Balance ────────────────────────────────────────────────────
ax = axes[1, 0]
bars = ax.bar([str(n) for n in N_values], balance, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Équilibre (min/max)")
ax.set_xlabel("$N$ (segments)")
ax.set_title("Équilibre des clusters (↑ = mieux)")
for i, (v, bar) in enumerate(zip(balance, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
ax.set_ylim(0, 0.7)

# ── Panel 4: Preference requise ─────────────────────────────────────────
ax = axes[1, 1]
pref_abs = [-p for p in preference]
bars = ax.bar([str(n) for n in N_values], pref_abs, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("|preference|")
ax.set_xlabel("$N$ (segments)")
ax.set_title("Preference requise pour $k \\approx 10$")
for i, (v, bar) in enumerate(zip(pref_abs, bars)):
    suffix = ""
    if i >= 2:
        suffix = "\n(k>" + str(k_values[i]) + ")"
    ax.text(bar.get_x() + bar.get_width()/2, v + 150, f"{v}" + suffix,
            ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 0 else "normal")
ax.set_ylim(0, max(pref_abs) * 1.3)

# ── Annotation globale ───────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Vert = configuration optimale retenue ($N$=3000, preference=−5000, $k$=10).  "
         "Pour $N$ ≥ 8000, AP ne peut plus atteindre $k$=10.",
         ha="center", fontsize=10, style="italic", color="#555555")

fig.tight_layout(rect=[0, 0.04, 1, 0.94])
out_path = OUTPUT_DIR / "fig_ap_sensitivity_N.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure sauvegardée : {out_path}")
