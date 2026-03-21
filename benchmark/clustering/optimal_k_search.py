#!/usr/bin/env python3
"""
optimal_k_search.py
===================
Recherche du nombre optimal de clusters (k) pour KMeans via :
  - Méthode du Coude (inertie)
  - Score Silhouette

Produit une figure Matplotlib (1×2) sauvegardée dans :
  output/benchmark_clustering/optimal_k_search.png
"""

import sys
import time
from pathlib import Path

# Résolution racine projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from dota_analytics.clustering import load_data
from mvc.config import COMPRESSED_DIR

# ── Paramètres ─────────────────────────────────────────────────────────────
K_RANGE = list(range(2, 21)) + list(range(25, 101, 5))  # 2..20 puis 25,30,...,100
SAMPLE_SIZE = 5000       # Sous-échantillon pour garder un temps raisonnable
MAX_FILES = 30           # Limiter le nombre de fichiers JSON chargés
RANDOM_STATE = 42

OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmark_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUTPUT_DIR / "optimal_k_search.png"

# ── Chargement des données ─────────────────────────────────────────────────
print("=" * 65)
print("  RECHERCHE DU K OPTIMAL — Méthode du Coude + Silhouette")
print("=" * 65)

target_folder = COMPRESSED_DIR / "w_error_12.0"
print(f"\n[1/4] Chargement des segments depuis : {target_folder}")
segments, _ = load_data(target_folder, limit=SAMPLE_SIZE, max_files=MAX_FILES)
print(f"       → {len(segments)} segments chargés (max_files={MAX_FILES}).")

# ── Extraction des features (identique au pipeline KMeans existant) ────────
print("[2/4] Extraction et normalisation des features...")
features = []
for s in segments:
    mid_x = (s.start.x + s.end.x) / 2.0
    mid_y = (s.start.y + s.end.y) / 2.0
    dx = s.end.x - s.start.x
    dy = s.end.y - s.start.y
    length = np.sqrt(dx**2 + dy**2)
    features.append([mid_x, mid_y, dx, dy, length])

X = np.array(features, dtype=np.float32)

# Sous-échantillonnage si trop de segments
if len(X) > SAMPLE_SIZE:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=SAMPLE_SIZE, replace=False)
    X = X[idx]
    print(f"       Sous-échantillon : {SAMPLE_SIZE} segments retenus.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"       Features shape : {X_scaled.shape}")

# ── Boucle sur les valeurs de k ───────────────────────────────────────────
print(f"[3/4] Test de {len(K_RANGE)} valeurs de k : {K_RANGE[0]}..{K_RANGE[-1]}")
print("-" * 65)

inertias = []
silhouettes = []
t_total_start = time.perf_counter()

for i, k in enumerate(K_RANGE, 1):
    t0 = time.perf_counter()

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=min(4096, len(X_scaled)),
        n_init=10,
    )
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_

    sil = silhouette_score(X_scaled, labels)

    elapsed = time.perf_counter() - t0
    inertias.append(inertia)
    silhouettes.append(sil)

    print(f"  [{i:>2}/{len(K_RANGE)}]  k={k:<4}  "
          f"Inertie={inertia:>12.1f}   Silhouette={sil:+.4f}   "
          f"({elapsed:.2f}s)")

t_total = time.perf_counter() - t_total_start
print("-" * 65)
print(f"  Temps total boucle : {t_total:.1f}s")

# ── Détection du coude (variation de la dérivée seconde) ──────────────────
inertias_arr = np.array(inertias)
k_arr = np.array(K_RANGE, dtype=float)

# Dérivée seconde discrète pour trouver le point d'inflexion
if len(inertias_arr) >= 3:
    d1 = np.diff(inertias_arr)
    d2 = np.diff(d1)
    # Le coude = là où la dérivée seconde est maximale (changement le plus fort)
    elbow_idx = np.argmax(d2) + 1  # +1 car diff décale d'un index
    k_elbow = K_RANGE[elbow_idx]
else:
    k_elbow = K_RANGE[0]

# Meilleur silhouette
best_sil_idx = np.argmax(silhouettes)
k_best_sil = K_RANGE[best_sil_idx]

print(f"\n  ★ Coude détecté à k = {k_elbow}")
print(f"  ★ Meilleur Silhouette à k = {k_best_sil} ({silhouettes[best_sil_idx]:+.4f})")

# ── Visualisation ──────────────────────────────────────────────────────────
print(f"\n[4/4] Génération de la figure → {PLOT_PATH}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Recherche du nombre optimal de clusters (k) — KMeans",
             fontsize=14, fontweight="bold", y=1.02)

# ── Subplot 1 : Méthode du Coude ──────────────────────────────────────────
ax1.plot(K_RANGE, inertias, "o-", color="#2563EB", linewidth=1.8,
         markersize=4, label="Inertie")
ax1.axvline(k_elbow, color="#DC2626", linestyle="--", alpha=0.8,
            label=f"Coude estimé (k={k_elbow})")
ax1.set_xlabel("Nombre de clusters (k)", fontsize=11)
ax1.set_ylabel("Inertie", fontsize=11)
ax1.set_title("Méthode du Coude", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=9)

# ── Subplot 2 : Score Silhouette ──────────────────────────────────────────
ax2.plot(K_RANGE, silhouettes, "s-", color="#059669", linewidth=1.8,
         markersize=4, label="Silhouette Score")
ax2.axvline(k_best_sil, color="#DC2626", linestyle="--", alpha=0.8,
            label=f"Meilleur k={k_best_sil} ({silhouettes[best_sil_idx]:.4f})")
ax2.set_xlabel("Nombre de clusters (k)", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=11)
ax2.set_title("Qualité des Clusters", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n  ✓ Figure sauvegardée : {PLOT_PATH}")
print("=" * 65)
