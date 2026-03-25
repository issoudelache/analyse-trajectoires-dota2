"""
Configuration centralisée pour tous les scripts de benchmark.

Toutes les constantes partagées (chemins, grilles, paramètres par défaut)
sont définies ici pour éviter la duplication entre les fichiers exp0–exp3,
sensitivity, calibrate_ap, etc.
"""

from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data-dota"
CANVAS_PATH = BASE_DIR / "canvas.png"

COMPRESSED_DIR = BASE_DIR / "output" / "compressed" / "w_error_12.0"
OUTPUT_BASE = BASE_DIR / "output"

# Sous-dossiers par expérience
OUTPUT_EXP0 = OUTPUT_BASE / "benchmark_exp0"
OUTPUT_EXP1 = OUTPUT_BASE / "benchmark_exp1"
OUTPUT_EXP2 = OUTPUT_BASE / "benchmark_exp2"
OUTPUT_EXP3 = OUTPUT_BASE / "benchmark_exp3"
OUTPUT_SENSITIVITY = OUTPUT_BASE / "benchmark_sensitivity"
OUTPUT_CLUSTERING = OUTPUT_BASE / "benchmark_clustering"
OUTPUT_FIGURES = OUTPUT_BASE / "rapport_figures"

# ═════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

W_ERROR = 12.0
MAX_FILES = 30
MIN_LENGTH = 5.0
SEED = 42

# Clustering
K_DEFAULT = 12

# Affinity Propagation
AP_PREFERENCE = -5000.0
AP_DAMPING = 0.7
AP_MAX_ITER = 500

# PrefixSpan
MIN_SUPPORT = 15
MAX_LENGTH = 5

# Sous-échantillonnage
N_SUBSAMPLE_DEFAULT = 3000
N_SUBSAMPLE_EXP1 = 5000
SEEDS_MULTI = list(range(7))

# ═════════════════════════════════════════════════════════════════════════════
# GRILLES DE PARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════

W_ERROR_GRID_26 = [
    0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
    25.0, 30.0, 40.0, 50.0, 75.0, 100.0,
]

K_GRID_30 = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 22, 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
]
