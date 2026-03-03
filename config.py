from pathlib import Path

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()

# Données d'entrée
DATA_DIR = BASE_DIR / "data-dota"
CANVAS_PATH = BASE_DIR / "canvas.png"
EXPORTED_DATA_MVC = BASE_DIR / "exported_data_mvc"

# Dossiers de sortie
OUTPUT_DIR = BASE_DIR / "output"
COMPRESSED_DIR = OUTPUT_DIR / "compressed"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"

# =============================================================================
# PARAMÈTRES PAR DÉFAUT (Tests & Benchmarks)
# =============================================================================
DEFAULT_MATCH_ID = "3841665963"
DEFAULT_PLAYER_ID = 0
DEFAULT_TICK_START = 66000
DEFAULT_TICK_END = 68000

# Création automatique des dossiers de base s'ils n'existent pas
for dir_path in [
    OUTPUT_DIR,
    COMPRESSED_DIR,
    VISUALIZATIONS_DIR,
    OVERLAYS_DIR,
    CLUSTERS_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)
