"""
Package d'analyse de trajectoires Dota 2.

Modules:
    - structures: Structures de données + Export JSON
    - geometry: Calculs géométriques vectorisés
    - compression: Algorithme MDL de compression de trajectoires
    - plotting: Visualisation (overlays, comparaisons, clusters, interactif)
    - clustering: Analyse de clusters de trajectoires
    - metrics / mining / recoding: Outils d'analyse complémentaires
"""

__version__ = "1.0.0"
__author__ = "Analyse Trajectoires Dota 2 Team"

from .structures import Trajectory, TrajectoryPoint, Segment, JSONExporter
from .compression import MDLCompressor, process_full_match
from .geometry import GeometryUtils
from .plotting import PLAYER_COLORS

__all__ = [
    "Trajectory",
    "TrajectoryPoint",
    "Segment",
    "MDLCompressor",
    "process_full_match",
    "JSONExporter",
    "GeometryUtils",
    "PLAYER_COLORS",
]
