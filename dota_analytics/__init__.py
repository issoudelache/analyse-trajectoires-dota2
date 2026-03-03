"""
Package d'analyse de trajectoires Dota 2.

Architecture plate (Flat Structure) pour une meilleure maintenabilité.

Modules:
    - structures: Structures de données + Export JSON
    - geometry: Calculs géométriques vectorisés
    - compression: Algorithme MDL de compression de trajectoires
    - plotting: Visualisations statiques et interactives
    - clustering: Analyse de clusters de trajectoires
"""

__version__ = "1.0.0"
__author__ = "Analyse Trajectoires Dota 2 Team"

# Imports principaux pour faciliter l'utilisation
from .structures import Trajectory, TrajectoryPoint, Segment, JSONExporter, export_match
from .compression import MDLCompressor, process_full_match
from .geometry import GeometryUtils

__all__ = [
    "Trajectory",
    "TrajectoryPoint",
    "Segment",
    "MDLCompressor",
    "process_full_match",
    "JSONExporter",
    "export_match",
    "GeometryUtils",
]
