"""Core module pour l'analyse de trajectoires DotA 2."""

from .structures import TrajectoryPoint, Segment, Trajectory
from .geometry import GeometryUtils, euclidean_distance, perpendicular_distance, angular_distance
from .compression import MDLCompressor, compress

__all__ = [
    'TrajectoryPoint',
    'Segment', 
    'Trajectory',
    'GeometryUtils',
    'euclidean_distance',
    'perpendicular_distance',
    'angular_distance',
    'MDLCompressor',
    'compress'
]
