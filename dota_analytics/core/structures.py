"""Structures de données pour les trajectoires."""

from dataclasses import dataclass
from typing import List
import numpy as np
import math


@dataclass
class TrajectoryPoint:
    """Point d'une trajectoire dans l'espace 2D."""
    x: float
    y: float
    tick: int
    
    def to_array(self) -> np.ndarray:
        """Convertit en array numpy (x, y)."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class Segment:
    """Segment reliant deux points d'une trajectoire."""
    start: TrajectoryPoint
    end: TrajectoryPoint
    
    def length(self) -> float:
        """Calcule la longueur euclidienne du segment."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float(np.sqrt(dx * dx + dy * dy))
    
    def angle(self) -> float:
        """Calcule l'angle du segment en radians [-π, π].
        
        Returns:
            Angle par rapport à l'axe X positif
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float(np.arctan2(dy, dx))
    
    def angle_degrees(self) -> float:
        """Calcule l'angle du segment en degrés [-180, 180]."""
        return math.degrees(self.angle())
    
    def vector(self) -> np.ndarray:
        """Retourne le vecteur directionnel du segment."""
        return np.array([self.end.x - self.start.x, 
                        self.end.y - self.start.y], dtype=np.float64)


class Trajectory:
    """Trajectoire composée d'une séquence de points."""
    
    def __init__(self, points: List[TrajectoryPoint]):
        """Initialise la trajectoire.
        
        Args:
            points: Liste ordonnée de points
        """
        if not points:
            raise ValueError("Trajectory must contain at least one point")
        self.points = points
    
    def to_numpy(self) -> np.ndarray:
        """Convertit la trajectoire en array numpy (N, 2).
        
        Returns:
            Array de forme (N, 2) contenant les coordonnées (x, y)
        """
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float64)
    
    def total_distance(self) -> float:
        """Calcule la distance totale parcourue.
        
        Returns:
            Somme des distances entre points consécutifs
        """
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i + 1].x - self.points[i].x
            dy = self.points[i + 1].y - self.points[i].y
            total += float(np.sqrt(dx * dx + dy * dy))
        return total
    
    def duration(self) -> int:
        """Calcule la durée en ticks.
        
        Returns:
            Différence entre dernier et premier tick
        """
        return self.points[-1].tick - self.points[0].tick
    
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Calcule la boîte englobante.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        coords = self.to_numpy()
        return (float(coords[:, 0].min()), float(coords[:, 1].min()),
                float(coords[:, 0].max()), float(coords[:, 1].max()))
    
    def __len__(self) -> int:
        """Retourne le nombre de points."""
        return len(self.points)
    
    def __getitem__(self, index: int) -> TrajectoryPoint:
        """Accès indexé aux points."""
        return self.points[index]
    
    def __repr__(self) -> str:
        """Représentation textuelle."""
        return f"Trajectory(points={len(self.points)})"
