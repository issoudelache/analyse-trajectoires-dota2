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
    
    def to_dict(self) -> dict:
        """Convertit le segment en dictionnaire pour exportation JSON."""
        import numpy as np
        return {
            "start": {
                "x": float(np.float64(self.start.x)),
                "y": float(np.float64(self.start.y)),
                "tick": int(np.int64(self.start.tick))
            },
            "end": {
                "x": float(np.float64(self.end.x)),
                "y": float(np.float64(self.end.y)),
                "tick": int(np.int64(self.end.tick))
            },
            "length": float(self.length()),
            "angle": float(self.angle_degrees())
        }


class Trajectory:
    """Trajectoire composée d'une séquence de points."""
    
    def __init__(self, points: List[TrajectoryPoint], player_id: int = 0):
        """Initialise la trajectoire.
        
        Args:
            points: Liste ordonnée de points
            player_id: Identifiant du joueur (0-9)
        """
        if not points:
            raise ValueError("Trajectory must contain at least one point")
        self.points = points
        self.player_id = player_id
    
    def __len__(self) -> int:
        """Retourne le nombre de points."""
        return len(self.points)
    
    def __getitem__(self, idx: int) -> TrajectoryPoint:
        """Accès aux points par index."""
        return self.points[idx]
    
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
    
    def bounding_box(self) -> tuple:
        """Calcule la boîte englobante.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        coords = self.to_numpy()
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)
