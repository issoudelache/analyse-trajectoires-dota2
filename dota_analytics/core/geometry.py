"""Fonctions géométriques optimisées avec numpy."""

import numpy as np
from typing import Tuple, Union


class GeometryUtils:
    """Utilitaires de calculs géométriques vectorisés."""
    
    @staticmethod
    def euclidean_distance(p1: Union[np.ndarray, Tuple[float, float]], 
                          p2: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calcule la distance euclidienne entre deux points.
        
        Args:
            p1: Premier point (x, y)
            p2: Deuxième point (x, y)
            
        Returns:
            Distance euclidienne
        """
        p1_arr = np.asarray(p1, dtype=np.float64)
        p2_arr = np.asarray(p2, dtype=np.float64)
        return float(np.linalg.norm(p1_arr - p2_arr))
    
    @staticmethod
    def perpendicular_distance(point: Union[np.ndarray, Tuple[float, float]],
                              line_start: Union[np.ndarray, Tuple[float, float]],
                              line_end: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calcule la distance perpendiculaire d'un point à une ligne infinie.
        
        Formule: d = ||(p - a) - ((p - a)·(b - a) / ||b - a||²) * (b - a)||
        où a = line_start, b = line_end, p = point
        
        Args:
            point: Point de test
            line_start: Point de départ de la ligne
            line_end: Point de fin de la ligne
            
        Returns:
            Distance perpendiculaire (0.0 si segment de longueur nulle)
        """
        p = np.asarray(point, dtype=np.float64)
        a = np.asarray(line_start, dtype=np.float64)
        b = np.asarray(line_end, dtype=np.float64)
        
        line_vec = b - a
        line_length_sq = np.dot(line_vec, line_vec)
        
        # Gestion des segments de longueur nulle
        if line_length_sq < 1e-10:
            return GeometryUtils.euclidean_distance(p, a)
        
        # Projection du point sur la ligne
        t = np.dot(p - a, line_vec) / line_length_sq
        projection = a + t * line_vec
        
        return float(np.linalg.norm(p - projection))
    
    @staticmethod
    def angular_distance(vector1: Union[np.ndarray, Tuple[float, float]],
                        vector2: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calcule la distance angulaire entre deux vecteurs.
        
        Mesure: angle = arccos(v1·v2 / (||v1|| * ||v2||))
        Retourne l'angle en radians [0, π]
        
        Args:
            vector1: Premier vecteur
            vector2: Deuxième vecteur
            
        Returns:
            Angle en radians (0.0 si un vecteur est nul)
        """
        v1 = np.asarray(vector1, dtype=np.float64)
        v2 = np.asarray(vector2, dtype=np.float64)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Gestion des vecteurs de longueur nulle
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        # Limitation pour éviter les erreurs numériques
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return float(np.arccos(cos_angle))


# Rétrocompatibilité: exposition comme fonctions au niveau module
euclidean_distance = GeometryUtils.euclidean_distance
perpendicular_distance = GeometryUtils.perpendicular_distance
angular_distance = GeometryUtils.angular_distance
