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

    @staticmethod
    def parallel_distance(s1_start: Union[np.ndarray, Tuple[float, float]],
                          s1_end: Union[np.ndarray, Tuple[float, float]],
                          s2_start: Union[np.ndarray, Tuple[float, float]],
                          s2_end: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calcule la distance parallèle (chevauchement) entre deux segments définis par leurs points.
        
        Args:
            s1_start, s1_end: Points du premier segment
            s2_start, s2_end: Points du second segment
        """
        # Conversion en numpy arrays
        p1_s = np.asarray(s1_start, dtype=np.float64)
        p1_e = np.asarray(s1_end, dtype=np.float64)
        p2_s = np.asarray(s2_start, dtype=np.float64)
        p2_e = np.asarray(s2_end, dtype=np.float64)

        # Calcul des vecteurs et longueurs
        v1 = p1_e - p1_s
        v2 = p2_e - p2_s
        l1_sq = np.dot(v1, v1)
        l2_sq = np.dot(v2, v2)

        # On identifie le segment de base (le plus long)
        if l1_sq > l2_sq:
            base_start, base_vec, base_len_sq = p1_s, v1, l1_sq
            other_start, other_end = p2_s, p2_e
        else:
            base_start, base_vec, base_len_sq = p2_s, v2, l2_sq
            other_start, other_end = p1_s, p1_e

        if base_len_sq < 1e-10: return 0.0

        # Vecteurs de projection
        v_start = other_start - base_start
        v_end = other_end - base_start

        # Projections scalaires
        t_start = np.dot(v_start, base_vec)
        t_end = np.dot(v_end, base_vec)

        # Calcul des distances des projections (formule TRACLUS simplifiée)
        p_dist_1 = min(abs(t_start), abs(t_start - base_len_sq))
        p_dist_2 = min(abs(t_end), abs(t_end - base_len_sq))

        # Normalisation
        return float(np.sqrt(p_dist_1**2 + p_dist_2**2) / np.sqrt(base_len_sq))



# Rétrocompatibilité: exposition comme fonctions au niveau module
euclidean_distance = GeometryUtils.euclidean_distance
perpendicular_distance = GeometryUtils.perpendicular_distance
angular_distance = GeometryUtils.angular_distance
parallel_distance = GeometryUtils.parallel_distance
