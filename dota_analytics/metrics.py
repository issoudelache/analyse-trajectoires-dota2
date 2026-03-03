"""Métriques pour évaluer la qualité de la compression de trajectoires.

Ce module fournit des outils pour mesurer :
- La précision de reconstruction (RMSE, erreur maximale)
- La génération de bruit gaussien pour tests de robustesse
"""

import numpy as np
from typing import List, Dict
from .structures import Trajectory, TrajectoryPoint, Segment


def calculate_reconstruction_error(
    original_traj: Trajectory, segments: List[Segment]
) -> Dict[str, float]:
    """Calcule l'erreur de reconstruction entre trajectoire originale et segments compressés.

    Pour chaque point de la trajectoire originale, calcule la distance perpendiculaire
    vers le segment compressé le plus proche (celui qui contient ce point temporellement).

    Args:
        original_traj: Trajectoire originale (liste de points)
        segments: Liste de segments compressés

    Returns:
        Dictionnaire contenant:
        - 'rmse': Root Mean Square Error (moyenne quadratique des erreurs)
        - 'max_error': Erreur maximale rencontrée
        - 'mean_error': Erreur moyenne
        - 'num_points': Nombre de points évalués

    Algorithme:
        Pour chaque point P(x, y, tick):
        1. Trouver le segment S dont [tick_start, tick_end] contient P.tick
        2. Calculer la distance perpendiculaire de P à la droite formée par S
        3. Agréger toutes les distances (RMSE, Max, Moyenne)
    """
    if not segments:
        return {"rmse": 0.0, "max_error": 0.0, "mean_error": 0.0, "num_points": 0}

    errors = []

    # Pour chaque point de la trajectoire originale
    for point in original_traj.points:
        # Trouver le segment qui contient ce point temporellement
        segment = None
        for seg in segments:
            if seg.start.tick <= point.tick <= seg.end.tick:
                segment = seg
                break

        if segment is None:
            # Point hors limites (avant premier segment ou après dernier)
            continue

        # Calculer distance perpendiculaire du point vers la droite du segment
        # Formule: |ax + by + c| / sqrt(a² + b²)
        # Où la droite est: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0

        x1, y1 = segment.start.x, segment.start.y
        x2, y2 = segment.end.x, segment.end.y
        px, py = point.x, point.y

        # Vecteur directeur du segment
        dx = x2 - x1
        dy = y2 - y1

        # Si le segment est un point unique (début = fin)
        if dx == 0 and dy == 0:
            distance = np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            # Distance perpendiculaire : |(x2-x1)(y1-py) - (x1-px)(y2-y1)| / sqrt((x2-x1)² + (y2-y1)²)
            numerator = abs(dx * (y1 - py) - (x1 - px) * dy)
            denominator = np.sqrt(dx * dx + dy * dy)
            distance = numerator / denominator

        errors.append(distance)

    if not errors:
        return {"rmse": 0.0, "max_error": 0.0, "mean_error": 0.0, "num_points": 0}

    # Calcul des métriques
    errors_array = np.array(errors)
    rmse = float(np.sqrt(np.mean(errors_array**2)))
    max_error = float(np.max(errors_array))
    mean_error = float(np.mean(errors_array))

    return {
        "rmse": rmse,
        "max_error": max_error,
        "mean_error": mean_error,
        "num_points": len(errors),
    }


def add_gaussian_noise(
    points: List[TrajectoryPoint], sigma: float
) -> List[TrajectoryPoint]:
    """Ajoute un bruit gaussien aux coordonnées spatiales des points.

    Simule le "jitter" ou tremblement de la souris en ajoutant un bruit aléatoire
    suivant une loi normale centrée (moyenne = 0, écart-type = sigma).

    Args:
        points: Liste de points originaux
        sigma: Écart-type du bruit gaussien (0 = pas de bruit)
               Valeurs typiques : 0.5 (léger), 1.0 (modéré), 3.0 (fort)

    Returns:
        Nouvelle liste de points avec coordonnées bruitées
        Les ticks restent inchangés

    Exemple:
        >>> original = [TrajectoryPoint(100, 200, 0), TrajectoryPoint(110, 210, 1)]
        >>> noisy = add_gaussian_noise(original, sigma=1.0)
        >>> # noisy[0].x ≈ 100 ± quelques unités
    """
    if sigma == 0:
        # Pas de bruit : copie simple
        return [TrajectoryPoint(p.x, p.y, p.tick) for p in points]

    noisy_points = []

    for point in points:
        # Générer bruit gaussien indépendant pour X et Y
        noise_x = np.random.normal(0, sigma)
        noise_y = np.random.normal(0, sigma)

        # Créer nouveau point avec coordonnées bruitées
        noisy_point = TrajectoryPoint(
            x=point.x + noise_x,
            y=point.y + noise_y,
            tick=point.tick,  # Le temps n'est pas bruité
        )

        noisy_points.append(noisy_point)

    return noisy_points


def calculate_compression_rate(num_original: int, num_compressed: int) -> float:
    """Calcule le taux de compression en pourcentage.

    Args:
        num_original: Nombre de points dans la trajectoire originale
        num_compressed: Nombre de segments après compression

    Returns:
        Taux de compression en % (ex: 95.2 signifie 95.2% de réduction)
    """
    if num_original == 0:
        return 0.0

    return (1 - num_compressed / num_original) * 100


def segment_length_statistics(segments: List[Segment]) -> Dict[str, float]:
    """Calcule des statistiques sur les longueurs des segments.

    Args:
        segments: Liste de segments compressés

    Returns:
        Dictionnaire avec min, max, mean, median des longueurs
    """
    if not segments:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    lengths = [seg.length() for seg in segments]

    return {
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
    }
