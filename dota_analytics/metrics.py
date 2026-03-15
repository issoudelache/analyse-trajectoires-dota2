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


# =============================================================================
# MÉTRIQUES DE COMPARAISON INTER-ALGORITHMES
# =============================================================================


def _point_to_segment_dist(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Distance d'un point (px, py) à un segment fini [(x1,y1)→(x2,y2)].

    Contrairement à la distance perpendiculaire (ligne infinie), projette
    le point sur le segment et le clamp entre les extrémités si besoin.

    Returns:
        Distance minimale en unités de carte
    """
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))


def rmse_segments_to_points(
    original_points: List[TrajectoryPoint], segments: List[Segment]
) -> float:
    """RMSE entre points originaux et trajectoire compressée (segments finis).

    Pour chaque point original, calcule la distance minimale à l'ensemble des
    segments (projection sur segment fini, bornée aux extrémités). Retourne
    la racine de la moyenne des carrés de ces distances.

    Utilisé pour comparer objectivement MDL, Douglas-Peucker et Uniforme
    à taux de compression égal.

    Args:
        original_points: Liste de TrajectoryPoint de la trajectoire brute
        segments: Segments compressés (MDL, DP ou Uniforme)

    Returns:
        RMSE en unités de distance (0.0 si données vides)
    """
    if not segments or not original_points:
        return 0.0

    sq_errors = []
    for pt in original_points:
        min_dist = min(
            _point_to_segment_dist(
                pt.x, pt.y,
                seg.start.x, seg.start.y,
                seg.end.x, seg.end.y,
            )
            for seg in segments
        )
        sq_errors.append(min_dist ** 2)

    return float(np.sqrt(np.mean(sq_errors)))


def hausdorff_distance(
    original_points: List[TrajectoryPoint], segments: List[Segment]
) -> float:
    """Distance de Hausdorff entre points originaux et trajectoire compressée.

    Calcule la distance maximale de chaque point original au segment le plus
    proche (finite segment). Représente l'erreur MAXIMALE garantie
    par la compression — un seul point mal placé suffit à dégrader ce score.

    Args:
        original_points: Liste de TrajectoryPoint de la trajectoire brute
        segments: Segments compressés

    Returns:
        Distance de Hausdorff (0.0 si données vides)
    """
    if not segments or not original_points:
        return 0.0

    max_dist = 0.0
    for pt in original_points:
        min_dist = min(
            _point_to_segment_dist(
                pt.x, pt.y,
                seg.start.x, seg.start.y,
                seg.end.x, seg.end.y,
            )
            for seg in segments
        )
        if min_dist > max_dist:
            max_dist = min_dist

    return float(max_dist)


def stop_preservation_rate(
    original_points: List[TrajectoryPoint],
    segments: List[Segment],
    speed_threshold: float = 2.0,
) -> "float | None":
    """Taux de préservation des points d'arrêt après compression.

    Un point P_i est un arrêt si la distance spatiale P_i → P_{i+1}
    est inférieure à speed_threshold (joueur quasi-immobile entre deux ticks).

    Mesure le % de ces ticks d'arrêt qui sont encore présents parmi les
    bornes (start.tick / end.tick) des segments compressés.

    Args:
        original_points: Points de la trajectoire originale (ordonnés par tick)
        segments: Segments compressés
        speed_threshold: Distance spatiale max entre 2 ticks pour qu'un point
                         soit considéré comme un arrêt (défaut: 2.0 unités)

    Returns:
        Pourcentage [0–100] d'arrêts préservés, ou None s'il n'y a aucun
        arrêt dans la fenêtre (à exclure des statistiques agrégées)
    """
    if len(original_points) < 2 or not segments:
        return None

    # Détection des arrêts : P_i est un stop si dist(P_i, P_{i+1}) < seuil
    stop_ticks: set[int] = set()
    for i in range(len(original_points) - 1):
        p_curr = original_points[i]
        p_next = original_points[i + 1]
        dist = np.sqrt(
            (p_next.x - p_curr.x) ** 2 + (p_next.y - p_curr.y) ** 2
        )
        if dist < speed_threshold:
            stop_ticks.add(p_curr.tick)

    if not stop_ticks:
        return None

    # Ticks conservés = bornes des segments compressés
    compressed_ticks: set[int] = set()
    for seg in segments:
        compressed_ticks.add(seg.start.tick)
        compressed_ticks.add(seg.end.tick)

    preserved = stop_ticks & compressed_ticks
    return float(len(preserved) / len(stop_ticks) * 100)


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
