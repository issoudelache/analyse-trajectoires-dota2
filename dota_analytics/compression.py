"""Compression de trajectoires MDL multi-joueurs.

Implémentation de l'algorithme greedy avec seuil de tolérance pour 10 joueurs.

Paramètre w_error (tolérance d'erreur):
    max_distance_threshold = w_error

    - w_error GRAND (ex: 20) → tolérance GRANDE → segments LONGS (compression forte)
    - w_error PETIT (ex: 0.5) → tolérance PETITE → segments COURTS (fidélité haute)

    Principe: On étend un segment tant que TOUTES les distances perpendiculaires
    des points intermédiaires restent < w_error.
"""

from typing import List, Dict
import numpy as np
import pandas as pd

from .structures import Trajectory, TrajectoryPoint, Segment
from .geometry import GeometryUtils


class MDLCompressor:
    """Compresseur avec seuil de tolérance vectorisé pour multi-joueurs."""

    def __init__(self, w_error: float = 2.0, verbose: bool = False):
        """Initialise le compresseur.

        Args:
            w_error: Tolérance d'erreur maximale (distance perpendiculaire)
                     GRAND (20) → tolérance grande → segments longs (compression forte)
                     PETIT (0.5) → tolérance petite → segments courts (haute fidélité)
            verbose: Afficher les informations de compression
        """
        self.w_error = w_error
        self.verbose = verbose
        self.geometry = GeometryUtils()

    def compress_player_trajectory(self, trajectory: Trajectory) -> List[Segment]:
        """Compresse une trajectoire avec seuil de tolerance.

        Algorithme glouton optimise :
        1. Extraire TOUTES les coordonnees en un seul tableau numpy (une fois pour toute
           la trajectoire) -- elimine les N*k acces Python a trajectory[i].x / .y
        2. Dans la boucle interne, passer une VUE numpy O(1) a
           perpendicular_distances_vectorized (slice = zero allocation, zero copie)
           au lieu d'une list-comprehension O(k).
           Meme chemin de calcul qu'avant -> resultats numeriquement IDENTIQUES bit a bit.
        3. Briser des que la tolerance est depassee (inchange)

        Note : la formule inlinee (pts-a) - outer(t,v) diverge de pts - (a + outer(t,v))
        a 1 ULP pres (IEEE 754 : x-(a+b) != (x-a)-b). En reutilisant exactement
        perpendicular_distances_vectorized on evite toute divergence numerique.

        Args:
            trajectory: Trajectoire du joueur a compresser

        Returns:
            Liste de segments comprimes
        """
        if len(trajectory) < 2:
            return []

        n = len(trajectory)
        # Extraction unique -- elimine k list comprehensions par iteration interne.
        # La boucle interne appelait sinon [[traj[i].x, traj[i].y] for i in range(...)]
        # soit O(k) Python pur a chaque end_idx.
        all_xy = np.array([[p.x, p.y] for p in trajectory.points])

        segments = []
        start_idx = 0

        while start_idx < n - 1:
            best_idx = start_idx + 1
            p_start = all_xy[start_idx]

            for end_idx in range(start_idx + 2, n):
                p_end = all_xy[end_idx]

                # Vue numpy O(1) : zero allocation, zero copie, zero list comprehension.
                # Equivalent a np.array([[traj[i].x, traj[i].y] for i in range(s+1,e)])
                # mais sans construire de nouvel objet Python.
                intermediates = all_xy[start_idx + 1 : end_idx]

                # Meme appel qu'avant -> resultats bit-a-bit identiques a l'original
                d_perps = self.geometry.perpendicular_distances_vectorized(
                    intermediates, p_start, p_end
                )

                if np.all(d_perps < self.w_error):
                    best_idx = end_idx
                else:
                    break

            segment = Segment(start=trajectory[start_idx], end=trajectory[best_idx])
            segments.append(segment)
            if self.verbose:
                print(
                    f"  Segment {len(segments)}: [{start_idx}->{best_idx}] "
                    f"length={segment.length():.2f}, skipped={best_idx-start_idx-1}"
                )
            start_idx = best_idx

        return segments

    def process_full_match(
        self, df: pd.DataFrame, match_id: str
    ) -> Dict[int, List[Segment]]:
        """Traite un match complet avec les 10 joueurs.

        Args:
            df: DataFrame avec colonnes tick, x0, y0, ..., x9, y9
            match_id: Identifiant du match

        Returns:
            Dictionnaire {player_id: [segments]} pour les 10 joueurs
        """
        results = {}

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Compression du match {match_id} avec w_error={self.w_error:.1f}")
            print(f"{'=' * 60}")

        for player_id in range(10):
            x_col = f"x{player_id}"
            y_col = f"y{player_id}"

            if x_col not in df.columns or y_col not in df.columns:
                continue

            # Créer la trajectoire
            points = []
            for _, row in df.iterrows():
                x, y, tick = row[x_col], row[y_col], row["tick"]
                if x != 0.0 or y != 0.0:
                    points.append(TrajectoryPoint(x=x, y=y, tick=tick))

            if len(points) < 2:
                continue

            trajectory = Trajectory(points=points, player_id=player_id)

            if self.verbose:
                print(f"\nJoueur {player_id}: {len(trajectory)} points")

            segments = self.compress_player_trajectory(trajectory)
            results[player_id] = segments

            if self.verbose:
                reduction = (1 - len(segments) / len(trajectory)) * 100
                print(f"  → {len(segments)} segments ({reduction:.1f}% réduction)")

        return results


def compress_player_trajectory(
    trajectory: Trajectory, w_error: float = 2.0, verbose: bool = False
) -> List[Segment]:
    """Compresse une trajectoire d'un joueur.

    Args:
        trajectory: Trajectoire à compresser
        w_error: Tolérance (GRAND=compression forte, PETIT=fidélité haute)
        verbose: Afficher les infos

    Returns:
        Liste de segments compressés
    """
    compressor = MDLCompressor(w_error=w_error, verbose=verbose)
    return compressor.compress_player_trajectory(trajectory)


class DouglasPeuckerCompressor:
    """Algorithme de simplification de Ramer-Douglas-Peucker.

    Méthode classique de référence : réduit récursivement le nombre de points
    en supprimant ceux qui sont sous le seuil epsilon de distance perpendiculaire.
    Renvoie une liste de Segment identique à MDLCompressor pour comparaison directe.
    """

    def __init__(self, epsilon: float = 5.0):
        """Initialise le compresseur Douglas-Peucker.

        Args:
            epsilon: Distance perpendiculaire maximale tolérée.
                     GRAND → compression forte (moins de segments)
                     PETIT → fidélité haute (plus de segments)
        """
        self.epsilon = epsilon
        self.geometry = GeometryUtils()

    def _rdp_indices(self, points: list, start: int, end: int, result: list) -> None:
        """Récursion RDP : trouve les indices des points à conserver.

        Args:
            points: Liste de TrajectoryPoint
            start: Indice de début
            end: Indice de fin
            result: Set des indices à conserver (modifié en place)
        """
        if end <= start + 1:
            return

        p_start = np.array([points[start].x, points[start].y])
        p_end = np.array([points[end].x, points[end].y])

        # Calculer distance perpendiculaire de tous les points intermédiaires
        intermediate = np.array(
            [[points[i].x, points[i].y] for i in range(start + 1, end)]
        )
        distances = self.geometry.perpendicular_distances_vectorized(
            intermediate, p_start, p_end
        )

        max_dist_idx = int(np.argmax(distances))
        max_dist = distances[max_dist_idx]
        split_idx = start + 1 + max_dist_idx

        if max_dist >= self.epsilon:
            result.append(split_idx)
            self._rdp_indices(points, start, split_idx, result)
            self._rdp_indices(points, split_idx, end, result)

    def compress_player_trajectory(self, trajectory: Trajectory) -> List[Segment]:
        """Compresse une trajectoire avec Douglas-Peucker.

        Args:
            trajectory: Trajectoire du joueur à compresser

        Returns:
            Liste de Segment (même interface que MDLCompressor)
        """
        if len(trajectory) < 2:
            return []

        points = trajectory.points
        kept_indices = [0, len(points) - 1]
        self._rdp_indices(points, 0, len(points) - 1, kept_indices)
        kept_indices = sorted(set(kept_indices))

        segments = [
            Segment(start=points[kept_indices[i]], end=points[kept_indices[i + 1]])
            for i in range(len(kept_indices) - 1)
        ]
        return segments


def process_full_match(
    df: pd.DataFrame, match_id: str, w_error: float = 2.0, verbose: bool = False
) -> Dict[int, List[Segment]]:
    """Traite un match complet (10 joueurs).

    Args:
        df: DataFrame avec colonnes tick, x0, y0, ..., x9, y9
        match_id: Identifiant du match
        w_error: Tolérance (GRAND=compression forte, PETIT=fidélité haute)
        verbose: Afficher les infos

    Returns:
        Dictionnaire {player_id: [segments]}
    """
    compressor = MDLCompressor(w_error=w_error, verbose=verbose)
    return compressor.process_full_match(df, match_id)
