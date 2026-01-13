"""Compression de trajectoires via MDL (Minimum Description Length).

Implémentation de l'algorithme greedy MDL du papier TraClus (Lee et al.).

Principe MDL:
    Cost(H) = L(H) + L(D|H)
    
    L(H) = log₂(length(segment))  # Hypothesis cost
    L(D|H) = Σ (d_perp + d_angle)  # Data encoding cost
    
    où:
    - d_perp: distance perpendiculaire du point au segment
    - d_angle: distance angulaire entre vecteurs consécutifs
"""

from typing import List
import numpy as np
import math

from .structures import Trajectory, TrajectoryPoint, Segment
from .geometry import GeometryUtils


class MDLCompressor:
    """Compresseur de trajectoires utilisant le principe MDL (Minimum Description Length)."""
    
    def __init__(self, verbose: bool = False):
        """Initialise le compresseur.
        
        Args:
            verbose: Afficher les informations de compression
        """
        self.verbose = verbose
        self.geometry = GeometryUtils()
    
    def _compute_mdl_cost(self, trajectory: Trajectory, 
                          start_idx: int, 
                          end_idx: int) -> float:
        """Calcule le coût MDL pour un segment candidat.
        
        MDL Cost = L(H) + L(D|H)
        
        L(H) = log₂(length) : coût de description du segment
        L(D|H) = Σ(d_perp + d_angle) : coût d'encodage des points intermédiaires
        
        Args:
            trajectory: Trajectoire complète
            start_idx: Index du point de départ
            end_idx: Index du point d'arrivée (inclusive)
            
        Returns:
            Coût MDL total
        """
        if end_idx - start_idx < 1:
            return 0.0
        
        p_start = trajectory[start_idx]
        p_end = trajectory[end_idx]
        
        # L(H): coût d'hypothèse
        segment_length = self.geometry.euclidean_distance(
            (p_start.x, p_start.y),
            (p_end.x, p_end.y)
        )
        
        if segment_length < 1e-10:
            return float('inf')
        
        l_h = math.log2(segment_length)
        
        # L(D|H): coût d'encodage des données
        l_d_h = 0.0
        
        # Points intermédiaires
        for i in range(start_idx + 1, end_idx):
            p_curr = trajectory[i]
            
            # Distance perpendiculaire au segment
            d_perp = self.geometry.perpendicular_distance(
                (p_curr.x, p_curr.y),
                (p_start.x, p_start.y),
                (p_end.x, p_end.y)
            )
            
            # Distance angulaire
            d_angle = 0.0
            if i > start_idx + 1:
                p_prev = trajectory[i - 1]
                
                # Vecteur précédent
                v_prev = np.array([p_curr.x - p_prev.x, 
                                  p_curr.y - p_prev.y])
                
                # Vecteur du segment principal
                v_seg = np.array([p_end.x - p_start.x, 
                                 p_end.y - p_start.y])
                
                d_angle = self.geometry.angular_distance(v_prev, v_seg)
            
            l_d_h += d_perp + d_angle
        
        return l_h + l_d_h
    
    def compress(self, trajectory: Trajectory) -> List[Segment]:
        """Compresse une trajectoire en segments via l'algorithme MDL greedy.
        
        Algorithme:
        1. Partir du premier point (anchor)
        2. Étendre progressivement vers les points suivants
        3. Calculer le coût MDL à chaque extension
        4. Quand le coût augmente, couper au point précédent
        5. Recommencer avec le nouveau point anchor
        
        Args:
            trajectory: Trajectoire à compresser
            
        Returns:
            Liste de segments compressés
        """
        if len(trajectory) < 2:
            return []
        
        segments: List[Segment] = []
        start_idx = 0
        
        while start_idx < len(trajectory) - 1:
            best_end_idx = start_idx + 1
            best_cost = self._compute_mdl_cost(trajectory, start_idx, best_end_idx)
            
            # Extension gloutonne
            curr_idx = start_idx + 2
            while curr_idx < len(trajectory):
                curr_cost = self._compute_mdl_cost(trajectory, start_idx, curr_idx)
                
                if curr_cost < best_cost:
                    # Amélioration: continuer l'extension
                    best_cost = curr_cost
                    best_end_idx = curr_idx
                    curr_idx += 1
                else:
                    # Coût augmente: stopper ici
                    break
            
            # Créer le segment optimal
            segment = Segment(
                start=trajectory[start_idx],
                end=trajectory[best_end_idx]
            )
            segments.append(segment)
            
            if self.verbose:
                print(f"Segment: [{start_idx}, {best_end_idx}], "
                      f"length={segment.length():.2f}, cost={best_cost:.2f}")
            
            # Nouveau point de départ
            start_idx = best_end_idx
        
        return segments


# Rétrocompatibilité: fonction au niveau module
def compress(trajectory: Trajectory, verbose: bool = False) -> List[Segment]:
    """Fonction wrapper pour la compatibilité.
    
    Args:
        trajectory: Trajectoire à compresser
        verbose: Afficher les infos de debug
        
    Returns:
        Liste de segments compressés
    """
    compressor = MDLCompressor(verbose=verbose)
    return compressor.compress(trajectory)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Compression MDL d'une trajectoire bruitée")
    print("=" * 60)
    
    # Génération d'une trajectoire avec bruit
    np.random.seed(42)
    n_points = 100
    
    # Ligne droite avec bruit gaussien
    t = np.linspace(0, 10, n_points)
    x = t + np.random.normal(0, 0.3, n_points)
    y = 2 * t + np.random.normal(0, 0.3, n_points)
    
    # Ajouter un coude à mi-chemin
    mid = n_points // 2
    y[mid:] = y[mid] - (t[mid:] - t[mid]) * 1.5
    
    points = [
        TrajectoryPoint(x=float(x[i]), y=float(y[i]), tick=i)
        for i in range(n_points)
    ]
    
    trajectory = Trajectory(points)
    
    print(f"Trajectoire originale: {len(trajectory)} points")
    print(f"Étendue X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Étendue Y: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Compression avec classe MDLCompressor
    print("Compression en cours...")
    compressor = MDLCompressor(verbose=True)
    segments = compressor.compress(trajectory)
    
    print()
    print(f"Résultat: {len(segments)} segments")
    print(f"Taux de compression: {len(trajectory)} -> {len(segments)} "
          f"({100 * len(segments) / len(trajectory):.1f}%)")
    
    print()
    print("Premiers segments:")
    for i, seg in enumerate(segments[:5]):
        print(f"  {i+1}. length={seg.length():.2f}, "
              f"angle={seg.angle_degrees():.1f}°, "
              f"start=({seg.start.x:.2f}, {seg.start.y:.2f}), "
              f"end=({seg.end.x:.2f}, {seg.end.y:.2f})")
    
    print()
    print("✓ Test terminé avec succès")
