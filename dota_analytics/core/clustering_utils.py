from geometry import GeometryUtils

class SegmentDistance:
    """Calcule la distance composite (TRACLUS) pour le clustering."""
    
    def __init__(self):
        self.geo = GeometryUtils()

    def compute_total_distance(self, s1, s2, w_perp=1.0, w_angle=1.0, w_par=1.0) -> float:
        """Calcule la distance pondérée combinant les 3 dimensions géométriques."""
        
        # 1. Distance Angulaire
        v1 = s1.vector()
        v2 = s2.vector()
        # On peut pondérer par la longueur moyenne pour donner plus de poids aux longs segments
        d_angle = self.geo.angular_distance(v1, v2) * (s1.length() + s2.length())

        # 2. Distance Parallèle (Maintenant via geometry.py)
        d_par = self.geo.parallel_distance(
            (s1.start.x, s1.start.y), (s1.end.x, s1.end.y),
            (s2.start.x, s2.start.y), (s2.end.x, s2.end.y)
        )

        # 3. Distance Perpendiculaire (TRACLUS Standard)
        # On projette toujours le plus court sur le plus long pour la cohérence
        if s1.length() > s2.length():
            base, other = s1, s2
        else:
            base, other = s2, s1
            
        d_perp_1 = self.geo.perpendicular_distance(
            (other.start.x, other.start.y), 
            (base.start.x, base.start.y), (base.end.x, base.end.y)
        )
        d_perp_2 = self.geo.perpendicular_distance(
            (other.end.x, other.end.y), 
            (base.start.x, base.start.y), (base.end.x, base.end.y)
        )
        
        # Moyenne quadratique des distances perpendiculaires
        if d_perp_1 + d_perp_2 == 0:
            d_perp = 0.0
        else:
            d_perp = (d_perp_1**2 + d_perp_2**2) / (d_perp_1 + d_perp_2)

        # Somme pondérée
        return w_perp * d_perp + w_angle * d_angle + w_par * d_par