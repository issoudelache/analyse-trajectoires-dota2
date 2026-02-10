"""Structures de données pour les trajectoires et export JSON."""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import numpy as np
import math
import json


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


# ============================================================================
# Export JSON
# ============================================================================

class JSONExporter:
    """Exporteur JSON pour les résultats de compression multi-joueurs.
    
    Format de sortie:
    {
        "match_id": "3841995251",
        "w_error": 10.0,
        "compression_info": {...},
        "players": [
            {
                "player_id": 0,
                "num_segments": 50,
                "compression_rate": 95.0,
                "segments": [...]
            }
        ]
    }
    """
    
    @staticmethod
    def export_match(results: Dict[int, List[Segment]], 
                     match_id: str,
                     output_path: Path,
                     w_error: float = 10.0,
                     original_points: Dict[int, int] = None) -> Path:
        """Exporte les résultats d'un match complet en JSON.
        
        Args:
            results: Dictionnaire {player_id: [segments]} pour les 10 joueurs
            match_id: Identifiant du match
            output_path: Chemin du fichier JSON de sortie
            w_error: Valeur w_error utilisée pour la compression
            original_points: Dict {player_id: nb_points} avant compression
            
        Returns:
            Path du fichier créé
        """
        # Construire la structure JSON
        players_data = []
        
        for player_id in sorted(results.keys()):
            segments = results[player_id]
            
            # Calculer les statistiques
            num_segments = len(segments)
            num_original = int(original_points.get(player_id, 0)) if original_points else 0
            compression_rate = 0.0
            if num_original > 0:
                compression_rate = (1 - num_segments / num_original) * 100
            
            # Convertir les segments en dictionnaires
            segments_data = [seg.to_dict() for seg in segments]
            
            player_data = {
                "player_id": int(player_id),
                "num_segments": int(num_segments),
                "num_original_points": int(num_original),
                "compression_rate": round(float(compression_rate), 2),
                "segments": segments_data
            }
            
            players_data.append(player_data)
        
        # Structure complète
        output_data = {
            "match_id": str(match_id),
            "w_error": float(w_error),
            "compression_info": {
                "total_players": int(len(results)),
                "timestamp": datetime.now().isoformat(),
                "algorithm": "MDL Greedy",
                "formula": "Cost = L(H) + w_error * L(D|H)"
            },
            "players": players_data
        }
        
        # Créer le dossier parent si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Écrire le fichier JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    @staticmethod
    def export_batch(matches_results: Dict[str, Dict[int, List[Segment]]],
                     output_dir: Path,
                     w_error: float = 10.0,
                     original_points_per_match: Dict[str, Dict[int, int]] = None) -> List[Path]:
        """Exporte plusieurs matchs en JSON.
        
        Args:
            matches_results: Dict {match_id: {player_id: [segments]}}
            output_dir: Dossier de sortie
            w_error: Valeur w_error utilisée
            original_points_per_match: Dict {match_id: {player_id: nb_points}}
            
        Returns:
            Liste des paths créés
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        created_files = []
        
        for match_id, results in matches_results.items():
            output_path = output_dir / f"{match_id}_compressed.json"
            
            original_points = None
            if original_points_per_match:
                original_points = original_points_per_match.get(match_id)
            
            created_file = JSONExporter.export_match(
                results=results,
                match_id=match_id,
                output_path=output_path,
                w_error=w_error,
                original_points=original_points
            )
            
            created_files.append(created_file)
        
        return created_files


def export_match(results: Dict[int, List[Segment]], 
                 match_id: str,
                 output_path: Path,
                 w_error: float = 10.0,
                 original_points: Dict[int, int] = None) -> Path:
    """Fonction utilitaire pour exporter un match.
    
    Args:
        results: Dictionnaire {player_id: [segments]} pour les 10 joueurs
        match_id: Identifiant du match
        output_path: Chemin du fichier JSON de sortie
        w_error: Valeur w_error utilisée pour la compression
        original_points: Dict {player_id: nb_points} avant compression
        
    Returns:
        Path du fichier créé
    """
    return JSONExporter.export_match(results, match_id, output_path, w_error, original_points)
