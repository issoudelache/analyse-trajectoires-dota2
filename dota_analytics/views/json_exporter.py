"""Exportation des résultats de compression en JSON.

Format de sortie:
{
    "match_id": "3841995251",
    "w_error": 10.0,
    "compression_info": {
        "total_players": 10,
        "timestamp": "2026-01-20T..."
    },
    "players": [
        {
            "player_id": 0,
            "num_segments": 50,
            "num_original_points": 1000,
            "compression_rate": 95.0,
            "segments": [...]
        },
        ...
    ]
}
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from ..models.structures import Segment


class JSONExporter:
    """Exporteur JSON pour les résultats de compression multi-joueurs."""
    
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
