from typing import Dict, List
from collections import defaultdict
import re

def reconstruct_sequences(cluster_map: Dict[str, int]) -> List[List[str]]:
    """
    Groupe les identifiants de clusters par joueur et les trie chronologiquement pour former des trajectoires.

    Args:
        cluster_map: Un dictionnaire associant les ID de segments (ex: 'P0_1') aux étiquettes de cluster (int).

    Returns:
        Une liste de séquences, où chaque séquence correspond aux clusters visités ordonnés
        pour un joueur spécifique.
        Exemple: [['5', '3', '5'], ['1', '1', '2']]
    """
    # Regroupe les tuples (index_segment, label_cluster) par player_id
    player_sequences = defaultdict(list)

    for segment_id, cluster_label in cluster_map.items():
        # Les ID de segments sont formatés comme P{player_id}_{sequence_index}
        match = re.match(r"P(\d+)_(\d+)", segment_id)
        if match:
            player_id, sequence_index = map(int, match.groups())
            # Stocker en chaîne immédiatement pour faciliter la jointure plus tard
            player_sequences[player_id].append((sequence_index, str(cluster_label)))

    # Trier les joueurs par ID pour assurer un ordre de sortie déterministe
    sorted_player_ids = sorted(player_sequences.keys())

    # Pour chaque joueur, trier ses clusters visités par index temporel
    ordered_sequences = []
    for pid in sorted_player_ids:
        # Trier par index (premier élément du tuple) et extraire le label du cluster
        sequence = [label for _, label in sorted(player_sequences[pid])]
        ordered_sequences.append(sequence)

    return ordered_sequences

def save_sequences_to_spmf(sequences: List[List[str]], output_path: str) -> None:
    """
    Écrit les séquences dans un fichier utilisant le format d'entrée SPMF.
    
    Le format SPMF nécessite que les éléments soient séparés par -1, et la séquence 
    terminée par -2.
    Exemple de ligne : 5 -1 3 -1 5 -1 -2

    Args:
        sequences: Liste de trajectoires (listes de chaînes d'ID de cluster).
        output_path: Chemin du fichier de destination.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            # Joint les éléments avec ' -1 ' et ajoute les marqueurs de fin de séquence
            line = " -1 ".join(seq) + " -1 -2\n"
            f.write(line)
