from typing import Dict, List
from collections import defaultdict
import re


def reconstruct_sequences(match_clusters: Dict[str, Dict[str, int]]) -> List[List[str]]:
    """
    Reconstruit une séquence de clusters par joueur par match.

    Args:
        match_clusters: Dictionnaire {match_id: {seg_id: cluster_label}}
                         où seg_id est formaté comme 'P{player_id}_{index}'.

    Returns:
        Une liste de séquences (une par joueur par match), triées chronologiquement.
        Exemple: [['5', '3', '5'], ['1', '1', '2']]
    """
    ordered_sequences = []

    # Itérer sur chaque match séparément pour ne pas mélanger les matchs
    for match_id in sorted(match_clusters.keys()):
        segments = match_clusters[match_id]

        # Regroupe les tuples (index_segment, label_cluster) par player_id
        player_sequences = defaultdict(list)

        for segment_id, cluster_label in segments.items():
            # Les ID de segments sont formatés comme P{player_id}_{sequence_index}
            m = re.match(r"P(\d+)_(\d+)", segment_id)
            if m:
                player_id, sequence_index = map(int, m.groups())
                player_sequences[player_id].append((sequence_index, str(cluster_label)))

        # Pour chaque joueur du match, trier ses segments par index temporel
        for pid in sorted(player_sequences.keys()):
            sequence = [label for _, label in sorted(player_sequences[pid])]
            if sequence:  # Ne pas ajouter les séquences vides
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
    with open(output_path, "w", encoding="utf-8") as f:
        for seq in sequences:
            # Joint les éléments avec ' -1 ' et ajoute les marqueurs de fin de séquence
            line = " -1 ".join(seq) + " -1 -2\n"
            f.write(line)
