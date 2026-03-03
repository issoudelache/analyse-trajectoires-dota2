"""
Implémentation de l'algorithme PrefixSpan pour la fouille de motifs séquentiels fréquents.
Compatible avec le format SPMF (Sequential Pattern Mining Framework).
Optimisé avec NumPy pour les performances mémoire (Slicing par vues).
"""

from typing import List, Tuple, Dict, Union
import numpy as np


class PrefixSpan:
    """
    Algorithme PrefixSpan de fouille de motifs séquentiels (Pattern Growth).
    """

    def __init__(self, min_support: int = 2):
        """
        Initialise le modèle avec un seuil de support minimum.

        Args:
            min_support: Nombre minimum d'occurrences pour qu'un motif soit conservé.
        """
        self.min_support = min_support
        self.results: Dict[Tuple[int, ...], int] = {}

    def load_spmf(self, filepath: str) -> List[np.ndarray]:
        """
        Charge une base de données de séquences depuis un fichier SPMF.
        Format attendu par ligne : item -1 item -1 ... -2
        """
        database = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Séparation de la ligne en symboles
                    tokens = line.split()
                    sequence = []

                    for token in tokens:
                        val = int(token)

                        # Convention SPMF : -2 indique la fin de la séquence globale
                        if val == -2:
                            break
                        # Convention SPMF : -1 indique la fin d'un itemset (ignoré en unidimensionnel)
                        if val == -1:
                            continue

                        sequence.append(val)

                    # Conversion immédiate en tableau NumPy pour la performance
                    if sequence:
                        database.append(np.array(sequence, dtype=np.int32))

        except FileNotFoundError:
            print(f"Fichier non trouvé: {filepath}")
            return []

        return database

    def _get_frequent_items(self, database: List[np.ndarray]) -> Dict[int, int]:
        """
        MÉTHODE SCOLAIRE & VECTORISÉE :
        Compte le support global de chaque item et filtre ceux dominants.
        Remplace les boucles 'for' Python par les opérations C de NumPy.
        """
        if not database:
            return {}

        # ETAPE 1 : Isoler les éléments uniques de chaque séquence (pas de doublons par ligne)
        unique_items_per_seq = [np.unique(seq) for seq in database]

        # ETAPE 2 : Concaténer en un seul immense tableau
        all_elements_flat = np.concatenate(unique_items_per_seq)

        # ETAPE 3 : Laisser NumPy scrupuleusement compter les occurrences
        items, counts = np.unique(all_elements_flat, return_counts=True)

        # ETAPE 4 : Filtrer mathématiquement via un masque booléen
        mask_frequent = counts >= self.min_support
        frequent_items = items[mask_frequent]
        frequent_counts = counts[mask_frequent]

        # ETAPE 5 : Construire un dictionnaire {item: support} pour accès direct
        return dict(zip(frequent_items.tolist(), frequent_counts.tolist()))

    def mine(
        self, database: List[Union[List[int], np.ndarray]]
    ) -> Dict[Tuple[int, ...], int]:
        """
        Point d'entrée principal de l'algorithme PrefixSpan.
        """
        self.results = {}

        if not database:
            return self.results

        # ---------------------------------------------------------
        # ETAPE A : CONVERSION EN NUMPY
        # ---------------------------------------------------------
        if isinstance(database[0], list):
            db_np = [np.array(seq, dtype=np.int32) for seq in database]
        else:
            db_np = database

        # ---------------------------------------------------------
        # ETAPE B : EXTRACTION DES PRIMITIFS (Taille 1)
        # ---------------------------------------------------------
        frequent_items_dict = self._get_frequent_items(db_np)

        # Sécurise un ordre d'exploration déterministe
        frequent_items_sorted = sorted(frequent_items_dict.keys())

        # ---------------------------------------------------------
        # ETAPE C : PATTERN GROWTH (Croissance des motifs)
        # ---------------------------------------------------------
        for item in frequent_items_sorted:
            # 1. Enregistrement du motif initial
            prefix = [item]
            support = frequent_items_dict[item]
            self.results[tuple(prefix)] = support

            # 2. Construction de la base de données projetée
            # (toutes les suites de séquences après cet item)
            projected_db = self._build_projected_database(db_np, item)

            # 3. Lancement de l'exploration en profondeur
            self._recursive_search(projected_db, prefix)

        return self.results

    def _recursive_search(self, database: List[np.ndarray], prefix: List[int]):
        """
        Étape récursive : Cherche de nouveaux items à ajouter au préfixe actuel.
        """
        # ---------------------------------------------------------
        # OPTIMISATION MATHEMATHIQUE : ÉLAGAGE PRÉCOCE (Early Pruning)
        # ---------------------------------------------------------
        # S'il y a moins de séquences projetées que le seuil min_support,
        # un enfant ne pourra jamais exister. On stoppe immédiatement la branche.
        if len(database) < self.min_support:
            return

        # 1. Trouver les extensions possibles et fréquentes pour ce préfixe
        frequent_items_dict = self._get_frequent_items(database)
        frequent_items_sorted = sorted(frequent_items_dict.keys())

        # 2. Explorer chaque nouvelle branche valide
        for item in frequent_items_sorted:
            # Nouveau motif = Ancien Motif + Nouvel item
            new_pattern = prefix + [item]
            support = frequent_items_dict[item]

            self.results[tuple(new_pattern)] = support

            # Création de la sous-base pour la génération suivante
            new_projected_db = self._build_projected_database(database, item)

            # Appel récursif
            self._recursive_search(new_projected_db, new_pattern)

    def _build_projected_database(
        self, database: List[np.ndarray], item_pivot: int
    ) -> List[np.ndarray]:
        """
        Construit une base projetée.
        """
        projected_db = []

        for seq in database:
            # ÉTAPE 1 : Trouver toutes les positions de l'item pivot dans la séquence
            # np.where est vectorisé (beaucoup plus rapide qu'une boucle 'in' Python)
            indices = np.where(seq == item_pivot)[0]

            if len(indices) > 0:
                # ÉTAPE 2 : Règle PrefixSpan -> Prendre la première apparition
                first_idx = indices[0]

                # ÉTAPE 3 : SLICING MEMOIRE (Aucune donnée copiée, c'est virtuel)
                # On recupère tout ce qu'il y a après le pivot
                suffix = seq[first_idx + 1 :]

                # ÉTAPE 4 : Ne conserver que les suffixes comportant des éléments restants
                if len(suffix) > 0:
                    projected_db.append(suffix)

        return projected_db

    def save_results_to_spmf(self, output_path: str):
        """Sauvegarde les résultats formatés : item -1 ... #SUP: N"""
        with open(output_path, "w", encoding="utf-8") as f:
            # Rangement cosmétique : par support décroissant, puis par longueur du motif
            sorted_patterns = sorted(
                self.results.items(), key=lambda x: (-x[1], len(x[0]))
            )

            for pattern, support in sorted_patterns:
                # Construction d'un itemset 1D SPMF
                items_str = " ".join([f"{item} -1" for item in pattern])
                line = f"{items_str} #SUP: {support}\n"
                f.write(line)
