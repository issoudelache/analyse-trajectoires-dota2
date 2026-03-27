"""
Implémentation de l'algorithme PrefixSpan pour la fouille de motifs séquentiels fréquents.
Compatible avec le format SPMF (Sequential Pattern Mining Framework).
Optimisé avec NumPy pour les performances mémoire (Slicing par vues).
"""

import time
from typing import List, Tuple, Dict, Union, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np


class PrefixSpan:
    """
    Algorithme PrefixSpan de fouille de motifs séquentiels (Pattern Growth).
    Version optimisée avec parallélisation et caching.
    """

    def __init__(self, min_support: Union[int, float] = 0.1, max_length: int = 10):
        """
        Initialise le modèle avec un seuil de support minimum.

        Args:
            min_support: Seuil de support minimum.
                - Si float (0.0 < x <= 1.0) : pourcentage de la base de données.
                  Ex: 0.1 = 10% des séquences doivent contenir le motif.
                - Si int (x >= 1) : nombre absolu de séquences.
                  Ex: 5 = le motif doit apparaître dans au moins 5 séquences.
            max_length: Longueur maximale des motifs extraits (évite la récursion infinie).
        """
        self._min_support_param = min_support
        self.min_support = min_support  # Sera recalculé dans mine() si pourcentage
        self.max_length = max_length
        self.results: Dict[Tuple[int, ...], int] = {}
        self._progress_callback: Optional[Callable[[int, int, float, int], None]] = None
        self._start_time: float = 0.0

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

                    tokens = line.split()
                    sequence = []

                    for token in tokens:
                        val = int(token)
                        if val == -2:
                            break
                        if val == -1:
                            continue
                        sequence.append(val)

                    if sequence:
                        database.append(np.array(sequence, dtype=np.int32))

        except FileNotFoundError:
            print(f"Fichier non trouvé: {filepath}")
            return []

        return database

    def _get_frequent_items_fast(self, database: List[np.ndarray]) -> Dict[int, int]:
        """
        Version optimisée: compte les items fréquents avec moins d'allocations.
        """
        if not database:
            return {}

        # Utiliser un dictionnaire pour compter directement
        item_counts: Dict[int, int] = {}

        for seq in database:
            # set() est plus rapide que np.unique pour les petites séquences
            seen = set()
            for item in seq:
                if item not in seen:
                    seen.add(item)
                    item_counts[item] = item_counts.get(item, 0) + 1

        # Filtrer les items fréquents
        return {k: v for k, v in item_counts.items() if v >= self.min_support}

    def mine(
        self,
        database: List[Union[List[int], np.ndarray]],
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
        parallel: bool = True,
    ) -> Dict[Tuple[int, ...], int]:
        """
        Point d'entrée principal de l'algorithme PrefixSpan.

        Args:
            database: Liste de séquences à analyser
            progress_callback: Callback optionnel (current, total, elapsed_sec, num_patterns)
            parallel: Utiliser le multiprocessing pour les items de niveau 1
        """
        self.results = {}
        self._progress_callback = progress_callback
        self._start_time = time.time()

        if not database:
            return self.results

        # Calcul du support absolu si pourcentage fourni
        n_sequences = len(database)
        if isinstance(self._min_support_param, float) and 0.0 < self._min_support_param <= 1.0:
            # Pourcentage : convertir en valeur absolue
            self.min_support = max(1, int(n_sequences * self._min_support_param))
        else:
            # Valeur absolue
            self.min_support = max(1, int(self._min_support_param))

        # Conversion en numpy si nécessaire
        if isinstance(database[0], list):
            db_np = [np.array(seq, dtype=np.int32) for seq in database]
        else:
            db_np = list(database)

        # Extraction des items fréquents de taille 1
        frequent_items_dict = self._get_frequent_items_fast(db_np)
        frequent_items_sorted = sorted(frequent_items_dict.keys())
        total_items = len(frequent_items_sorted)

        if total_items == 0:
            return self.results

        # Décider si on parallélise (seulement si assez d'items et pas de callback GUI)
        use_parallel = parallel and total_items >= 4 and mp.cpu_count() > 1

        if use_parallel:
            self._mine_parallel(db_np, frequent_items_dict, frequent_items_sorted)
        else:
            self._mine_sequential(db_np, frequent_items_dict, frequent_items_sorted)

        return self.results

    def _mine_sequential(
        self,
        db_np: List[np.ndarray],
        frequent_items_dict: Dict[int, int],
        frequent_items_sorted: List[int],
    ):
        """Fouille séquentielle avec callbacks de progression."""
        total_items = len(frequent_items_sorted)
        callback_interval = max(1, total_items // 20)  # Callback tous les 5%

        for idx, item in enumerate(frequent_items_sorted):
            prefix = [item]
            support = frequent_items_dict[item]
            self.results[tuple(prefix)] = support

            projected_db = self._build_projected_database_fast(db_np, item)
            self._recursive_search(projected_db, prefix)

            # Callback moins fréquent pour réduire l'overhead GUI
            if self._progress_callback and (idx % callback_interval == 0 or idx == total_items - 1):
                elapsed = time.time() - self._start_time
                self._progress_callback(idx + 1, total_items, elapsed, len(self.results))

    def _mine_parallel(
        self,
        db_np: List[np.ndarray],
        frequent_items_dict: Dict[int, int],
        frequent_items_sorted: List[int],
    ):
        """Fouille parallèle des items de niveau 1."""
        total_items = len(frequent_items_sorted)
        n_workers = min(mp.cpu_count(), 4, total_items)

        # Convertir en listes pour la sérialisation
        db_lists = [seq.tolist() for seq in db_np]

        # Préparer les tâches
        tasks = [
            (item, frequent_items_dict[item], db_lists, self.min_support, self.max_length)
            for item in frequent_items_sorted
        ]

        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_mine_single_item, *task): task[0] for task in tasks}

            for future in as_completed(futures):
                item = futures[future]
                try:
                    partial_results = future.result()
                    self.results.update(partial_results)
                except Exception as e:
                    print(f"Erreur pour item {item}: {e}")

                completed += 1
                if self._progress_callback:
                    elapsed = time.time() - self._start_time
                    self._progress_callback(completed, total_items, elapsed, len(self.results))

    def _recursive_search(self, database: List[np.ndarray], prefix: List[int]):
        """Étape récursive optimisée."""
        # Élagage précoce
        if len(database) < self.min_support:
            return

        if len(prefix) >= self.max_length:
            return

        # Trouver les extensions fréquentes
        frequent_items_dict = self._get_frequent_items_fast(database)

        for item in sorted(frequent_items_dict.keys()):
            new_pattern = prefix + [item]
            support = frequent_items_dict[item]
            self.results[tuple(new_pattern)] = support

            new_projected_db = self._build_projected_database_fast(database, item)
            self._recursive_search(new_projected_db, new_pattern)

    def _build_projected_database_fast(
        self, database: List[np.ndarray], item_pivot: int
    ) -> List[np.ndarray]:
        """
        Version optimisée de la projection.
        Utilise une recherche plus efficace pour la première occurrence.
        """
        projected_db = []

        for seq in database:
            # Recherche de la première occurrence
            # Pour les arrays numpy, on utilise une approche hybride
            if len(seq) < 20:
                # Pour les petites séquences, boucle simple plus rapide
                idx = -1
                for i, val in enumerate(seq):
                    if val == item_pivot:
                        idx = i
                        break
            else:
                # Pour les grandes séquences, numpy est plus efficace
                matches = np.where(seq == item_pivot)[0]
                idx = matches[0] if len(matches) > 0 else -1

            if idx >= 0 and idx < len(seq) - 1:
                suffix = seq[idx + 1:]
                if len(suffix) > 0:
                    projected_db.append(suffix)

        return projected_db

    def save_results_to_spmf(self, output_path: str):
        """Sauvegarde les résultats formatés : item -1 ... #SUP: N"""
        with open(output_path, "w", encoding="utf-8") as f:
            sorted_patterns = sorted(
                self.results.items(), key=lambda x: (-x[1], len(x[0]))
            )

            for pattern, support in sorted_patterns:
                items_str = " ".join([f"{item} -1" for item in pattern])
                line = f"{items_str} #SUP: {support}\n"
                f.write(line)


# === Fonction externe pour le multiprocessing ===

def _mine_single_item(
    item: int,
    support: int,
    db_lists: List[List[int]],
    min_support: int,
    max_length: int,
) -> Dict[Tuple[int, ...], int]:
    """
    Mine un seul item de niveau 1 (pour parallélisation).
    Fonction externe car les méthodes d'instance ne sont pas picklables.
    """
    results = {(item,): support}

    # Convertir en numpy
    database = [np.array(seq, dtype=np.int32) for seq in db_lists]

    # Construire la base projetée
    projected_db = []
    for seq in database:
        idx = -1
        for i, val in enumerate(seq):
            if val == item:
                idx = i
                break
        if idx >= 0 and idx < len(seq) - 1:
            suffix = seq[idx + 1:]
            if len(suffix) > 0:
                projected_db.append(suffix)

    # Fouille récursive
    _recursive_search_standalone(projected_db, [item], results, min_support, max_length)

    return results


def _get_frequent_items_standalone(database: List[np.ndarray], min_support: int) -> Dict[int, int]:
    """Version standalone pour le multiprocessing."""
    if not database:
        return {}

    item_counts: Dict[int, int] = {}
    for seq in database:
        seen = set()
        for item in seq:
            if item not in seen:
                seen.add(item)
                item_counts[item] = item_counts.get(item, 0) + 1

    return {k: v for k, v in item_counts.items() if v >= min_support}


def _recursive_search_standalone(
    database: List[np.ndarray],
    prefix: List[int],
    results: Dict[Tuple[int, ...], int],
    min_support: int,
    max_length: int,
):
    """Version standalone de la recherche récursive pour le multiprocessing."""
    if len(database) < min_support:
        return

    if len(prefix) >= max_length:
        return

    frequent_items_dict = _get_frequent_items_standalone(database, min_support)

    for item in sorted(frequent_items_dict.keys()):
        new_pattern = prefix + [item]
        support = frequent_items_dict[item]
        results[tuple(new_pattern)] = support

        # Projection
        new_projected_db = []
        for seq in database:
            idx = -1
            for i, val in enumerate(seq):
                if val == item:
                    idx = i
                    break
            if idx >= 0 and idx < len(seq) - 1:
                suffix = seq[idx + 1:]
                if len(suffix) > 0:
                    new_projected_db.append(suffix)

        _recursive_search_standalone(new_projected_db, new_pattern, results, min_support, max_length)
