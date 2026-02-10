
"""
Implémentation de l'algorithme PrefixSpan pour la fouille de motifs séquentiels fréquents.
Compatible avec le format SPMF (Sequential Pattern Mining Framework).
Optimisé avec NumPy pour les performances mémoire (Slicing par vues).
"""

from typing import List, Tuple, Dict, Union
import collections
import numpy as np

class PrefixSpan:
    """
    Miner de motifs séquentiels fréquents utilisant l'approche de projection de motifs (Pattern Growth).
    Utilise des tableaux NumPy pour éviter les copies mémoire coûteuses lors des projections.
    """

    def __init__(self, min_support: int = 2):
        """
        Args:
            min_support: Nombre minimum d'occurrences pour qu'un motif soit considéré fréquent.
        """
        self.min_support = min_support
        self.results: Dict[Tuple[int, ...], int] = {} 

    def load_spmf(self, filepath: str) -> List[np.ndarray]:
        """
        Charge une base de données de séquences depuis un fichier SPMF.
        Retourne une liste de tableaux NumPy.
        """
        database = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
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
                        # Conversion immédiate en NumPy array (int32 suffit généralement)
                        database.append(np.array(sequence, dtype=np.int32))
        except FileNotFoundError:
            print(f"Fichier non trouvé: {filepath}")
            return []
            
        return database

    def mine(self, database: List[Union[List[int], np.ndarray]]) -> Dict[Tuple[int, ...], int]:
        """
        Exécute l'algorithme PrefixSpan. Convertit automatiquement en NumPy si nécessaire.
        """
        self.results = {}
        
        # Conversion préventive en liste de numpy arrays si ce sont des listes python
        # Cela permet de bénéficier des "vues" mémoire dès le début
        if database and isinstance(database[0], list):
             db_working = [np.array(seq, dtype=np.int32) for seq in database]
        else:
             db_working = database

        # 1. Compter les items fréquents (L1)
        item_counts = collections.defaultdict(int)
        
        # np.unique est très rapide
        for seq in db_working:
            unique_items = np.unique(seq)
            for item in unique_items:
                item_counts[item] += 1
        
        frequent_items = [
            item for item, count in item_counts.items() 
            if count >= self.min_support
        ]
        frequent_items.sort() # Ordre déterministe
        
        # 2. Récursion
        for item in frequent_items:
            prefix = [item]
            self.results[tuple(prefix)] = item_counts[item]
            
            projected_db = self._build_projected_database(db_working, item)
            self._recursive_search(projected_db, prefix)
            
        return self.results

    def _recursive_search(self, database: List[np.ndarray], prefix: List[int]):
        """Étape récursive avec des tableaux NumPy."""
        if not database:
            return

        item_counts = collections.defaultdict(int)
        for seq in database:
            # np.unique sur chaque suffixe projeté
            unique_items = np.unique(seq)
            for item in unique_items:
                item_counts[item] += 1
                
        frequent_items = [
            item for item, count in item_counts.items() 
            if count >= self.min_support
        ]
        frequent_items.sort()
        
        for item in frequent_items:
            new_pattern = prefix + [item]
            self.results[tuple(new_pattern)] = item_counts[item]
            
            new_projected_db = self._build_projected_database(database, item)
            
            if new_projected_db:
                self._recursive_search(new_projected_db, new_pattern)

    def _build_projected_database(self, database: List[np.ndarray], item_pivot: int) -> List[np.ndarray]:
        """
        Crée une base projetée.
        Grâce aux tableaux NumPy, 'seq[idx+1:]' est une VUE et non une copie.
        """
        projected_db = []
        for seq in database:
            # np.where est vectorisé (plus rapide que .index() sur de grandes séquences)
            indices = np.where(seq == item_pivot)[0]
            
            if indices.size > 0:
                # On prend la première occurrence
                idx = indices[0]
                
                # Slicing NumPy = Vue (Performance++)
                suffix = seq[idx+1:]
                
                # On ne garde que les suffixes non vides
                if suffix.size > 0:
                    projected_db.append(suffix)
                
        return projected_db

    def save_results_to_spmf(self, output_path: str):

        """Sauvegarde les résultats au format: pattern -1 #SUP: support"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Tri des résultats par support décroissant, puis par longueur
            sorted_patterns = sorted(
                self.results.items(), 
                key=lambda x: (-x[1], len(x[0]))
            )
            
            for pattern, support in sorted_patterns:
                # Construction ligne: item -1 item -1 ... #SUP: N
                items_str = " ".join([f"{item} -1" for item in pattern])
                line = f"{items_str} #SUP: {support}\n"
                f.write(line)
 