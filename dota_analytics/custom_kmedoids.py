"""
Algorithme des K-Médoïdes (Partitioning Around Medoids - PAM).
Implémentation maison vectorisée avec Numpy uniquement.
"""

import numpy as np


class CustomKMedoids:
    """
    Algorithme des K-Médoïdes (Partitioning Around Medoids - PAM).

    Contrairement aux K-Means qui utilise la moyenne (centroïde), PAM choisit
    comme représentant de chaque cluster un point RÉEL du jeu de données
    (le médoïde), ce qui le rend plus robuste aux outliers et compatible
    avec n'importe quelle métrique de distance (même non euclidienne).

    Ici, l'entrée est directement une matrice de distances pré-calculée D (N×N),
    ce qui permet de réutiliser la matrice TRACLUS déjà construite dans
    clustering.py (distance perpendiculaire + parallèle + angulaire).
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: int = None):
        """
        Args:
            n_clusters   : Nombre de clusters K voulus.
            max_iter     : Nombre maximum d'itérations avant arrêt forcé.
            random_state : Graine aléatoire pour la reproductibilité des résultats.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        # Attributs résultats (remplis par fit())
        self.labels_ = None           # shape (N,) : cluster de chaque point
        self.medoid_indices_ = None   # shape (K,) : indices des K médoïdes

    def fit(self, D: np.ndarray) -> "CustomKMedoids":
        """
        Ajuste le modèle sur une matrice de distances pré-calculée.

        Args:
            D : Matrice de distances carrée (N, N), symétrique, diagonale nulle.
                D[i, j] = distance entre le segment i et le segment j.

        Returns:
            self (pour permettre le chaînage : km.fit(D).labels_)
        """
        N = D.shape[0]
        rng = np.random.RandomState(self.random_state)

        # ----------------------------------------------------------------
        # ÉTAPE 1 — INITIALISATION ALÉATOIRE DES MÉDOÏDES
        # On tire K indices uniques au hasard parmi les N points.
        # Ces K points deviennent les représentants initiaux de chaque cluster.
        # ----------------------------------------------------------------
        medoids = rng.choice(N, size=self.n_clusters, replace=False)

        for iteration in range(self.max_iter):

            # ------------------------------------------------------------
            # ÉTAPE 2a — ASSIGNATION (vectorisée, O(N·K))
            # Pour chaque point i, on regarde sa distance à chacun des K
            # médoïdes courants, et on l'assigne au médoïde le plus proche.
            #
            # D[:, medoids] est une sous-matrice (N, K) :
            #   colonne k = distances de tous les N points vers le médoïde k.
            # np.argmin sur axis=1 donne, pour chaque ligne (point), la colonne
            # (l'index local dans medoids) du médoïde le plus proche.
            # ------------------------------------------------------------
            labels = np.argmin(D[:, medoids], axis=1)  # shape (N,)

            # ------------------------------------------------------------
            # ÉTAPE 2b — MISE À JOUR DES MÉDOÏDES (O(N²/K) amorti)
            # Pour chaque cluster c, on cherche le point réel qui minimise
            # la somme de ses distances vers tous les autres membres du cluster.
            # C'est le critère PAM : minimiser l'inertie intra-cluster.
            # ------------------------------------------------------------
            new_medoids = medoids.copy()

            for c in range(self.n_clusters):
                # Indices globaux de tous les points du cluster c
                cluster_indices = np.where(labels == c)[0]

                if len(cluster_indices) == 0:
                    # Cluster vide : on garde le médoïde actuel (cas rare)
                    continue

                # Sous-matrice carrée des distances intra-cluster.
                # np.ix_ construit un indexage croisé : sub_D[i, j] =
                # distance entre le i-ème et le j-ème membre du cluster c.
                sub_D = D[np.ix_(cluster_indices, cluster_indices)]

                # Somme des distances de chaque membre vers tous les autres.
                # np.sum(sub_D, axis=1)[i] = coût total si le point i est médoïde.
                total_distances = np.sum(sub_D, axis=1)

                # Le candidat médoïde est celui qui minimise ce coût global.
                best_local_idx = np.argmin(total_distances)

                # Conversion de l'index local (dans cluster_indices) → index global
                new_medoids[c] = cluster_indices[best_local_idx]

            # ------------------------------------------------------------
            # ÉTAPE 2c — TEST DE CONVERGENCE
            # Si aucun médoïde n'a changé, l'algorithme a convergé :
            # une nouvelle itération donnerait exactement le même résultat.
            # ------------------------------------------------------------
            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                print(f"   Convergence atteinte à l'itération {iteration + 1}.")
                break

            medoids = new_medoids

        else:
            print(f"   ⚠️  Max itérations ({self.max_iter}) atteint avant convergence.")

        # ----------------------------------------------------------------
        # RÉSULTATS FINAUX
        # On recalcule les labels finaux avec les médoïdes convergés.
        # labels_[i] = index local du cluster (0..K-1)
        # medoid_indices_[k] = index global du représentant du cluster k
        # ----------------------------------------------------------------
        self.medoid_indices_ = medoids                      # shape (K,)
        self.labels_ = np.argmin(D[:, medoids], axis=1)    # shape (N,)

        return self
