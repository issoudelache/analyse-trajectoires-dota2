"""Tests unitaires pour dota_analytics.custom_ap — CustomAffinityPropagation."""

import numpy as np
import pytest

from dota_analytics.custom_ap import CustomAffinityPropagation


def _make_similarity_matrix(centers, n_per_cluster=20, noise=0.5, seed=42):
    """Génère une matrice de similarité à partir de clusters synthétiques.

    Crée des points groupés autour de `centers`, puis calcule S = -distance².
    """
    rng = np.random.RandomState(seed)
    points = []
    for cx, cy in centers:
        pts = rng.normal(loc=[cx, cy], scale=noise, size=(n_per_cluster, 2))
        points.append(pts)
    X = np.vstack(points)
    # Similarité = négation de la distance euclidienne au carré
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    S = -np.sum(diff ** 2, axis=2)
    return S, X


# ── Instanciation ────────────────────────────────────────────────────────────


class TestAPInit:
    def test_defaults(self):
        ap = CustomAffinityPropagation()
        assert ap.damping == 0.9
        assert ap.max_iter == 200
        assert ap.convergence_iter == 15

    def test_custom_params(self):
        ap = CustomAffinityPropagation(damping=0.7, max_iter=500)
        assert ap.damping == 0.7
        assert ap.max_iter == 500


# ── Fit basique ──────────────────────────────────────────────────────────────


class TestAPFit:
    def test_returns_self(self):
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=10)
        ap = CustomAffinityPropagation(verbose=False)
        result = ap.fit(S)
        assert result is ap

    def test_labels_shape(self):
        n_per = 15
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=n_per)
        ap = CustomAffinityPropagation(verbose=False)
        ap.fit(S)
        assert ap.labels_ is not None
        assert len(ap.labels_) == n_per * 2

    def test_labels_are_contiguous(self):
        """Les labels doivent être remappés 0,1,2,...,k-1."""
        S, _ = _make_similarity_matrix([(0, 0), (10, 10), (20, 0)], n_per_cluster=15)
        ap = CustomAffinityPropagation(verbose=False)
        ap.fit(S)
        unique = np.unique(ap.labels_)
        # Labels consécutifs à partir de 0
        np.testing.assert_array_equal(unique, np.arange(len(unique)))

    def test_exemplars_exist(self):
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=15)
        ap = CustomAffinityPropagation(verbose=False)
        ap.fit(S)
        assert ap.cluster_centers_indices_ is not None
        assert len(ap.cluster_centers_indices_) > 0


# ── Qualité du clustering ───────────────────────────────────────────────────


class TestAPClustering:
    def test_two_well_separated_clusters(self):
        """Deux groupes très séparés → exactement 2 clusters."""
        S, _ = _make_similarity_matrix(
            [(0, 0), (100, 100)], n_per_cluster=20, noise=0.5
        )
        # Préférence = médiane des similarités (encourage peu de clusters)
        pref = np.median(S)
        np.fill_diagonal(S, pref)
        ap = CustomAffinityPropagation(damping=0.9, max_iter=300, verbose=False)
        ap.fit(S)
        k = len(np.unique(ap.labels_))
        assert k == 2

    def test_cluster_assignment_correctness(self):
        """Chaque point doit être assigné au cluster le plus proche."""
        n_per = 20
        S, X = _make_similarity_matrix(
            [(0, 0), (100, 100)], n_per_cluster=n_per, noise=0.5
        )
        pref = np.median(S)
        np.fill_diagonal(S, pref)
        ap = CustomAffinityPropagation(damping=0.9, max_iter=300, verbose=False)
        ap.fit(S)
        # Les 20 premiers points (autour de (0,0)) doivent avoir le même label
        labels_group1 = ap.labels_[:n_per]
        labels_group2 = ap.labels_[n_per:]
        assert len(np.unique(labels_group1)) == 1
        assert len(np.unique(labels_group2)) == 1
        assert labels_group1[0] != labels_group2[0]

    def test_three_clusters(self):
        """Trois groupes séparés → 3 clusters."""
        S, _ = _make_similarity_matrix(
            [(0, 0), (50, 0), (0, 50)], n_per_cluster=25, noise=1.0
        )
        pref = np.median(S)
        np.fill_diagonal(S, pref)
        ap = CustomAffinityPropagation(damping=0.9, max_iter=300, verbose=False)
        ap.fit(S)
        k = len(np.unique(ap.labels_))
        assert k == 3


# ── Cas limites ──────────────────────────────────────────────────────────────


class TestAPEdgeCases:
    def test_single_point(self):
        """Un seul point → 1 cluster ou label -1."""
        S = np.array([[0.0]])
        ap = CustomAffinityPropagation(verbose=False, max_iter=50)
        ap.fit(S)
        assert len(ap.labels_) == 1

    def test_two_identical_points(self):
        """Deux points identiques → même cluster."""
        S = np.array([[0.0, 0.0], [0.0, 0.0]])
        ap = CustomAffinityPropagation(verbose=False, max_iter=50)
        ap.fit(S)
        assert len(ap.labels_) == 2

    def test_damping_effect(self):
        """Un damping plus élevé ne doit pas crasher."""
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=10)
        ap = CustomAffinityPropagation(damping=0.95, verbose=False)
        ap.fit(S)
        assert ap.labels_ is not None


# ── Convergence ──────────────────────────────────────────────────────────────


class TestAPConvergence:
    def test_n_iter_set(self):
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=10)
        ap = CustomAffinityPropagation(verbose=False, max_iter=200)
        ap.fit(S)
        assert ap.n_iter_ >= 0
        assert ap.n_iter_ < 200  # devrait converger avant le max

    def test_max_iter_reached(self):
        """Avec très peu d'itérations, on atteint le max."""
        S, _ = _make_similarity_matrix([(0, 0), (10, 10)], n_per_cluster=10)
        ap = CustomAffinityPropagation(verbose=False, max_iter=2, convergence_iter=15)
        ap.fit(S)
        assert ap.n_iter_ == 1  # 0-indexed, 2 iterations = index 1
