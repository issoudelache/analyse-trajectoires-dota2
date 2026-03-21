"""Tests unitaires pour dota_analytics.geometry — GeometryUtils."""

import math
import numpy as np
import pytest

from dota_analytics.geometry import GeometryUtils


# ── euclidean_distance ───────────────────────────────────────────────────────

class TestEuclideanDistance:
    def test_same_point(self):
        assert GeometryUtils.euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_horizontal(self):
        assert GeometryUtils.euclidean_distance((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_vertical(self):
        assert GeometryUtils.euclidean_distance((0, 0), (0, 4)) == pytest.approx(4.0)

    def test_diagonal_345(self):
        assert GeometryUtils.euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_negative_coords(self):
        d = GeometryUtils.euclidean_distance((-1, -1), (2, 3))
        assert d == pytest.approx(5.0)

    def test_accepts_numpy(self):
        d = GeometryUtils.euclidean_distance(np.array([1.0, 2.0]), np.array([4.0, 6.0]))
        assert d == pytest.approx(5.0)


# ── perpendicular_distance ──────────────────────────────────────────────────

class TestPerpendicularDistance:
    def test_point_on_line(self):
        """Un point sur la droite a une distance perpendiculaire de 0."""
        d = GeometryUtils.perpendicular_distance((1, 1), (0, 0), (2, 2))
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_point_above_horizontal_line(self):
        d = GeometryUtils.perpendicular_distance((5, 3), (0, 0), (10, 0))
        assert d == pytest.approx(3.0)

    def test_point_right_of_vertical_line(self):
        d = GeometryUtils.perpendicular_distance((4, 5), (0, 0), (0, 10))
        assert d == pytest.approx(4.0)

    def test_zero_length_segment(self):
        """Segment de longueur nulle → distance au point."""
        d = GeometryUtils.perpendicular_distance((3, 4), (0, 0), (0, 0))
        assert d == pytest.approx(5.0)

    def test_projection_outside_segment(self):
        """La projection tombe en-dehors du segment, mais c'est une ligne infinie."""
        d = GeometryUtils.perpendicular_distance((5, 1), (0, 0), (2, 0))
        assert d == pytest.approx(1.0)


# ── perpendicular_distances_vectorized ──────────────────────────────────────

class TestVectorizedDistances:
    def test_single_point(self):
        pts = np.array([[5.0, 3.0]])
        d = GeometryUtils.perpendicular_distances_vectorized(
            pts, np.array([0.0, 0.0]), np.array([10.0, 0.0])
        )
        assert d.shape == (1,)
        assert d[0] == pytest.approx(3.0)

    def test_multiple_points(self):
        pts = np.array([[5.0, 2.0], [5.0, 4.0], [5.0, 0.0]])
        d = GeometryUtils.perpendicular_distances_vectorized(
            pts, np.array([0.0, 0.0]), np.array([10.0, 0.0])
        )
        np.testing.assert_allclose(d, [2.0, 4.0, 0.0], atol=1e-10)

    def test_zero_length_segment(self):
        pts = np.array([[3.0, 4.0], [0.0, 0.0]])
        d = GeometryUtils.perpendicular_distances_vectorized(
            pts, np.array([0.0, 0.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_allclose(d, [5.0, 0.0], atol=1e-10)

    def test_consistent_with_scalar(self):
        """La version vectorisée doit donner le même résultat que la version scalaire."""
        pts = np.array([[1.0, 3.0], [4.0, -1.0], [7.0, 2.0]])
        a, b = np.array([0.0, 0.0]), np.array([10.0, 5.0])
        vec = GeometryUtils.perpendicular_distances_vectorized(pts, a, b)
        for i, pt in enumerate(pts):
            scalar = GeometryUtils.perpendicular_distance(pt, a, b)
            assert vec[i] == pytest.approx(scalar, abs=1e-10)


# ── angular_distance ────────────────────────────────────────────────────────

class TestAngularDistance:
    def test_same_direction(self):
        d = GeometryUtils.angular_distance((1, 0), (2, 0))
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular(self):
        d = GeometryUtils.angular_distance((1, 0), (0, 1))
        assert d == pytest.approx(math.pi / 2)

    def test_opposite(self):
        d = GeometryUtils.angular_distance((1, 0), (-1, 0))
        assert d == pytest.approx(math.pi)

    def test_zero_vector(self):
        d = GeometryUtils.angular_distance((0, 0), (1, 0))
        assert d == pytest.approx(0.0)
