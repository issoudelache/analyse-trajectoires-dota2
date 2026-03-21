"""Tests unitaires pour dota_analytics.structures — TrajectoryPoint, Segment, Trajectory."""

import json
import math
import numpy as np
import pytest

from dota_analytics.structures import TrajectoryPoint, Segment, Trajectory


# ── helpers ─────────────────────────────────────────────────────────────────

def _pts(*coords):
    """Crée une liste de TrajectoryPoint à partir de (x, y, tick) tuples."""
    return [TrajectoryPoint(x=x, y=y, tick=t) for x, y, t in coords]


# ── TrajectoryPoint ─────────────────────────────────────────────────────────

class TestTrajectoryPoint:
    def test_to_array(self):
        p = TrajectoryPoint(x=3.0, y=4.0, tick=10)
        arr = p.to_array()
        np.testing.assert_array_equal(arr, [3.0, 4.0])
        assert arr.dtype == np.float64


# ── Segment ──────────────────────────────────────────────────────────────────

class TestSegment:
    def test_length_345(self):
        s = Segment(
            start=TrajectoryPoint(0, 0, 0),
            end=TrajectoryPoint(3, 4, 10),
        )
        assert s.length() == pytest.approx(5.0)

    def test_length_zero(self):
        p = TrajectoryPoint(1, 1, 0)
        s = Segment(start=p, end=p)
        assert s.length() == pytest.approx(0.0)

    def test_angle_horizontal(self):
        s = Segment(
            start=TrajectoryPoint(0, 0, 0),
            end=TrajectoryPoint(5, 0, 10),
        )
        assert s.angle() == pytest.approx(0.0)

    def test_angle_vertical(self):
        s = Segment(
            start=TrajectoryPoint(0, 0, 0),
            end=TrajectoryPoint(0, 5, 10),
        )
        assert s.angle() == pytest.approx(math.pi / 2)

    def test_angle_degrees(self):
        s = Segment(
            start=TrajectoryPoint(0, 0, 0),
            end=TrajectoryPoint(0, 5, 10),
        )
        assert s.angle_degrees() == pytest.approx(90.0)

    def test_vector(self):
        s = Segment(
            start=TrajectoryPoint(1, 2, 0),
            end=TrajectoryPoint(4, 6, 10),
        )
        np.testing.assert_array_equal(s.vector(), [3.0, 4.0])

    def test_to_dict_roundtrip(self):
        s = Segment(
            start=TrajectoryPoint(1.5, 2.5, 100),
            end=TrajectoryPoint(4.5, 6.5, 200),
        )
        d = s.to_dict()
        assert d["start"]["x"] == pytest.approx(1.5)
        assert d["start"]["y"] == pytest.approx(2.5)
        assert d["start"]["tick"] == 100
        assert d["end"]["x"] == pytest.approx(4.5)
        assert d["end"]["y"] == pytest.approx(6.5)
        assert d["end"]["tick"] == 200
        assert "length" in d
        assert "angle" in d
        # Doit être sérialisable en JSON
        json.dumps(d)


# ── Trajectory ───────────────────────────────────────────────────────────────

class TestTrajectory:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            Trajectory([])

    def test_len(self):
        t = Trajectory(_pts((0, 0, 0), (1, 0, 1), (2, 0, 2)))
        assert len(t) == 3

    def test_getitem(self):
        points = _pts((0, 0, 0), (3, 4, 10))
        t = Trajectory(points)
        assert t[0].x == 0
        assert t[1].x == 3

    def test_to_numpy(self):
        t = Trajectory(_pts((1, 2, 0), (3, 4, 1)))
        arr = t.to_numpy()
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])

    def test_total_distance_straight(self):
        t = Trajectory(_pts((0, 0, 0), (3, 0, 1), (6, 0, 2)))
        assert t.total_distance() == pytest.approx(6.0)

    def test_total_distance_single_point(self):
        t = Trajectory(_pts((5, 5, 0),))
        assert t.total_distance() == 0.0

    def test_duration(self):
        t = Trajectory(_pts((0, 0, 10), (1, 1, 50)))
        assert t.duration() == 40

    def test_bounding_box(self):
        t = Trajectory(_pts((1, 3, 0), (5, 1, 1), (3, 7, 2)))
        min_x, min_y, max_x, max_y = t.bounding_box()
        assert min_x == pytest.approx(1.0)
        assert min_y == pytest.approx(1.0)
        assert max_x == pytest.approx(5.0)
        assert max_y == pytest.approx(7.0)

    def test_player_id_default(self):
        t = Trajectory(_pts((0, 0, 0),))
        assert t.player_id == 0

    def test_player_id_custom(self):
        t = Trajectory(_pts((0, 0, 0),), player_id=5)
        assert t.player_id == 5
