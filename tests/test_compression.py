"""Tests unitaires pour dota_analytics.compression — MDLCompressor."""


from dota_analytics.compression import MDLCompressor
from dota_analytics.structures import TrajectoryPoint, Trajectory, Segment


def _make_trajectory(coords, player_id=0):
    """Crée une Trajectory à partir de [(x, y, tick), ...]."""
    pts = [TrajectoryPoint(x=x, y=y, tick=t) for x, y, t in coords]
    return Trajectory(pts, player_id=player_id)


class TestMDLCompressor:
    def test_straight_line_full_compression(self):
        """Points parfaitement alignés → un seul segment."""
        traj = _make_trajectory([(0, 0, i) for i in range(20)])
        # w_error=1 ; tous les points sont à distance 0 de la ligne
        c = MDLCompressor(w_error=1.0)
        segments = c.compress_player_trajectory(traj)
        assert len(segments) == 1
        assert segments[0].start.tick == 0
        assert segments[0].end.tick == 19

    def test_two_points(self):
        """Deux points → un seul segment."""
        traj = _make_trajectory([(0, 0, 0), (10, 10, 100)])
        segments = MDLCompressor(w_error=1.0).compress_player_trajectory(traj)
        assert len(segments) == 1

    def test_single_point(self):
        """Un seul point → aucun segment."""
        traj = _make_trajectory([(0, 0, 0)])
        segments = MDLCompressor(w_error=1.0).compress_player_trajectory(traj)
        assert len(segments) == 0

    def test_zigzag_no_compression(self):
        """Zig-zag prononcé avec tolérance basse → presque autant de segments que de points."""
        coords = [(i, ((-1) ** i) * 100, i) for i in range(10)]
        traj = _make_trajectory(coords)
        segments = MDLCompressor(w_error=0.01).compress_player_trajectory(traj)
        # Avec un zig-zag de 200 unités et w_error=0.01, quasiment aucune fusion
        assert len(segments) >= 8

    def test_high_tolerance_merges(self):
        """Grande tolérance → fusion agressive."""
        coords = [(i, ((-1) ** i) * 0.5, i) for i in range(10)]
        traj = _make_trajectory(coords)
        segments = MDLCompressor(w_error=10.0).compress_player_trajectory(traj)
        assert len(segments) < len(coords) - 1

    def test_segments_cover_trajectory(self):
        """Les segments doivent couvrir du premier au dernier point."""
        coords = [(i * 2, i * 3 + ((-1) ** i) * 0.5, i * 10) for i in range(15)]
        traj = _make_trajectory(coords)
        segments = MDLCompressor(w_error=2.0).compress_player_trajectory(traj)
        assert segments[0].start.tick == traj[0].tick
        assert segments[-1].end.tick == traj[-1].tick
        # Les segments sont contigus
        for i in range(len(segments) - 1):
            assert segments[i].end.tick == segments[i + 1].start.tick

    def test_reduction_ratio(self):
        """Le nombre de segments doit être inférieur au nombre de points - 1."""
        coords = [(i, i + ((-1) ** i) * 0.2, i) for i in range(50)]
        traj = _make_trajectory(coords)
        segments = MDLCompressor(w_error=5.0).compress_player_trajectory(traj)
        assert len(segments) < len(traj) - 1

    def test_all_segments_are_segments(self):
        """Vérifie que le résultat contient bien des objets Segment."""
        traj = _make_trajectory([(0, 0, 0), (5, 5, 10), (10, 0, 20)])
        segments = MDLCompressor(w_error=1.0).compress_player_trajectory(traj)
        for s in segments:
            assert isinstance(s, Segment)
