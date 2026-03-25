"""Tests unitaires pour dota_analytics.mining — PrefixSpan."""

import numpy as np
import pytest

from dota_analytics.mining import PrefixSpan


# ── Construction de bases de test ────────────────────────────────────────────


def _db(seqs):
    """Convertit une liste de listes en base numpy."""
    return [np.array(s, dtype=np.int32) for s in seqs]


# ── Instanciation ────────────────────────────────────────────────────────────


class TestPrefixSpanInit:
    def test_defaults(self):
        ps = PrefixSpan()
        assert ps.min_support == 2
        assert ps.max_length == 10

    def test_custom(self):
        ps = PrefixSpan(min_support=5, max_length=3)
        assert ps.min_support == 5
        assert ps.max_length == 3


# ── Base vide / triviale ─────────────────────────────────────────────────────


class TestPrefixSpanEmpty:
    def test_empty_database(self):
        ps = PrefixSpan(min_support=1)
        result = ps.mine([], parallel=False)
        assert result == {}

    def test_single_item(self):
        ps = PrefixSpan(min_support=1, max_length=3)
        result = ps.mine(_db([[1]]), parallel=False)
        assert (1,) in result
        assert result[(1,)] == 1

    def test_min_support_filters(self):
        """Un item qui n'atteint pas le support minimum est exclu."""
        ps = PrefixSpan(min_support=3, max_length=5)
        result = ps.mine(_db([[1, 2], [1, 3], [2, 3]]), parallel=False)
        # item 1 et 3 apparaissent 2 fois chacun → en dessous de 3
        # item 2 apparaît 2 fois → aussi sous 3
        assert result == {}


# ── Motifs connus ────────────────────────────────────────────────────────────


class TestPrefixSpanPatterns:
    @pytest.fixture
    def simple_db(self):
        """Base classique : 5 séquences, motifs attendus bien définis."""
        return _db([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [2, 3, 4],
            [1, 3],
        ])

    def test_singleton_supports(self, simple_db):
        ps = PrefixSpan(min_support=2, max_length=5)
        result = ps.mine(simple_db, parallel=False)
        # item 1 dans seq 0,1,2,4 → 4
        assert result[(1,)] == 4
        # item 2 dans seq 0,1,2,3 → 4
        assert result[(2,)] == 4
        # item 3 dans seq 0,2,3,4 → 4
        assert result[(3,)] == 4

    def test_bigram_12(self, simple_db):
        ps = PrefixSpan(min_support=2, max_length=5)
        result = ps.mine(simple_db, parallel=False)
        # 1→2 dans seq 0,1,2 → 3
        assert result[(1, 2)] == 3

    def test_trigram_123(self, simple_db):
        ps = PrefixSpan(min_support=2, max_length=5)
        result = ps.mine(simple_db, parallel=False)
        # 1→2→3 dans seq 0,2 → 2
        assert result[(1, 2, 3)] == 2

    def test_max_length_respected(self, simple_db):
        ps = PrefixSpan(min_support=1, max_length=2)
        result = ps.mine(simple_db, parallel=False)
        # Aucun motif de longueur > 2
        for pattern in result:
            assert len(pattern) <= 2


# ── Cohérence séquentiel vs parallèle ───────────────────────────────────────


class TestPrefixSpanParallel:
    def test_same_results(self):
        """Le mode parallèle et séquentiel doivent donner les mêmes motifs."""
        db = _db([
            [1, 2, 3, 1],
            [2, 1, 3, 2],
            [1, 3, 2, 1],
            [3, 1, 2, 3],
            [1, 2, 1, 3],
        ])
        ps_seq = PrefixSpan(min_support=2, max_length=4)
        ps_par = PrefixSpan(min_support=2, max_length=4)

        r_seq = ps_seq.mine(db, parallel=False)
        r_par = ps_par.mine(db, parallel=True)

        assert r_seq == r_par


# ── SPMF : chargement / sauvegarde ──────────────────────────────────────────


class TestSPMF:
    def test_load_spmf(self, tmp_path):
        spmf_file = tmp_path / "test.spmf"
        spmf_file.write_text("1 -1 2 -1 3 -1 -2\n4 -1 5 -1 -2\n")

        ps = PrefixSpan(min_support=1)
        db = ps.load_spmf(str(spmf_file))

        assert len(db) == 2
        np.testing.assert_array_equal(db[0], [1, 2, 3])
        np.testing.assert_array_equal(db[1], [4, 5])

    def test_load_nonexistent(self):
        ps = PrefixSpan()
        db = ps.load_spmf("nonexistent_file.spmf")
        assert db == []

    def test_roundtrip(self, tmp_path):
        """Mine → save → reload → re-mine doit donner les mêmes résultats."""
        db = _db([[1, 2, 3], [1, 2], [2, 3], [1, 3]])

        ps = PrefixSpan(min_support=2, max_length=3)
        result1 = ps.mine(db, parallel=False)

        # Sauvegarder
        out_path = str(tmp_path / "patterns.spmf")
        ps.save_results_to_spmf(out_path)

        # Le fichier existe et n'est pas vide
        with open(out_path) as f:
            content = f.read()
        assert len(content) > 0
        assert "#SUP:" in content
