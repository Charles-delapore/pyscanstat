"""Tests for the space-time cylindrical scan (Poisson model)."""

import numpy as np
import pytest

from pyscanstat.llr import poisson_llr
from pyscanstat.spacetime import scan_spacetime_poisson


def st_fixture():
    """6x6 grid, T=6 periods, pop 1000/cell, baseline 5 cases/cell/period,
    planted 3x3 block (rows/cols 2..4) with 30 cases/cell in periods 3..4
    only."""
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    counts = np.full((6, 36), 5.0)
    block = [14, 15, 16, 20, 21, 22, 26, 27, 28]
    counts[3:5, block] = 30.0
    population = np.full(36, 1000.0)
    return coords, counts, population, block


def test_planted_spacetime_cluster_detected():
    coords, counts, population, block = st_fixture()
    result = scan_spacetime_poisson(coords, counts, population)
    assert result.model == "spacetime-poisson"
    mlc = result.most_likely_cluster
    assert mlc is not None
    assert mlc.center_index in block
    assert set(mlc.location_indices.tolist()) <= set(block)
    assert mlc.time_start == 3
    assert mlc.time_end == 4
    assert mlc.duration == 2
    assert mlc.llr > 100.0
    assert mlc.relative_risk > 2.0


def test_cylinder_llr_matches_scalar_formula():
    # Spot-check the MLC cylinder LLR against the scalar Poisson formula.
    coords, counts, population, _ = st_fixture()
    result = scan_spacetime_poisson(coords, counts, population)
    mlc = result.most_likely_cluster
    total = float(counts.sum())
    assert mlc.llr == pytest.approx(
        poisson_llr(mlc.cases, mlc.expected, total), rel=1e-9
    )
    # Hand values for the planted cylinder: c=540, e=202.5, C=1530.
    assert mlc.cases == pytest.approx(540.0)
    assert mlc.expected == pytest.approx(202.5)
    assert mlc.llr == pytest.approx(poisson_llr(540, 202.5, 1530),
                                    rel=1e-9)


def test_prospective_only_windows_ending_at_last_period():
    coords, counts, population, _ = st_fixture()
    result = scan_spacetime_poisson(coords, counts, population,
                                    prospective=True)
    assert result.most_likely_cluster.time_end == 5


def test_max_duration_limits_windows():
    coords, counts, population, _ = st_fixture()
    result = scan_spacetime_poisson(coords, counts, population,
                                    max_duration=1)
    assert result.most_likely_cluster.duration == 1


def test_spacetime_monte_carlo_significant_and_reproducible():
    coords, counts, population, _ = st_fixture()
    r1 = scan_spacetime_poisson(coords, counts, population,
                                n_replicates=199, random_state=3)
    r2 = scan_spacetime_poisson(coords, counts, population,
                                n_replicates=199, random_state=3)
    assert r1.p_value == r2.p_value
    assert np.array_equal(r1.replicates_max_llr, r2.replicates_max_llr)
    assert r1.p_value is not None
    assert r1.p_value <= 0.05


def test_secondary_clusters_disjoint():
    coords, counts, population, _ = st_fixture()
    result = scan_spacetime_poisson(coords, counts, population,
                                    n_secondary=3)
    seen = set(result.most_likely_cluster.location_indices.tolist())
    for sec in result.secondary_clusters:
        assert not (seen & set(sec.location_indices.tolist()))
        seen |= set(sec.location_indices.tolist())


def test_validation():
    coords = np.zeros((4, 2))
    with pytest.raises(ValueError):
        scan_spacetime_poisson(coords, np.zeros((3, 5)),
                               np.full(4, 100.0))
    with pytest.raises(ValueError):
        scan_spacetime_poisson(coords, np.zeros((3, 4)),
                               np.full(4, 100.0), max_duration=4)
    with pytest.raises(ValueError):
        scan_spacetime_poisson(coords, -np.ones((3, 4)),
                               np.full(4, 100.0))
