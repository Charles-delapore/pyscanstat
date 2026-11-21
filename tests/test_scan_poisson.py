"""Tests for the Poisson circular scan."""

import numpy as np
import pytest

from pyscanstat._core import build_zones, scan_zones_poisson
from pyscanstat.llr import poisson_llr
from pyscanstat.scan import _expected_poisson, scan_poisson


def grid_fixture():
    """6x6 grid, pop 1000/cell, baseline 10 cases/cell, planted 3x3 block
    of 40 cases/cell at rows/cols 2..4 (indices 14,15,16,20,21,22,26,27,28
    in row-major order)."""
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    cases = np.full(36, 10.0)
    block = [14, 15, 16, 20, 21, 22, 26, 27, 28]
    cases[block] = 40.0
    population = np.full(36, 1000.0)
    return coords, cases, population, block


def test_kernel_matches_scalar_llr_every_zone():
    coords, cases, population, _ = grid_fixture()
    expected = _expected_poisson(cases, population)
    zones = build_zones(coords, population, 0.5)
    llr = scan_zones_poisson(zones, cases, expected)
    total = float(cases.sum())
    for pos in range(zones.n_zones):
        members = zones.members(pos)
        c = float(cases[members].sum())
        e = float(expected[members].sum())
        assert llr[pos] == pytest.approx(poisson_llr(c, e, total), abs=1e-9)


def test_planted_cluster_is_most_likely():
    coords, cases, population, block = grid_fixture()
    result = scan_poisson(coords, cases, population)
    mlc = result.most_likely_cluster
    assert mlc is not None
    assert mlc.llr > 30.0
    assert mlc.relative_risk > 2.0
    # The MLC must lie inside the planted block (centre + members).
    assert mlc.center_index in block
    assert set(mlc.location_indices.tolist()) <= set(block)
    assert mlc.cases > mlc.expected


def test_zone_population_cap_respected():
    coords, cases, population, _ = grid_fixture()
    zones = build_zones(coords, population, 0.5)
    assert np.all(zones.zone_population <= 0.5 * population.sum() + 1e-9)
    # 6x6 grid, cap 50% of 36000 = 18000 -> max 18 locations per zone.
    assert zones.size.max() <= 18


def test_uniform_rate_gives_zero_llr():
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    cases = np.full(16, 5.0)
    population = np.full(16, 100.0)
    result = scan_poisson(coords, cases, population)
    assert result.most_likely_cluster.llr == pytest.approx(0.0)
    assert result.most_likely_cluster.relative_risk == pytest.approx(1.0)


def test_no_cases_gives_zero_llr():
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    result = scan_poisson(coords, np.zeros(16), np.full(16, 100.0))
    assert result.most_likely_cluster.llr == pytest.approx(0.0)


def test_secondary_clusters_are_disjoint():
    coords, cases, population, _ = grid_fixture()
    result = scan_poisson(coords, cases, population, n_secondary=4)
    seen = set(result.most_likely_cluster.location_indices.tolist())
    for sec in result.secondary_clusters:
        assert not (seen & set(sec.location_indices.tolist()))
        seen |= set(sec.location_indices.tolist())
        assert sec.llr > 0.0


def test_input_validation():
    coords = np.zeros((4, 2))
    with pytest.raises(ValueError):
        scan_poisson(coords, [1, 2, 3], [10, 10, 10, 10])
    with pytest.raises(ValueError):
        scan_poisson(coords, [1, -1, 2, 3], [10, 10, 10, 10])
    with pytest.raises(ValueError):
        scan_poisson(coords, [1, 2, 3, 4], [10, 0, 10, 10])
    with pytest.raises(ValueError):
        scan_poisson(coords, [1, 2, 3, 4], [10, 10, 10, 10],
                     max_population_fraction=1.5)
