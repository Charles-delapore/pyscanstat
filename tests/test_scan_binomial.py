"""Tests for the binomial (Bernoulli) circular scan."""

import numpy as np
import pytest

from pyscanstat._core import build_zones, scan_zones_binomial
from pyscanstat.llr import binomial_llr
from pyscanstat.scan import scan_binomial


def grid_fixture():
    """6x6 grid, 1000 people/cell, baseline 100 cases/cell (10%),
    planted 3x3 block of 300 cases/cell (30%) at rows/cols 2..4."""
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    cases = np.full(36, 100.0)
    block = [14, 15, 16, 20, 21, 22, 26, 27, 28]
    cases[block] = 300.0
    population = np.full(36, 1000.0)
    return coords, cases, population, block


def test_kernel_matches_scalar_llr_every_zone():
    coords, cases, population, _ = grid_fixture()
    zones = build_zones(coords, population, 0.5)
    llr = scan_zones_binomial(zones, cases, population)
    total_cases = float(cases.sum())
    total_pop = float(population.sum())
    for pos in range(zones.n_zones):
        members = zones.members(pos)
        c = float(cases[members].sum())
        n = float(population[members].sum())
        assert llr[pos] == pytest.approx(
            binomial_llr(c, n, total_cases, total_pop), abs=1e-9
        )


def test_planted_cluster_is_most_likely():
    coords, cases, population, block = grid_fixture()
    result = scan_binomial(coords, cases, population)
    assert result.model == "binomial"
    mlc = result.most_likely_cluster
    assert mlc is not None
    assert mlc.llr > 30.0
    assert mlc.relative_risk > 2.0
    assert mlc.center_index in block
    assert set(mlc.location_indices.tolist()) <= set(block)
    # Null-expected cases in the MLC zone: pop * overall rate.
    assert mlc.expected == pytest.approx(
        mlc.population * cases.sum() / population.sum()
    )


def test_uniform_rate_gives_zero_llr():
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    result = scan_binomial(coords, np.full(16, 50.0), np.full(16, 500.0))
    assert result.most_likely_cluster.llr == pytest.approx(0.0)


def test_cases_above_population_rejected():
    coords = np.zeros((4, 2))
    with pytest.raises(ValueError):
        scan_binomial(coords, [11, 2, 3, 4], [10, 10, 10, 10])
