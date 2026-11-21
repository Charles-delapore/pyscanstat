"""Tests for Monte Carlo significance testing."""

import numpy as np
import pytest

from pyscanstat.montecarlo import monte_carlo_pvalue, monte_carlo_scan
from pyscanstat.scan import scan_binomial, scan_poisson
from tests.test_scan_binomial import grid_fixture as binomial_fixture
from tests.test_scan_poisson import grid_fixture as poisson_fixture


def test_pvalue_formula():
    # (1 + #{rep >= obs}) / (1 + R)
    assert monte_carlo_pvalue(10.0, [1, 2, 12, 9]) == pytest.approx(2 / 5)
    assert monte_carlo_pvalue(10.0, [1, 2, 3]) == pytest.approx(1 / 4)
    assert monte_carlo_pvalue(10.0, [11, 12, 13]) == pytest.approx(1.0)
    assert monte_carlo_pvalue(10.0, []) is None


def test_seed_reproducibility_poisson():
    coords, cases, population, _ = poisson_fixture()
    r1 = scan_poisson(coords, cases, population, n_replicates=99,
                      random_state=7)
    r2 = scan_poisson(coords, cases, population, n_replicates=99,
                      random_state=7)
    assert r1.p_value == r2.p_value
    assert np.array_equal(r1.replicates_max_llr, r2.replicates_max_llr)
    assert r1.most_likely_cluster.llr == r2.most_likely_cluster.llr
    r3 = scan_poisson(coords, cases, population, n_replicates=99,
                      random_state=8)
    assert not np.array_equal(r1.replicates_max_llr,
                              r3.replicates_max_llr)


def test_planted_cluster_is_significant_poisson():
    coords, cases, population, _ = poisson_fixture()
    result = scan_poisson(coords, cases, population, n_replicates=999,
                          random_state=7)
    assert result.n_replicates == 999
    assert result.p_value is not None
    assert result.p_value <= 0.01
    assert result.most_likely_cluster.p_value == result.p_value
    assert len(result.replicates_max_llr) == 999
    for sec in result.secondary_clusters:
        assert sec.p_value is not None
        assert 0.0 < sec.p_value <= 1.0


def test_uniform_rate_pvalue_is_one():
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    result = scan_poisson(coords, np.full(16, 5.0), np.full(16, 100.0),
                          n_replicates=99, random_state=1)
    # Observed LLR is 0; every replicate max is >= 0.
    assert result.p_value == pytest.approx(1.0)


def test_no_replicates_means_no_pvalue():
    coords, cases, population, _ = poisson_fixture()
    result = scan_poisson(coords, cases, population)
    assert result.p_value is None
    assert result.n_replicates == 0
    assert result.replicates_max_llr is None


def test_planted_cluster_is_significant_binomial():
    coords, cases, population, _ = binomial_fixture()
    result = scan_binomial(coords, cases, population, n_replicates=999,
                           random_state=7)
    assert result.p_value is not None
    assert result.p_value <= 0.01


def test_monte_carlo_scan_validates_model():
    coords, cases, population, _ = poisson_fixture()
    with pytest.raises(ValueError):
        monte_carlo_scan(coords, cases, population, model="gaussian",
                         n_replicates=9)
    with pytest.raises(ValueError):
        monte_carlo_scan(coords, cases, population, n_replicates=0)
