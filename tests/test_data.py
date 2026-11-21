"""Tests for the shipped synthetic fixtures."""

import numpy as np

from pyscanstat import (
    make_clustered,
    make_null,
    make_spacetime_clustered,
    scan_poisson,
    scan_spacetime_poisson,
)


def test_make_clustered_is_deterministic():
    a = make_clustered(seed=42)
    b = make_clustered(seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    c = make_clustered(seed=43)
    assert not np.array_equal(a[1], c[1])


def test_make_clustered_detected_with_small_p():
    coords, cases, pop = make_clustered(seed=42)
    result = scan_poisson(coords, cases, pop, n_replicates=199,
                          random_state=7)
    assert result.most_likely_cluster.llr > 50.0
    assert result.p_value <= 0.01


def test_make_null_has_no_significant_cluster():
    coords, cases, pop = make_null(seed=42)
    result = scan_poisson(coords, cases, pop, n_replicates=199,
                          random_state=7)
    # Uniform risk: the max LLR stays modest and the p-value is not small.
    assert result.most_likely_cluster.llr < 20.0
    assert result.p_value > 0.05


def test_make_spacetime_clustered_detected():
    coords, counts, pop = make_spacetime_clustered(seed=42)
    assert counts.shape == (6, 64)
    result = scan_spacetime_poisson(coords, counts, pop,
                                    n_replicates=99, random_state=7)
    mlc = result.most_likely_cluster
    assert mlc.time_start == 2
    assert mlc.time_end == 3
    assert mlc.llr > 20.0
    assert result.p_value <= 0.05
