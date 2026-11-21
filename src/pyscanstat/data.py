"""Small synthetic fixtures shipped with the package.

All fixtures are deterministic given ``seed`` and small enough for the
test suite. Locations are uniform in a 10 x 10 square (continuous
coordinates, so distance ties -- which can make zone membership
order-dependent -- do not occur).
"""

import numpy as np

__all__ = ["make_clustered", "make_null", "make_spacetime_clustered"]


def _base_locations(n, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    population = rng.integers(800, 1201, size=n).astype(np.float64)
    return rng, coords, population


def make_clustered(n=64, seed=42, cluster_center=(3.0, 3.0),
                   cluster_radius=2.0, base_rate=0.01,
                   rate_multiplier=4.0):
    """Spatial fixture with one planted circular cluster.

    Returns ``(coords, cases, population)``. Cases are Poisson with rate
    ``base_rate`` per person, multiplied by ``rate_multiplier`` inside
    the planted circle.
    """
    rng, coords, population = _base_locations(n, seed)
    dist = np.sqrt(np.sum((coords - np.asarray(cluster_center)) ** 2,
                          axis=1))
    rate = np.full(n, base_rate)
    rate[dist <= cluster_radius] *= rate_multiplier
    cases = rng.poisson(population * rate).astype(np.float64)
    return coords, cases, population


def make_null(n=64, seed=42, base_rate=0.01):
    """Spatial fixture with no cluster (uniform risk)."""
    rng, coords, population = _base_locations(n, seed)
    cases = rng.poisson(population * base_rate).astype(np.float64)
    return coords, cases, population


def make_spacetime_clustered(n=64, n_times=6, seed=42,
                             cluster_center=(3.0, 3.0),
                             cluster_radius=2.0, base_rate=0.01,
                             rate_multiplier=4.0, active_periods=(2, 3)):
    """Space-time fixture: planted cluster active in ``active_periods``.

    Returns ``(coords, counts, population)`` with ``counts`` of shape
    ``(n_times, n)``.
    """
    rng, coords, population = _base_locations(n, seed)
    dist = np.sqrt(np.sum((coords - np.asarray(cluster_center)) ** 2,
                          axis=1))
    in_cluster = dist <= cluster_radius
    counts = np.empty((n_times, n), dtype=np.float64)
    for t in range(n_times):
        rate = np.full(n, base_rate)
        if active_periods[0] <= t <= active_periods[1]:
            rate[in_cluster] *= rate_multiplier
        counts[t] = rng.poisson(population * rate)
    return coords, counts, population
