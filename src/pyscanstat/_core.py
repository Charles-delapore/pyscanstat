"""Zone enumeration and Numba-compiled LLR kernels.

Design (see README "Performance design"): the candidate-zone structure
depends only on coordinates and population, so it is built once with
vectorised NumPy and reused for every Monte Carlo replicate. Only the
per-zone LLR evaluation -- a tight loop over flat arrays -- is compiled
with Numba. Cumulative case/expected counts per centre are NumPy cumsums
along the distance-ordered neighbour axis.
"""

import numpy as np

from ._numba import njit

__all__ = [
    "ZoneSet",
    "build_zones",
    "scan_zones_poisson",
    "scan_zones_binomial",
    "poisson_llr_kernel",
    "binomial_llr_kernel",
]


class ZoneSet:
    """Circular candidate zones: one centre per location.

    For centre ``i`` the neighbours are ordered by distance and the zone
    grows prefix by prefix until the cumulative population would exceed
    ``max_population_fraction`` of the total. Only population-valid
    prefixes are kept (flat arrays ``center`` / ``size``); member indices
    for a zone are ``order[centre, :size]``.
    """

    def __init__(self, coords, population, max_population_fraction=0.5):
        coords = np.asarray(coords, dtype=np.float64)
        population = np.asarray(population, dtype=np.float64)
        n = coords.shape[0]
        self.n = n
        self.coords = coords
        self.total_population = float(population.sum())
        # Full distance matrix (N x N) and per-centre neighbour order.
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        self.order = np.argsort(dist, axis=1)
        self.sorted_dist = np.take_along_axis(dist, self.order, axis=1)
        sorted_pop = population[self.order]
        cum_pop = np.cumsum(sorted_pop, axis=1)
        valid = cum_pop <= max_population_fraction * self.total_population
        # A zone must contain at least its centre; a centre whose own
        # population already exceeds the cap yields no zone (as in
        # SaTScan's ubpop rule).
        centre_idx, size_idx = np.nonzero(valid)
        self.center = centre_idx.astype(np.int64)
        self.size = (size_idx + 1).astype(np.int64)
        self.zone_population = cum_pop[centre_idx, size_idx]
        self.n_zones = int(self.center.shape[0])
        # Cumulative population per centre, for fast zone lookups.
        self._cum_pop = cum_pop

    def members(self, zone_pos):
        """Location indices of the ``zone_pos``-th valid zone."""
        c = int(self.center[zone_pos])
        s = int(self.size[zone_pos])
        return self.order[c, :s]

    def radius(self, zone_pos):
        c = int(self.center[zone_pos])
        s = int(self.size[zone_pos])
        return float(self.sorted_dist[c, s - 1])


def build_zones(coords, population, max_population_fraction=0.5):
    return ZoneSet(coords, population, max_population_fraction)


@njit
def poisson_llr_kernel(c_in, e_in, total_cases, total_expected, out):
    """LLR per zone (Poisson model); ``out`` is filled in place."""
    for i in range(c_in.shape[0]):
        c = c_in[i]
        e = e_in[i]
        co = total_cases - c
        eo = total_expected - e
        if c > e and c > 0.0 and e > 0.0:
            v = c * np.log(c / e)
            if co > 0.0 and eo > 0.0:
                v += co * np.log(co / eo)
            out[i] = v if v > 0.0 else 0.0
        else:
            out[i] = 0.0


@njit
def binomial_llr_kernel(c_in, n_in, total_cases, total_population, out):
    """LLR per zone (binomial/Bernoulli model); ``out`` filled in place."""
    p0 = total_cases / total_population
    for i in range(c_in.shape[0]):
        c = c_in[i]
        n = n_in[i]
        if n <= 0.0 or n >= total_population or c / n <= p0:
            out[i] = 0.0
            continue
        v = 0.0
        if c > 0.0:
            v += c * np.log(c / n)
        if n - c > 0.0:
            v += (n - c) * np.log((n - c) / n)
        co = total_cases - c
        no = total_population - n
        if co > 0.0:
            v += co * np.log(co / no)
        if no - co > 0.0:
            v += (no - co) * np.log((no - co) / no)
        if total_cases > 0.0:
            v -= total_cases * np.log(p0)
        if total_population - total_cases > 0.0:
            v -= (total_population - total_cases) * np.log(
                1.0 - p0
            )
        out[i] = v if v > 0.0 else 0.0


def _zone_sums(zones, values):
    """Cumulative sums of ``values`` over each valid zone (flat array)."""
    ordered = values[zones.order]
    cum = np.cumsum(ordered, axis=1)
    return cum[zones.center, zones.size - 1]


def scan_zones_poisson(zones, cases, expected):
    """Poisson LLR for every valid zone. Returns an ``(n_zones,)`` array."""
    cases = np.asarray(cases, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    c_in = _zone_sums(zones, cases)
    e_in = _zone_sums(zones, expected)
    total_cases = float(cases.sum())
    total_expected = float(expected.sum())
    out = np.empty(zones.n_zones, dtype=np.float64)
    poisson_llr_kernel(c_in, e_in, total_cases, total_expected, out)
    return out


def scan_zones_binomial(zones, cases, population):
    """Binomial LLR for every valid zone. Returns an ``(n_zones,)`` array."""
    cases = np.asarray(cases, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    c_in = _zone_sums(zones, cases)
    n_in = zones.zone_population
    total_cases = float(cases.sum())
    total_population = float(population.sum())
    out = np.empty(zones.n_zones, dtype=np.float64)
    binomial_llr_kernel(c_in, n_in, total_cases, total_population, out)
    return out
