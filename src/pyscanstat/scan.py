"""Public spatial scan API: circular scan, Poisson and binomial models."""

import numpy as np

from ._core import build_zones, scan_zones_binomial, scan_zones_poisson
from .results import Cluster, ScanResult

__all__ = ["scan_poisson", "scan_binomial"]


def _validate_spatial(coords, cases, population):
    coords = np.asarray(coords, dtype=np.float64)
    cases = np.asarray(cases, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_locations, 2)")
    n = coords.shape[0]
    if cases.shape != (n,) or population.shape != (n,):
        raise ValueError("cases and population must have length n_locations")
    if np.any(cases < 0):
        raise ValueError("cases must be non-negative")
    if np.any(population <= 0):
        raise ValueError("population must be positive at every location")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite")
    return coords, cases, population


def _expected_poisson(cases, population):
    """Expected counts under the null: proportional to population."""
    total_cases = float(cases.sum())
    total_pop = float(population.sum())
    return population * (total_cases / total_pop)


def _relative_risk_poisson(c, e, total_cases, total_expected):
    c_out = total_cases - c
    e_out = total_expected - e
    if e <= 0.0 or e_out <= 0.0:
        return float("inf") if c > e else 1.0
    if c_out <= 0.0:
        return float("inf") if c > e else 1.0
    return (c / e) / (c_out / e_out)


def _relative_risk_binomial(c, n, total_cases, total_population):
    if n <= 0.0 or total_population - n <= 0.0:
        return 1.0
    outside = (total_cases - c) / (total_population - n)
    if outside <= 0.0:
        return float("inf") if c > 0.0 else 1.0
    return (c / n) / outside


def _make_cluster(zones, zone_pos, llr, cases_in, expected_in, pop_in,
                  total_cases, total_expected, rr, time_span=None):
    centre = int(zones.center[zone_pos])
    members = zones.members(zone_pos)
    cluster = Cluster(
        center_index=centre,
        center=(float(zones.coords[centre, 0]),
                float(zones.coords[centre, 1])),
        radius=zones.radius(zone_pos),
        location_indices=members,
        n_locations=int(members.shape[0]),
        cases=float(cases_in),
        expected=float(expected_in),
        population=float(pop_in),
        llr=float(llr),
        relative_risk=float(rr),
    )
    if time_span is not None:
        cluster.time_start, cluster.time_end, cluster.duration = time_span
    return cluster


def _secondary_clusters(zones, llr, cases_in, expected_in, pop_in,
                        total_cases, total_expected, rr_func, n_secondary):
    """Greedy non-overlapping secondary clusters by descending LLR."""
    if n_secondary <= 0 or zones.n_zones == 0:
        return []
    order = np.argsort(-llr, kind="stable")
    chosen = []
    used = np.zeros(zones.n, dtype=bool)
    for pos in order:
        if len(chosen) >= n_secondary:
            break
        if llr[pos] <= 0.0:
            break
        members = zones.members(int(pos))
        if used[members].any():
            continue
        used[members] = True
        rr = rr_func(cases_in[pos], expected_in[pos], pop_in[pos])
        chosen.append(_make_cluster(
            zones, int(pos), llr[pos], cases_in[pos], expected_in[pos],
            pop_in[pos], total_cases, total_expected, rr,
        ))
    return chosen


def _zone_case_sums(zones, cases, expected):
    ordered_cases = cases[zones.order]
    ordered_exp = expected[zones.order]
    cum_cases = np.cumsum(ordered_cases, axis=1)
    cum_exp = np.cumsum(ordered_exp, axis=1)
    idx = (zones.center, zones.size - 1)
    return cum_cases[idx], cum_exp[idx]


def scan_poisson(coords, cases, population, *,
                 max_population_fraction=0.5, n_secondary=4,
                 n_replicates=0, random_state=None):
    """Circular spatial scan, Poisson model (Kulldorff 1997).

    Parameters
    ----------
    coords : (n, 2) array of location coordinates (any planar units).
    cases : (n,) array of observed case counts.
    population : (n,) array of at-risk population per location.
    max_population_fraction : largest share of the total population a
        candidate zone may contain (SaTScan default 0.5).
    n_secondary : number of non-overlapping secondary clusters to report.
    n_replicates : Monte Carlo replicates for the p-value (0 = no test).
    random_state : seed for the Monte Carlo replicates.

    Returns a :class:`ScanResult` with the most likely cluster, secondary
    clusters, LLR and relative risk per cluster; with ``n_replicates > 0``
    also the Monte Carlo p-value (Dwass 1957).
    """
    if n_replicates and n_replicates > 0:
        from .montecarlo import monte_carlo_scan

        return monte_carlo_scan(
            coords, cases, population, model="poisson",
            n_replicates=n_replicates, random_state=random_state,
            max_population_fraction=max_population_fraction,
            n_secondary=n_secondary,
        )
    coords, cases, population = _validate_spatial(coords, cases, population)
    if not (0.0 < max_population_fraction <= 1.0):
        raise ValueError("max_population_fraction must be in (0, 1]")
    expected = _expected_poisson(cases, population)
    zones = build_zones(coords, population, max_population_fraction)
    llr = scan_zones_poisson(zones, cases, expected)
    total_cases = float(cases.sum())
    total_expected = float(expected.sum())
    cases_in, expected_in = _zone_case_sums(zones, cases, expected)

    if zones.n_zones == 0:
        return ScanResult(model="poisson", most_likely_cluster=None,
                          n_zones=0, total_cases=total_cases,
                          total_population=float(population.sum()))

    best = int(np.argmax(llr))
    rr = _relative_risk_poisson(cases_in[best], expected_in[best],
                                total_cases, total_expected)
    mlc = _make_cluster(zones, best, llr[best], cases_in[best],
                        expected_in[best], zones.zone_population[best],
                        total_cases, total_expected, rr)
    secondary = _secondary_clusters(
        zones, llr, cases_in, expected_in, zones.zone_population,
        total_cases, total_expected,
        lambda c, e, p: _relative_risk_poisson(c, e, total_cases,
                                              total_expected),
        n_secondary,
    )
    # The MLC itself is the first secondary pick; drop it from the list.
    secondary = [s for s in secondary
                 if s.center_index != mlc.center_index
                 or s.n_locations != mlc.n_locations
                 or abs(s.llr - mlc.llr) > 1e-12]
    return ScanResult(
        model="poisson", most_likely_cluster=mlc,
        secondary_clusters=secondary, n_zones=zones.n_zones,
        total_cases=total_cases,
        total_population=float(population.sum()),
    )


def scan_binomial(coords, cases, population, *,
                  max_population_fraction=0.5, n_secondary=4,
                  n_replicates=0, random_state=None):
    """Circular spatial scan, binomial (Bernoulli) model (Kulldorff 1997).

    Here ``population`` is the total number of individuals per location
    (cases + controls) and ``cases`` the cases among them. Otherwise
    identical in structure to :func:`scan_poisson`. With
    ``n_replicates > 0`` a Monte Carlo p-value is computed.
    """
    if n_replicates and n_replicates > 0:
        from .montecarlo import monte_carlo_scan

        return monte_carlo_scan(
            coords, cases, population, model="binomial",
            n_replicates=n_replicates, random_state=random_state,
            max_population_fraction=max_population_fraction,
            n_secondary=n_secondary,
        )
    coords, cases, population = _validate_spatial(coords, cases, population)
    if np.any(cases > population):
        raise ValueError("cases must not exceed population (binomial model)")
    if not (0.0 < max_population_fraction <= 1.0):
        raise ValueError("max_population_fraction must be in (0, 1]")
    zones = build_zones(coords, population, max_population_fraction)
    llr = scan_zones_binomial(zones, cases, population)
    total_cases = float(cases.sum())
    total_population = float(population.sum())
    cases_in = np.cumsum(cases[zones.order], axis=1)[
        zones.center, zones.size - 1]

    if zones.n_zones == 0:
        return ScanResult(model="binomial", most_likely_cluster=None,
                          n_zones=0, total_cases=total_cases,
                          total_population=total_population)

    best = int(np.argmax(llr))
    rr = _relative_risk_binomial(cases_in[best],
                                 zones.zone_population[best],
                                 total_cases, total_population)
    # For the binomial model the "expected" field is the null-expected
    # case count in the zone: population * overall rate.
    expected_in_zone = float(zones.zone_population[best]
                             * total_cases / total_population)
    mlc = _make_cluster(zones, best, llr[best], cases_in[best],
                        expected_in_zone,
                        zones.zone_population[best],
                        total_cases, total_population, rr)
    secondary = _secondary_clusters(
        zones, llr, cases_in,
        zones.zone_population * total_cases / total_population,
        zones.zone_population, total_cases, total_population,
        lambda c, e, p: _relative_risk_binomial(c, p, total_cases,
                                               total_population),
        n_secondary,
    )
    secondary = [s for s in secondary
                 if s.center_index != mlc.center_index
                 or s.n_locations != mlc.n_locations
                 or abs(s.llr - mlc.llr) > 1e-12]
    for s in secondary:
        s.expected = float(s.population * total_cases / total_population)
    return ScanResult(
        model="binomial", most_likely_cluster=mlc,
        secondary_clusters=secondary, n_zones=zones.n_zones,
        total_cases=total_cases, total_population=total_population,
    )
