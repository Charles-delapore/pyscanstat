"""Space-time cylindrical scan (Poisson model, Kulldorff 1997).

A candidate cylinder is a spatial circular zone (as in
:mod:`pyscanstat.scan`) crossed with a time window ``[t0, t1]``. Expected
counts are proportional to population and to the time-period total:
``E[t, i] = pop[i] * C[t] / P``. The scan statistic is the maximum
Poisson LLR over all (zone, end-time, duration) cylinders; significance
comes from the same Monte Carlo scheme as the spatial scan (multinomial
replicates per time slice, p-value against the replicate maxima).
"""

import numpy as np

from ._core import build_zones
from ._numba import njit
from .montecarlo import monte_carlo_pvalue, replicate_cases_poisson
from .results import Cluster, ScanResult

__all__ = ["scan_spacetime_poisson"]


@njit
def _spacetime_llr_kernel(csum_c, csum_e, total_cases, max_duration,
                          only_last_end, out):
    """Fill ``out[z, t1, d-1]`` with the Poisson LLR of each cylinder."""
    n_zones = csum_c.shape[0]
    n_times = csum_c.shape[1] - 1
    for z in range(n_zones):
        for t1 in range(n_times):
            if only_last_end and t1 != n_times - 1:
                continue
            max_d = max_duration
            if max_d > t1 + 1:
                max_d = t1 + 1
            for d in range(1, max_d + 1):
                c = csum_c[z, t1 + 1] - csum_c[z, t1 + 1 - d]
                e = csum_e[z, t1 + 1] - csum_e[z, t1 + 1 - d]
                co = total_cases - c
                eo = total_cases - e
                if c > e and c > 0.0 and e > 0.0:
                    v = c * np.log(c / e)
                    if co > 0.0 and eo > 0.0:
                        v += co * np.log(co / eo)
                    out[z, t1, d - 1] = v if v > 0.0 else 0.0
                else:
                    out[z, t1, d - 1] = 0.0


def _validate_spacetime(coords, counts, population):
    coords = np.asarray(coords, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_locations, 2)")
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_times, n_locations)")
    n_times, n = counts.shape
    if n != coords.shape[0] or population.shape != (n,):
        raise ValueError("counts/population must match n_locations")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if np.any(population <= 0):
        raise ValueError("population must be positive at every location")
    return coords, counts, population


def _expected_spacetime(counts, population):
    """E[t, i] = pop[i] * C[t] / P (null: risk follows population x time)."""
    totals = counts.sum(axis=1)
    return population[None, :] * (totals / population.sum())[:, None]


def _zone_time_cumsums(zones, counts, expected):
    """Time-cumsummed case/expected counts per zone, shape (Z, T+1)."""
    n_zones = zones.n_zones
    n_loc = zones.n
    membership = np.zeros((n_zones, n_loc), dtype=np.float64)
    for pos in range(n_zones):
        membership[pos, zones.members(pos)] = 1.0
    zone_cases = membership @ counts.T
    zone_exp = membership @ expected.T
    zeros = np.zeros((n_zones, 1), dtype=np.float64)
    csum_c = np.concatenate(
        [zeros, np.cumsum(zone_cases, axis=1)], axis=1)
    csum_e = np.concatenate(
        [zeros, np.cumsum(zone_exp, axis=1)], axis=1)
    return csum_c, csum_e


def _all_cylinder_llr(zones, counts, expected, max_duration, prospective):
    csum_c, csum_e = _zone_time_cumsums(zones, counts, expected)
    n_times = counts.shape[0]
    out = np.zeros((zones.n_zones, n_times, max_duration),
                   dtype=np.float64)
    _spacetime_llr_kernel(csum_c, csum_e, float(counts.sum()),
                          max_duration, prospective, out)
    return out


def _relative_risk(c, e, total_cases):
    c_out = total_cases - c
    e_out = total_cases - e
    if e <= 0.0 or e_out <= 0.0 or c_out <= 0.0:
        return float("inf") if c > e else 1.0
    return (c / e) / (c_out / e_out)


def _make_st_cluster(zones, zone_pos, t1, d, llr, csum_c, csum_e,
                     total_cases):
    centre = int(zones.center[zone_pos])
    members = zones.members(zone_pos)
    t0 = t1 - d + 1
    c = float(csum_c[zone_pos, t1 + 1] - csum_c[zone_pos, t0])
    e = float(csum_e[zone_pos, t1 + 1] - csum_e[zone_pos, t0])
    return Cluster(
        center_index=centre,
        center=(float(zones.coords[centre, 0]),
                float(zones.coords[centre, 1])),
        radius=zones.radius(zone_pos),
        location_indices=members,
        n_locations=int(members.shape[0]),
        cases=c,
        expected=e,
        population=float(zones.zone_population[zone_pos]),
        llr=float(llr),
        relative_risk=_relative_risk(c, e, total_cases),
        time_start=int(t0),
        time_end=int(t1),
        duration=int(d),
    )


def _secondary_st_clusters(zones, llr_3d, csum_c, csum_e, total_cases,
                           n_secondary):
    """Greedy non-overlapping (by location) secondary cylinders."""
    if n_secondary <= 0:
        return []
    flat = np.argsort(-llr_3d.ravel(), kind="stable")
    chosen = []
    used = np.zeros(zones.n, dtype=bool)
    n_times = llr_3d.shape[1]
    max_dur = llr_3d.shape[2]
    for flat_pos in flat:
        if len(chosen) >= n_secondary:
            break
        z, t1, d0 = np.unravel_index(
            int(flat_pos), (llr_3d.shape[0], n_times, max_dur))
        value = llr_3d[z, t1, d0]
        if value <= 0.0:
            break
        members = zones.members(int(z))
        if used[members].any():
            continue
        used[members] = True
        chosen.append(_make_st_cluster(
            zones, int(z), int(t1), int(d0) + 1, float(value),
            csum_c, csum_e, total_cases))
    return chosen


def scan_spacetime_poisson(coords, counts, population, *,
                           max_duration=None, prospective=False,
                           max_population_fraction=0.5, n_replicates=0,
                           random_state=None, n_secondary=4):
    """Space-time cylindrical scan, Poisson model (Kulldorff 1997).

    Parameters
    ----------
    coords : (n, 2) array of location coordinates.
    counts : (n_times, n) array of case counts per time period.
    population : (n,) array of at-risk population per location.
    max_duration : longest time window in periods (default: all periods).
    prospective : if True, only windows ending at the last time period
        are considered (prospective surveillance); otherwise all end
        times (retrospective).
    max_population_fraction : largest share of the total population a
        candidate zone may contain (default 0.5).
    n_replicates : Monte Carlo replicates for the p-value (0 = no test).
    random_state : seed for the Monte Carlo replicates.
    n_secondary : number of non-overlapping secondary clusters.
    """
    coords, counts, population = _validate_spacetime(
        coords, counts, population)
    n_times = counts.shape[0]
    if max_duration is None:
        max_duration = n_times
    if not (1 <= max_duration <= n_times):
        raise ValueError("max_duration must be in [1, n_times]")
    if not (0.0 < max_population_fraction <= 1.0):
        raise ValueError("max_population_fraction must be in (0, 1]")

    zones = build_zones(coords, population, max_population_fraction)
    expected = _expected_spacetime(counts, population)
    total_cases = float(counts.sum())
    llr_3d = _all_cylinder_llr(zones, counts, expected, max_duration,
                               prospective)
    csum_c, csum_e = _zone_time_cumsums(zones, counts, expected)

    rep_max = None
    if n_replicates and n_replicates > 0:
        rng = np.random.default_rng(random_state)
        rep_max = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            rep_counts = np.vstack([
                replicate_cases_poisson(rng, expected[t])
                for t in range(n_times)
            ])
            rep_llr = _all_cylinder_llr(zones, rep_counts, expected,
                                        max_duration, prospective)
            rep_max[r] = float(np.max(rep_llr))

    if zones.n_zones == 0:
        return ScanResult(model="spacetime-poisson",
                          most_likely_cluster=None, n_zones=0,
                          total_cases=total_cases,
                          total_population=float(population.sum()),
                          n_replicates=int(n_replicates or 0),
                          replicates_max_llr=rep_max)

    best = np.unravel_index(int(np.argmax(llr_3d)), llr_3d.shape)
    best_z, best_t1, best_d0 = (int(v) for v in best)
    mlc = _make_st_cluster(zones, best_z, best_t1, best_d0 + 1,
                           float(llr_3d[best_z, best_t1, best_d0]),
                           csum_c, csum_e, total_cases)
    if rep_max is not None:
        mlc.p_value = monte_carlo_pvalue(mlc.llr, rep_max)

    secondary = _secondary_st_clusters(zones, llr_3d, csum_c, csum_e,
                                       total_cases, n_secondary)
    # First greedy pick is the MLC itself; drop it.
    secondary = [s for s in secondary
                 if not (s.center_index == mlc.center_index
                         and s.n_locations == mlc.n_locations
                         and s.time_start == mlc.time_start
                         and s.time_end == mlc.time_end)]
    if rep_max is not None:
        for s in secondary:
            s.p_value = monte_carlo_pvalue(s.llr, rep_max)

    return ScanResult(
        model="spacetime-poisson", most_likely_cluster=mlc,
        secondary_clusters=secondary,
        p_value=mlc.p_value, n_replicates=int(n_replicates or 0),
        n_zones=zones.n_zones, total_cases=total_cases,
        total_population=float(population.sum()),
        replicates_max_llr=rep_max,
    )
