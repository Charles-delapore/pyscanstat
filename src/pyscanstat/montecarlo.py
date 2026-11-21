"""Monte Carlo significance testing for scan statistics.

Null hypothesis (Poisson model): cases are multinomially distributed
over locations with probabilities proportional to the expected counts.
Null hypothesis (binomial model): the fixed total number of cases is
allocated to individuals uniformly at random, i.e. a multivariate
hypergeometric draw over locations (conditional on the total).

The Monte Carlo p-value follows Dwass (1957):

    p = (1 + #{replicate max-LLR >= observed max-LLR}) / (1 + R)

The same null distribution of the *maximum* LLR is used for the
secondary clusters, which makes their p-values conservative and
accounts for the search over zones (multiple-testing awareness). The
primary p-value therefore already adjusts for having searched all
candidate zones.
"""

import numpy as np

from ._core import build_zones, scan_zones_binomial, scan_zones_poisson
from .results import ScanResult
from .scan import (
    _expected_poisson,
    _make_cluster,
    _relative_risk_binomial,
    _relative_risk_poisson,
    _secondary_clusters,
    _validate_spatial,
    _zone_case_sums,
)

__all__ = [
    "monte_carlo_pvalue",
    "monte_carlo_scan",
    "replicate_cases_poisson",
    "replicate_cases_binomial",
]


def monte_carlo_pvalue(observed, replicates):
    """Dwass Monte Carlo p-value from replicate maxima."""
    replicates = np.asarray(replicates, dtype=np.float64)
    if replicates.size == 0:
        return None
    return float(
        (1.0 + np.sum(replicates >= float(observed)))
        / (1.0 + replicates.size)
    )


def replicate_cases_poisson(rng, expected):
    """One null replicate of case counts (Poisson model)."""
    total = float(expected.sum())
    probs = expected / total
    return rng.multinomial(int(round(total)), probs).astype(np.float64)


def replicate_cases_binomial(rng, population, total_cases):
    """One null replicate of case counts (binomial model)."""
    if not np.all(population == np.floor(population)):
        raise ValueError(
            "binomial Monte Carlo needs integer-valued population"
        )
    return rng.multivariate_hypergeometric(
        population.astype(np.int64), int(total_cases)
    ).astype(np.float64)


def monte_carlo_scan(coords, cases, population, *, model="poisson",
                     n_replicates=999, random_state=None,
                     max_population_fraction=0.5, n_secondary=4,
                     min_cases=1):
    """Circular scan with Monte Carlo significance (Poisson/binomial).

    Same scan as :func:`pyscanstat.scan.scan_poisson` /
    :func:`pyscanstat.scan.scan_binomial`, plus ``p_value`` for the most
    likely cluster (and for each secondary cluster) and the vector of
    replicate maxima in ``replicates_max_llr``.
    """
    if model not in ("poisson", "binomial"):
        raise ValueError("model must be 'poisson' or 'binomial'")
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    coords, cases, population = _validate_spatial(coords, cases, population)
    if model == "binomial" and np.any(cases > population):
        raise ValueError("cases must not exceed population (binomial model)")
    rng = np.random.default_rng(random_state)
    zones = build_zones(coords, population, max_population_fraction)
    total_cases = float(cases.sum())
    total_population = float(population.sum())

    if model == "poisson":
        expected = _expected_poisson(cases, population)
        scan_fn = lambda cs: scan_zones_poisson(zones, cs, expected,
                                                min_cases=min_cases)
        replicate_fn = lambda: replicate_cases_poisson(rng, expected)
        total_expected = float(expected.sum())
    else:
        expected = population * (total_cases / total_population)
        scan_fn = lambda cs: scan_zones_binomial(zones, cs, population,
                                                 min_cases=min_cases)
        replicate_fn = lambda: replicate_cases_binomial(
            rng, population, total_cases)
        total_expected = total_population

    llr = scan_fn(cases)
    cases_in, expected_in = _zone_case_sums(zones, cases, expected)

    rep_max = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        rep_cases = replicate_fn()
        rep_max[r] = float(np.max(scan_fn(rep_cases)))

    if zones.n_zones == 0:
        return ScanResult(model=model, most_likely_cluster=None,
                          n_replicates=n_replicates, n_zones=0,
                          total_cases=total_cases,
                          total_population=total_population,
                          replicates_max_llr=rep_max)

    best = int(np.argmax(llr))
    if model == "poisson":
        rr = _relative_risk_poisson(cases_in[best], expected_in[best],
                                    total_cases, total_expected)
    else:
        rr = _relative_risk_binomial(cases_in[best],
                                     zones.zone_population[best],
                                     total_cases, total_population)
    mlc = _make_cluster(zones, best, llr[best], cases_in[best],
                        expected_in[best], zones.zone_population[best],
                        total_cases, total_expected, rr)
    mlc.p_value = monte_carlo_pvalue(llr[best], rep_max)

    if model == "poisson":
        rr_func = lambda c, e, p: _relative_risk_poisson(
            c, e, total_cases, total_expected)
    else:
        rr_func = lambda c, e, p: _relative_risk_binomial(
            c, p, total_cases, total_population)
    secondary = _secondary_clusters(
        zones, llr, cases_in, expected_in, zones.zone_population,
        total_cases, total_expected, rr_func, n_secondary,
    )
    secondary = [s for s in secondary
                 if s.center_index != mlc.center_index
                 or s.n_locations != mlc.n_locations
                 or abs(s.llr - mlc.llr) > 1e-12]
    for s in secondary:
        s.p_value = monte_carlo_pvalue(s.llr, rep_max)
        if model == "binomial":
            s.expected = float(s.population * total_cases
                               / total_population)

    return ScanResult(
        model=model, most_likely_cluster=mlc,
        secondary_clusters=secondary,
        p_value=mlc.p_value, n_replicates=n_replicates,
        n_zones=zones.n_zones, total_cases=total_cases,
        total_population=total_population,
        replicates_max_llr=rep_max,
    )
