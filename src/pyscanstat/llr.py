"""Exact scalar log-likelihood ratios (Kulldorff 1997).

These are the reference scalar formulas. The vectorised/compiled
equivalents live in :mod:`pyscanstat._core` and must agree with these to
within floating-point rounding; ``tests/test_llr.py`` checks both against
hand-computed values.
"""

import math

__all__ = ["poisson_llr", "binomial_llr"]


def _xlogx_ratio(a, b):
    """Return ``a * log(a / b)`` with the ``0 * log 0 = 0`` convention."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return a * math.log(a / b)


def poisson_llr(c, expected, total_cases, total_expected=None):
    """Poisson log-likelihood ratio for one candidate zone.

    ``c`` cases observed in the zone, ``expected`` expected under the null,
    ``total_cases`` cases overall. Only elevated zones count: the ratio is
    0 when ``c <= expected`` (Kulldorff 1997, eq. for the Poisson model).
    """
    if total_expected is None:
        total_expected = float(total_cases)
    c = float(c)
    expected = float(expected)
    total_cases = float(total_cases)
    total_expected = float(total_expected)
    if c <= expected or c <= 0.0 or expected <= 0.0:
        return 0.0
    llr = _xlogx_ratio(c, expected) + _xlogx_ratio(
        total_cases - c, total_expected - expected
    )
    return llr if llr > 0.0 else 0.0


def binomial_llr(c, n, total_cases, total_population):
    """Binomial (Bernoulli) log-likelihood ratio for one candidate zone.

    ``c`` cases among ``n`` individuals in the zone, ``total_cases`` cases
    among ``total_population`` individuals overall. Only zones with a
    higher in-zone rate than the overall rate count.
    """
    c = float(c)
    n = float(n)
    total_cases = float(total_cases)
    total_population = float(total_population)
    if n <= 0.0 or n >= total_population or total_population <= 0.0:
        return 0.0
    if c / n <= total_cases / total_population:
        return 0.0
    # LLR = loglik(zone rate, outside rate) - loglik(common rate), up to
    # binomial coefficients that cancel in the ratio. Each term is
    # a*log(a/b) with the 0*log0 = 0 convention.
    llr = (
        _xlogx_ratio(c, n)
        + _xlogx_ratio(n - c, n)
        + _xlogx_ratio(total_cases - c, total_population - n)
        + _xlogx_ratio(
            total_population - n - (total_cases - c), total_population - n
        )
        - _xlogx_ratio(total_cases, total_population)
        - _xlogx_ratio(total_population - total_cases, total_population)
    )
    # Bernoulli log-likelihood terms, e.g. c*log(c/n) is _xlogx_ratio(c, n).
    return llr if llr > 0.0 else 0.0
