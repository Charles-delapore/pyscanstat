"""Tests for the scalar LLR formulas against hand-computed values."""

import math

import pytest

from pyscanstat.llr import binomial_llr, poisson_llr


def test_poisson_llr_hand_value():
    # C=100 total cases, zone c=30, E=10:
    # 30*ln(30/10) + 70*ln(70/90) = 15.366358680379868
    assert poisson_llr(30, 10, 100) == pytest.approx(15.366358680379868)
    # RR = (30/10) / (70/90) = 3.857142857142857 (checked in scan tests).


def test_poisson_llr_not_elevated_is_zero():
    assert poisson_llr(5, 10, 100) == 0.0
    assert poisson_llr(10, 10, 100) == 0.0
    assert poisson_llr(0, 10, 100) == 0.0


def test_poisson_llr_matches_direct_formula():
    for c, e, total in [(7, 3.5, 50), (41, 12.0, 200), (3, 0.25, 17)]:
        direct = c * math.log(c / e) + (total - c) * math.log(
            (total - c) / (total - e)
        )
        assert poisson_llr(c, e, total) == pytest.approx(direct)


def test_binomial_llr_hand_value():
    # N=1000, C=100, zone n=200, c=40 -> 11.894918069474272
    assert binomial_llr(40, 200, 100, 1000) == pytest.approx(
        11.894918069474272
    )


def test_binomial_llr_not_elevated_is_zero():
    assert binomial_llr(10, 200, 100, 1000) == 0.0
    assert binomial_llr(20, 200, 100, 1000) == 0.0  # rate == overall


def test_binomial_llr_matches_direct_formula():
    def direct(c, n, total_cases, total_pop):
        p_in = c / n
        p_out = (total_cases - c) / (total_pop - n)
        p0 = total_cases / total_pop

        def t(a, p):
            return a * math.log(p) if a > 0 else 0.0

        return (
            t(c, p_in)
            + t(n - c, 1 - p_in)
            + t(total_cases - c, p_out)
            + t(total_pop - n - (total_cases - c), 1 - p_out)
            - t(total_cases, p0)
            - t(total_pop - total_cases, 1 - p0)
        )

    for c, n, total_cases, total_pop in [
        (40, 200, 100, 1000),
        (15, 50, 60, 500),
        (9, 30, 25, 400),
    ]:
        assert binomial_llr(c, n, total_cases, total_pop) == pytest.approx(
            direct(c, n, total_cases, total_pop)
        )


def test_binomial_llr_whole_map_is_zero():
    assert binomial_llr(100, 1000, 100, 1000) == 0.0
