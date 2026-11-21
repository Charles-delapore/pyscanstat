"""Cross-validation against R smerc (CRAN 1.8.6, GPL >= 2).

Compares the Poisson circular scan on the shared ``make_clustered``
fixture: smerc::scan.test (nsim=99, min.cases=2, ubpop=0.5, Euclidean
distances) vs :func:`pyscanstat.scan_poisson` with the same settings.
The two implementations use different RNG streams for the Monte Carlo
replicates, so the p-value is compared with a tolerance of 0.02 while
the LLR, relative risk and MLC location set must agree tightly.

Skipped when Rscript or the smerc package is unavailable.
"""

import shutil
import subprocess

import numpy as np
import pytest

from pyscanstat import make_clustered, scan_poisson

R_SCRIPT = __file__.rsplit("/", 1)[0] + "/smerc_compare.R"


def _smerc_available():
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e",
         'quit(status = if (requireNamespace("smerc", quietly=TRUE)) 0 else 1)'],
        capture_output=True, text=True, timeout=120,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _smerc_available(),
                    reason="Rscript or R package smerc unavailable")
def test_poisson_scan_matches_smerc(tmp_path):
    coords, cases, pop = make_clustered(n=64, seed=42)
    fixture = tmp_path / "fixture.csv"
    np.savetxt(fixture, np.column_stack([coords, cases, pop]),
               delimiter=",", header="x,y,cases,pop", comments="")
    out = tmp_path / "smerc_out.csv"
    proc = subprocess.run(
        ["Rscript", R_SCRIPT, str(fixture), str(out), "99", "123"],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        pytest.skip(f"smerc run failed: {proc.stderr.strip()[:300]}")
    smerc = np.genfromtxt(out, delimiter=",", names=True, dtype=None,
                          encoding="utf-8")
    locids = [int(v) for v in
              str(smerc["locids"]).strip('"').split(";")]

    result = scan_poisson(coords, cases, pop, n_replicates=99,
                          random_state=123, min_cases=2)
    mlc = result.most_likely_cluster

    assert mlc.llr == pytest.approx(float(smerc["loglikrat"]), rel=1e-6)
    assert mlc.relative_risk == pytest.approx(float(smerc["rr"]),
                                              rel=1e-6)
    assert mlc.cases == pytest.approx(float(smerc["cases"]))
    assert mlc.expected == pytest.approx(float(smerc["expected"]),
                                         rel=1e-6)
    assert sorted((mlc.location_indices + 1).tolist()) == locids
    assert result.p_value == pytest.approx(float(smerc["pvalue"]),
                                           abs=0.02)
