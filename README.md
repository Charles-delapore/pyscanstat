# pyscanstat

Exact-likelihood spatial and space-time scan statistics (Kulldorff 1997)
in pure Python with Numba-accelerated scan loops. No SaTScan binary needed.

## Scope (narrowed, by design)

This library implements the **exact likelihood-ratio + Monte Carlo
significance** line of Kulldorff's spatial scan statistic:

- Circular spatial scan, **Poisson** and **binomial (Bernoulli)** models,
  exact log-likelihood ratios (Kulldorff 1997, *Communications in
  Statistics: Theory and Methods* 26(6), 1481–1496,
  DOI 10.1080/03610929708831995).
- Monte Carlo significance testing: `p = (1 + #{replicates >= observed}) / (1 + R)`
  (Dwass 1957), with a reproducible seed, and secondary-cluster p-values
  computed against the same null distribution of the maximum.
- Space-time cylindrical scan (Poisson) with Monte Carlo p-values.
- Results: most likely cluster, non-overlapping secondary clusters,
  relative risk (RR) and log-likelihood ratio (LLR) per cluster.

It deliberately does **not** implement the approximate-maximisation line:
`pyscan` / `pyscan-stats` 1.0.2 (2025-08-19) already covers approximate
Kulldorff-discrepancy maximisation over halfplanes/disks/rectangles with a
C++ backend, and its documentation and PyPI page show no Monte Carlo
significance path. `pyscanstat` is the complementary exact route. It is
also not a SaTScan wrapper: SaTScan is closed-source freeware under a
custom proprietary licence with no Python binding (only the R `rsatscan`
wrapper, which still needs a SaTScan install). The R package `smerc`
(CRAN 1.8.6, 2026-08-20) is the closest reference implementation; the
test suite cross-validates the Poisson scan against it where R is
available.

## Install

```bash
pip install pyscanstat
```

Requires Python >= 3.9 and NumPy. Numba is a hard dependency for the
accelerated path; if it is unavailable the package falls back to the same
kernels executed as plain Python (much slower, identical results).

## Quick start

```python
import numpy as np
from pyscanstat import scan_poisson, make_clustered

coords, cases, population = make_clustered(n=100, seed=42)
result = scan_poisson(coords, cases, population, n_replicates=999,
                      random_state=7)
print(result.most_likely_cluster)
print("p =", result.p_value)
```

## Performance design

Scanning is O(N^2) candidate zones (one centre per location, radii grown
through the distance-ordered neighbours) and Monte Carlo multiplies that
by the number of replicates, so the hot loops are compiled:

- **Candidate generation is vectorised**: the full N x N distance matrix
  and the per-centre neighbour ordering are built once with NumPy; the
  population-based zone mask (default: at most 50% of the total
  population, the SaTScan default) is computed once and reused for every
  Monte Carlo replicate.
- **LLR evaluation is Numba-compiled** (`src/pyscanstat/_core.py`): the
  per-zone Poisson/binomial log-likelihood ratio loop runs as `njit`
  machine code over flat arrays, with cumulative case/expected counts
  precomputed by NumPy cumsums.
- **Monte Carlo replicates are generated vectorised**: replicate case
  vectors come from a single NumPy `Generator.multinomial` /
  `multivariate_hypergeometric` call per replicate, and each replicate
  reuses the precomputed zone structure, so only the LLR kernel is
  re-executed. No Python-level loop over zones exists anywhere.
- **Space-time scan** reuses the same spatial zones and evaluates all
  (zone, end-time, duration) cylinders in one compiled kernel over
  time-cumsummed counts.

Indicative timing on this machine (Python 3.12, Numba 0.68, N=200
locations, 19,923 candidate zones):

- One Poisson scan: **1.2 ms** compiled vs **59 ms** pure-Python fallback
  (~49x; identical max LLR 567.24) — the fallback exists for
  correctness, not speed.
- Full scan + 999 Monte Carlo replicates: **1.7 s**.
- Space-time scan, N=64, T=6, + 199 replicates: **1.6 s** (membership
  matrix cached on the zone set; 3.4x faster than rebuilding it per
  replicate).

Scaling notes: candidate zones are O(N^2) and the distance matrix is
N x N, so N in the low thousands is comfortable (N=2000: ~2M zones,
~32 MB distance matrix); beyond that, chunked zone evaluation is the
known next step (see REPORT.md).

## Multiple testing

The scan statistic is the *maximum* LLR over all candidate zones, and the
Monte Carlo p-value is computed against the distribution of that maximum
under the null, so the primary p-value already accounts for searching
over zones. Secondary clusters are reported for exploration; their
p-values use the same max-null distribution and are conservative.

## Licence

MIT. See `LICENSE`.
