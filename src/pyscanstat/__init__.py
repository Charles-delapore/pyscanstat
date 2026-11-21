"""pyscanstat: exact-likelihood spatial and space-time scan statistics.

Implements Kulldorff's (1997) circular spatial scan statistic with exact
Poisson and binomial log-likelihood ratios and Monte Carlo significance
testing, plus a space-time cylindrical scan. Pure Python + Numba; no
SaTScan binary required.
"""

from .llr import binomial_llr, poisson_llr
from .montecarlo import monte_carlo_pvalue, monte_carlo_scan
from .results import Cluster, ScanResult
from .scan import scan_binomial, scan_poisson
from .spacetime import scan_spacetime_poisson

__version__ = "0.1.0"

__all__ = [
    "Cluster",
    "ScanResult",
    "scan_poisson",
    "scan_binomial",
    "scan_spacetime_poisson",
    "monte_carlo_pvalue",
    "monte_carlo_scan",
    "poisson_llr",
    "binomial_llr",
    "__version__",
]
