"""pyscanstat: exact-likelihood spatial and space-time scan statistics.

Implements Kulldorff's (1997) circular spatial scan statistic with exact
Poisson and binomial log-likelihood ratios and Monte Carlo significance
testing, plus a space-time cylindrical scan. Pure Python + Numba; no
SaTScan binary required.
"""

from .llr import binomial_llr, poisson_llr
from .results import Cluster, ScanResult
from .scan import scan_binomial, scan_poisson

__version__ = "0.1.0"

__all__ = [
    "Cluster",
    "ScanResult",
    "scan_poisson",
    "scan_binomial",
    "poisson_llr",
    "binomial_llr",
    "__version__",
]
