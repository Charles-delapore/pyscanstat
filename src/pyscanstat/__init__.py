"""pyscanstat: exact-likelihood spatial and space-time scan statistics.

Implements Kulldorff's (1997) circular spatial scan statistic with exact
Poisson and binomial log-likelihood ratios and Monte Carlo significance
testing, plus a space-time cylindrical scan. Pure Python + Numba; no
SaTScan binary required.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
