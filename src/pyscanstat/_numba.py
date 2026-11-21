"""Optional-Numba shim.

Numba is a hard dependency of the distributed package, but the library is
written so that it still imports and runs (as plain Python, much slower)
when Numba is unavailable. All compiled kernels are written in the
Numba/Python intersection: plain loops over NumPy arrays, no objmode.
"""

try:
    from numba import njit as _njit
    from numba import prange as _prange

    HAS_NUMBA = True
    njit = _njit
    prange = _prange
except ImportError:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity stand-in for :func:`numba.njit`."""

        def wrap(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap

    prange = range
