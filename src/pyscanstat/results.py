"""Result containers for scan statistics."""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Cluster", "ScanResult"]


@dataclass
class Cluster:
    """One candidate cluster (most likely or secondary)."""

    center_index: int
    center: tuple
    radius: float
    location_indices: np.ndarray
    n_locations: int
    cases: float
    expected: float
    population: float
    llr: float
    relative_risk: float
    p_value: float | None = None
    # Space-time fields; None for purely spatial scans.
    time_start: int | None = None
    time_end: int | None = None
    duration: int | None = None

    def __repr__(self):  # pragma: no cover - display helper
        base = (
            f"Cluster(centre={self.center_index}, n_loc={self.n_locations}, "
            f"cases={self.cases:g}, expected={self.expected:.2f}, "
            f"LLR={self.llr:.3f}, RR={self.relative_risk:.3f}"
        )
        if self.p_value is not None:
            base += f", p={self.p_value:.4f}"
        if self.time_start is not None:
            base += f", t={self.time_start}..{self.time_end}"
        return base + ")"

    def to_dict(self):
        return {
            "center_index": self.center_index,
            "center": tuple(float(v) for v in self.center),
            "radius": float(self.radius),
            "location_indices": [int(i) for i in self.location_indices],
            "n_locations": self.n_locations,
            "cases": float(self.cases),
            "expected": float(self.expected),
            "population": float(self.population),
            "llr": float(self.llr),
            "relative_risk": float(self.relative_risk),
            "p_value": self.p_value,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "duration": self.duration,
        }


@dataclass
class ScanResult:
    """Result of a spatial or space-time scan."""

    model: str
    most_likely_cluster: Cluster | None
    secondary_clusters: list = field(default_factory=list)
    p_value: float | None = None
    n_replicates: int = 0
    n_zones: int = 0
    total_cases: float = 0.0
    total_population: float = 0.0
    statistic: str = "max_llr"
    replicates_max_llr: np.ndarray | None = None

    def __repr__(self):  # pragma: no cover - display helper
        mlc = self.most_likely_cluster
        return (
            f"ScanResult(model={self.model!r}, "
            f"max_LLR={mlc.llr if mlc else 0.0:.3f}, "
            f"p={self.p_value}, n_replicates={self.n_replicates}, "
            f"n_zones={self.n_zones}, "
            f"n_secondary={len(self.secondary_clusters)})"
        )
