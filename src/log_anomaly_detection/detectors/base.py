"""The detector interface shared by every model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CountDetector(ABC):
    """An anomaly detector over event count vectors (one row per session or window).

    ``fit`` sees training rows (normal sessions in the benchmarks); ``score`` returns a value per
    row where higher means more anomalous; ``predict`` applies the detector's own decision rule.
    """

    name: str = "detector"
    threshold_: float | None = None

    @abstractmethod
    def fit(self, counts: NDArray[np.floating]) -> CountDetector: ...

    @abstractmethod
    def score(self, counts: NDArray[np.floating]) -> NDArray[np.float64]: ...

    def predict(self, counts: NDArray[np.floating]) -> NDArray[np.bool_]:
        if self.threshold_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        return self.score(counts) > self.threshold_

    def params(self) -> dict[str, Any]:
        """Hyperparameters, for reports."""

        return {}


def check_counts(counts: NDArray[np.floating], width: int | None = None) -> NDArray[np.float64]:
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a 2-D count matrix")
    if width is not None and x.shape[1] != width:
        raise ValueError(f"expected {width} columns, got {x.shape[1]}")
    if not np.all(np.isfinite(x)):
        raise ValueError("count matrix contains NaN or infinity")
    return x
