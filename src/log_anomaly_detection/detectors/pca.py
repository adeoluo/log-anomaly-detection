"""PCA anomaly detection (Xu et al., SOSP 2009) with the squared prediction error statistic.

Training rows are TF-IDF weighted and centred; the principal subspace keeps the components that
explain ``variance`` of the training variance. A row's anomaly score is its squared distance from
that subspace (the SPE, or Q statistic), and the threshold is the Jackson and Mudholkar (1979)
upper limit of the Q statistic at confidence ``1 - alpha``, as in the LogPAI implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.detectors.base import CountDetector, check_counts
from log_anomaly_detection.features import FeatureTransform

# Standard normal quantiles for the supported significance levels.
_Z = {0.05: 1.6449, 0.01: 2.3263, 0.001: 3.0902, 0.0001: 3.7190}


class PCADetector(CountDetector):
    name = "PCA"

    def __init__(self, variance: float = 0.95, alpha: float = 0.001, tfidf: bool = True) -> None:
        if not 0 < variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        if alpha not in _Z:
            raise ValueError(f"alpha must be one of {sorted(_Z)}")
        self.variance = variance
        self.alpha = alpha
        self.transform = FeatureTransform(tfidf=tfidf)
        self.residual_: NDArray[np.float64] | None = None
        self.components_: int = 0

    def params(self) -> dict[str, Any]:
        return {"variance": self.variance, "alpha": self.alpha, "components": self.components_}

    def fit(self, counts: NDArray[np.floating]) -> PCADetector:
        x = self.transform.fit_transform(check_counts(counts))
        cov = x.T @ x / x.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        total = eigvals.sum()
        if total <= 0:
            k = 0
        else:
            k = int(np.searchsorted(np.cumsum(eigvals) / total, self.variance - 1e-12) + 1)
            k = min(k, eigvals.size)
        self.components_ = k
        p = eigvecs[:, :k]
        self.residual_ = np.eye(x.shape[1]) - p @ p.T
        rest = eigvals[k:]
        phi = [float(np.sum(rest**i)) for i in (1, 2, 3)]
        if phi[0] <= 0 or phi[1] <= 0:
            self.threshold_ = 1e-12
            return self
        h0 = 1.0 - 2.0 * phi[0] * phi[2] / (3.0 * phi[1] ** 2)
        if h0 <= 0:
            h0 = 1e-3
        c = _Z[self.alpha]
        base = c * np.sqrt(2 * phi[1] * h0**2) / phi[0] + 1 + phi[1] * h0 * (h0 - 1) / phi[0] ** 2
        self.threshold_ = float(phi[0] * max(base, 0.0) ** (1.0 / h0))
        return self

    def score(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.residual_ is None:
            raise RuntimeError("PCA is not fitted")
        x = self.transform.transform(check_counts(counts, self.residual_.shape[0]))
        r = x @ self.residual_
        out: NDArray[np.float64] = np.einsum("ij,ij->i", r, r)
        return out
