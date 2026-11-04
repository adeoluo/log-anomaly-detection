"""LogCluster (Lin et al., ICSE 2016): cluster normal sequences, flag rows far from every cluster.

Rows are IDF-weighted event count vectors. Training sequences are clustered online: each vector
joins the nearest cluster when its cosine distance to the cluster's representative (the mean
vector) is at most ``max_distance``, otherwise it starts a new cluster. At test time a row's score
is its cosine distance to the nearest representative, and it is anomalous when that exceeds
``anomaly_threshold``. Training runs on deduplicated rows, which is exact for this procedure
because identical vectors always join the same cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.detectors.base import CountDetector, check_counts
from log_anomaly_detection.features import FeatureTransform, unique_rows


def _normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


class LogClusterDetector(CountDetector):
    name = "LogCluster"

    def __init__(self, max_distance: float = 0.3, anomaly_threshold: float = 0.3) -> None:
        if not 0 <= max_distance <= 2 or not 0 <= anomaly_threshold <= 2:
            raise ValueError("distances must be in [0, 2]")
        self.max_distance = max_distance
        self.threshold_ = anomaly_threshold
        self.transform = FeatureTransform(tfidf=True, zero_mean=False)
        self.representatives_: NDArray[np.float64] | None = None
        self.cluster_sizes_: NDArray[np.int64] | None = None

    def params(self) -> dict[str, Any]:
        n = 0 if self.representatives_ is None else int(self.representatives_.shape[0])
        return {"max_distance": self.max_distance, "threshold": self.threshold_, "clusters": n}

    def fit(self, counts: NDArray[np.floating]) -> LogClusterDetector:
        x = self.transform.fit_transform(check_counts(counts))
        rows, weights, _ = unique_rows(x)
        # Visit frequent patterns first so representatives settle on common behaviour.
        order = np.argsort(-weights, kind="stable")
        rows, weights = rows[order], weights[order]
        unit = _normalize(rows)
        sums = np.zeros((0, rows.shape[1]))
        reps = np.zeros((0, rows.shape[1]))
        sizes: list[int] = []
        for vec, u, w in zip(rows, unit, weights, strict=True):
            if reps.shape[0]:
                dist = 1.0 - reps @ u
                best = int(np.argmin(dist))
                if dist[best] <= self.max_distance:
                    sums[best] += vec * w
                    sizes[best] += int(w)
                    reps[best] = _normalize(sums[best][None, :])[0]
                    continue
            sums = np.vstack([sums, vec * w])
            reps = np.vstack([reps, u])
            sizes.append(int(w))
        self.representatives_ = reps
        self.cluster_sizes_ = np.asarray(sizes, dtype=np.int64)
        return self

    def score(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.representatives_ is None:
            raise RuntimeError("LogCluster is not fitted")
        x = self.transform.transform(check_counts(counts, self.representatives_.shape[1]))
        unit = _normalize(x)
        if self.representatives_.shape[0] == 0:
            return np.ones(x.shape[0])
        out = np.empty(x.shape[0])
        for start in range(0, x.shape[0], 65536):
            block = unit[start : start + 65536]
            out[start : start + 65536] = 1.0 - np.max(block @ self.representatives_.T, axis=1)
        # A zero vector (no events) is maximally far from every cluster.
        out[np.linalg.norm(x, axis=1) == 0] = 1.0
        return out
