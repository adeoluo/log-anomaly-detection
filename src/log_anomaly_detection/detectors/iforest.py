"""Isolation forest (Liu, Ting and Zhou, ICDM 2008), implemented with numpy.

Each tree isolates a subsample of ``max_samples`` rows by random axis-aligned splits up to depth
``ceil(log2(max_samples))``. A row's path length is averaged over trees and normalized by
``c(n)``, the average path length of an unsuccessful binary-search-tree lookup, giving the score
``s = 2 ** (-E[h(x)] / c(n))``. Trees are stored as flat arrays, and scoring walks every row down
a tree level by level with vectorized indexing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.detectors.base import CountDetector, check_counts

_EULER = 0.5772156649015329


def average_path_length(n: NDArray[np.floating] | float) -> NDArray[np.float64]:
    """``c(n)``: expected path length of an unsuccessful search in a BST of ``n`` nodes."""

    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + _EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class _Tree:
    __slots__ = ("feature", "left", "right", "size", "threshold")

    def __init__(self, x: NDArray[np.float64], rng: np.random.Generator, max_depth: int) -> None:
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        size: list[int] = []
        stack: list[tuple[NDArray[np.float64], int, int]] = []  # rows, depth, node id

        def new(rows: NDArray[np.float64], depth: int) -> int:
            node = len(feature)
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            size.append(rows.shape[0])
            stack.append((rows, depth, node))
            return node

        new(x, 0)
        while stack:
            rows, depth, node = stack.pop()
            if depth >= max_depth or rows.shape[0] <= 1:
                continue
            lo, hi = rows.min(axis=0), rows.max(axis=0)
            candidates = np.flatnonzero(hi > lo)
            if candidates.size == 0:
                continue
            f = int(rng.choice(candidates))
            t = float(rng.uniform(lo[f], hi[f]))
            mask = rows[:, f] < t
            feature[node] = f
            threshold[node] = t
            left[node] = new(rows[mask], depth + 1)
            right[node] = new(rows[~mask], depth + 1)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.size = np.asarray(size, dtype=np.float64)

    def path_lengths(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        node = np.zeros(x.shape[0], dtype=np.int64)
        depth = np.zeros(x.shape[0], dtype=np.float64)
        active = np.flatnonzero(self.feature[node] >= 0)
        while active.size:
            cur = node[active]
            go_left = x[active, self.feature[cur]] < self.threshold[cur]
            node[active] = np.where(go_left, self.left[cur], self.right[cur])
            depth[active] += 1.0
            active = active[self.feature[node[active]] >= 0]
        return depth + average_path_length(self.size[node])


class IsolationForestDetector(CountDetector):
    name = "Isolation forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        if n_estimators < 1 or max_samples < 2:
            raise ValueError("need at least one tree and two samples per tree")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.threshold_ = threshold
        self.trees_: list[_Tree] = []
        self.sample_size_ = 0
        self.width_ = 0

    def params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "threshold": self.threshold_,
            "seed": self.seed,
        }

    def fit(self, counts: NDArray[np.floating]) -> IsolationForestDetector:
        x = check_counts(counts)
        if x.shape[0] == 0:
            raise ValueError("cannot fit on zero rows")
        rng = np.random.default_rng(self.seed)
        n = min(self.max_samples, x.shape[0])
        depth = max(1, math.ceil(math.log2(max(n, 2))))
        self.sample_size_ = n
        self.width_ = x.shape[1]
        self.trees_ = [
            _Tree(x[rng.choice(x.shape[0], size=n, replace=False)], rng, depth)
            for _ in range(self.n_estimators)
        ]
        return self

    def score(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        if not self.trees_:
            raise RuntimeError("isolation forest is not fitted")
        x = check_counts(counts, self.width_)
        total = np.zeros(x.shape[0], dtype=np.float64)
        for tree in self.trees_:
            total += tree.path_lengths(x)
        mean = total / len(self.trees_)
        c = float(average_path_length(np.asarray([self.sample_size_], dtype=np.float64))[0])
        return np.power(2.0, -mean / max(c, 1e-12))
