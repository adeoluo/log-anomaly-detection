"""Invariant mining (Lou et al., USENIX ATC 2010): sparse integer relations among event counts.

In normal executions many event counts obey linear invariants, for example every "open" is
followed by one "close", so ``count(open) - count(close) == 0``. The miner:

1. estimates the dimension ``r`` of the invariant space from the spectrum of the training count
   matrix (eigenvalues of its Gram matrix that are numerically zero);
2. searches column subsets of growing size (1, 2, ... ``max_size``) for an integer vector ``v``
   with ``counts[:, subset] @ v == 0`` on at least ``support`` of the training rows, skipping
   supersets of invariants already found, until ``r`` invariants are found;
3. flags a row as anomalous when it violates any invariant.

The search works on deduplicated rows with weights, and candidate vectors come from the smallest
eigenvector of the subset's weighted Gram matrix, so each candidate costs a tiny eigensolve.
Size-one invariants say "this event never happens in normal runs".
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.detectors.base import CountDetector, check_counts
from log_anomaly_detection.features import unique_rows


def _integer_vector(v: NDArray[np.float64], max_coef: int) -> NDArray[np.int64] | None:
    nz = np.abs(v) > 1e-6
    if not np.all(nz):
        return None
    scaled = v / np.min(np.abs(v))
    for mult in range(1, max_coef + 1):
        cand = scaled * mult
        rounded = np.round(cand)
        if np.all(np.abs(cand - rounded) < 0.05) and np.max(np.abs(rounded)) <= max_coef:
            out: NDArray[np.int64] = rounded.astype(np.int64)
            return -out if out[0] < 0 else out
    return None


class InvariantsDetector(CountDetector):
    name = "Invariant mining"

    def __init__(
        self,
        support: float = 0.98,
        max_size: int = 3,
        max_coef: int = 3,
        rank_tol: float = 1e-6,
        max_columns: int = 200,
    ) -> None:
        if not 0.5 < support <= 1.0:
            raise ValueError("support must be in (0.5, 1]")
        if max_size < 1 or max_coef < 1:
            raise ValueError("max_size and max_coef must be at least 1")
        self.support = support
        self.max_size = max_size
        self.max_coef = max_coef
        self.rank_tol = rank_tol
        self.max_columns = max_columns
        self.invariants_: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
        self.expected_rank_ = 0
        self.width_ = 0
        self.threshold_ = 0.5

    def params(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "max_size": self.max_size,
            "invariants": len(self.invariants_),
            "invariant_space": self.expected_rank_,
        }

    def fit(self, counts: NDArray[np.floating]) -> InvariantsDetector:
        x = check_counts(counts)
        if x.shape[0] == 0:
            raise ValueError("cannot fit on zero rows")
        self.width_ = x.shape[1]
        rows, weights, _ = unique_rows(x)
        w = weights.astype(np.float64)
        total = w.sum()
        gram = (rows * w[:, None]).T @ rows
        eig = np.linalg.eigvalsh(gram)
        scale = max(float(np.max(eig)), 1.0)
        self.expected_rank_ = int(np.count_nonzero(eig <= self.rank_tol * scale))
        found: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
        found_sets: list[frozenset[int]] = []

        def holds(cols: NDArray[np.int64], vec: NDArray[np.int64]) -> bool:
            ok = np.abs(rows[:, cols] @ vec) < 1e-9
            return bool(float(w[ok].sum()) >= self.support * total)

        # Size one: events (almost) never seen in training.
        for c in range(self.width_):
            cols = np.array([c])
            if holds(cols, np.array([1])):
                found.append((cols, np.array([1], dtype=np.int64)))
                found_sets.append(frozenset([c]))
        active = [c for c in range(self.width_) if frozenset([c]) not in found_sets][
            : self.max_columns
        ]
        for size in range(2, self.max_size + 1):
            if len(found) >= max(self.expected_rank_, 1) and size > 2:
                break
            for combo in itertools.combinations(active, size):
                cs = frozenset(combo)
                if any(f <= cs for f in found_sets):
                    continue
                cols = np.asarray(combo, dtype=np.int64)
                vec = self._candidate(rows, w, gram, cols)
                if vec is None or not holds(cols, vec):
                    continue
                found.append((cols, vec))
                found_sets.append(cs)
        self.invariants_ = found
        return self

    def _candidate(
        self,
        rows: NDArray[np.float64],
        w: NDArray[np.float64],
        gram: NDArray[np.float64],
        cols: NDArray[np.int64],
    ) -> NDArray[np.int64] | None:
        """An integer null vector for ``cols``, refitted once on the rows it fits best."""

        vec = np.linalg.eigh(gram[np.ix_(cols, cols)])[1][:, 0]
        candidate = _integer_vector(vec, self.max_coef)
        if candidate is not None:
            return candidate
        # A few violating rows can tilt the null vector: refit on the best-fitting rows.
        sub = rows[:, cols]
        residual = np.abs(sub @ vec)
        order = np.argsort(residual, kind="stable")
        cum = np.cumsum(w[order])
        keep = order[: int(np.searchsorted(cum, self.support * cum[-1])) + 1]
        trimmed = (sub[keep] * w[keep, None]).T @ sub[keep]
        return _integer_vector(np.linalg.eigh(trimmed)[1][:, 0], self.max_coef)

    def violations(self, counts: NDArray[np.floating]) -> NDArray[np.int64]:
        """How many invariants each row violates."""

        if not self.width_:
            raise RuntimeError("invariant miner is not fitted")
        x = check_counts(counts, self.width_)
        out = np.zeros(x.shape[0], dtype=np.int64)
        for cols, vec in self.invariants_:
            out += np.abs(x[:, cols] @ vec) > 1e-9
        return out

    def score(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        return self.violations(counts).astype(np.float64)
