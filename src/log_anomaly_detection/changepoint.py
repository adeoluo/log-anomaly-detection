"""Change-point detection for performance-baseline regressions: CUSUM and Bayesian online CPD.

* :class:`CUSUM` is Page's two-sided tabular cumulative-sum test on a series standardized against
  a baseline window: ``S+ = max(0, S+ + z - k)`` and ``S- = max(0, S- - z - k)`` raise an alarm
  when either exceeds ``h``; the change is placed where the alarming sum last left zero, and the
  baseline is re-estimated after each change.
* :class:`BOCPD` is Bayesian online change-point detection (Adams and MacKay, 2007) with a
  Normal-Gamma conjugate model and a constant hazard. It keeps the full run-length posterior
  (vectorized over run lengths, with negligible tails pruned); offline, change points come from
  walking the most probable run lengths back from the end of the series.

Both run online (``update`` one value at a time, for streaming metrics) or offline over an array
(``detect``). :func:`detect_regressions` turns change points into regressions with before and
after means. :func:`f_measure` and :func:`covering` score change points against several human
annotators exactly as the Turing Change Point Detection benchmark defines them (van den Burg and
Williams, 2020): a margin of 5 points, the start of the series counted as a change point, and no
double counting.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _standardize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    sd = float(np.std(x))
    return (x - float(np.mean(x))) / (sd if sd > 0 else 1.0)


class CUSUM:
    """Two-sided tabular CUSUM for shifts in the mean, with automatic re-baselining."""

    def __init__(
        self, threshold: float = 10.0, drift: float = 0.5, baseline: int = 20, min_sd: float = 1e-9
    ) -> None:
        if threshold <= 0 or drift < 0 or baseline < 2:
            raise ValueError("threshold must be positive, drift non-negative, baseline >= 2")
        self.h = threshold
        self.k = drift
        self.baseline = baseline
        self.min_sd = min_sd
        self.reset(0)

    def reset(self, t: int) -> None:
        """Start a new segment at index ``t``: collect a fresh baseline."""

        self._buffer: list[float] = []
        self._mean = 0.0
        self._sd = 1.0
        self._pos = 0.0
        self._neg = 0.0
        self._pos_start = t
        self._neg_start = t
        self._segment_start = t

    def update(self, t: int, x: float) -> int | None:
        """Feed value ``x`` at index ``t``; return the change index if an alarm fires."""

        if len(self._buffer) < self.baseline:
            self._buffer.append(float(x))
            if len(self._buffer) == self.baseline:
                arr = np.asarray(self._buffer)
                self._mean = float(arr.mean())
                self._sd = max(float(arr.std(ddof=1)), self.min_sd, 1e-6 * abs(self._mean))
                self._pos_start = self._neg_start = t + 1
            return None
        z = (float(x) - self._mean) / self._sd
        if self._pos == 0.0:
            self._pos_start = t
        if self._neg == 0.0:
            self._neg_start = t
        self._pos = max(0.0, self._pos + z - self.k)
        self._neg = max(0.0, self._neg - z - self.k)
        if self._pos > self.h or self._neg > self.h:
            change = self._pos_start if self._pos > self.h else self._neg_start
            self.reset(t + 1)
            return change
        return None

    def detect(self, values: ArrayLike) -> list[int]:
        """Change-point indices (0-based) of a whole series.

        After each change the detector restarts at the change and re-baselines on the values that
        follow it, so consecutive shifts are found as separate changes.
        """

        x = np.asarray(values, dtype=np.float64).reshape(-1)
        out: list[int] = []
        start = 0
        while start < x.size:
            self.reset(start)
            change = None
            for t in range(start, x.size):
                change = self.update(t, float(x[t]))
                if change is not None:
                    break
            if change is None:
                break
            out.append(change)
            start = change
        return out


class BOCPD:
    """Bayesian online change-point detection with a Normal-Gamma model and constant hazard."""

    def __init__(
        self,
        hazard: float = 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        prune: float = 1e-10,
        max_run: int | None = None,
    ) -> None:
        if hazard <= 1 or kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("hazard must exceed 1 and the prior parameters must be positive")
        self.hazard = hazard
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.prune = prune
        self.max_run = max_run
        self._lg = np.empty(0)
        self.reset()

    def reset(self) -> None:
        mu0, kappa0, alpha0, beta0 = self.prior
        self._probs = np.array([1.0])
        self._mu = np.array([mu0])
        self._kappa = np.array([kappa0])
        self._alpha = np.array([alpha0])
        self._beta = np.array([beta0])
        self._t = 0
        self._last_map = 0

    def _lgamma(self, size: int) -> NDArray[np.float64]:
        """``lgamma(alpha0 + j / 2)`` for ``j < size``; run length r has alpha0 + r / 2."""

        if self._lg.size < size:
            alpha0 = self.prior[2]
            n = max(size, 2 * self._lg.size)
            self._lg = np.array([math.lgamma(alpha0 + 0.5 * j) for j in range(n)])
        return self._lg

    def _predictive(self, x: float) -> NDArray[np.float64]:
        runs = self._alpha.size
        lg = self._lgamma(runs + 1)
        df = 2.0 * self._alpha
        scale2 = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        z2 = (x - self._mu) ** 2 / scale2
        log_pdf = (
            lg[1 : runs + 1]
            - lg[:runs]
            - 0.5 * np.log(np.pi * df * scale2)
            - (df + 1.0) / 2.0 * np.log1p(z2 / df)
        )
        out: NDArray[np.float64] = np.exp(log_pdf)
        return out

    def update(self, x: float) -> tuple[int, float]:
        """Feed one value; return (most probable run length, probability of a change now)."""

        h = 1.0 / self.hazard
        pred = self._predictive(float(x))
        growth = self._probs * pred * (1.0 - h)
        change = float(np.sum(self._probs * pred * h))
        probs = np.concatenate([[change], growth])
        total = probs.sum()
        probs = probs / total if total > 0 and np.isfinite(total) else np.eye(1, probs.size)[0]
        mu0, kappa0, alpha0, beta0 = self.prior
        new_mu = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        new_beta = self._beta + self._kappa * (x - self._mu) ** 2 / (2.0 * (self._kappa + 1.0))
        self._mu = np.concatenate([[mu0], new_mu])
        self._kappa = np.concatenate([[kappa0], self._kappa + 1.0])
        self._alpha = np.concatenate([[alpha0], self._alpha + 0.5])
        self._beta = np.concatenate([[beta0], new_beta])
        keep = probs >= self.prune
        keep[0] = True
        if self.max_run is not None:
            keep[self.max_run + 1 :] = False
        if not keep.all():
            last = int(np.flatnonzero(keep)[-1]) + 1
            probs, keep = probs[:last], keep[:last]
            probs = np.where(keep, probs, 0.0)
            probs = probs / probs.sum()
            self._mu, self._kappa = self._mu[:last], self._kappa[:last]
            self._alpha, self._beta = self._alpha[:last], self._beta[:last]
        self._probs = probs
        self._t += 1
        return int(np.argmax(probs)), float(probs[0])

    def detect(self, values: ArrayLike, standardize: bool = True) -> list[int]:
        """Change-point indices of a whole series, from the MAP run-length path.

        Starting at the last point, the most probable run length says where the current segment
        began; that start is a change point, and the walk repeats from the point just before it.
        """

        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if standardize and x.size:
            x = _standardize(x)
        self.reset()
        runs = [self.update(float(value))[0] for value in x]
        out: list[int] = []
        t = x.size - 1
        while t > 0:
            start = t - runs[t] + 1
            if start <= 0:
                break
            out.append(start)
            t = start - 1
        return sorted(out)


@dataclass(frozen=True)
class Regression:
    """A change in a metric's level, with before and after means over the adjacent segments."""

    index: int
    before: float
    after: float

    @property
    def change(self) -> float:
        return self.after - self.before

    @property
    def relative(self) -> float:
        return self.change / abs(self.before) if self.before else math.inf

    def as_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["change"] = self.change
        out["relative"] = self.relative
        return out


def detect_regressions(
    values: ArrayLike,
    method: str = "cusum",
    direction: str = "up",
    min_relative: float = 0.05,
    min_segment: int = 3,
    **kwargs: Any,
) -> list[Regression]:
    """Level shifts of at least ``min_relative`` in ``direction`` ("up", "down" or "both").

    For latency or memory "up" is a regression; for throughput "down" is. Segments shorter than
    ``min_segment`` points are spikes, not level shifts: their boundaries are dropped.
    """

    if direction not in ("up", "down", "both"):
        raise ValueError("direction must be 'up', 'down' or 'both'")
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if method == "cusum":
        cps = CUSUM(**kwargs).detect(x)
    elif method == "bocpd":
        cps = BOCPD(**kwargs).detect(x)
    else:
        raise ValueError("method must be 'cusum' or 'bocpd'")
    cps = drop_short_segments(cps, x.size, min_segment)
    bounds = [0, *cps, x.size]
    out = []
    for i, cp in enumerate(cps, start=1):
        before = float(x[bounds[i - 1] : cp].mean())
        after = float(x[cp : bounds[i + 1]].mean())
        reg = Regression(cp, before, after)
        rel = reg.relative
        if direction == "up" and rel < min_relative:
            continue
        if direction == "down" and rel > -min_relative:
            continue
        if direction == "both" and abs(rel) < min_relative:
            continue
        out.append(reg)
    return out


def drop_short_segments(cps: Sequence[int], n: int, min_segment: int) -> list[int]:
    """Remove the boundaries of every segment shorter than ``min_segment`` (isolated spikes)."""

    if min_segment <= 1:
        return list(cps)
    bounds = [0, *sorted(set(cps)), n]
    drop: set[int] = set()
    for i in range(len(bounds) - 1):
        if bounds[i + 1] - bounds[i] < min_segment:
            drop.update(b for b in (bounds[i], bounds[i + 1]) if 0 < b < n)
    return [c for c in bounds[1:-1] if c not in drop]


# ---------------------------------------------------------------- TCPD evaluation metrics
def _true_positives(truth: Iterable[int], predicted: Iterable[int], margin: int) -> int:
    available = sorted(set(predicted))
    hits = 0
    for tau in sorted(set(truth)):
        best = None
        for x in available:
            d = abs(tau - x)
            if d <= margin and (best is None or (d, x) < best):
                best = (d, x)
        if best is not None:
            hits += 1
            available.remove(best[1])
    return hits


def f_measure(
    annotations: Mapping[str, Sequence[int]], predictions: Iterable[int], margin: int = 5
) -> tuple[float, float, float]:
    """(F1, precision, recall) of predicted change points against several annotators."""

    if not annotations:
        raise ValueError("need at least one annotator")
    sets = [{0, *map(int, a)} for a in annotations.values()]
    x = {0, *map(int, predictions)}
    union: set[int] = set().union(*sets)
    precision = _true_positives(union, x, margin) / len(x)
    recall = sum(_true_positives(t, x, margin) / len(t) for t in sets) / len(sets)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _segments(cps: Iterable[int], n: int) -> list[tuple[int, int]]:
    cuts = sorted({c for c in cps if 0 < c < n})
    bounds = [0, *cuts, n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def covering(annotations: Mapping[str, Sequence[int]], predictions: Iterable[int], n: int) -> float:
    """Average segmentation covering of each annotator's partition by the predicted partition."""

    if not annotations:
        raise ValueError("need at least one annotator")
    predicted = _segments(predictions, n)
    total = 0.0
    for cps in annotations.values():
        score = 0.0
        for a0, a1 in _segments(cps, n):
            best = 0.0
            for b0, b1 in predicted:
                inter = max(0, min(a1, b1) - max(a0, b0))
                if inter:
                    best = max(best, inter / (max(a1, b1) - min(a0, b0)))
            score += (a1 - a0) * best
        total += score / n
    return total / len(annotations)
