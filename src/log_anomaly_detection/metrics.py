"""Detection metrics: precision, recall, F1, ROC AUC and average precision, vectorized."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class DetectionScores:
    """Binary detection quality, with anomalies as the positive class."""

    n: int
    anomalies: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    average_precision: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def prf(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int, float, float, float]:
    """True positives, false positives, false negatives, precision, recall and F1."""

    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp = int(np.count_nonzero(t & p))
    fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float | None:
    """Area under the ROC curve (Mann-Whitney statistic, ties count one half)."""

    t = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = int(t.sum()), int((~t).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    # Average ranks over ties.
    uniq_start = np.flatnonzero(np.concatenate([[True], sorted_s[1:] != sorted_s[:-1]]))
    uniq_end = np.concatenate([uniq_start[1:], [s.size]])
    avg = (uniq_start + uniq_end + 1) / 2.0
    ranks[order] = np.repeat(avg, uniq_end - uniq_start)
    return float((ranks[t].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float | None:
    """Average precision (area under the precision-recall curve, step interpolation)."""

    t = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    pos = int(t.sum())
    if pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted, t_sorted = s[order], t[order]
    last = np.flatnonzero(np.concatenate([s_sorted[1:] != s_sorted[:-1], [True]]))
    tps = np.cumsum(t_sorted)[last]
    fps = (last + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / pos
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scores: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> DetectionScores:
    """Score predictions; ``weights`` repeats rows (deduplicated sequences with counts)."""

    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    s = None if scores is None else np.asarray(scores, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.int64)
        t, p = np.repeat(t, w), np.repeat(p, w)
        s = None if s is None else np.repeat(s, w)
    tp, fp, fn, precision, recall, f1 = prf(t, p)
    return DetectionScores(
        n=int(t.size),
        anomalies=int(t.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=None if s is None else roc_auc(t, s),
        average_precision=None if s is None else average_precision(t, s),
    )


def best_f1_threshold(y_true: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """The score threshold (predict ``score > threshold``) that maximizes F1, and that F1."""

    t = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    pos = int(t.sum())
    if pos == 0 or s.size == 0:
        return float("inf"), 0.0
    order = np.argsort(-s, kind="mergesort")
    s_sorted, t_sorted = s[order], t[order]
    last = np.flatnonzero(np.concatenate([s_sorted[1:] != s_sorted[:-1], [True]]))
    tps = np.cumsum(t_sorted)[last]
    predicted = last + 1
    f1 = 2 * tps / (predicted + pos)
    best = int(np.argmax(f1))
    nxt = s_sorted[last[best] + 1] if last[best] + 1 < s.size else -np.inf
    threshold = (
        float((s_sorted[last[best]] + nxt) / 2)
        if np.isfinite(nxt)
        else float(s_sorted[last[best]] - 1.0)
    )
    return threshold, float(f1[best])
