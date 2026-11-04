"""Feature extraction: event sequences per session or time window, and event count matrices.

Sequences are stored in compressed form (:class:`Sequences`): one flat array of event ids and an
offsets array, so 575,000 HDFS sessions or millions of BGL lines cost a few arrays, not millions
of Python lists. Count matrices, TF-IDF weighting and window building are vectorized with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class Sequences:
    """Event-id sequences in CSR form: sequence ``i`` is ``events[offsets[i]:offsets[i + 1]]``."""

    events: NDArray[np.int64]
    offsets: NDArray[np.int64]

    def __post_init__(self) -> None:
        self.events = np.asarray(self.events, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if self.offsets.ndim != 1 or self.offsets.size == 0 or self.offsets[0] != 0:
            raise ValueError("offsets must be a 1-D array starting at 0")
        if self.offsets[-1] != self.events.size or np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be non-decreasing and end at len(events)")

    @classmethod
    def from_lists(cls, seqs: list[list[int]]) -> Sequences:
        lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat = [e for s in seqs for e in s]
        return cls(np.asarray(flat, dtype=np.int64), offsets)

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def __getitem__(self, i: int) -> NDArray[np.int64]:
        return self.events[self.offsets[i] : self.offsets[i + 1]]

    @property
    def lengths(self) -> NDArray[np.int64]:
        return np.diff(self.offsets)

    def rows(self) -> NDArray[np.int64]:
        """The sequence index of every element of ``events``."""

        return np.repeat(np.arange(len(self), dtype=np.int64), self.lengths)

    def subset(self, index: NDArray[np.integer] | NDArray[np.bool_]) -> Sequences:
        """The sequences selected by an index array or boolean mask, in that order."""

        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        lengths = self.lengths[idx]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        if idx.size == 0:
            return Sequences(np.empty(0, np.int64), offsets)
        starts = self.offsets[idx]
        gather = np.repeat(starts - offsets[:-1], lengths) + np.arange(offsets[-1])
        return Sequences(self.events[gather], offsets)

    def to_lists(self) -> list[list[int]]:
        return [self[i].tolist() for i in range(len(self))]


def group_by_key(keys: ArrayLike, events: ArrayLike, n_keys: int) -> Sequences:
    """Group ``events`` by ``keys`` (one key per event), keeping each group's original order."""

    k: NDArray[np.int64] = np.asarray(keys, dtype=np.int64)
    ev: NDArray[np.int64] = np.asarray(events, dtype=np.int64)
    if k.shape != ev.shape:
        raise ValueError("keys and events must have the same length")
    if k.size and (int(np.min(k)) < 0 or int(np.max(k)) >= n_keys):
        raise ValueError("keys must lie in [0, n_keys)")
    order = np.argsort(k, kind="stable")
    counts = np.bincount(k, minlength=n_keys)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Sequences(ev[order], offsets)


def time_windows(
    times: ArrayLike,
    events: ArrayLike,
    labels: ArrayLike | None,
    size: int,
    step: int | None = None,
) -> tuple[Sequences, NDArray[np.uint8], NDArray[np.int64]]:
    """Group a time-ordered event stream into fixed (``step == size``) or sliding windows.

    Returns the non-empty windows, a label per window (1 if any line in it is anomalous) and each
    window's start time. ``times`` are integer seconds.
    """

    step = size if step is None else step
    if size <= 0 or step <= 0:
        raise ValueError("window size and step must be positive")
    if step > size:
        raise ValueError("step must not exceed the window size")
    t: NDArray[np.int64] = np.asarray(times, dtype=np.int64)
    ev: NDArray[np.int64] = np.asarray(events, dtype=np.int64)
    if t.shape != ev.shape:
        raise ValueError("times and events must have the same length")
    line_labels = (
        np.zeros(t.size, dtype=np.uint8) if labels is None else np.asarray(labels, np.uint8)
    )
    if t.size == 0:
        return (
            Sequences(np.empty(0, np.int64), np.zeros(1, np.int64)),
            np.empty(0, np.uint8),
            np.empty(0, np.int64),
        )
    order = np.argsort(t, kind="stable")
    t, ev, line_labels = t[order], ev[order], line_labels[order]
    start = t[0]
    rel = t - start
    copies = -(-size // step)
    window_ids = []
    pos = []
    for k in range(copies):
        # A line at relative time r belongs to windows w with w*step <= r < w*step + size.
        w = rel // step - k
        ok = (w >= 0) & (rel < w * step + size)
        window_ids.append(w[ok])
        pos.append(np.flatnonzero(ok))
    wid = np.concatenate(window_ids)
    idx = np.concatenate(pos)
    order2 = np.lexsort((idx, wid))
    wid, idx = wid[order2], idx[order2]
    uniq, starts, counts = np.unique(wid, return_index=True, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    seqs = Sequences(ev[idx], offsets)
    win_labels = np.maximum.reduceat(line_labels[idx], starts).astype(np.uint8)
    return seqs, win_labels, start + uniq * step


def count_windows(
    events: ArrayLike,
    labels: ArrayLike | None,
    size: int,
    step: int | None = None,
) -> tuple[Sequences, NDArray[np.uint8]]:
    """Group an event stream into windows of ``size`` consecutive lines, advancing by ``step``."""

    step = size if step is None else step
    if size <= 0 or step <= 0:
        raise ValueError("window size and step must be positive")
    ev: NDArray[np.int64] = np.asarray(events, dtype=np.int64)
    line_labels = (
        np.zeros(ev.size, dtype=np.uint8) if labels is None else np.asarray(labels, np.uint8)
    )
    n = ev.size
    if n == 0:
        return Sequences(np.empty(0, np.int64), np.zeros(1, np.int64)), np.empty(0, np.uint8)
    starts = np.arange(0, max(n - size, 0) + 1, step, dtype=np.int64)
    ends = np.minimum(starts + size, n)
    lengths = ends - starts
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    gather = np.repeat(starts - offsets[:-1], lengths) + np.arange(offsets[-1])
    seqs = Sequences(ev[gather], offsets)
    win_labels = np.maximum.reduceat(line_labels[gather], offsets[:-1]).astype(np.uint8)
    return seqs, win_labels


def count_matrix(seqs: Sequences, n_events: int) -> NDArray[np.float32]:
    """Event count matrix: ``X[i, e]`` is how often event ``e`` occurs in sequence ``i``.

    Event ids at or above ``n_events`` (events never seen in training) are counted in an extra
    last column, so the matrix has ``n_events + 1`` columns.
    """

    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    ev = np.minimum(seqs.events, n_events)
    if ev.size and ev.min() < 0:
        raise ValueError("event ids must be non-negative")
    width = n_events + 1
    flat = np.bincount(seqs.rows() * width + ev, minlength=len(seqs) * width)
    return flat.reshape(len(seqs), width).astype(np.float32)


@dataclass
class FeatureTransform:
    """TF-IDF weighting and zero-mean normalization fitted on training counts (as in LogPAI)."""

    idf: NDArray[np.float64] | None = None
    mean: NDArray[np.float64] | None = None
    tfidf: bool = True
    zero_mean: bool = True

    def fit(self, counts: NDArray[np.floating]) -> FeatureTransform:
        x = np.asarray(counts, dtype=np.float64)
        n = x.shape[0]
        if n == 0:
            raise ValueError("cannot fit on zero rows")
        if self.tfidf:
            df = np.count_nonzero(x > 0, axis=0)
            self.idf = np.log(n / (df + 1e-8))
            x = x * self.idf
        self.mean = x.mean(axis=0) if self.zero_mean else np.zeros(x.shape[1])
        return self

    def transform(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.mean is None:
            raise RuntimeError("FeatureTransform is not fitted")
        x = np.asarray(counts, dtype=np.float64)
        if self.tfidf:
            assert self.idf is not None
            x = x * self.idf
        return x - self.mean

    def fit_transform(self, counts: NDArray[np.floating]) -> NDArray[np.float64]:
        return self.fit(counts).transform(counts)


def unique_rows(
    x: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Distinct rows of ``x``, how often each occurs, and the inverse index."""

    uniq, inverse, counts = np.unique(
        np.asarray(x, dtype=np.float64), axis=0, return_inverse=True, return_counts=True
    )
    return uniq, counts.astype(np.int64), inverse.reshape(-1).astype(np.int64)
