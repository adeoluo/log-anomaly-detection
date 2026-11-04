from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from log_anomaly_detection.features import (
    FeatureTransform,
    Sequences,
    count_matrix,
    count_windows,
    group_by_key,
    time_windows,
    unique_rows,
)

SEQS = st.lists(st.lists(st.integers(0, 6), max_size=8), min_size=1, max_size=20)


@given(SEQS)
def test_round_trip_and_subset(seqs: list[list[int]]) -> None:
    s = Sequences.from_lists(seqs)
    assert s.to_lists() == seqs and len(s) == len(seqs)
    idx = np.arange(len(seqs))[::-1]
    assert s.subset(idx).to_lists() == seqs[::-1]
    mask = np.array([i % 2 == 0 for i in range(len(seqs))])
    assert s.subset(mask).to_lists() == [q for q, m in zip(seqs, mask, strict=True) if m]
    assert s.subset(np.array([], dtype=np.int64)).to_lists() == []


@given(SEQS)
def test_count_matrix_matches_python(seqs: list[list[int]]) -> None:
    x = count_matrix(Sequences.from_lists(seqs), 5)
    assert x.shape == (len(seqs), 6)
    for i, q in enumerate(seqs):
        expected = np.zeros(6)
        for e in q:
            expected[min(e, 5)] += 1
        assert np.array_equal(x[i], expected)


def test_sequences_validation() -> None:
    with pytest.raises(ValueError):
        Sequences(np.array([1, 2]), np.array([1, 2]))
    with pytest.raises(ValueError):
        Sequences(np.array([1, 2]), np.array([0, 3]))
    with pytest.raises(ValueError):
        count_matrix(Sequences.from_lists([[-1]]), 3)
    with pytest.raises(ValueError):
        count_matrix(Sequences.from_lists([[1]]), -1)


def test_group_by_key_keeps_order() -> None:
    s = group_by_key(np.array([1, 0, 1, 2, 0]), np.array([10, 11, 12, 13, 14]), 4)
    assert s.to_lists() == [[11, 14], [10, 12], [13], []]
    with pytest.raises(ValueError):
        group_by_key(np.array([5]), np.array([1]), 2)
    with pytest.raises(ValueError):
        group_by_key(np.array([0, 1]), np.array([1]), 2)


def test_fixed_and_sliding_time_windows() -> None:
    times = np.array([100, 101, 159, 160, 230, 400])
    events = np.arange(6)
    labels = np.array([0, 0, 0, 1, 0, 0])
    seqs, lab, starts = time_windows(times, events, labels, 60)
    assert seqs.to_lists() == [[0, 1, 2], [3], [4], [5]]
    assert lab.tolist() == [0, 1, 0, 0] and starts.tolist() == [100, 160, 220, 400]
    seqs, lab, starts = time_windows(times, events, labels, 60, 30)
    assert seqs.to_lists()[:3] == [[0, 1, 2], [2, 3], [3]]
    assert lab.tolist()[:3] == [0, 1, 1]
    seqs, _, starts = time_windows(times, events, None, 50, 20)  # size not a multiple of step
    for w, start in zip(seqs.to_lists(), starts, strict=True):
        assert all(start <= times[e] < start + 50 for e in w)
    empty, lab, _ = time_windows(np.array([]), np.array([]), None, 10)
    assert len(empty) == 0 and lab.size == 0
    for bad in [(0, None), (10, 0), (10, 20)]:
        with pytest.raises(ValueError):
            time_windows(times, events, None, *bad)
    with pytest.raises(ValueError):
        time_windows(times, events[:2], None, 10)


@given(
    st.lists(st.integers(0, 1000), min_size=1, max_size=60), st.integers(1, 50), st.integers(1, 50)
)
def test_time_windows_cover_lines(ts: list[int], size: int, step: int) -> None:
    step = min(step, size)
    times = np.array(sorted(ts))
    seqs, _, starts = time_windows(times, np.arange(times.size), None, size, step)
    seen = set()
    for w, start in zip(seqs.to_lists(), starts, strict=True):
        assert w and all(start <= times[e] < start + size for e in w)
        seen.update(w)
    assert seen == set(range(times.size))


def test_count_windows() -> None:
    seqs, lab = count_windows(np.arange(7), np.array([0, 0, 0, 0, 0, 1, 0]), 3)
    assert seqs.to_lists() == [[0, 1, 2], [3, 4, 5]] and lab.tolist() == [0, 1]
    seqs, lab = count_windows(np.arange(5), None, 3, 1)
    assert len(seqs) == 3 and lab.tolist() == [0, 0, 0]
    seqs, _ = count_windows(np.arange(2), None, 5)
    assert seqs.to_lists() == [[0, 1]]
    assert len(count_windows(np.array([]), None, 3)[0]) == 0
    with pytest.raises(ValueError):
        count_windows(np.arange(3), None, 0)


def test_feature_transform() -> None:
    x = np.array([[1, 0, 2], [0, 0, 2], [1, 1, 2]], dtype=float)
    t = FeatureTransform().fit(x)
    out = t.transform(x)
    assert np.allclose(out.mean(axis=0), 0)
    assert np.allclose(out[:, 2], 0)  # an event in every row carries no weight
    raw = FeatureTransform(tfidf=False, zero_mean=False).fit_transform(x)
    assert np.array_equal(raw, x)
    with pytest.raises(RuntimeError):
        FeatureTransform().transform(x)
    with pytest.raises(ValueError):
        FeatureTransform().fit(np.zeros((0, 3)))


def test_unique_rows() -> None:
    rows, counts, inverse = unique_rows(np.array([[1, 2], [0, 1], [1, 2]]))
    assert counts.sum() == 3 and np.array_equal(rows[inverse], [[1, 2], [0, 1], [1, 2]])
