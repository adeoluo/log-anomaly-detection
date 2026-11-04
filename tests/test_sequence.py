from __future__ import annotations

import numpy as np
import pytest

from log_anomaly_detection.detectors.sequence import make_windows, unique_sequences
from log_anomaly_detection.features import Sequences


def test_unique_sequences() -> None:
    seqs = Sequences.from_lists([[1, 2], [3], [1, 2], [], [3], []])
    uniq, counts, inverse = unique_sequences(seqs)
    assert uniq.to_lists() == [[1, 2], [3], []]
    assert counts.tolist() == [2, 2, 2]
    assert inverse.tolist() == [0, 1, 0, 2, 1, 2]


def test_make_windows_left_pads_within_each_sequence() -> None:
    hist, nxt, rows = make_windows(Sequences.from_lists([[5, 6, 7], [8]]), 2, pad=9)
    assert hist.tolist() == [[9, 9], [9, 5], [5, 6], [9, 9]]
    assert nxt.tolist() == [5, 6, 7, 8]
    assert rows.tolist() == [0, 0, 0, 1]
    h, n, r = make_windows(Sequences.from_lists([[], []]), 3, pad=0)
    assert h.shape == (0, 3) and n.size == 0 and r.size == 0


torch = pytest.importorskip("torch")

from log_anomaly_detection.detectors.sequence import (  # noqa: E402
    SEQUENCE_DETECTORS,
    DeepLogDetector,
    TransformerDetector,
)

TINY = {
    "window": 3,
    "top_g": 1,
    "embedding": 8,
    "hidden": 16,
    "layers": 1,
    "epochs": 12,
    "batch_size": 64,
    "samples_per_epoch": 1500,
    "learning_rate": 0.02,
    "device": "cpu",
}


def workflow() -> tuple[Sequences, Sequences, np.ndarray]:
    normal = [[0, 1, 2, 3, 4]] * 40
    test = [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 2, 3, 4, 6], [0, 1, 2], [0, 1, 2, 3, 4]]
    labels = np.array([0, 1, 1, 1, 0])
    return Sequences.from_lists(normal), Sequences.from_lists(test), labels


@pytest.mark.parametrize("name", sorted(SEQUENCE_DETECTORS))
def test_learns_the_workflow(name: str) -> None:
    train, test, labels = workflow()
    det = SEQUENCE_DETECTORS[name](seed=1, **TINY).fit(train, n_events=5)
    pred = det.predict(test)
    assert pred.tolist() == labels.astype(bool).tolist()
    scores = det.score(test)
    assert scores[2] == 6  # the unseen event 6 ranks below every known event and the end token
    assert scores[0] == scores[4] == 0
    params = det.params()
    assert params["distinct_windows"] <= params["train_windows"] and params["final_loss"] < 1.0


def test_without_end_token_truncation_goes_unnoticed() -> None:
    train, test, _ = workflow()
    det = DeepLogDetector(seed=1, end_token=False, **TINY).fit(train, n_events=5)
    assert det.predict(test).tolist() == [False, True, True, False, False]
    assert det.params()["end_token"] is False


def test_training_is_reproducible() -> None:
    train, test, _ = workflow()
    a = DeepLogDetector(seed=3, **TINY).fit(train, 5).score(test)
    b = DeepLogDetector(seed=3, **TINY).fit(train, 5).score(test)
    assert np.array_equal(a, b)


def test_errors() -> None:
    with pytest.raises(ValueError):
        DeepLogDetector(window=0)
    with pytest.raises(ValueError):
        TransformerDetector(hidden=10, heads=4)
    det = DeepLogDetector(device="cpu")
    with pytest.raises(RuntimeError):
        det.score(Sequences.from_lists([[1]]))
    with pytest.raises(RuntimeError):
        det._ranks(np.zeros((1, 10), np.int32), np.zeros(1, np.int32))
    with pytest.raises(ValueError):
        det.fit(Sequences.from_lists([[]]), 3)
    with pytest.raises(ValueError):
        det.fit(Sequences.from_lists([[-1, 2]]), 3)
    assert TransformerDetector().params()["heads"] == 4
