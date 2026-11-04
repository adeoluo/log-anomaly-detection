from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from log_anomaly_detection.metrics import (
    average_precision,
    best_f1_threshold,
    evaluate,
    prf,
    roc_auc,
)


def test_prf_hand_computed() -> None:
    assert prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == (1, 1, 1, 0.5, 0.5, 0.5)
    assert prf(np.array([0, 0]), np.array([0, 0]))[3:] == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        prf(np.array([1]), np.array([1, 0]))


def test_auc_and_ap_known_values() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert average_precision(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(
        (1 / 2) * 1.0 + (1 / 2) * (2 / 3)
    )
    assert roc_auc(y, np.array([1, 1, 1, 1])) == pytest.approx(0.5)
    assert roc_auc(np.array([1, 1]), np.array([0.1, 0.2])) is None
    assert average_precision(np.array([0, 0]), np.array([0.1, 0.2])) is None


@given(
    st.lists(st.tuples(st.booleans(), st.integers(0, 5)), min_size=2, max_size=40).filter(
        lambda r: any(a for a, _ in r) and not all(a for a, _ in r)
    )
)
def test_auc_matches_pairwise_definition(rows: list[tuple[bool, int]]) -> None:
    y = np.array([a for a, _ in rows])
    s = np.array([b for _, b in rows], dtype=float)
    pos, neg = s[y], s[~y]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert roc_auc(y, s) == pytest.approx(expected)


def test_evaluate_with_weights_and_scores() -> None:
    r = evaluate(np.array([1, 0]), np.array([1, 1]), np.array([0.9, 0.2]), weights=np.array([2, 3]))
    assert (r.n, r.anomalies, r.tp, r.fp) == (5, 2, 2, 3)
    assert r.roc_auc == 1.0 and r.as_dict()["f1"] == pytest.approx(r.f1)
    assert evaluate(np.array([1]), np.array([1])).roc_auc is None


def test_best_f1_threshold() -> None:
    y = np.array([0, 0, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.85])
    thr, f1 = best_f1_threshold(y, s)
    assert f1 == pytest.approx(prf(y, s > thr)[5])
    assert f1 == pytest.approx(0.8)
    assert best_f1_threshold(np.array([0, 0]), np.array([0.1, 0.2])) == (float("inf"), 0.0)
    thr, f1 = best_f1_threshold(np.array([1, 1]), np.array([0.3, 0.3]))
    assert f1 == 1.0 and np.all(np.array([0.3, 0.3]) > thr)
