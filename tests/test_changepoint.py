from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from log_anomaly_detection.changepoint import (
    BOCPD,
    CUSUM,
    Regression,
    covering,
    detect_regressions,
    f_measure,
)


def steps(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, 200), rng.normal(4, 1, 200), rng.normal(-1, 1, 200)])


def test_tcpd_metric_reference_values() -> None:
    # Values from the docstrings of the TCPDBench reference implementation.
    assert f_measure({"1": [10, 20], "2": [11, 20], "3": [10], "4": [0, 5]}, [10, 20])[0] == 1.0
    assert f_measure({"1": [], "2": [10], "3": [50]}, [10])[0] == pytest.approx(0.9090909090909091)
    assert f_measure({"1": [], "2": [10], "3": [50]}, [])[0] == pytest.approx(0.8)
    assert covering({"1": [10, 20], "2": [10], "3": [0, 5]}, [10, 20], 45) == pytest.approx(
        0.7962962962962963
    )
    assert covering({"1": [], "2": [10], "3": [40]}, [10], 45) == pytest.approx(0.7954144620811286)
    assert covering({"1": [], "2": [10], "3": [40]}, [], 45) == pytest.approx(0.8189300411522634)


def test_metric_no_double_counting() -> None:
    _, p, r = f_measure({"a": [10]}, [9, 11])
    assert p == pytest.approx(2 / 3) and r == 1.0
    with pytest.raises(ValueError):
        f_measure({}, [1])
    with pytest.raises(ValueError):
        covering({}, [1], 10)


def test_cusum_and_bocpd_find_level_shifts() -> None:
    x = steps()
    for cps in (CUSUM().detect(x), BOCPD().detect(x)):
        assert any(abs(c - 200) <= 5 for c in cps)
        assert any(abs(c - 400) <= 5 for c in cps)
        assert f_measure({"t": [200, 400]}, cps)[0] > 0.7


def test_no_change_means_few_alarms() -> None:
    x = np.random.default_rng(3).normal(10, 1, 600)
    assert len(CUSUM().detect(x)) <= 1
    assert len(BOCPD().detect(x)) <= 1
    assert BOCPD().detect([]) == []
    assert CUSUM().detect(np.ones(50)) == []


def test_online_interfaces() -> None:
    x = steps(1)
    cusum = CUSUM(baseline=30)
    alarms = [cusum.update(t, v) for t, v in enumerate(x)]
    changes = [a for a in alarms if a is not None]
    assert changes and abs(changes[0] - 200) <= 10
    z = (x - x.mean()) / x.std()
    bocpd = BOCPD()
    runs = [bocpd.update(float(v))[0] for v in z]
    assert runs[199] > 100 and runs[230] < 40
    capped = BOCPD(max_run=150)
    assert max(capped.update(float(v))[0] for v in z) <= 150


def test_bocpd_pruning_keeps_a_distribution() -> None:
    b = BOCPD(prune=1e-3)
    for v in steps(2)[:300]:
        _, p0 = b.update(float((v - 1.0) / 2.0))
        assert 0 <= p0 <= 1
    assert b._probs.sum() == pytest.approx(1.0)


def test_detect_regressions_directions() -> None:
    rng = np.random.default_rng(5)
    latency = np.concatenate(
        [rng.normal(100, 2, 120), rng.normal(115, 2, 120), rng.normal(100, 2, 120)]
    )
    up = detect_regressions(latency, direction="up")
    assert len(up) == 1 and abs(up[0].index - 120) <= 5 and up[0].relative > 0.1
    down = detect_regressions(latency, direction="down")
    assert len(down) == 1 and abs(down[0].index - 240) <= 5
    both = detect_regressions(latency, method="bocpd", direction="both")
    assert {abs(r.index - 120) <= 5 or abs(r.index - 240) <= 5 for r in both} == {True}
    assert detect_regressions(latency, direction="up", min_relative=0.5) == []
    assert Regression(3, 0.0, 1.0).relative == float("inf")
    assert Regression(3, 2.0, 3.0).as_dict()["relative"] == 0.5
    with pytest.raises(ValueError):
        detect_regressions(latency, direction="sideways")
    with pytest.raises(ValueError):
        detect_regressions(latency, method="magic")


def test_spikes_are_not_level_shifts() -> None:
    from log_anomaly_detection.changepoint import drop_short_segments

    assert drop_short_segments([90, 91, 180], 300, 3) == [180]
    assert drop_short_segments([1, 150], 300, 3) == [150]
    assert drop_short_segments([150, 299], 300, 3) == [150]
    assert drop_short_segments([5, 6], 300, 1) == [5, 6]
    x = np.full(200, 100.0)
    x[50] = 160.0
    x[120:] = 112.0
    found = detect_regressions(x, method="bocpd", direction="both")
    assert [r.index for r in found] == [120]


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        CUSUM(threshold=0)
    with pytest.raises(ValueError):
        CUSUM(baseline=1)
    with pytest.raises(ValueError):
        BOCPD(hazard=1)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3), min_size=0, max_size=120))
def test_detectors_return_sorted_in_range_indices(values: list[float]) -> None:
    x = np.asarray(values)
    for cps in (CUSUM().detect(x), BOCPD().detect(x)):
        assert cps == sorted(set(cps))
        assert all(0 < c < max(len(values), 1) for c in cps)
    if values:
        assert 0 <= covering({"a": [len(values) // 2]}, BOCPD().detect(x), len(values)) <= 1


@pytest.mark.validation
def test_agrees_with_ruptures_on_clear_shifts() -> None:
    rpt = pytest.importorskip("ruptures")
    x = steps(4)
    ref = rpt.Pelt(model="l2").fit((x - x.mean()) / x.std()).predict(pen=2 * np.log(x.size))[:-1]
    ours = BOCPD().detect(x)
    assert f_measure({"ruptures": ref}, ours)[0] > 0.9
