from __future__ import annotations

import numpy as np
import pytest

from log_anomaly_detection.detectors import (
    COUNT_DETECTORS,
    InvariantsDetector,
    IsolationForestDetector,
    LogClusterDetector,
    PCADetector,
)
from log_anomaly_detection.detectors.iforest import average_path_length
from log_anomaly_detection.detectors.invariants import _integer_vector
from log_anomaly_detection.metrics import evaluate, roc_auc
from tests.synthetic import workflow_counts


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, _ = workflow_counts(3000, 0, seed=1)
    test, y = workflow_counts(2000, 200, seed=2)
    return train, test, y, np.zeros(1)


def test_invariant_mining_finds_the_workflow_invariants(data: tuple) -> None:
    train, test, y, _ = data
    det = InvariantsDetector().fit(train)
    found = {(tuple(c.tolist()), tuple(v.tolist())) for c, v in det.invariants_}
    assert ((6,), (1,)) in found  # errors never happen
    assert ((0, 2), (1, -1)) in found  # open == close
    assert ((3, 4), (1, -1)) in found  # replicate == ack
    r = evaluate(y, det.predict(test))
    assert r.precision == 1.0 and r.recall == 1.0
    assert det.params()["invariants"] >= 3
    assert det.violations(test[:1]).tolist() == [0]


def test_pca_separates_anomalies(data: tuple) -> None:
    train, test, y, _ = data
    det = PCADetector(tfidf=False).fit(train)
    assert 0 < det.components_ < train.shape[1]
    assert roc_auc(y, det.score(test)) > 0.95
    r = evaluate(y, det.predict(test))
    assert r.precision > 0.9 and r.recall > 0.5
    # With TF-IDF (the LogPAI default) events present in every run get zero weight, so only the
    # never-seen error event stands out: high precision, lower recall.
    tfidf = PCADetector().fit(train)
    r = evaluate(y, tfidf.predict(test))
    assert r.precision == 1.0 and 0.2 < r.recall < 0.5
    assert tfidf.params()["components"] == tfidf.components_


def test_logcluster_and_iforest_rank_anomalies_higher(data: tuple) -> None:
    train, test, y, _ = data
    lc = LogClusterDetector(max_distance=0.05, anomaly_threshold=0.05).fit(train)
    assert lc.params()["clusters"] >= 1
    assert roc_auc(y, lc.score(test)) > 0.65
    # Isolation forest barely ranks these subtle count violations (scikit-learn scores the same,
    # see the validation test), so only check it is deterministic here.
    forest = IsolationForestDetector(n_estimators=50, seed=3).fit(train)
    assert np.array_equal(forest.score(test[:50]), forest.score(test[:50]))


def test_iforest_scores_outliers_high() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    det = IsolationForestDetector(seed=1).fit(x)
    s = det.score(np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]]))
    assert s[1] > 0.6 > s[0]
    assert det.predict(np.array([[8.0, 8.0, 8.0]])).tolist() == [True]
    again = IsolationForestDetector(seed=1).fit(x)
    assert np.allclose(again.score(x[:20]), det.score(x[:20]))
    assert det.params()["n_estimators"] == 100


def test_average_path_length() -> None:
    assert average_path_length(np.array([0.0, 1.0, 2.0])).tolist() == [0.0, 0.0, 1.0]
    c256 = float(average_path_length(np.array([256.0]))[0])
    assert c256 == pytest.approx(2 * (np.log(255) + 0.5772156649) - 2 * 255 / 256)


def test_integer_vector() -> None:
    assert _integer_vector(np.array([0.5, -0.5]), 3).tolist() == [1, -1]
    assert _integer_vector(np.array([-0.3, 0.9]), 3).tolist() == [1, -3]
    assert _integer_vector(np.array([0.0, 1.0]), 3) is None
    assert _integer_vector(np.array([0.1, 0.71]), 3) is None


@pytest.mark.parametrize("name", sorted(COUNT_DETECTORS))
def test_unfitted_and_bad_input(name: str) -> None:
    det = COUNT_DETECTORS[name]()
    with pytest.raises(RuntimeError):
        det.score(np.zeros((2, 3)))
    if name not in ("invariants", "iforest", "logcluster"):
        with pytest.raises(RuntimeError):
            det.predict(np.zeros((2, 3)))
    det.fit(np.abs(np.random.default_rng(0).normal(size=(40, 3))).round())
    with pytest.raises(ValueError):
        det.score(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        det.score(np.zeros(3))
    with pytest.raises(ValueError):
        det.score(np.full((1, 3), np.nan))


def test_constructor_validation() -> None:
    for build in (
        lambda: PCADetector(variance=0),
        lambda: PCADetector(alpha=0.2),
        lambda: InvariantsDetector(support=0.4),
        lambda: InvariantsDetector(max_size=0),
        lambda: LogClusterDetector(max_distance=3),
        lambda: IsolationForestDetector(n_estimators=0),
        lambda: IsolationForestDetector(threshold=1.5),
    ):
        with pytest.raises(ValueError):
            build()
    with pytest.raises(ValueError):
        IsolationForestDetector().fit(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        InvariantsDetector().fit(np.zeros((0, 2)))


def test_degenerate_training_data() -> None:
    zeros = np.zeros((10, 3))
    pca = PCADetector().fit(zeros)
    assert pca.threshold_ is not None and pca.score(zeros).max() == 0
    lc = LogClusterDetector().fit(zeros)
    assert np.all(lc.score(np.ones((2, 3))) >= 0)
    constant = np.ones((10, 2))
    forest = IsolationForestDetector(n_estimators=3).fit(constant)
    assert forest.score(constant).shape == (10,)


@pytest.mark.validation
def test_isolation_forest_agrees_with_scikit_learn(data: tuple) -> None:
    sklearn = pytest.importorskip("sklearn.ensemble")
    train, test, y, _ = data
    ours = roc_auc(y, IsolationForestDetector(seed=0).fit(train).score(test))
    ref = sklearn.IsolationForest(random_state=0).fit(train)
    theirs = roc_auc(y, -ref.score_samples(test))
    assert ours is not None and theirs is not None
    assert abs(ours - theirs) < 0.08


@pytest.mark.validation
def test_pca_residual_matches_scikit_learn() -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6))
    det = PCADetector(tfidf=False).fit(x)
    ref = decomposition.PCA(n_components=det.components_).fit(x)
    recon = ref.inverse_transform(ref.transform(x))
    assert np.allclose(det.score(x), ((x - recon) ** 2).sum(axis=1), atol=1e-8)
