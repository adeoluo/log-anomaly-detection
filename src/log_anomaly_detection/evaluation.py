"""Anomaly-detection evaluation: chronological split, normal-only training, test scoring.

The protocol is the same for every detector:

1. sequences (HDFS sessions or BGL windows) are ordered by time; the first ``train_fraction``
   form the training period and the rest the test part; optionally the last
   ``validation_fraction`` (of all sequences) of the training period is held out for validation;
2. detectors are fitted on the *normal* sequences of the remaining training part only (labels are
   used just to drop anomalies, the semi-supervised setting of DeepLog);
3. each detector labels every test sequence with its own decision rule; precision, recall and F1
   use those decisions, and ROC AUC and average precision use the raw scores;
4. with a validation part, a second set of decisions uses the score threshold that maximizes F1
   on the validation sequences (labels from the training period only, never from the test).

Identical test rows are scored once and the results expanded, which is exact and makes scoring
the 287,000 HDFS test sessions cheap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.datasets import SessionDataset, chronological_split
from log_anomaly_detection.detectors import COUNT_DETECTORS, CountDetector
from log_anomaly_detection.features import count_matrix, unique_rows
from log_anomaly_detection.metrics import DetectionScores, best_f1_threshold, evaluate


@dataclass
class DetectorResult:
    """One detector's test performance and cost."""

    detector: str
    scores: DetectionScores
    fit_seconds: float
    score_seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    tuned: DetectionScores | None = None
    tuned_threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "detector": self.detector,
            **self.scores.as_dict(),
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            "params": self.params,
        }
        if self.tuned is not None:
            out["tuned"] = {**self.tuned.as_dict(), "threshold": self.tuned_threshold}
        return out


@dataclass
class Split:
    """Index arrays of a chronological split; ``train_normal`` excludes training anomalies."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]
    train_normal: NDArray[np.int64]
    validation: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))

    @classmethod
    def make(
        cls, labels: NDArray[np.uint8], train_fraction: float, validation_fraction: float = 0.0
    ) -> Split:
        n = int(labels.shape[0])
        train, test = chronological_split(n, train_fraction)
        if test.size == 0:
            raise ValueError("the test split is empty")
        if not 0 <= validation_fraction < train_fraction:
            raise ValueError("validation_fraction must be in [0, train_fraction)")
        held = round(n * validation_fraction)
        validation = train[train.size - held :] if held else np.empty(0, np.int64)
        fit = train[: train.size - held]
        normal = fit[labels[fit] == 0]
        if normal.size == 0:
            raise ValueError("no normal sequences in the training split")
        return cls(fit, test, normal, validation)

    def describe(self, labels: NDArray[np.uint8]) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "train_normal": int(self.train_normal.size),
            "validation": int(self.validation.size),
            "validation_anomalies": int(labels[self.validation].sum()),
            "test": int(self.test.size),
            "test_anomalies": int(labels[self.test].sum()),
        }


def _tune(
    labels: NDArray[np.uint8],
    val_scores: NDArray[np.float64],
    test_labels: NDArray[np.uint8],
    test_scores: NDArray[np.float64],
) -> tuple[DetectionScores | None, float | None]:
    """Decisions at the F1-best validation threshold, or None without validation anomalies."""

    if val_scores.size == 0 or int(labels.sum()) == 0:
        return None, None
    threshold, _ = best_f1_threshold(labels, val_scores)
    return evaluate(test_labels, test_scores > threshold, test_scores), threshold


def build_detector(name: str, **kwargs: Any) -> CountDetector:
    try:
        cls = COUNT_DETECTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown detector {name!r}; choose from {sorted(COUNT_DETECTORS)}"
        ) from exc
    return cls(**kwargs)


def run_count_detector(
    detector: CountDetector,
    counts: NDArray[np.float32],
    labels: NDArray[np.uint8],
    split: Split,
) -> DetectorResult:
    start = time.perf_counter()
    detector.fit(counts[split.train_normal])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rows, _, inverse = unique_rows(counts[split.test])
    raw = detector.score(rows)
    assert detector.threshold_ is not None
    decided = raw > detector.threshold_
    score_seconds = time.perf_counter() - start
    result = evaluate(labels[split.test], decided[inverse], raw[inverse])
    val_scores = np.empty(0)
    if split.validation.size:
        val_rows, _, val_inverse = unique_rows(counts[split.validation])
        val_scores = detector.score(val_rows)[val_inverse]
    tuned, threshold = _tune(labels[split.validation], val_scores, labels[split.test], raw[inverse])
    return DetectorResult(
        detector.name, result, fit_seconds, score_seconds, detector.params(), tuned, threshold
    )


def run_sequence_detector(
    name: str,
    dataset: SessionDataset,
    split: Split,
    **kwargs: Any,
) -> DetectorResult:
    from log_anomaly_detection.detectors.sequence import SEQUENCE_DETECTORS

    detector = SEQUENCE_DETECTORS[name](**kwargs)
    start = time.perf_counter()
    detector.fit(dataset.sequences.subset(split.train_normal), dataset.n_events)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    raw = detector.score(dataset.sequences.subset(split.test))
    decided = raw > detector.threshold_
    score_seconds = time.perf_counter() - start
    result = evaluate(dataset.labels[split.test], decided, raw)
    val_scores = (
        detector.score(dataset.sequences.subset(split.validation))
        if split.validation.size
        else np.empty(0)
    )
    tuned, threshold = _tune(
        dataset.labels[split.validation], val_scores, dataset.labels[split.test], raw
    )
    return DetectorResult(
        detector.name, result, fit_seconds, score_seconds, detector.params(), tuned, threshold
    )


SEQUENCE_NAMES = ("deeplog", "transformer")


def evaluate_dataset(
    dataset: SessionDataset,
    detectors: list[str],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    detector_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit and score each named detector on ``dataset``; return a JSON-ready report."""

    split = Split.make(dataset.labels, train_fraction, validation_fraction)
    counts = count_matrix(dataset.sequences, dataset.n_events)
    detector_kwargs = detector_kwargs or {}
    results = []
    for name in detectors:
        kwargs = dict(detector_kwargs.get(name, {}))
        if name in ("iforest", *SEQUENCE_NAMES):
            kwargs.setdefault("seed", seed)
        if name in SEQUENCE_NAMES:
            results.append(run_sequence_detector(name, dataset, split, **kwargs))
        else:
            results.append(
                run_count_detector(build_detector(name, **kwargs), counts, dataset.labels, split)
            )
    return {
        "dataset": dataset.summary(),
        "split": {
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            **split.describe(dataset.labels),
        },
        "results": [r.as_dict() for r in results],
    }
