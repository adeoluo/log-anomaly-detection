"""Anomaly-detection evaluation: chronological split, normal-only training, test scoring.

The protocol is the same for every detector:

1. sequences (HDFS sessions or BGL windows) are ordered by time; the first ``train_fraction``
   form the training part and the rest the test part;
2. detectors are fitted on the *normal* training sequences only (labels are used just to drop
   anomalies from training, the semi-supervised setting of DeepLog);
3. each detector labels every test sequence with its own decision rule (no threshold is tuned on
   test labels); precision, recall and F1 use those decisions, and ROC AUC and average precision
   use the raw scores.

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
from log_anomaly_detection.metrics import DetectionScores, evaluate


@dataclass
class DetectorResult:
    """One detector's test performance and cost."""

    detector: str
    scores: DetectionScores
    fit_seconds: float
    score_seconds: float
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            **self.scores.as_dict(),
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            "params": self.params,
        }


@dataclass
class Split:
    """Index arrays of a chronological split; ``train_normal`` excludes training anomalies."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]
    train_normal: NDArray[np.int64]

    @classmethod
    def make(cls, labels: NDArray[np.uint8], train_fraction: float) -> Split:
        train, test = chronological_split(int(labels.shape[0]), train_fraction)
        if test.size == 0:
            raise ValueError("the test split is empty")
        normal = train[labels[train] == 0]
        if normal.size == 0:
            raise ValueError("no normal sequences in the training split")
        return cls(train, test, normal)

    def describe(self, labels: NDArray[np.uint8]) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "train_normal": int(self.train_normal.size),
            "test": int(self.test.size),
            "test_anomalies": int(labels[self.test].sum()),
        }


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
    return DetectorResult(detector.name, result, fit_seconds, score_seconds, detector.params())


def evaluate_dataset(
    dataset: SessionDataset,
    detectors: list[str],
    *,
    train_fraction: float = 0.5,
    seed: int = 0,
    detector_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit and score each named count detector on ``dataset``; return a JSON-ready report."""

    split = Split.make(dataset.labels, train_fraction)
    counts = count_matrix(dataset.sequences, dataset.n_events)
    detector_kwargs = detector_kwargs or {}
    results = []
    for name in detectors:
        kwargs = dict(detector_kwargs.get(name, {}))
        if name == "iforest":
            kwargs.setdefault("seed", seed)
        results.append(
            run_count_detector(build_detector(name, **kwargs), counts, dataset.labels, split)
        )
    return {
        "dataset": dataset.summary(),
        "split": {"train_fraction": train_fraction, **split.describe(dataset.labels)},
        "results": [r.as_dict() for r in results],
    }
