"""Anomaly detectors over event counts (PCA, invariant mining, LogCluster, isolation forest)."""

from log_anomaly_detection.detectors.base import CountDetector
from log_anomaly_detection.detectors.iforest import IsolationForestDetector
from log_anomaly_detection.detectors.invariants import InvariantsDetector
from log_anomaly_detection.detectors.logcluster import LogClusterDetector
from log_anomaly_detection.detectors.pca import PCADetector

COUNT_DETECTORS: dict[str, type[CountDetector]] = {
    "pca": PCADetector,
    "invariants": InvariantsDetector,
    "logcluster": LogClusterDetector,
    "iforest": IsolationForestDetector,
}

__all__ = [
    "COUNT_DETECTORS",
    "CountDetector",
    "InvariantsDetector",
    "IsolationForestDetector",
    "LogClusterDetector",
    "PCADetector",
]
