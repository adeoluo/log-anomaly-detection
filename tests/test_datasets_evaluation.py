from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pytest

from log_anomaly_detection.datasets import (
    chronological_split,
    load_bgl,
    load_hdfs,
    read_hdfs_labels,
)
from log_anomaly_detection.evaluation import Split, build_detector, evaluate_dataset


@pytest.fixture
def hdfs_labels(tmp_path: Path, loghub_2k: Path) -> Path:
    with gzip.open(loghub_2k / "HDFS_2k.log.gz", "rt") as handle:
        blocks = sorted({b for line in handle for b in re.findall(r"blk_-?\d+", line)})
    path = tmp_path / "anomaly_label.csv"
    rows = [f"{b},{'Anomaly' if i % 7 == 0 else 'Normal'}" for i, b in enumerate(blocks[:-5])]
    path.write_text("BlockId,Label\n" + "\n".join(rows) + "\n")
    return path


def test_load_hdfs_sessions(loghub_2k: Path, hdfs_labels: Path, backend: str) -> None:
    ds = load_hdfs(loghub_2k / "HDFS_2k.log.gz", hdfs_labels, backend=backend)
    labels = read_hdfs_labels(hdfs_labels)
    assert len(ds.sequences) == len(labels) == len(ds.keys)
    assert ds.info["unlabelled_blocks"] == 5
    assert ds.labels.tolist() == [labels[k] for k in ds.keys]
    assert ds.lines == 2000 and ds.n_events == len(ds.templates)
    # Every line mentioning a block contributes exactly one event to that block's session.
    with gzip.open(loghub_2k / "HDFS_2k.log.gz", "rt") as handle:
        mentions = sum(
            len({b for b in re.findall(r"blk_-?\d+", line) if b in labels}) for line in handle
        )
    assert int(ds.sequences.lengths.sum()) == mentions
    assert ds.summary()["sequences"] == len(ds.sequences)


def test_load_bgl_windows(loghub_2k: Path, backend: str) -> None:
    ds = load_bgl(loghub_2k / "BGL_2k.log.gz", window_seconds=3600, backend=backend)
    assert ds.lines == 2000 and int(ds.sequences.lengths.sum()) == 2000
    assert ds.labels.sum() >= 1 and ds.info["anomalous_lines"] > 0
    sliding = load_bgl(loghub_2k / "BGL_2k.log.gz", window_seconds=3600, step_seconds=1800)
    assert len(sliding.sequences) >= len(ds.sequences)
    by_lines = load_bgl(loghub_2k / "BGL_2k.log.gz", window_lines=100, backend=backend)
    assert len(by_lines.sequences) == 20
    with pytest.raises(ValueError):
        load_bgl(loghub_2k / "BGL_2k.log.gz", window_seconds=None)


def test_splits() -> None:
    train, test = chronological_split(10, 0.3)
    assert train.tolist() == [0, 1, 2] and test.tolist() == list(range(3, 10))
    with pytest.raises(ValueError):
        chronological_split(10, 1.0)
    labels = np.array([1, 0, 0, 1, 0, 1], dtype=np.uint8)
    split = Split.make(labels, 0.5)
    assert split.train_normal.tolist() == [1, 2]
    assert split.describe(labels) == {"train": 3, "train_normal": 2, "test": 3, "test_anomalies": 2}
    with pytest.raises(ValueError, match="no normal"):
        Split.make(np.ones(4, np.uint8), 0.5)
    with pytest.raises(ValueError, match="empty"):
        Split.make(np.zeros(2, np.uint8), 0.9)


def test_evaluate_dataset_on_bgl(loghub_2k: Path) -> None:
    ds = load_bgl(loghub_2k / "BGL_2k.log.gz", window_lines=20)
    report = evaluate_dataset(
        ds, ["pca", "invariants", "logcluster", "iforest"], train_fraction=0.5
    )
    assert report["split"]["test"] == 50
    names = [r["detector"] for r in report["results"]]
    assert names == ["PCA", "Invariant mining", "LogCluster", "Isolation forest"]
    for r in report["results"]:
        assert 0 <= r["f1"] <= 1 and r["n"] == 50 and r["fit_seconds"] >= 0


def test_unknown_detector() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("svm")
