"""Anomaly-detection datasets: HDFS sessions and BGL windows, parsed straight from the raw logs.

* **HDFS** (Loghub HDFS_v1, 11,175,629 lines): every line is parsed with Drain and attached to
  each distinct block id (``blk_...``) it mentions; a block's events in log order form a session,
  labelled by ``anomaly_label.csv`` (575,061 blocks, 16,838 anomalous).
* **BGL** (Loghub BGL, 4,747,963 lines): lines carry their own label (``-`` is normal) and a Unix
  timestamp; the stream is cut into time windows, and a window is anomalous if any line is.

Parsing runs in the Rust core and streams the file, so the 1.5 GB HDFS log never sits in memory.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.features import Sequences, count_windows, group_by_key, time_windows
from log_anomaly_detection.formats import settings_for
from log_anomaly_detection.parser import UNMATCHED, LogParser

HDFS_SESSION_REGEX = r"blk_-?\d+"


@dataclass
class SessionDataset:
    """Labelled event sequences (sessions or windows) and how they were produced."""

    name: str
    sequences: Sequences
    labels: NDArray[np.uint8]
    templates: list[str]
    lines: int
    parse_seconds: float
    keys: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.templates)

    def summary(self) -> dict[str, object]:
        return {
            "dataset": self.name,
            "lines": self.lines,
            "sequences": len(self.sequences),
            "anomalous": int(self.labels.sum()),
            "templates": self.n_events,
            "parse_seconds": round(self.parse_seconds, 2),
            **self.info,
        }


def read_hdfs_labels(path: str | Path) -> dict[str, int]:
    """``anomaly_label.csv``: block id -> 1 if anomalous, 0 if normal."""

    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[row["BlockId"]] = int(row["Label"].strip().lower() == "anomaly")
    return out


def load_hdfs(
    log_path: str | Path,
    label_path: str | Path,
    *,
    max_lines: int | None = None,
    backend: str = "auto",
) -> SessionDataset:
    """Parse HDFS and build one sequence per block id (blocks without a label are dropped)."""

    parser = LogParser.for_dataset(settings_for("HDFS"), backend=backend)
    start = time.perf_counter()
    parsed = parser.parse_file(log_path, session_regex=HDFS_SESSION_REGEX, max_lines=max_lines)
    seconds = time.perf_counter() - start
    labels_by_key = read_hdfs_labels(label_path)
    events = parsed.event_ids[parsed.session_lines].astype(np.int64)
    sequences = group_by_key(parsed.session_ids, events, len(parsed.session_keys))
    known = np.array([k in labels_by_key for k in parsed.session_keys], dtype=bool)
    labels = np.array([labels_by_key.get(k, 0) for k in parsed.session_keys], dtype=np.uint8)
    keys = [k for k, ok in zip(parsed.session_keys, known, strict=True) if ok]
    return SessionDataset(
        name="HDFS",
        sequences=sequences.subset(known),
        labels=labels[known],
        templates=parsed.templates,
        lines=parsed.num_lines,
        parse_seconds=seconds,
        keys=keys,
        info={"grouping": "block id", "unlabelled_blocks": int((~known).sum())},
    )


def load_bgl(
    log_path: str | Path,
    *,
    window_seconds: int | None = 3600,
    step_seconds: int | None = None,
    window_lines: int | None = None,
    max_lines: int | None = None,
    backend: str = "auto",
) -> SessionDataset:
    """Parse BGL and cut it into time windows (or windows of ``window_lines`` lines)."""

    parser = LogParser.for_dataset(settings_for("BGL"), backend=backend)
    start = time.perf_counter()
    parsed = parser.parse_file(
        log_path, label_field="Label", time_field="Timestamp", max_lines=max_lines
    )
    seconds = time.perf_counter() - start
    assert parsed.labels is not None and parsed.times is not None
    keep = parsed.event_ids != UNMATCHED
    events = parsed.event_ids[keep].astype(np.int64)
    line_labels = parsed.labels[keep]
    times = parsed.times[keep]
    if window_lines is not None:
        sequences, labels = count_windows(events, line_labels, window_lines, step_seconds)
        info: dict[str, object] = {"grouping": f"{window_lines}-line windows"}
        starts = np.arange(len(sequences))
    else:
        if window_seconds is None:
            raise ValueError("give window_seconds or window_lines")
        sequences, labels, starts = time_windows(
            times, events, line_labels, window_seconds, step_seconds
        )
        info = {"grouping": f"{window_seconds} s windows, step {step_seconds or window_seconds} s"}
    info["anomalous_lines"] = int(line_labels.sum())
    info["unmatched_lines"] = int((~keep).sum())
    return SessionDataset(
        name="BGL",
        sequences=sequences,
        labels=labels,
        templates=parsed.templates,
        lines=parsed.num_lines,
        parse_seconds=seconds,
        keys=[str(int(s)) for s in starts],
        info=info,
    )


def chronological_split(
    n: int, train_fraction: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """First ``train_fraction`` of the sequences (in time order) for training, the rest for test."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    cut = round(n * train_fraction)
    idx = np.arange(n, dtype=np.int64)
    return idx[:cut], idx[cut:]
