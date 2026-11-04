"""Loghub parsing benchmarks: locate datasets, parse them, score them, time them.

Two layouts are supported:

* **Loghub-2k** (2,000 annotated lines per dataset): ``<root>/<Name>/<Name>_2k.log`` with
  ``<Name>_2k.log_structured.csv`` beside it, or flat ``<root>/<Name>_2k.log[.gz]`` files (the
  bundled test fixtures).
* **Loghub-2.0** (Jiang et al., ISSTA 2024; 3.6 million annotated lines per dataset on average):
  the Zenodo archives ``<root>/<Name>.zip`` holding ``<Name>/<Name>_full.log`` and its structured
  CSV. Both are streamed straight out of the archive, so nothing is extracted to disk.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from log_anomaly_detection.formats import settings_for
from log_anomaly_detection.parser import CHUNK, UNMATCHED, LogParser, iter_lines
from log_anomaly_detection.parsing_metrics import ParsingScores, evaluate_clusters, read_truth


@dataclass(frozen=True)
class DatasetRun:
    """One parse of one dataset: accuracy and wall-clock time."""

    dataset: str
    scores: ParsingScores
    parse_seconds: float
    unmatched: int

    @property
    def lines_per_second(self) -> float:
        return self.scores.lines / self.parse_seconds if self.parse_seconds > 0 else float("inf")

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"dataset": self.dataset}
        out.update(self.scores.as_dict())
        out["parse_seconds"] = round(self.parse_seconds, 4)
        out["lines_per_second"] = round(self.lines_per_second, 1)
        out["unmatched"] = self.unmatched
        return out


def find_2k(root: str | Path, name: str) -> tuple[Path, Path]:
    """Return the log and structured-CSV paths of a Loghub-2k dataset under ``root``."""

    root = Path(root)
    candidates = [
        (root / name / f"{name}_2k.log", root / name / f"{name}_2k.log_structured.csv"),
        (root / f"{name}_2k.log", root / f"{name}_2k.log_structured.csv"),
        (root / f"{name}_2k.log.gz", root / f"{name}_2k.log_structured.csv.gz"),
    ]
    for log, truth in candidates:
        if log.exists() and truth.exists():
            return log, truth
    raise FileNotFoundError(f"no Loghub-2k files for {name} under {root}")


def _score(
    name: str, parser: LogParser, ids: np.ndarray, truth: np.ndarray, strings: list[str]
) -> tuple[ParsingScores, int]:
    unmatched = int(np.count_nonzero(ids == UNMATCHED))
    kept = ids[ids != UNMATCHED]
    if kept.shape[0] != truth.shape[0]:
        raise ValueError(
            f"{name}: parsed {kept.shape[0]} matched lines but ground truth has {truth.shape[0]}"
        )
    return evaluate_clusters(truth, strings, kept, parser.templates()), unmatched


def run_2k(name: str, root: str | Path, backend: str = "auto") -> DatasetRun:
    """Parse and score one Loghub-2k dataset with its benchmark settings."""

    log, truth_path = find_2k(root, name)
    parser = LogParser.for_dataset(settings_for(name), skip_unmatched=True, backend=backend)
    lines = list(iter_lines(log))
    start = time.perf_counter()
    ids = parser.add_lines(lines)
    seconds = time.perf_counter() - start
    truth, strings = read_truth(truth_path)
    scores, unmatched = _score(name, parser, ids, truth, strings)
    return DatasetRun(name, scores, seconds, unmatched)


def zip_members(archive: zipfile.ZipFile, name: str) -> tuple[str, str]:
    """The log and structured-CSV member names of a Loghub-2.0 archive."""

    log = truth = None
    for member in archive.namelist():
        if member.endswith(f"{name}_full.log"):
            log = member
        elif member.endswith(f"{name}_full.log_structured.csv"):
            truth = member
    if log is None or truth is None:
        raise FileNotFoundError(f"{name}: archive lacks {name}_full.log or its structured CSV")
    return log, truth


def iter_zip_lines(path: str | Path, member: str) -> Iterator[str]:
    """Stream the lines of one archive member without extracting it."""

    with zipfile.ZipFile(path) as archive, archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="\n")
        for line in text:
            yield line[:-1] if line.endswith("\n") else line


def run_full(name: str, root: str | Path, backend: str = "auto") -> DatasetRun:
    """Parse and score one Loghub-2.0 dataset streamed from ``<root>/<name>.zip``.

    The timed span covers reading, decompressing and parsing every line, so it is an end-to-end
    throughput figure; ground truth loading and scoring are not timed.
    """

    path = Path(root) / f"{name}.zip"
    with zipfile.ZipFile(path) as archive:
        log_member, truth_member = zip_members(archive, name)
    parser = LogParser.for_dataset(settings_for(name), skip_unmatched=True, backend=backend)
    start = time.perf_counter()
    parts = []
    batch: list[str] = []
    for line in iter_zip_lines(path, log_member):
        batch.append(line)
        if len(batch) == CHUNK * 4:
            parts.append(parser.add_lines(batch))
            batch = []
    if batch:
        parts.append(parser.add_lines(batch))
    ids = np.concatenate(parts) if parts else np.empty(0, np.uint32)
    seconds = time.perf_counter() - start
    del batch, parts
    truth, strings = read_truth(path, truth_member)
    scores, unmatched = _score(name, parser, ids, truth, strings)
    return DatasetRun(name, scores, seconds, unmatched)
