"""Streaming detection: parse lines as they arrive, close sessions, score them, raise alerts.

:class:`StreamMonitor` keeps one Drain parser for its whole life. It is trained on a historical
log (typically normal operation): the parser learns templates, sessions are built and a count
detector is fitted. Afterwards every new line is parsed with the same parser (so event ids stay
consistent; templates first seen after training count as "unknown"), appended to its open
session, and sessions are scored when they close:

* with a ``session_regex`` (HDFS block ids), a session closes after ``idle_lines`` further lines
  without activity, or on :meth:`flush`;
* without one, consecutive lines form fixed windows of ``window_lines`` lines.

A CUSUM watches the fraction of lines with templates never seen in training (a common sign of a
new failure mode) and raises a ``rate`` alert when it shifts up.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from log_anomaly_detection.changepoint import CUSUM
from log_anomaly_detection.detectors import COUNT_DETECTORS, CountDetector
from log_anomaly_detection.features import Sequences, count_matrix
from log_anomaly_detection.formats import DatasetSettings
from log_anomaly_detection.parser import UNMATCHED, LogParser


@dataclass
class Alert:
    """One alert: an anomalous session or window, or a shift in the unknown-template rate."""

    kind: str
    key: str
    score: float
    threshold: float
    line: int
    events: int = 0
    templates: list[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Open:
    events: list[int] = field(default_factory=list)
    last_line: int = 0


class StreamMonitor:
    """Online parser, sessionizer and detector with alerting."""

    def __init__(
        self,
        parser: LogParser,
        *,
        session_regex: str | None = None,
        idle_lines: int = 20_000,
        window_lines: int = 100,
        detector: str = "pca",
        detector_kwargs: dict[str, Any] | None = None,
        rate_window: int = 1000,
    ) -> None:
        if idle_lines < 1 or window_lines < 1 or rate_window < 1:
            raise ValueError("idle_lines, window_lines and rate_window must be positive")
        if detector not in COUNT_DETECTORS:
            raise ValueError(
                f"unknown detector {detector!r}; choose from {sorted(COUNT_DETECTORS)}"
            )
        self.parser = parser
        self.session_regex = re.compile(session_regex) if session_regex else None
        self.idle_lines = idle_lines
        self.window_lines = window_lines
        self.detector_name = detector
        self.detector_kwargs = detector_kwargs or {}
        self.detector: CountDetector | None = None
        self.n_events = 0
        self.rate_window = rate_window
        self._sessions: OrderedDict[str, _Open] = OrderedDict()
        self._window: list[int] = []
        self._window_start = 0
        self._line = 0
        self._rate_unknown = 0
        self._rate_seen = 0
        self._rate_blocks = 0
        self._cusum = CUSUM(threshold=10.0, baseline=10)
        self.stats: dict[str, int] = {
            "lines": 0,
            "sessions_scored": 0,
            "alerts": 0,
            "unknown_lines": 0,
            "unmatched_lines": 0,
        }

    @classmethod
    def for_dataset(cls, settings: DatasetSettings, **kwargs: Any) -> StreamMonitor:
        return cls(LogParser.for_dataset(settings), **kwargs)

    # ------------------------------------------------------------------ training
    def _keys(self, line: str) -> list[str]:
        if self.session_regex is None:
            return []
        return list(dict.fromkeys(m.group(0) for m in self.session_regex.finditer(line)))

    def train(self, lines: Iterable[str], exclude_keys: set[str] | None = None) -> dict[str, int]:
        """Learn templates and fit the detector on historical lines.

        ``exclude_keys`` drops known-anomalous sessions (for example labelled HDFS blocks) from the
        detector's training data; they still teach the parser their templates.
        """

        batch = list(lines)
        ids = self.parser.add_lines(batch)
        seqs: list[list[int]] = []
        if self.session_regex is not None:
            by_key: dict[str, list[int]] = {}
            for line, event in zip(batch, ids.tolist(), strict=True):
                if event == int(UNMATCHED):
                    continue
                for key in self._keys(line):
                    by_key.setdefault(key, []).append(event)
            exclude = exclude_keys or set()
            seqs = [v for k, v in by_key.items() if k not in exclude]
        else:
            kept = [e for e in ids.tolist() if e != int(UNMATCHED)]
            seqs = [kept[i : i + self.window_lines] for i in range(0, len(kept), self.window_lines)]
        if not seqs:
            raise ValueError("no training sessions found")
        self.n_events = self.parser.num_clusters
        counts = count_matrix(Sequences.from_lists(seqs), self.n_events)
        self.detector = COUNT_DETECTORS[self.detector_name](**self.detector_kwargs).fit(counts)
        return {"lines": len(batch), "sessions": len(seqs), "templates": self.n_events}

    @property
    def ready(self) -> bool:
        return self.detector is not None

    # ------------------------------------------------------------------ scoring
    def _score(self, key: str, events: list[int], line: int, kind: str) -> Alert | None:
        assert self.detector is not None and self.detector.threshold_ is not None
        counts = count_matrix(Sequences.from_lists([events]), self.n_events)
        score = float(self.detector.score(counts)[0])
        self.stats["sessions_scored"] += 1
        if score <= self.detector.threshold_:
            return None
        templates = [
            self.parser.template(e) + (" [new]" if e >= self.n_events else "")
            for e in sorted(set(events))
        ][:10]
        self.stats["alerts"] += 1
        return Alert(
            kind, key, score, float(self.detector.threshold_), line, len(events), templates
        )

    def _rate(self, unknown: bool) -> Alert | None:
        self._rate_seen += 1
        self._rate_unknown += int(unknown)
        if self._rate_seen < self.rate_window:
            return None
        fraction = self._rate_unknown / self._rate_seen
        self._rate_seen = self._rate_unknown = 0
        block = self._rate_blocks
        self._rate_blocks += 1
        change = self._cusum.update(block, fraction)
        if change is None or fraction == 0:
            return None
        self.stats["alerts"] += 1
        return Alert("rate", f"block-{block}", fraction, 0.0, self._line)

    def process(self, line: str) -> list[Alert]:
        """Parse one raw line; return any alerts it triggers."""

        if self.detector is None:
            raise RuntimeError("the monitor is not trained")
        alerts: list[Alert] = []
        self._line += 1
        self.stats["lines"] += 1
        event = int(self.parser.add_lines([line])[0])
        if event == int(UNMATCHED):
            self.stats["unmatched_lines"] += 1
            return alerts
        unknown = event >= self.n_events
        self.stats["unknown_lines"] += int(unknown)
        rate = self._rate(unknown)
        if rate is not None:
            alerts.append(rate)
        if self.session_regex is None:
            if not self._window:
                self._window_start = self._line
            self._window.append(event)
            if len(self._window) >= self.window_lines:
                alert = self._score(
                    f"lines-{self._window_start}", self._window, self._line, "window"
                )
                self._window = []
                if alert is not None:
                    alerts.append(alert)
            return alerts
        for key in self._keys(line):
            session = self._sessions.pop(key, None) or _Open()
            session.events.append(event)
            session.last_line = self._line
            self._sessions[key] = session  # most recently active last
        while self._sessions:
            key, oldest = next(iter(self._sessions.items()))
            if self._line - oldest.last_line < self.idle_lines:
                break
            self._sessions.popitem(last=False)
            alert = self._score(key, oldest.events, self._line, "session")
            if alert is not None:
                alerts.append(alert)
        return alerts

    def process_many(self, lines: Iterable[str]) -> Iterator[Alert]:
        for line in lines:
            yield from self.process(line)

    def flush(self) -> list[Alert]:
        """Score every open session or partial window."""

        alerts: list[Alert] = []
        if self.detector is None:
            return alerts
        while self._sessions:
            key, session = self._sessions.popitem(last=False)
            alert = self._score(key, session.events, self._line, "session")
            if alert is not None:
                alerts.append(alert)
        if self._window:
            alert = self._score(f"lines-{self._window_start}", self._window, self._line, "window")
            self._window = []
            if alert is not None:
                alerts.append(alert)
        return alerts

    @property
    def open_sessions(self) -> int:
        return len(self._sessions) + int(bool(self._window))

    def score_sessions(self, sessions: list[list[str]]) -> list[dict[str, Any]]:
        """Score independent sessions of raw lines (the parser still learns from them)."""

        if self.detector is None:
            raise RuntimeError("the monitor is not trained")
        seqs = []
        for lines in sessions:
            ids = self.parser.add_lines(lines)
            seqs.append([int(e) for e in ids.tolist() if e != int(UNMATCHED)])
        counts = count_matrix(Sequences.from_lists(seqs), self.n_events)
        scores = self.detector.score(counts)
        assert self.detector.threshold_ is not None
        threshold = float(self.detector.threshold_)
        return [
            {"score": float(s), "anomalous": bool(s > threshold), "threshold": threshold}
            for s in np.asarray(scores).tolist()
        ]
