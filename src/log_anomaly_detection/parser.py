"""High-level log parsing: one API over the Rust core and the Python reference.

:class:`LogParser` takes raw log lines, strips them, splits off the header with the dataset's log
format, masks variables and assigns each message to a Drain cluster. It parses lists of lines
(online, for streaming) and whole files (streamed in chunks, never loaded at once), optionally
extracting session keys such as HDFS block ids, per-line labels and timestamps on the way.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.backend import extension, resolve_backend
from log_anomaly_detection.drain import Drain, DrainConfig
from log_anomaly_detection.formats import DatasetSettings, LogFormat

UNMATCHED = np.uint32(0xFFFFFFFF)
CHUNK = 65536


@dataclass
class ParsedLog:
    """Columnar result of parsing a file: one entry per line unless noted."""

    event_ids: NDArray[np.uint32]
    templates: list[str]
    session_lines: NDArray[np.uint32] = field(default_factory=lambda: np.empty(0, np.uint32))
    session_ids: NDArray[np.uint32] = field(default_factory=lambda: np.empty(0, np.uint32))
    session_keys: list[str] = field(default_factory=list)
    labels: NDArray[np.uint8] | None = None
    times: NDArray[np.int64] | None = None
    unmatched: int = 0

    @property
    def num_lines(self) -> int:
        return int(self.event_ids.shape[0])


def iter_lines(path: str | Path, max_lines: int | None = None) -> Iterator[str]:
    """Yield the lines of ``path`` (gzip if it ends in ``.gz``) without terminators.

    Invalid UTF-8 is decoded leniently, as the Rust reader does.
    """

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="\n") as handle:
        for count, line in enumerate(handle):
            if max_lines is not None and count >= max_lines:
                return
            yield line[:-1] if line.endswith("\n") else line


class _PythonPipeline:
    """The Python reference pipeline, matching ``rust/src/pipeline.rs`` line for line."""

    def __init__(self, config: DrainConfig, log_format: str | None, skip_unmatched: bool) -> None:
        self.drain = Drain(config)
        self.format = LogFormat(log_format) if log_format else None
        self.skip_unmatched = skip_unmatched

    def content(self, line: str) -> tuple[str | None, dict[str, str] | None]:
        line = line.strip()
        if self.format is None:
            return line, None
        fields = self.format.match(line)
        if fields is None:
            return (None if self.skip_unmatched else line), None
        return fields["Content"], fields

    def add_lines(self, lines: Iterable[str]) -> NDArray[np.uint32]:
        ids = []
        for line in lines:
            content, _ = self.content(line)
            ids.append(int(UNMATCHED) if content is None else self.drain.add(content))
        return np.asarray(ids, dtype=np.uint32)

    def parse_file(
        self,
        path: str,
        session_regex: str | None,
        label_field: str | None,
        normal_label: str,
        time_field: str | None,
        max_lines: int | None,
    ) -> dict[str, Any]:
        for name in (label_field, time_field):
            if name is not None and (self.format is None or name not in self.format.headers):
                raise ValueError(f"log format has no field {name!r}")
        session = re.compile(session_regex) if session_regex else None
        ids: list[int] = []
        labels: list[int] = []
        times: list[int] = []
        s_lines: list[int] = []
        s_ids: list[int] = []
        keys: dict[str, int] = {}
        unmatched = 0
        for index, line in enumerate(iter_lines(path, max_lines)):
            content, fields = self.content(line)
            if fields is not None:
                if label_field is not None:
                    labels.append(int(fields[label_field] != normal_label))
                if time_field is not None:
                    try:
                        times.append(int(fields[time_field].strip()))
                    except ValueError:
                        times.append(0)
            else:
                if label_field is not None:
                    labels.append(0)
                if time_field is not None:
                    times.append(0)
            if content is None:
                unmatched += 1
                ids.append(int(UNMATCHED))
                continue
            if session is not None:
                seen: list[str] = []
                for match in session.finditer(content):
                    if match.group(0) not in seen:
                        seen.append(match.group(0))
                for key in seen:
                    s_lines.append(index)
                    s_ids.append(keys.setdefault(key, len(keys)))
            ids.append(self.drain.add(content))
        return {
            "event_ids": np.asarray(ids, np.uint32).tobytes(),
            "session_lines": np.asarray(s_lines, np.uint32).tobytes(),
            "session_ids": np.asarray(s_ids, np.uint32).tobytes(),
            "session_keys": list(keys),
            "labels": np.asarray(labels, np.uint8).tobytes(),
            "times": np.asarray(times, np.int64).tobytes(),
            "unmatched": unmatched,
        }


class LogParser:
    """Drain log parser over raw lines, backed by the Rust core or the Python reference.

    Example::

        parser = LogParser.for_dataset("HDFS")
        ids = parser.add_lines(open("HDFS.log"))
        print(parser.templates()[ids[0]])
    """

    def __init__(
        self,
        *,
        log_format: str | None = None,
        masks: Iterable[str] = (),
        depth: int = 4,
        similarity: float = 0.5,
        max_children: int = 100,
        skip_unmatched: bool = False,
        backend: str = "auto",
    ) -> None:
        self.config = DrainConfig(
            depth=depth, similarity=similarity, max_children=max_children, masks=tuple(masks)
        )
        self.log_format = log_format
        self.skip_unmatched = skip_unmatched
        self.backend = resolve_backend(backend)
        self._rust: Any = None
        self._py: _PythonPipeline | None = None
        if self.backend == "rust":
            if log_format is not None:
                LogFormat(log_format)  # same validation errors as the Python backend
            self._rust = extension().RustDrain(
                depth=depth,
                similarity=similarity,
                max_children=max_children,
                masks=list(self.config.masks),
                log_format=log_format,
                skip_unmatched=skip_unmatched,
            )
        else:
            self._py = _PythonPipeline(self.config, log_format, skip_unmatched)

    @classmethod
    def for_dataset(
        cls,
        settings: DatasetSettings,
        *,
        skip_unmatched: bool = False,
        backend: str = "auto",
    ) -> LogParser:
        """A parser configured with one dataset's format, masks and Drain settings."""

        return cls(
            log_format=settings.log_format,
            masks=settings.masks,
            depth=settings.depth,
            similarity=settings.similarity,
            max_children=settings.max_children,
            skip_unmatched=skip_unmatched,
            backend=backend,
        )

    # ------------------------------------------------------------------ parsing
    def add(self, content: str) -> int:
        """Add one message content (no header); return its cluster id."""

        if self._py is None:
            return int(self._rust.add(content))
        return self._py.drain.add(content)

    def add_lines(self, lines: Iterable[str]) -> NDArray[np.uint32]:
        """Parse raw lines in order; unmatched lines get ``UNMATCHED`` if skipped."""

        if self._py is None:
            out = []
            batch: list[str] = []
            for line in lines:
                batch.append(line)
                if len(batch) == CHUNK:
                    out.append(np.frombuffer(self._rust.add_lines(batch), dtype=np.uint32))
                    batch = []
            if batch or not out:
                out.append(np.frombuffer(self._rust.add_lines(batch), dtype=np.uint32))
            return np.concatenate(out) if len(out) > 1 else out[0].copy()
        return self._py.add_lines(lines)

    def parse_file(
        self,
        path: str | Path,
        *,
        session_regex: str | None = None,
        label_field: str | None = None,
        normal_label: str = "-",
        time_field: str | None = None,
        max_lines: int | None = None,
    ) -> ParsedLog:
        """Stream ``path`` through the parser, extracting sessions, labels and times if asked."""

        if max_lines is not None and max_lines < 0:
            raise ValueError("max_lines must be non-negative")
        raw: dict[str, Any]
        if self._py is None:
            try:
                raw = self._rust.parse_file(
                    str(path),
                    session_regex=session_regex,
                    label_field=label_field,
                    normal_label=normal_label,
                    time_field=time_field,
                    max_lines=max_lines,
                )
            except RuntimeError as exc:
                if "has no field" in str(exc) or "needs a log format" in str(exc):
                    raise ValueError(str(exc)) from exc
                if "cannot open" in str(exc):
                    raise FileNotFoundError(str(exc)) from exc
                raise
        else:
            if not Path(path).exists():
                raise FileNotFoundError(f"cannot open {path}")
            raw = self._py.parse_file(
                str(path), session_regex, label_field, normal_label, time_field, max_lines
            )
        return ParsedLog(
            event_ids=np.frombuffer(raw["event_ids"], dtype=np.uint32).copy(),
            templates=self.templates(),
            session_lines=np.frombuffer(raw["session_lines"], dtype=np.uint32).copy(),
            session_ids=np.frombuffer(raw["session_ids"], dtype=np.uint32).copy(),
            session_keys=list(raw["session_keys"]),
            labels=(
                np.frombuffer(raw["labels"], dtype=np.uint8).copy()
                if label_field is not None
                else None
            ),
            times=(
                np.frombuffer(raw["times"], dtype=np.int64).copy()
                if time_field is not None
                else None
            ),
            unmatched=int(raw["unmatched"]),
        )

    # ------------------------------------------------------------------ state
    @property
    def num_clusters(self) -> int:
        if self._py is None:
            return int(self._rust.num_clusters)
        return self._py.drain.num_clusters

    def template(self, cid: int) -> str:
        if cid < 0 or cid >= self.num_clusters:
            raise ValueError(f"unknown cluster id {cid}")
        if self._py is None:
            return str(self._rust.template(cid))
        return self._py.drain.template(cid)

    def templates(self) -> list[str]:
        if self._py is None:
            return [str(t) for t in self._rust.templates()]
        return self._py.drain.templates()

    def sizes(self) -> list[int]:
        if self._py is None:
            return [int(s) for s in self._rust.sizes()]
        return self._py.drain.sizes()

    def tokenize(self, content: str) -> list[str]:
        """Mask and split ``content`` the way the parser does."""

        if self._py is None:
            return [str(t) for t in self._rust.tokenize(content)]
        return self._py.drain.tokenize(content)
