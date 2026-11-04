"""Log line formats and the per-dataset parser settings used by the Loghub benchmarks.

A log format such as ``<Date> <Time> <Level> <Component>: <Content>`` names the header fields
of a line. It compiles to an anchored regular expression with one lazy named group per field
(runs of spaces become ``\\s+``), the construction used by the LogPAI benchmark tooling, so a
line either matches and yields its ``Content`` or is reported as unmatched.

The settings in :data:`BENCHMARK_SETTINGS` (format, masking regexes, similarity threshold and
tree depth for each Loghub dataset) are the values published with the LogPAI Drain benchmark
(https://github.com/logpai/logparser, Apache-2.0), so results are comparable with the
published Drain numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FIELD = re.compile(r"(<[^<>]+>)")


def format_to_regex(log_format: str) -> tuple[list[str], str]:
    """Return the header names and the anchored regex source for ``log_format``."""

    headers: list[str] = []
    parts: list[str] = []
    for index, piece in enumerate(_FIELD.split(log_format)):
        if index % 2 == 0:
            parts.append(re.sub(" +", r"\\s+", piece))
        else:
            name = piece.strip("<>")
            if not name.isidentifier():
                raise ValueError(f"invalid field name {name!r} in log format")
            if name in headers:
                raise ValueError(f"duplicate field {name!r} in log format")
            headers.append(name)
            parts.append(f"(?P<{name}>.*?)")
    if "Content" not in headers:
        raise ValueError("a log format must contain a <Content> field")
    return headers, "^" + "".join(parts) + "$"


class LogFormat:
    """A compiled log format that splits a line into header fields and content."""

    def __init__(self, log_format: str) -> None:
        self.source = log_format
        self.headers, self.pattern = format_to_regex(log_format)
        self._regex = re.compile(self.pattern)

    def match(self, line: str) -> dict[str, str] | None:
        """Return the header fields of ``line`` (already stripped), or None if it does not match."""

        found = self._regex.search(line)
        if found is None:
            return None
        return {name: found.group(name) for name in self.headers}

    def content(self, line: str) -> str | None:
        """Return only the ``Content`` field of ``line``, or None if it does not match."""

        found = self._regex.search(line)
        return None if found is None else found.group("Content")


@dataclass(frozen=True)
class DatasetSettings:
    """Parser settings for one dataset."""

    log_format: str
    masks: tuple[str, ...] = ()
    similarity: float = 0.5
    depth: int = 4
    max_children: int = 100
    extra: dict[str, str] = field(default_factory=dict)


BENCHMARK_SETTINGS: dict[str, DatasetSettings] = {
    "HDFS": DatasetSettings(
        "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        (r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"),
        0.5,
        4,
    ),
    "Hadoop": DatasetSettings(
        r"<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        (r"(\d+\.){3}\d+",),
        0.5,
        4,
    ),
    "Spark": DatasetSettings(
        "<Date> <Time> <Level> <Component>: <Content>",
        (r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"),
        0.5,
        4,
    ),
    "Zookeeper": DatasetSettings(
        r"<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        (r"(/|)(\d+\.){3}\d+(:\d+)?",),
        0.5,
        4,
    ),
    "BGL": DatasetSettings(
        "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> "
        "<Content>",
        (r"core\.\d+",),
        0.5,
        4,
    ),
    "HPC": DatasetSettings(
        "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        (r"=\d+",),
        0.5,
        4,
    ),
    "Thunderbird": DatasetSettings(
        "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> "
        r"<Component>(\[<PID>\])?: <Content>",
        (r"(\d+\.){3}\d+",),
        0.5,
        4,
    ),
    "Windows": DatasetSettings(
        "<Date> <Time>, <Level>                  <Component>    <Content>",
        (r"0x.*?\s",),
        0.7,
        5,
    ),
    "Linux": DatasetSettings(
        r"<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        (r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"),
        0.39,
        6,
    ),
    "Android": DatasetSettings(
        "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        (
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
        ),
        0.2,
        6,
    ),
    "HealthApp": DatasetSettings(r"<Time>\|<Component>\|<Pid>\|<Content>", (), 0.2, 4),
    "Apache": DatasetSettings(r"\[<Time>\] \[<Level>\] <Content>", (r"(\d+\.){3}\d+",), 0.5, 4),
    "Proxifier": DatasetSettings(
        r"\[<Time>\] <Program> - <Content>",
        (r"<\d+\ssec", r"([\w-]+\.)+[\w-]+(:\d+)?", r"\d{2}:\d{2}(:\d{2})*", r"[KGTM]B"),
        0.6,
        3,
    ),
    "OpenSSH": DatasetSettings(
        r"<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        (r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"),
        0.6,
        5,
    ),
    "OpenStack": DatasetSettings(
        r"<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        (r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"),
        0.5,
        5,
    ),
    "Mac": DatasetSettings(
        r"<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        (r"([\w-]+\.){2,}[\w-]+",),
        0.7,
        6,
    ),
}

# The 14 datasets with Loghub-2.0 annotations (Jiang et al., ISSTA 2024).
LOGHUB2_DATASETS: tuple[str, ...] = (
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
)


def settings_for(dataset: str) -> DatasetSettings:
    """Look up the benchmark settings for ``dataset`` (case-insensitive)."""

    for name, value in BENCHMARK_SETTINGS.items():
        if name.lower() == dataset.lower():
            return value
    known = ", ".join(sorted(BENCHMARK_SETTINGS))
    raise KeyError(f"unknown dataset {dataset!r}; known datasets: {known}")
