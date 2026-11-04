"""The Rust core must assign every line to the same cluster with the same template as Python."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from log_anomaly_detection.backend import rust_available
from log_anomaly_detection.formats import LOGHUB2_DATASETS, settings_for
from log_anomaly_detection.parser import LogParser, iter_lines

pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


def _pair(**kwargs: object) -> tuple[LogParser, LogParser]:
    return LogParser(backend="python", **kwargs), LogParser(backend="rust", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", LOGHUB2_DATASETS)
def test_loghub_2k_parity(name: str, loghub_2k: Path) -> None:
    lines = list(iter_lines(loghub_2k / f"{name}_2k.log.gz"))
    py = LogParser.for_dataset(settings_for(name), skip_unmatched=True, backend="python")
    rs = LogParser.for_dataset(settings_for(name), skip_unmatched=True, backend="rust")
    assert np.array_equal(py.add_lines(lines), rs.add_lines(lines))
    assert py.templates() == rs.templates()
    assert py.sizes() == rs.sizes()


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("HDFS", {"session_regex": r"blk_-?\d+"}),
        ("BGL", {"label_field": "Label", "time_field": "Timestamp"}),
        ("Thunderbird", {"label_field": "Label", "time_field": "Timestamp", "max_lines": 777}),
    ],
)
def test_parse_file_parity(name: str, kwargs: dict[str, object], loghub_2k: Path) -> None:
    path = loghub_2k / f"{name}_2k.log.gz"
    py = LogParser.for_dataset(settings_for(name), backend="python").parse_file(path, **kwargs)  # type: ignore[arg-type]
    rs = LogParser.for_dataset(settings_for(name), backend="rust").parse_file(path, **kwargs)  # type: ignore[arg-type]
    assert np.array_equal(py.event_ids, rs.event_ids)
    assert py.templates == rs.templates
    assert np.array_equal(py.session_lines, rs.session_lines)
    assert np.array_equal(py.session_ids, rs.session_ids)
    assert py.session_keys == rs.session_keys
    for a, b in ((py.labels, rs.labels), (py.times, rs.times)):
        assert (a is None and b is None) or np.array_equal(a, b)  # type: ignore[arg-type]
    assert py.unmatched == rs.unmatched


TOKENS = st.sampled_from(
    ["a", "b", "blk_-12", "10.0.0.1:80", "x1", "ERROR", "<*>", "é", "ü2", "=", "", "Z"]
)
SPACES = st.sampled_from([" ", "  ", "\t", "\u00a0", "\x1c", "\u3000", " \r"])


@st.composite
def raw_line(draw: st.DrawFn) -> str:
    parts = draw(st.lists(TOKENS, max_size=8))
    out = draw(st.sampled_from(["", " ", "\x1f"]))
    for part in parts:
        out += part + draw(SPACES)
    return out


@settings(max_examples=200, deadline=None)
@given(
    st.lists(raw_line(), min_size=1, max_size=80),
    st.integers(3, 6),
    st.sampled_from([0.0, 0.3, 0.5, 0.7, 1.0]),
    st.integers(2, 5),
    st.sampled_from([(), (r"blk_-?\d+",), (r"(\d+\.){3}\d+(:\d+)?", r"\d+")]),
)
def test_random_lines_parity(
    lines: list[str], depth: int, sim: float, children: int, masks: tuple[str, ...]
) -> None:
    py, rs = _pair(depth=depth, similarity=sim, max_children=children, masks=masks)
    assert py.add_lines(lines).tolist() == rs.add_lines(lines).tolist()
    assert py.templates() == rs.templates()
    for line in lines[:5]:
        assert py.tokenize(line) == rs.tokenize(line)


@settings(max_examples=60, deadline=None)
@given(st.lists(raw_line(), min_size=1, max_size=40))
def test_format_and_unmatched_parity(lines: list[str]) -> None:
    for skip in (True, False):
        py, rs = _pair(log_format="<Level> <Content>", skip_unmatched=skip)
        assert py.add_lines(lines).tolist() == rs.add_lines(lines).tolist()
        assert py.templates() == rs.templates()


def test_large_batches_cross_chunk_boundaries(loghub_2k: Path) -> None:
    lines = list(iter_lines(loghub_2k / "BGL_2k.log.gz")) * 40  # 80,000 lines > one chunk
    py = LogParser.for_dataset(settings_for("BGL"), backend="python")
    rs = LogParser.for_dataset(settings_for("BGL"), backend="rust")
    assert np.array_equal(py.add_lines(lines), rs.add_lines(iter(lines)))
    assert py.templates() == rs.templates()
