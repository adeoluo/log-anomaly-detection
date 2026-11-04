from __future__ import annotations

import re

import pytest

from log_anomaly_detection.formats import (
    BENCHMARK_SETTINGS,
    LOGHUB2_DATASETS,
    LogFormat,
    format_to_regex,
    settings_for,
)


def test_format_to_regex_builds_lazy_named_groups() -> None:
    headers, pattern = format_to_regex("<Date> <Time>  <Level>: <Content>")
    assert headers == ["Date", "Time", "Level", "Content"]
    assert pattern == r"^(?P<Date>.*?)\s+(?P<Time>.*?)\s+(?P<Level>.*?):\s+(?P<Content>.*?)$"


def test_log_format_matches_and_extracts() -> None:
    fmt = LogFormat(r"<Date> <Time> <Level> \[<Process>\] <Component>: <Content>")
    line = "2015-10-18 18:01:47,978 INFO [main] org.Foo: Created MRAppMaster"
    fields = fmt.match(line)
    assert fields is not None
    assert fields["Process"] == "main"
    assert fmt.content(line) == "Created MRAppMaster"
    assert fmt.match("no structure here") is None
    assert fmt.content("no structure here") is None


@pytest.mark.parametrize(
    "bad", ["<Date> <Time>", "<1bad> <Content>", "<A> <A> <Content>", "<a-b> <Content>"]
)
def test_invalid_formats_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        LogFormat(bad)


def test_every_benchmark_setting_compiles() -> None:
    assert set(LOGHUB2_DATASETS) <= set(BENCHMARK_SETTINGS)
    for name, settings in BENCHMARK_SETTINGS.items():
        LogFormat(settings.log_format)
        for mask in settings.masks:
            re.compile(mask)
        assert settings_for(name.lower()) is settings
        assert 3 <= settings.depth <= 6


def test_unknown_dataset() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        settings_for("nope")
