"""Smoke tests - replace with real unit and integration tests.

These cover every starter module so the template is green (at least 80 percent)
out of the box; treat them as the pattern for your own tests, not as sufficient
coverage.
"""

from __future__ import annotations

import pytest

import log_anomaly_detection.__main__ as entrypoint
from log_anomaly_detection import __version__
from log_anomaly_detection.configuration import Settings
from log_anomaly_detection.evaluation import main as evaluation_main
from log_anomaly_detection.evaluation import run_evaluation


def test_version() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_evaluation_returns_metrics() -> None:
    results = run_evaluation()
    assert "primary_metric" in results
    assert isinstance(results["primary_metric"], float)


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.seed == 42


def test_entrypoint_runs(capsys: pytest.CaptureFixture[str]) -> None:
    entrypoint.main()
    assert "environment=" in capsys.readouterr().out


def test_evaluation_main_prints_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    evaluation_main()
    assert "primary_metric" in capsys.readouterr().out
