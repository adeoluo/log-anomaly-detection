"""Shared fixtures: bundled Loghub-2k samples and the available parser backends."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from log_anomaly_detection.backend import rust_available

FIXTURES = Path(__file__).parent / "fixtures"
LOGHUB_2K = FIXTURES / "loghub_2k"

if os.environ.get("LAD_REQUIRE_RUST") == "1" and not rust_available():  # pragma: no cover
    raise RuntimeError("LAD_REQUIRE_RUST=1 but the Rust extension is not importable")

BACKENDS = ["python", "rust"] if rust_available() else ["python"]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def loghub_2k() -> Path:
    return LOGHUB_2K
