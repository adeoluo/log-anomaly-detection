"""Entrypoint: `python -m log_anomaly_detection`."""

from __future__ import annotations

import sys

from log_anomaly_detection.cli import main

if __name__ == "__main__":
    sys.exit(main())
