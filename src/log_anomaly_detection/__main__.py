"""Entrypoint: `python -m log_anomaly_detection`."""

from __future__ import annotations

from log_anomaly_detection import __version__
from log_anomaly_detection.configuration import settings


def main() -> None:
    header = f"Log Anomaly Detection v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
