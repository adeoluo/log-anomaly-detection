"""Synthetic workloads with known structure, shared by detector tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def workflow_counts(
    n_normal: int, n_anomalous: int, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    """Event counts of a file-transfer workflow with invariants, plus anomalies that break them.

    Columns: 0 open, 1 write, 2 close, 3 replicate, 4 ack, 5 heartbeat, 6 error.
    Normal runs: open == close, replicate == ack == 3 * open, writes vary, no errors.
    """

    rng = np.random.default_rng(seed)

    def normal(n: int) -> NDArray[np.float64]:
        opens = rng.integers(1, 3, n)
        x = np.zeros((n, 7))
        x[:, 0] = opens
        x[:, 1] = rng.integers(1, 6, n)
        x[:, 2] = opens
        x[:, 3] = 3 * opens
        x[:, 4] = 3 * opens
        x[:, 5] = rng.integers(0, 3, n)
        return x

    x_norm = normal(n_normal)
    x_anom = normal(n_anomalous)
    kind = rng.integers(0, 3, n_anomalous)
    x_anom[kind == 0, 2] -= 1  # a missing close
    x_anom[kind == 1, 4] -= rng.integers(1, 3, int((kind == 1).sum()))  # lost acks
    x_anom[kind == 2, 6] += 1  # an error event
    x = np.vstack([x_norm, x_anom])
    y = np.concatenate([np.zeros(n_normal, np.uint8), np.ones(n_anomalous, np.uint8)])
    return x, y
