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


def hdfs_like_lines(n_blocks: int, anomalous: set[int], seed: int = 0) -> list[str]:
    """Interleaved HDFS-format lines for ``n_blocks`` block sessions; listed blocks fail."""

    rng = np.random.default_rng(seed)
    head = "081109 2035{:02d} {} INFO dfs.DataNode$PacketResponder: "
    out: list[tuple[float, str]] = []
    for b in range(n_blocks):
        t0 = b * 3.0
        blk = f"blk_{1000 + b}"
        steps = [
            f"Receiving block {blk} src: /10.0.0.{b % 250}:50010 dest: /10.0.1.{b % 250}:50010",
            f"Received block {blk} of size {int(rng.integers(1000, 9000))} from /10.0.0.{b % 250}",
            f"PacketResponder {int(rng.integers(0, 3))} for block {blk} terminating",
        ]
        if b in anomalous:
            steps[1] = (
                f"Exception in receiveBlock for block {blk} java.io.IOException: Connection reset"
            )
        for i, text in enumerate(steps):
            out.append((t0 + i * 2.5, head.format(int(t0) % 60, 100 + b % 50) + text))
    out.sort(key=lambda item: item[0])
    return [line for _, line in out]
