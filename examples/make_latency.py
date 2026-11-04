"""Write examples/latency.csv: a synthetic per-build p95 latency series with two level shifts.

Builds 0 to 179 run at about 120 ms, a regression at build 180 adds about 12 percent, and a fix at
build 260 brings it back to about 121 ms; noise is 3 percent with occasional spikes. Seeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
level = np.concatenate([np.full(180, 120.0), np.full(80, 134.5), np.full(60, 121.0)])
p95 = level * np.exp(rng.normal(0, 0.03, level.size))
p95[rng.random(level.size) < 0.01] *= 1.4
rows = ["build,commit,p95_ms"] + [
    f"{i},{rng.integers(0, 16**7):07x},{v:.2f}" for i, v in enumerate(p95)
]
Path(__file__).with_name("latency.csv").write_text("\n".join(rows) + "\n")
