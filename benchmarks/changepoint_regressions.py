"""Synthetic performance-regression benchmark for the change-point detectors.

Each series imitates a per-build latency metric: a 100 ms baseline with multiplicative log-normal
noise (2 to 6 percent), 1 percent outlier spikes, and 0 to 3 level shifts of 3 to 30 percent up or
down at random positions at least 40 points apart. Detectors run with default settings, on the raw
series and after a 3-point running median (which removes isolated spikes); a detected
change counts if it is within 5 points of a true shift (the TCPD F1 with one annotator). Scores are
reported per shift-size band. Seeded, so reruns give the same series.

    uv run --no-sync python benchmarks/changepoint_regressions.py [--series 300]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from log_anomaly_detection.changepoint import BOCPD, CUSUM, f_measure


def make_series(
    rng: np.random.Generator, n: int = 400
) -> tuple[np.ndarray, list[int], list[float]]:
    k = int(rng.integers(0, 4))
    cps: list[int] = []
    while len(cps) < k:
        c = int(rng.integers(40, n - 40))
        if all(abs(c - o) >= 40 for o in cps):
            cps.append(c)
    cps.sort()
    sizes = [float(rng.uniform(0.03, 0.30) * rng.choice([-1, 1])) for _ in cps]
    level = np.full(n, 100.0)
    for c, s in zip(cps, sizes, strict=True):
        level[c:] *= 1 + s
    noise = float(rng.uniform(0.02, 0.06))
    x = level * np.exp(rng.normal(0, noise, n))
    spikes = rng.random(n) < 0.01
    x[spikes] *= 1.5
    return x, cps, sizes


def median3(x: np.ndarray) -> np.ndarray:
    """A 3-point running median, which removes isolated spikes."""

    p = np.concatenate([[x[0]], x, [x[-1]]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def ruptures_pelt(x: np.ndarray) -> list[int]:
    import ruptures as rpt

    z = (x - x.mean()) / (x.std() or 1.0)
    return [
        int(b) for b in rpt.Pelt(model="l2", min_size=2).fit(z).predict(pen=2 * np.log(x.size))[:-1]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-ruptures", action="store_true")
    ap.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/changepoint_regressions.json")
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data = [make_series(rng) for _ in range(args.series)]
    base = {
        "CUSUM": lambda x: CUSUM().detect((x - np.median(x)) / (np.std(x) or 1.0)),
        "BOCPD": lambda x: BOCPD().detect(x),
    }
    if not args.no_ruptures:
        base["ruptures Pelt (l2)"] = ruptures_pelt
    methods = dict(base)
    for name, fn in base.items():
        methods[f"{name} + 3-point median"] = lambda x, fn=fn: fn(median3(x))
    bands = {
        "small (3-10%)": (0.03, 0.10),
        "medium (10-20%)": (0.10, 0.20),
        "large (20-30%)": (0.20, 0.31),
    }
    summary: dict[str, dict[str, float]] = {}
    for name, method in methods.items():
        start = time.perf_counter()
        preds = [method(x) for x, _, _ in data]
        seconds = time.perf_counter() - start
        f1s, ps, rs = zip(
            *(f_measure({"truth": c}, p) for (_, c, _), p in zip(data, preds, strict=True)),
            strict=True,
        )
        row: dict[str, float] = {
            "f1": round(float(np.mean(f1s)), 3),
            "precision": round(float(np.mean(ps)), 3),
            "recall": round(float(np.mean(rs)), 3),
            "false_alarms_per_series": round(
                float(
                    np.mean(
                        [
                            len(p) - sum(any(abs(q - c) <= 5 for c in cs) for q in p)
                            for (_, cs, _), p in zip(data, preds, strict=True)
                        ]
                    )
                ),
                3,
            ),
            "seconds": round(seconds, 3),
        }
        for band, (lo, hi) in bands.items():
            hit = total = 0
            for (_, cs, sz), p in zip(data, preds, strict=True):
                for c, s in zip(cs, sz, strict=True):
                    if lo <= abs(s) < hi:
                        total += 1
                        hit += any(abs(q - c) <= 5 for q in p)
            row[f"recall {band}"] = round(hit / total, 3) if total else float("nan")
        summary[name] = row
        print(name, row)
    shifts = sum(len(c) for _, c, _ in data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"series": args.series, "seed": args.seed, "shifts": shifts, "summary": summary},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
