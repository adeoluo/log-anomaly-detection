"""Anomaly detection on full HDFS and BGL: every detector, one protocol, several seeds.

    uv sync --extra deep
    D=../.bench-data/log-anomaly-detection
    uv run --no-sync python benchmarks/anomaly.py hdfs \
        --log $D/hdfs_v1/HDFS.log --labels $D/hdfs_v1/preprocessed/anomaly_label.csv
    uv run --no-sync python benchmarks/anomaly.py bgl --log $D/bgl_v1/BGL.log

Protocol (log_anomaly_detection.evaluation): the raw log is parsed with Drain (Rust core),
grouped into HDFS block sessions or fixed BGL time windows and ordered by time: the first 70
percent is for fitting (normal sequences only), the next 10 percent for validation and the last
20 percent for test. Every detector decides with its own rule; a second set of decisions uses the
threshold that maximizes F1 on the validation part. Test labels never set a threshold.
Stochastic detectors (isolation forest, DeepLog, transformer) run with seeds 0, 1 and 2 and
report the mean and standard deviation. Results go to benchmarks/results/anomaly_<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from log_anomaly_detection.datasets import load_bgl, load_hdfs
from log_anomaly_detection.evaluation import Split, build_detector, run_count_detector
from log_anomaly_detection.features import count_matrix


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"detector": runs[0]["detector"], "seeds": len(runs)}
    for key in (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "fit_seconds",
        "score_seconds",
    ):
        vals = [r[key] for r in runs if r[key] is not None]
        out[key] = round(float(np.mean(vals)), 4) if vals else None
        out[f"{key}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
    for key in ("precision", "recall", "f1"):
        vals = [r["tuned"][key] for r in runs if r.get("tuned")]
        out[f"tuned_{key}"] = round(float(np.mean(vals)), 4) if vals else None
        out[f"tuned_{key}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
    out["runs"] = runs
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["hdfs", "bgl"])
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--window", type=int, default=300, help="BGL window in seconds")
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--detectors",
        nargs="+",
        default=["pca", "invariants", "logcluster", "iforest", "deeplog", "transformer"],
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-lines", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    start = time.perf_counter()
    if args.dataset == "hdfs":
        if args.labels is None:
            ap.error("hdfs needs --labels")
        ds = load_hdfs(args.log, args.labels, max_lines=args.max_lines)
    else:
        ds = load_bgl(args.log, window_seconds=args.window, max_lines=args.max_lines)
    load_seconds = time.perf_counter() - start
    print(ds.summary(), f"loaded in {load_seconds:.1f} s", flush=True)
    split = Split.make(ds.labels, args.train_fraction, args.validation_fraction)
    counts = count_matrix(ds.sequences, ds.n_events)
    results = []
    for name in args.detectors:
        seeds = args.seeds if name in ("iforest", "deeplog", "transformer") else [0]
        runs = []
        for seed in seeds:
            if name in ("deeplog", "transformer"):
                from log_anomaly_detection.evaluation import run_sequence_detector

                res = run_sequence_detector(
                    name, ds, split, seed=seed, epochs=args.epochs, device=args.device
                )
            else:
                kwargs = {"seed": seed} if name == "iforest" else {}
                res = run_count_detector(build_detector(name, **kwargs), counts, ds.labels, split)
            row = res.as_dict()
            row["seed"] = seed
            runs.append(row)
            print(
                f"{row['detector']:<18} seed {seed}: P {row['precision']:.3f} "
                f"R {row['recall']:.3f} F1 {row['f1']:.3f} AUC {row['roc_auc']} "
                f"tuned F1 {row.get('tuned', {}).get('f1')} "
                f"fit {row['fit_seconds']:.1f}s "
                f"score {row['score_seconds']:.1f}s rss {peak_rss_mb():.0f} MB",
                flush=True,
            )
        results.append(summarize(runs))
    report = {
        "meta": {"machine": platform.machine(), "platform": platform.platform()},
        "dataset": ds.summary(),
        "load_seconds": round(load_seconds, 2),
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            **split.describe(ds.labels),
        },
        "results": results,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    out = args.out or Path(f"benchmarks/results/anomaly_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
