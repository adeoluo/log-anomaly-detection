"""Command line interface: ``python -m log_anomaly_detection <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from log_anomaly_detection import __version__
from log_anomaly_detection.formats import BENCHMARK_SETTINGS, LOGHUB2_DATASETS, settings_for
from log_anomaly_detection.parser import LogParser


def _parser_for(args: argparse.Namespace) -> LogParser:
    if args.dataset:
        settings = settings_for(args.dataset)
        return LogParser.for_dataset(settings, backend=args.backend)
    return LogParser(
        log_format=args.format,
        masks=args.mask or (),
        depth=args.depth,
        similarity=args.similarity,
        backend=args.backend,
    )


def cmd_parse(args: argparse.Namespace) -> int:
    parser = _parser_for(args)
    start = time.perf_counter()
    result = parser.parse_file(args.input, max_lines=args.max_lines)
    seconds = time.perf_counter() - start
    sizes = parser.sizes()
    order = sorted(range(len(sizes)), key=lambda c: -sizes[c])
    if args.json:
        payload = {
            "lines": result.num_lines,
            "templates": len(sizes),
            "seconds": round(seconds, 4),
            "backend": parser.backend,
            "top": [
                {"id": c, "count": sizes[c], "template": result.templates[c]}
                for c in order[: args.top]
            ],
        }
        print(json.dumps(payload))
        return 0
    rate = result.num_lines / seconds if seconds > 0 else float("inf")
    print(
        f"{result.num_lines:,} lines -> {len(sizes):,} templates in {seconds:.2f} s "
        f"({rate:,.0f} lines/s, {parser.backend} backend)"
    )
    for c in order[: args.top]:
        print(f"{sizes[c]:>10,}  E{c:<5} {result.templates[c]}")
    return 0


def cmd_parse_eval(args: argparse.Namespace) -> int:
    from log_anomaly_detection.loghub import run_2k, run_full

    names = args.datasets or list(LOGHUB2_DATASETS)
    rows = []
    runs = []
    print(
        f"{'dataset':<12} {'lines':>11} {'GA':>6} {'FGA':>6} {'PA':>6} {'FTA':>6} {'lines/s':>12}"
    )
    for name in names:
        run = (run_full if args.full else run_2k)(name, args.data_dir, args.backend)
        s = run.scores
        rows.append(run.as_dict())
        runs.append(run)
        print(
            f"{name:<12} {s.lines:>11,} {s.ga:6.3f} {s.fga:6.3f} {s.pa:6.3f} {s.fta:6.3f} "
            f"{run.lines_per_second:>12,.0f}"
        )
    n = len(rows)
    means = {
        k: sum(float(getattr(r.scores, k)) for r in runs) / n for k in ("ga", "fga", "pa", "fta")
    }
    print(
        f"{'average':<12} {'':>11} {means['ga']:6.3f} {means['fga']:6.3f} "
        f"{means['pa']:6.3f} {means['fta']:6.3f}"
    )
    if args.output:
        Path(args.output).write_text(json.dumps({"runs": rows, "average": means}, indent=2) + "\n")
    return 0


def load_dataset(args: argparse.Namespace) -> object:
    from log_anomaly_detection.datasets import load_bgl, load_hdfs

    if args.dataset == "hdfs":
        if not args.labels:
            raise ValueError("HDFS needs --labels anomaly_label.csv")
        return load_hdfs(args.log, args.labels, max_lines=args.max_lines, backend=args.backend)
    return load_bgl(
        args.log,
        window_seconds=args.window,
        step_seconds=args.step,
        window_lines=args.window_lines,
        max_lines=args.max_lines,
        backend=args.backend,
    )


def print_results(results: list[dict[str, object]]) -> None:
    print(f"{'detector':<18} {'precision':>9} {'recall':>7} {'F1':>6} {'ROC AUC':>8} {'AP':>6}")
    for r in results:
        auc = r.get("roc_auc")
        ap = r.get("average_precision")
        print(
            f"{r['detector']!s:<18} {r['precision']:>9.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
            f"{'n/a' if auc is None else f'{auc:.3f}':>8} {'n/a' if ap is None else f'{ap:.3f}':>6}"
        )


def cmd_detect(args: argparse.Namespace) -> int:
    from log_anomaly_detection.datasets import SessionDataset
    from log_anomaly_detection.evaluation import evaluate_dataset

    dataset = load_dataset(args)
    assert isinstance(dataset, SessionDataset)
    summary = dataset.summary()
    print(
        f"{summary['dataset']}: {summary['lines']:,} lines, {summary['sequences']:,} sequences "
        f"({summary['anomalous']:,} anomalous), {summary['templates']} templates, "
        f"parsed in {summary['parse_seconds']} s"
    )
    report = evaluate_dataset(
        dataset, args.detectors, train_fraction=args.train_fraction, seed=args.seed
    )
    split = report["split"]
    print(
        f"train {split['train']:,} ({split['train_normal']:,} normal), "
        f"test {split['test']:,} ({split['test_anomalies']:,} anomalous)"
    )
    print_results(report["results"])
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=["hdfs", "bgl"])
    p.add_argument("--log", required=True, help="raw log file (.gz allowed)")
    p.add_argument("--labels", help="HDFS anomaly_label.csv")
    p.add_argument("--window", type=int, default=3600, help="BGL window, seconds")
    p.add_argument("--step", type=int, help="BGL window step, seconds (default: the window)")
    p.add_argument("--window-lines", type=int, help="BGL windows of this many lines instead")
    p.add_argument("--max-lines", type=int)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=["auto", "rust", "python"])


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m log_anomaly_detection",
        description="Log parsing and anomaly detection for large clusters.",
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="parse a log file with Drain and list its templates")
    p.add_argument("input", help="log file (.gz allowed)")
    p.add_argument("--dataset", choices=sorted(BENCHMARK_SETTINGS), help="use a dataset's settings")
    p.add_argument("--format", help="log format, e.g. '<Date> <Time> <Level>: <Content>'")
    p.add_argument("--mask", action="append", help="masking regex (repeatable)")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--similarity", type=float, default=0.5)
    p.add_argument("--max-lines", type=int)
    p.add_argument("--top", type=int, default=10, help="templates to show")
    p.add_argument("--json", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto", "rust", "python"])
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("parse-eval", help="score Drain on Loghub-2k or Loghub-2.0 ground truth")
    p.add_argument("--data-dir", required=True, help="Loghub-2k folder or Loghub-2.0 zip folder")
    p.add_argument("--full", action="store_true", help="Loghub-2.0 zips instead of Loghub-2k")
    p.add_argument("--datasets", nargs="+", choices=list(LOGHUB2_DATASETS))
    p.add_argument("--output", help="write the results as JSON")
    p.add_argument("--backend", default="auto", choices=["auto", "rust", "python"])
    p.set_defaults(func=cmd_parse_eval)

    p = sub.add_parser("detect", help="train and score anomaly detectors on HDFS or BGL")
    add_dataset_args(p)
    p.add_argument(
        "--detectors",
        nargs="+",
        default=["pca", "invariants", "logcluster", "iforest"],
        choices=["pca", "invariants", "logcluster", "iforest"],
    )
    p.add_argument("--output", help="write the report as JSON")
    p.set_defaults(func=cmd_detect)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
