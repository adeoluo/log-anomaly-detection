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
    print(
        f"{'detector':<18} {'precision':>9} {'recall':>7} {'F1':>6} {'ROC AUC':>8} {'AP':>6}"
        f" {'tuned F1':>9}"
    )
    for r in results:
        auc = r.get("roc_auc")
        ap = r.get("average_precision")
        tuned = r.get("tuned")
        tf1 = "n/a" if not isinstance(tuned, dict) else f"{tuned['f1']:.3f}"
        print(
            f"{r['detector']!s:<18} {r['precision']:>9.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
            f"{'n/a' if auc is None else f'{auc:.3f}':>8} {'n/a' if ap is None else f'{ap:.3f}':>6}"
            f" {tf1:>9}"
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
    neural = {"epochs": args.epochs, "device": args.device}
    report = evaluate_dataset(
        dataset,
        args.detectors,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        detector_kwargs={"deeplog": neural, "transformer": neural},
    )
    split = report["split"]
    print(
        f"train {split['train']:,} ({split['train_normal']:,} normal), "
        f"validation {split['validation']:,}, "
        f"test {split['test']:,} ({split['test_anomalies']:,} anomalous)"
    )
    print_results(report["results"])
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def read_series(path: str, column: str | None) -> list[float]:
    """Numbers from a one-column text file or a CSV column (blank and non-numeric lines skipped)."""

    import csv

    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        if column is None:
            for line in handle:
                try:
                    values.append(float(line.strip()))
                except ValueError:
                    continue
            return values
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column {column!r} not found in {path}")
        for row in reader:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                continue
    return values


def cmd_regress(args: argparse.Namespace) -> int:
    from log_anomaly_detection.changepoint import detect_regressions

    values = read_series(args.input, args.column)
    if len(values) < 3:
        raise ValueError("need at least 3 numeric values")
    found = detect_regressions(
        values,
        method=args.method,
        direction=args.direction,
        min_relative=args.min_relative,
        min_segment=args.min_segment,
    )
    if args.json:
        print(json.dumps({"points": len(values), "regressions": [r.as_dict() for r in found]}))
        return 0
    print(f"{len(values)} points, {args.method}: {len(found)} level shift(s) ({args.direction})")
    for r in found:
        print(f"  at index {r.index:>5}: {r.before:,.3f} -> {r.after:,.3f} ({r.relative:+.1%})")
    return 0


DEFAULT_SESSION_REGEX = {"HDFS": r"blk_-?\d+"}


def build_monitor(args: argparse.Namespace) -> tuple[object, dict[str, int] | None, dict[str, int]]:
    """A StreamMonitor trained as the stream and serve commands ask; plus the labels if given."""

    from itertools import islice

    from log_anomaly_detection.datasets import read_hdfs_labels
    from log_anomaly_detection.parser import iter_lines
    from log_anomaly_detection.streaming import StreamMonitor

    settings = settings_for(args.dataset)
    regex = args.session_regex or DEFAULT_SESSION_REGEX.get(args.dataset)
    monitor = StreamMonitor(
        LogParser.for_dataset(settings, backend=args.backend),
        session_regex=regex,
        idle_lines=args.idle_lines,
        window_lines=args.window_lines,
        detector=args.detector,
    )
    labels = read_hdfs_labels(args.labels) if args.labels else None
    info: dict[str, int] = {}
    if args.train:
        lines = iter_lines(args.train, args.train_lines)
        exclude = {k for k, v in labels.items() if v} if labels else None
        info = monitor.train(islice(lines, None), exclude_keys=exclude)
    return monitor, labels, info


def cmd_stream(args: argparse.Namespace) -> int:
    from itertools import islice

    from log_anomaly_detection.parser import iter_lines
    from log_anomaly_detection.streaming import StreamMonitor

    if not args.train:
        raise ValueError("stream needs --train to fit a detector first")
    monitor, labels, info = build_monitor(args)
    assert isinstance(monitor, StreamMonitor)
    print(
        f"trained {args.detector} on {info['lines']:,} lines: {info['sessions']:,} sessions, "
        f"{info['templates']} templates",
        flush=True,
    )
    source = sys.stdin if args.input == "-" else iter_lines(args.input)
    stop = None if args.max_lines is None else args.skip_lines + args.max_lines
    stream = islice((line.rstrip("\n") for line in source), args.skip_lines, stop)
    start = time.perf_counter()
    alerts = []
    for alert in monitor.process_many(stream):
        alerts.append(alert)
    alerts += monitor.flush()
    seconds = time.perf_counter() - start
    shown = 0
    for alert in alerts:
        if args.json:
            print(json.dumps(alert.as_dict()))
            continue
        if shown < args.show:
            tag = ""
            if labels is not None and alert.kind == "session":
                tag = " [labelled anomaly]" if labels.get(alert.key) else " [labelled normal]"
            print(
                f"ALERT {alert.kind:<7} {alert.key:<28} score {alert.score:10.3f} > "
                f"{alert.threshold:.3f}  events {alert.events}{tag}"
            )
            shown += 1
    stats = monitor.stats
    rate = stats["lines"] / seconds if seconds > 0 else float("inf")
    summary = (
        f"{stats['lines']:,} lines in {seconds:.1f} s ({rate:,.0f} lines/s), "
        f"{stats['sessions_scored']:,} sessions scored, {len(alerts):,} alerts"
    )
    if labels is not None:
        flagged = [a for a in alerts if a.kind == "session"]
        hits = sum(1 for a in flagged if labels.get(a.key))
        summary += f"; {hits:,} of {len(flagged):,} session alerts are labelled anomalies"
    print(summary, file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocking server
    from log_anomaly_detection.configuration import settings
    from log_anomaly_detection.service import App, serve
    from log_anomaly_detection.streaming import StreamMonitor

    monitor, _, info = build_monitor(args)
    assert isinstance(monitor, StreamMonitor)
    if info:
        print(f"trained {args.detector} on {info['lines']:,} lines ({info['templates']} templates)")
    serve(
        App(monitor),
        args.host or settings.host,
        args.port or settings.port,
        settings.max_body_bytes,
    )
    return 0


def add_monitor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="HDFS", choices=sorted(BENCHMARK_SETTINGS))
    p.add_argument("--train", help="historical log to learn templates and fit the detector")
    p.add_argument("--train-lines", type=int, help="use only the first N training lines")
    p.add_argument(
        "--labels", help="HDFS anomaly_label.csv: keep labelled anomalies out of training"
    )
    p.add_argument(
        "--detector", default="pca", choices=["pca", "invariants", "logcluster", "iforest"]
    )
    p.add_argument("--session-regex", help="session key regex (default: block ids for HDFS)")
    p.add_argument(
        "--idle-lines", type=int, default=20_000, help="close a session after N idle lines"
    )
    p.add_argument("--window-lines", type=int, default=100, help="window size without sessions")
    p.add_argument("--backend", default="auto", choices=["auto", "rust", "python"])


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=["hdfs", "bgl"])
    p.add_argument("--log", required=True, help="raw log file (.gz allowed)")
    p.add_argument("--labels", help="HDFS anomaly_label.csv")
    p.add_argument("--window", type=int, default=3600, help="BGL window, seconds")
    p.add_argument("--step", type=int, help="BGL window step, seconds (default: the window)")
    p.add_argument("--window-lines", type=int, help="BGL windows of this many lines instead")
    p.add_argument("--max-lines", type=int)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="end of the training period held out to tune thresholds (0 to skip)",
    )
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
        choices=["pca", "invariants", "logcluster", "iforest", "deeplog", "transformer"],
    )
    p.add_argument("--epochs", type=int, default=10, help="DeepLog and transformer epochs")
    p.add_argument("--device", default="auto", help="torch device for neural detectors")
    p.add_argument("--output", help="write the report as JSON")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("regress", help="find level shifts (regressions) in a numeric metric")
    p.add_argument("input", help="text file with one number per line, or a CSV with --column")
    p.add_argument("--column", help="CSV column holding the metric")
    p.add_argument("--method", default="cusum", choices=["cusum", "bocpd"])
    p.add_argument("--direction", default="up", choices=["up", "down", "both"])
    p.add_argument("--min-relative", type=float, default=0.05, help="smallest relative shift")
    p.add_argument("--min-segment", type=int, default=3, help="shorter segments are spikes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_regress)

    p = sub.add_parser("stream", help="train on history, then stream lines and print alerts")
    add_monitor_args(p)
    p.add_argument("--input", default="-", help="log to stream ('-' for stdin)")
    p.add_argument("--skip-lines", type=int, default=0, help="skip the first N input lines")
    p.add_argument("--max-lines", type=int, help="stop after N input lines")
    p.add_argument("--show", type=int, default=20, help="alerts to print")
    p.add_argument("--json", action="store_true", help="every alert as a JSON line")
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("serve", help="run the HTTP service")
    add_monitor_args(p)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
