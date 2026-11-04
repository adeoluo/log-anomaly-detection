"""Parsing accuracy and throughput on Loghub-2k and Loghub-2.0, against drain3 and published Drain.

Reproduce (downloads in scripts/download_data.sh):

    uv sync --group validation
    uv run --no-sync python benchmarks/parse_loghub.py \
        --loghub2k .bench-data/loghub2/2k_dataset --loghub2 .bench-data/loghub2_full --drain3

Every engine sees the same lines: each line is stripped and split by the dataset's log format, and
lines that do not match are skipped (the LogPAI convention), then the content is masked with the
dataset's regexes and parsed with its depth and similarity threshold. Timings are end to end
(read, decompress, split headers, mask, parse) on one machine; ground truth loading and scoring are
not timed. Results go to benchmarks/results/parsing_{2k,full}.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import zipfile
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from published import DRAIN_PUBLISHED, DRAIN_PUBLISHED_SECONDS

from log_anomaly_detection.formats import LOGHUB2_DATASETS, LogFormat, settings_for
from log_anomaly_detection.loghub import (
    find_2k,
    iter_zip_lines,
    run_2k,
    run_full,
    zip_members,
)
from log_anomaly_detection.parser import iter_lines
from log_anomaly_detection.parsing_metrics import evaluate_clusters, read_truth


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def drain3_parse(name: str, lines: Iterable[str]) -> tuple[np.ndarray, list[str], float]:
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    settings = settings_for(name)
    config = TemplateMinerConfig()
    config.drain_depth = settings.depth
    config.drain_sim_th = settings.similarity
    config.drain_max_children = settings.max_children
    config.masking_instructions = [MaskingInstruction(p, "*") for p in settings.masks]
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    fmt = LogFormat(settings.log_format)
    ids = array("I")
    start = time.perf_counter()
    for line in lines:
        content = fmt.content(line.strip())
        if content is None:
            continue
        ids.append(miner.add_log_message(content)["cluster_id"])
    seconds = time.perf_counter() - start
    top = max(ids) + 1 if ids else 1
    templates = [""] * top
    for cluster in miner.drain.clusters:
        templates[cluster.cluster_id] = cluster.get_template()
    return np.frombuffer(ids, dtype=np.uint32).astype(np.int64), templates, seconds


def score_drain3(name: str, lines: Iterator[str], truth_path: Path, member: str | None) -> dict:
    ids, templates, seconds = drain3_parse(name, lines)
    truth, strings = read_truth(truth_path, member)
    s = evaluate_clusters(truth, strings, ids, templates)
    out = {"dataset": name, **s.as_dict()}
    out["parse_seconds"] = round(seconds, 4)
    out["lines_per_second"] = round(s.lines / seconds, 1)
    return out


def average(rows: list[dict], keys: Iterable[str]) -> dict[str, float]:
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--loghub2k", type=Path, help="Loghub-2k folder (2k_dataset)")
    ap.add_argument("--loghub2", type=Path, help="folder of Loghub-2.0 Zenodo zips")
    ap.add_argument("--datasets", nargs="+", default=list(LOGHUB2_DATASETS))
    ap.add_argument("--drain3", action="store_true", help="also run the drain3 library")
    ap.add_argument("--python", action="store_true", help="also time the Python reference")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    metrics = ("ga", "fga", "pa", "fta")

    if args.loghub2k:
        rows: dict[str, list[dict]] = {"rust": [], "python": [], "drain3": []}
        for name in args.datasets:
            rows["rust"].append(run_2k(name, args.loghub2k, "rust").as_dict())
            if args.python:
                rows["python"].append(run_2k(name, args.loghub2k, "python").as_dict())
            if args.drain3:
                log, truth = find_2k(args.loghub2k, name)
                rows["drain3"].append(score_drain3(name, iter_lines(log), truth, None))
            print("2k", name, {k: round(rows["rust"][-1][k], 3) for k in metrics}, flush=True)
        result = {
            "meta": meta,
            "engines": {k: v for k, v in rows.items() if v},
            "average": {k: average(v, metrics) for k, v in rows.items() if v},
            "published_drain": {
                n: {m: DRAIN_PUBLISHED[n][m][0] for m in ("GA", "FGA", "PA", "FTA")}
                for n in [*args.datasets, "Average"]
            },
        }
        (args.out / "parsing_2k.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.loghub2:
        engines: dict[str, list[dict]] = {"rust": [], "drain3": []}
        for name in args.datasets:
            run = run_full(name, args.loghub2, "rust").as_dict()
            run["peak_rss_mb"] = round(peak_rss_mb(), 1)
            engines["rust"].append(run)
            print("full", name, run, flush=True)
            if args.drain3:
                path = args.loghub2 / f"{name}.zip"
                with zipfile.ZipFile(path) as archive:
                    log_member, truth_member = zip_members(archive, name)
                row = score_drain3(name, iter_zip_lines(path, log_member), path, truth_member)
                engines["drain3"].append(row)
                print("full drain3", name, row, flush=True)
            partial = {"meta": meta, "engines": {k: v for k, v in engines.items() if v}}
            (args.out / "parsing_full.json").write_text(json.dumps(partial, indent=2) + "\n")
        result = {
            "meta": meta,
            "engines": {k: v for k, v in engines.items() if v},
            "average": {k: average(v, metrics) for k, v in engines.items() if v},
            "published_drain": {
                n: {m: DRAIN_PUBLISHED[n][m][1] for m in ("GA", "FGA", "PA", "FTA")}
                for n in [*args.datasets, "Average"]
            },
            "published_drain_seconds": {n: DRAIN_PUBLISHED_SECONDS[n][1] for n in args.datasets},
        }
        (args.out / "parsing_full.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
