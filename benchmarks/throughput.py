"""Parser throughput on the raw HDFS and BGL logs: Rust core, Python reference, drain3.

    uv sync --group validation
    uv run --no-sync python benchmarks/throughput.py \
        --hdfs .bench-data/hdfs_v1/HDFS.log --bgl .bench-data/bgl_v1/BGL.log

Every engine reads the plain file, strips each line, splits off the header with the dataset's log
format, masks variables and parses the content with the same Drain settings. The Rust core streams
the whole file (``LogParser.parse_file``: parallel chunked preprocessing, sequential tree
updates) with 8 threads and with 1 thread. The Python reference and drain3 are timed on the first
``--sample`` lines, since a full run takes many minutes; their lines-per-second figure is what is
compared. Peak memory is the process's maximum resident set size after each run (it only grows,
so the Rust rows, which run first, are the meaningful ones). Results go to
benchmarks/results/throughput.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from log_anomaly_detection.formats import LogFormat, settings_for
from log_anomaly_detection.parser import LogParser, iter_lines


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def rust_run(name: str, path: Path, threads: int) -> dict[str, object]:
    """Run in a subprocess so RAYON_NUM_THREADS takes effect and memory is measured alone."""

    code = (
        "import json,resource,sys,time\n"
        "from log_anomaly_detection.formats import settings_for\n"
        "from log_anomaly_detection.parser import LogParser\n"
        f"p = LogParser.for_dataset(settings_for({name!r}), backend='rust')\n"
        "t = time.perf_counter()\n"
        f"r = p.parse_file({str(path)!r})\n"
        "s = time.perf_counter() - t\n"
        "rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "rss = rss / 1e6 if sys.platform == 'darwin' else rss / 1e3\n"
        "print(json.dumps({'lines': r.num_lines, 'seconds': s, 'templates': len(r.templates),"
        " 'peak_rss_mb': rss}))\n"
    )
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    row = json.loads(out.stdout.strip().splitlines()[-1])
    row.update(engine=f"Rust core, {threads} thread{'s' if threads > 1 else ''}", dataset=name)
    row["lines_per_second"] = row["lines"] / row["seconds"]
    return row


def python_run(name: str, path: Path, sample: int) -> dict[str, object]:
    parser = LogParser.for_dataset(settings_for(name), backend="python")
    lines = list(iter_lines(path, sample))
    start = time.perf_counter()
    parser.add_lines(lines)
    seconds = time.perf_counter() - start
    return {
        "engine": "Python reference",
        "dataset": name,
        "lines": len(lines),
        "seconds": seconds,
        "lines_per_second": len(lines) / seconds,
        "templates": parser.num_clusters,
    }


def drain3_run(name: str, path: Path, sample: int) -> dict[str, object]:
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    s = settings_for(name)
    config = TemplateMinerConfig()
    config.drain_depth = s.depth
    config.drain_sim_th = s.similarity
    config.masking_instructions = [MaskingInstruction(p, "*") for p in s.masks]
    miner = TemplateMiner(config=config)
    fmt = LogFormat(s.log_format)
    lines = list(iter_lines(path, sample))
    start = time.perf_counter()
    for line in lines:
        content = fmt.content(line.strip())
        miner.add_log_message(line.strip() if content is None else content)
    seconds = time.perf_counter() - start
    return {
        "engine": "drain3 0.9.11",
        "dataset": name,
        "lines": len(lines),
        "seconds": seconds,
        "lines_per_second": len(lines) / seconds,
        "templates": len(miner.drain.clusters),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--hdfs", type=Path)
    ap.add_argument("--bgl", type=Path)
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--no-drain3", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/throughput.json"))
    args = ap.parse_args()
    rows = []
    for name, path in (("HDFS", args.hdfs), ("BGL", args.bgl)):
        if path is None:
            continue
        for threads in (8, 1):
            rows.append(rust_run(name, path, threads))
            print(rows[-1], flush=True)
        rows.append(python_run(name, path, args.sample))
        print(rows[-1], flush=True)
        if not args.no_drain3:
            rows.append(drain3_run(name, path, args.sample))
            print(rows[-1], flush=True)
    meta = {"machine": platform.machine(), "platform": platform.platform(), "cpus": os.cpu_count()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"meta": meta, "sample": args.sample, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
