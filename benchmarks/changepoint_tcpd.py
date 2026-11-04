"""Change-point detection on the Turing Change Point Dataset (TCPD), default settings.

Reproduce:

    git clone https://github.com/alan-turing-institute/TCPD && cd TCPD && make   # or build_tcpd.py
    uv sync --group validation
    uv run --no-sync python benchmarks/changepoint_tcpd.py --tcpd path/to/TCPD

Every method runs once per univariate series with fixed default parameters (no tuning on the
annotations), as in the "Default" experiment of van den Burg and Williams (2020). Scores are the
benchmark's F1 (margin 5) and segmentation covering, averaged over series. Missing values are
linearly interpolated. Results go to benchmarks/results/changepoint_tcpd.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from log_anomaly_detection.changepoint import BOCPD, CUSUM, covering, f_measure

# Default-experiment univariate means printed in the TCPDBench paper table
# (analysis/output/tables/aggregate_scores_wide.tex, complete-case analysis); quoted, not rerun.
PUBLISHED = {
    "binseg (R changepoint)": {"cover": 0.672, "f1": 0.698},
    "pelt (R changepoint)": {"cover": 0.652, "f1": 0.674},
    "bocpd (R ocp)": {"cover": 0.594, "f1": 0.662},
    "zero (no change points)": {"cover": 0.566, "f1": 0.645},
}
PER_SERIES = Path(__file__).with_name("published_tcpd.json")


def load_series(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    raw = np.array(
        [np.nan if v is None else float(v) for v in data["series"][0]["raw"]], dtype=np.float64
    )
    bad = np.isnan(raw)
    if bad.any() and not bad.all():
        idx = np.arange(raw.size)
        raw[bad] = np.interp(idx[bad], idx[~bad], raw[~bad])
    return raw


def ruptures_detector(algo: str):  # type: ignore[no-untyped-def]
    import ruptures as rpt

    def run(x: np.ndarray) -> list[int]:
        z = (x - x.mean()) / (x.std() or 1.0)
        pen = 2.0 * np.log(max(z.size, 2))
        model = (
            rpt.Pelt(model="l2", min_size=2)
            if algo == "pelt"
            else rpt.Binseg(model="l2", min_size=2)
        )
        return [int(b) for b in model.fit(z).predict(pen=pen)[:-1]]

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tcpd", type=Path, required=True)
    ap.add_argument("--no-ruptures", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/changepoint_tcpd.json"))
    args = ap.parse_args()
    annotations = json.loads((args.tcpd / "annotations.json").read_text())
    methods = {
        "zero": lambda x: [],
        "BOCPD (this package)": lambda x: BOCPD().detect(x),
        "CUSUM (this package)": lambda x: CUSUM().detect((x - x.mean()) / (x.std() or 1.0)),
    }
    if not args.no_ruptures:
        methods["ruptures Pelt (l2, pen 2 log n)"] = ruptures_detector("pelt")
        methods["ruptures Binseg (l2, pen 2 log n)"] = ruptures_detector("binseg")
    rows = []
    for folder in sorted((args.tcpd / "datasets").iterdir()):
        path = folder / f"{folder.name}.json"
        if not path.exists() or folder.name not in annotations:
            continue
        meta = json.loads(path.read_text())
        if meta.get("n_dim", 1) != 1:
            continue
        x = load_series(path)
        for name, method in methods.items():
            start = time.perf_counter()
            cps = method(x)
            seconds = time.perf_counter() - start
            f1, p, r = f_measure(annotations[folder.name], cps)
            cov = covering(annotations[folder.name], cps, x.size)
            rows.append(
                {
                    "dataset": folder.name,
                    "n": int(x.size),
                    "method": name,
                    "f1": f1,
                    "precision": p,
                    "recall": r,
                    "cover": cov,
                    "change_points": len(cps),
                    "seconds": round(seconds, 4),
                }
            )
    summary = {}
    for name in methods:
        mine = [r for r in rows if r["method"] == name]
        summary[name] = {
            "series": len(mine),
            "f1": round(float(np.mean([r["f1"] for r in mine])), 3),
            "cover": round(float(np.mean([r["cover"] for r in mine])), 3),
            "seconds": round(float(np.sum([r["seconds"] for r in mine])), 3),
        }
        print(
            f"{name:<36} F1 {summary[name]['f1']:.3f}  cover {summary[name]['cover']:.3f}  "
            f"({summary[name]['series']} series, {summary[name]['seconds']} s)"
        )
    # Published per-series scores averaged over exactly the series run here (like for like).
    per_series = json.loads(PER_SERIES.read_text())
    names = sorted({r["dataset"] for r in rows})
    same_series = {}
    for m in per_series["methods"]:
        f1s = [
            per_series["f1"][d][m] for d in names if per_series["f1"].get(d, {}).get(m) is not None
        ]
        covs = [
            per_series["cover"][d][m]
            for d in names
            if per_series["cover"].get(d, {}).get(m) is not None
        ]
        same_series[m] = {
            "series": len(f1s),
            "f1": round(float(np.mean(f1s)), 3),
            "cover": round(float(np.mean(covs)), 3),
        }
        print(
            f"published {m:<26} F1 {same_series[m]['f1']:.3f}  "
            f"cover {same_series[m]['cover']:.3f}  ({len(f1s)} series)"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "summary": summary,
                "published_same_series": same_series,
                "published_paper_table": PUBLISHED,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
