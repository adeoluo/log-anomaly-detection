"""Print the README results tables (Markdown) from benchmarks/results/*.json.

uv run --no-sync python benchmarks/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).with_name("results")


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def parsing() -> None:
    for name, label, col in (
        ("parsing_2k.json", "Loghub-2k", 0),
        ("parsing_full.json", "Loghub-2.0", 1),
    ):
        data = load(name)
        if not data:
            continue
        rust = {r["dataset"]: r for r in data["engines"]["rust"]}
        d3 = {r["dataset"]: r for r in data["engines"].get("drain3", [])}
        pub = data["published_drain"]
        print(f"\n#### {label}\n")
        print(
            "| Dataset | Lines | GA | FGA | PA | FTA | Published Drain GA / FGA / PA / FTA | drain3 GA / FGA / PA / FTA |"
        )
        print("|---|---:|---:|---:|---:|---:|---|---|")
        for ds, r in rust.items():
            p = pub.get(ds, {})
            q = d3.get(ds)
            ptxt = " / ".join(f3(p.get(m)) for m in ("GA", "FGA", "PA", "FTA"))
            qtxt = " / ".join(f3(q[m]) for m in ("ga", "fga", "pa", "fta")) if q else "n/a"
            print(
                f"| {ds} | {r['lines']:,} | {f3(r['ga'])} | {f3(r['fga'])} | {f3(r['pa'])} | {f3(r['fta'])} | {ptxt} | {qtxt} |"
            )
        avg = data["average"]["rust"]
        p = pub["Average"]
        q = data["average"].get("drain3")
        qtxt = " / ".join(f3(q[m]) for m in ("ga", "fga", "pa", "fta")) if q else "n/a"
        print(
            f"| **Average** | | **{f3(avg['ga'])}** | **{f3(avg['fga'])}** | **{f3(avg['pa'])}** | **{f3(avg['fta'])}** | {' / '.join(f3(p[m]) for m in ('GA', 'FGA', 'PA', 'FTA'))} | {qtxt} |"
        )
        if col == 1:
            print(
                "\n| Dataset | Lines | This package (s) | Lines/s | drain3 (s) | drain3 lines/s | Speedup | Published Drain (s, not rerun) |"
            )
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
            secs = data.get("published_drain_seconds", {})
            tot = [0.0, 0.0, 0]
            for ds, r in rust.items():
                q = d3.get(ds)
                tot[0] += r["parse_seconds"]
                tot[2] += r["lines"]
                if q:
                    tot[1] += q["parse_seconds"]
                sp = f"{q['parse_seconds'] / r['parse_seconds']:.1f}x" if q else "n/a"
                print(
                    f"| {ds} | {r['lines']:,} | {r['parse_seconds']:.1f} | {r['lines_per_second']:,.0f} | "
                    f"{q['parse_seconds']:.1f} | {q['lines_per_second']:,.0f} | {sp} | {secs.get(ds, 'n/a')} |"
                    if q
                    else f"| {ds} | {r['lines']:,} | {r['parse_seconds']:.1f} | {r['lines_per_second']:,.0f} | n/a | n/a | n/a | {secs.get(ds, 'n/a')} |"
                )
            print(
                f"| **Total** | {tot[2]:,} | {tot[0]:.1f} | {tot[2] / tot[0]:,.0f} | {tot[1]:.1f} | {tot[2] / tot[1]:,.0f} | {tot[1] / tot[0]:.1f}x | |"
            )


def throughput() -> None:
    data = load("throughput.json")
    if not data:
        return
    print("\n#### Raw-file throughput\n")
    print("| Dataset | Engine | Lines timed | Seconds | Lines per second | Peak RSS (MB) |")
    print("|---|---|---:|---:|---:|---:|")
    for r in data["rows"]:
        rss = f"{r['peak_rss_mb']:.0f}" if "peak_rss_mb" in r else ""
        print(
            f"| {r['dataset']} | {r['engine']} | {r['lines']:,} | {r['seconds']:.2f} | {r['lines_per_second']:,.0f} | {rss} |"
        )


def anomaly() -> None:
    for ds in ("hdfs", "bgl"):
        data = load(f"anomaly_{ds}.json")
        if not data:
            continue
        d = data["dataset"]
        s = data["split"]
        print(
            f"\n#### {d['dataset']}: {d['lines']:,} lines, {d['sequences']:,} sequences, test {s['test']:,} ({s['test_anomalies']:,} anomalous)\n"
        )
        print(
            "| Detector | Precision | Recall | F1, own rule | F1, validation threshold | ROC AUC | Average precision |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for r in data["results"]:

            def pm(k: str, r: dict = r) -> str:
                v = r[k]
                if v is None:
                    return "n/a"
                return f"{v:.3f} ± {r[k + '_std']:.3f}" if r["seeds"] > 1 else f"{v:.3f}"

            print(
                f"| {r['detector']} | {pm('precision')} | {pm('recall')} | {pm('f1')} | {pm('tuned_f1')} | {pm('roc_auc')} | {pm('average_precision')} |"
            )


def changepoints() -> None:
    data = load("changepoint_tcpd.json")
    if data:
        print("\n#### TCPD (38 univariate series, default settings)\n")
        print("| Method | F1 | Cover | Source |")
        print("|---|---:|---:|---|")
        for name, r in data["summary"].items():
            print(f"| {name} | {r['f1']:.3f} | {r['cover']:.3f} | run here |")
        for name, r in data["published_same_series"].items():
            print(
                f"| {name} (TCPDBench) | {r['f1']:.3f} | {r['cover']:.3f} | published, {r['series']} series |"
            )
    data = load("changepoint_regressions.json")
    if data:
        print(f"\n#### Synthetic regressions ({data['series']} series, {data['shifts']} shifts)\n")
        print(
            "| Method | F1 | Precision | Recall | False alarms per series | Recall 3-10% | Recall 10-20% | Recall 20-30% | Seconds |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in data["summary"].items():
            print(
                f"| {name} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['false_alarms_per_series']:.2f} | "
                f"{r['recall small (3-10%)']:.3f} | {r['recall medium (10-20%)']:.3f} | {r['recall large (20-30%)']:.3f} | {r['seconds']:.1f} |"
            )


if __name__ == "__main__":
    parsing()
    throughput()
    anomaly()
    changepoints()
