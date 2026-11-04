"""Accuracy on the bundled Loghub-2k samples, pinned to the LogPAI Drain reference."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from log_anomaly_detection.loghub import find_2k, run_2k, run_full

# GA, FGA, PA, FTA of the LogPAI Drain implementation (loghub-2.0 repository) on these files,
# rerun locally with its own evaluator; this package must reproduce them exactly.
REFERENCE = {
    "Proxifier": (0.526, 0.538, 0.000, 0.000),
    "Linux": (0.690, 0.930, 0.183, 0.435),
    "Apache": (1.000, 1.000, 0.694, 0.500),
    "Zookeeper": (0.967, 0.854, 0.497, 0.354),
    "Hadoop": (0.948, 0.820, 0.269, 0.276),
    "HealthApp": (0.780, 0.351, 0.231, 0.109),
    "OpenStack": (0.733, 0.117, 0.018, 0.018),
    "HPC": (0.887, 0.753, 0.635, 0.366),
    "Mac": (0.786, 0.797, 0.217, 0.199),
    "OpenSSH": (0.789, 0.880, 0.508, 0.440),
    "Spark": (0.920, 0.769, 0.359, 0.369),
    "Thunderbird": (0.955, 0.773, 0.046, 0.248),
    "BGL": (0.963, 0.833, 0.342, 0.211),
    "HDFS": (0.998, 0.867, 0.354, 0.267),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_loghub_2k_matches_reference(name: str, loghub_2k: Path, backend: str) -> None:
    run = run_2k(name, loghub_2k, backend)
    s = run.scores
    got = tuple(float(f"{v:.3f}") for v in (s.ga, s.fga, s.pa, s.fta))
    assert got == REFERENCE[name]
    assert run.unmatched == 0 and s.lines == 2000
    assert run.as_dict()["dataset"] == name and run.lines_per_second > 0


def test_find_2k_layouts(tmp_path: Path) -> None:
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "A_2k.log").write_text("x\n")
    (tmp_path / "A" / "A_2k.log_structured.csv").write_text("EventTemplate\nx\n")
    assert find_2k(tmp_path, "A")[0].name == "A_2k.log"
    with pytest.raises(FileNotFoundError):
        find_2k(tmp_path, "B")


def test_run_full_streams_from_zip(tmp_path: Path, backend: str) -> None:
    log = "\n".join(
        f"[Sun Dec 04 04:47:{i % 60:02d} 2005] [notice] workerEnv.init() ok /etc/x{i}.conf"
        for i in range(50)
    )
    truth = "LineId,EventTemplate\n" + "".join(f"{i},workerEnv.init() ok <*>\n" for i in range(50))
    archive = tmp_path / "Apache.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Apache/Apache_full.log", log + "\n")
        z.writestr("Apache/Apache_full.log_structured.csv", truth)
    run = run_full("Apache", tmp_path, backend)
    assert run.scores.lines == 50 and run.scores.ga == 1.0 and run.scores.pa == 1.0


def test_run_full_errors(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "Linux.zip", "w") as z:
        z.writestr("Linux/readme.txt", "x")
    with pytest.raises(FileNotFoundError):
        run_full("Linux", tmp_path)
    with zipfile.ZipFile(tmp_path / "Apache.zip", "w") as z:
        z.writestr("Apache/Apache_full.log", "[t] [notice] a\n[t] [notice] b\n")
        z.writestr("Apache/Apache_full.log_structured.csv", "LineId,EventTemplate\n1,a\n")
    with pytest.raises(ValueError, match="ground truth"):
        run_full("Apache", tmp_path)
