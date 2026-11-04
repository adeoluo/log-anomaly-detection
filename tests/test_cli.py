from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from log_anomaly_detection.cli import main


def test_parse_command(loghub_2k: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["parse", str(loghub_2k / "HDFS_2k.log.gz"), "--dataset", "HDFS", "--top", "3"]) == 0
    )
    out = capsys.readouterr().out
    assert "2,000 lines" in out and "templates" in out


def test_parse_command_json_and_custom_format(
    loghub_2k: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "parse",
        str(loghub_2k / "Apache_2k.log.gz"),
        "--format",
        r"\[<Time>\] \[<Level>\] <Content>",
    ]
    assert main([*args, "--mask", r"(\d+\.){3}\d+", "--json", "--max-lines", "100"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lines"] == 100 and payload["top"]


def test_parse_eval_command(
    loghub_2k: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_file = tmp_path / "r.json"
    rc = main(
        [
            "parse-eval",
            "--data-dir",
            str(loghub_2k),
            "--datasets",
            "HDFS",
            "Apache",
            "--output",
            str(out_file),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "average" in text and "HDFS" in text
    assert len(json.loads(out_file.read_text())["runs"]) == 2


def test_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", str(tmp_path / "missing.log")]) == 2
    assert "error:" in capsys.readouterr().err


def test_module_entrypoint(monkeypatch: pytest.MonkeyPatch, loghub_2k: Path) -> None:
    monkeypatch.setattr(
        sys, "argv", ["x", "parse", str(loghub_2k / "HDFS_2k.log.gz"), "--top", "1"]
    )
    with pytest.raises(SystemExit) as info:
        runpy.run_module("log_anomaly_detection", run_name="__main__")
    assert info.value.code == 0


def test_detect_command_on_bgl(
    loghub_2k: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_file = tmp_path / "d.json"
    args = ["detect", "--dataset", "bgl", "--log", str(loghub_2k / "BGL_2k.log.gz")]
    assert (
        main(
            [
                *args,
                "--window-lines",
                "20",
                "--detectors",
                "pca",
                "iforest",
                "--output",
                str(out_file),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "BGL: 2,000 lines" in text and "Isolation forest" in text
    assert len(json.loads(out_file.read_text())["results"]) == 2


def test_detect_hdfs_needs_labels(loghub_2k: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["detect", "--dataset", "hdfs", "--log", str(loghub_2k / "HDFS_2k.log.gz")]) == 2
    assert "--labels" in capsys.readouterr().err


def test_detect_with_neural_detectors(loghub_2k: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    args = ["detect", "--dataset", "bgl", "--log", str(loghub_2k / "BGL_2k.log.gz")]
    extra = ["--window-lines", "20", "--epochs", "1", "--device", "cpu"]
    assert main([*args, *extra, "--detectors", "deeplog", "transformer"]) == 0
    out = capsys.readouterr().out
    assert "DeepLog" in out and "Transformer" in out
