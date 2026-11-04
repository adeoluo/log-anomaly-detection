from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from log_anomaly_detection.parsing_metrics import (
    TemplateCodes,
    evaluate_clusters,
    evaluate_codes,
    evaluate_templates,
    read_truth,
)


def test_hand_computed_example() -> None:
    truth = ["A <*>", "A <*>", "B", "B", "C <*>"]
    parsed = ["A <*>", "A <*>", "X", "X", "X"]
    s = evaluate_templates(truth, parsed)
    # Group A is reproduced exactly (2 lines); B and C are merged into X.
    assert s.ga == pytest.approx(2 / 5)
    assert s.parsed_templates == 2 and s.truth_templates == 3
    assert s.fga == pytest.approx(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3))
    assert s.pa == pytest.approx(2 / 5)
    assert s.fta == pytest.approx(s.fga)
    assert s.as_dict()["lines"] == 5


def test_right_groups_wrong_strings() -> None:
    s = evaluate_templates(["a <*>", "a <*>", "b"], ["a x", "a x", "b"])
    assert s.ga == 1.0 and s.fga == 1.0
    assert s.pa == pytest.approx(1 / 3)
    assert s.fta == pytest.approx(0.5)


def test_empty_truth_rows_are_skipped_for_grouping_but_count_for_pa() -> None:
    s = evaluate_templates(["", "a", "a"], ["", "a", "a"])
    assert s.ga == 1.0 and s.truth_templates == 1
    assert s.pa == pytest.approx(2 / 3)


def test_identical_cluster_templates_merge() -> None:
    truth, strings = np.array([0, 0]), ["x <*>"]
    s = evaluate_clusters(truth, strings, np.array([0, 1]), ["x <*>", "x <*>"])
    assert s.ga == 1.0 and s.parsed_templates == 1


def test_errors() -> None:
    with pytest.raises(ValueError, match="rows"):
        evaluate_codes(np.array([0]), ["a"], np.array([0, 0]), ["a"])
    with pytest.raises(ValueError, match="nothing"):
        evaluate_codes(np.array([], dtype=np.int64), [], np.array([], dtype=np.int64), [])
    with pytest.raises(ValueError, match="no message"):
        evaluate_templates([""], ["x"])


@given(st.lists(st.sampled_from(["a", "b <*>", "c d", "e"]), min_size=1, max_size=50))
def test_perfect_parse_scores_one(truth: list[str]) -> None:
    s = evaluate_templates(truth, truth)
    assert s.ga == s.fga == s.pa == s.fta == 1.0


@given(st.lists(st.integers(0, 4), min_size=1, max_size=60), st.permutations(range(5)))
def test_grouping_ignores_parsed_names(groups: list[int], perm: list[int]) -> None:
    truth = [f"t{g}" for g in groups]
    parsed = [f"p{perm[g]}" for g in groups]
    s = evaluate_templates(truth, parsed)
    assert s.ga == 1.0 and s.fga == 1.0 and s.pa == 0.0 and s.fta == 0.0


def test_read_truth_plain_gzip_and_zip(tmp_path: Path, loghub_2k: Path) -> None:
    gz = loghub_2k / "Apache_2k.log_structured.csv.gz"
    codes, strings = read_truth(gz)
    assert codes.shape == (2000,)
    plain = tmp_path / "x.csv"
    plain.write_text('LineId,EventTemplate\n1,"a, <*>"\n2,b\n3,"a, <*>"\n4\n')
    codes, strings = read_truth(plain)
    assert codes.tolist() == [0, 1, 0, 2] and strings == ["a, <*>", "b", ""]
    archive = tmp_path / "x.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(plain, "X/X_full.log_structured.csv")
    assert read_truth(archive, "X/X_full.log_structured.csv")[1] == strings


def test_template_codes() -> None:
    codes = TemplateCodes()
    assert codes.encode(["a", "b", "a"]).tolist() == [0, 1, 0]
    assert codes.strings == ["a", "b"]
