"""Log parsing accuracy metrics, computed exactly as the Loghub-2.0 benchmark computes them.

* **GA** (grouping accuracy, Zhu et al. ICSE-SEIP 2019): the fraction of messages whose ground
  truth group is reproduced exactly, i.e. all messages of a ground truth template land in one
  parsed group that holds nothing else.
* **FGA** (F1 of grouping accuracy, Jiang et al. ISSTA 2024): F1 of the precision and recall of
  correctly grouped templates.
* **PA** (parsing accuracy): the fraction of messages whose parsed template string equals the
  ground truth template string.
* **FTA** (F1 of template accuracy): a parsed group is correct when it holds exactly one ground
  truth template and its template string equals it; F1 over parsed and ground truth groups.

As in the published evaluator, groups are keyed by template *string*, and messages whose ground
truth template is empty are left out of GA, FGA and FTA (but count as misses for PA).

Everything is vectorized over integer template codes, so scoring 16 million lines takes seconds.
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ParsingScores:
    """Accuracy of one parse against ground truth."""

    lines: int
    parsed_templates: int
    truth_templates: int
    ga: float
    fga: float
    pa: float
    fta: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class TemplateCodes:
    """Interns template strings as consecutive integer codes."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.strings: list[str] = []

    def code(self, template: str) -> int:
        found = self.index.get(template)
        if found is None:
            found = len(self.strings)
            self.index[template] = found
            self.strings.append(template)
        return found

    def encode(self, templates: Iterable[str]) -> NDArray[np.int64]:
        return np.fromiter((self.code(t) for t in templates), dtype=np.int64)


def _f1(p: float, r: float) -> float:
    return 0.0 if p == 0 and r == 0 else 2 * p * r / (p + r)


def evaluate_codes(
    truth: NDArray[np.int64],
    truth_strings: Sequence[str],
    parsed: NDArray[np.int64],
    parsed_strings: Sequence[str],
) -> ParsingScores:
    """Score parsed template codes against ground truth codes (one entry per message).

    ``truth_strings[c]`` and ``parsed_strings[c]`` give each code's template string; parsed codes
    must already be deduplicated by string (two clusters with the same template are one group).
    """

    truth = np.asarray(truth, dtype=np.int64)
    parsed = np.asarray(parsed, dtype=np.int64)
    if truth.shape != parsed.shape:
        raise ValueError(f"{truth.shape[0]} ground truth rows but {parsed.shape[0]} parsed rows")
    total = int(truth.shape[0])
    if total == 0:
        raise ValueError("nothing to evaluate")

    # PA: string equality per message, over every message.
    lookup = {s: i for i, s in enumerate(truth_strings)}
    parsed_as_truth = np.array([lookup.get(s, -1) for s in parsed_strings], dtype=np.int64)
    empty_truth = {i for i, s in enumerate(truth_strings) if s == ""}
    pa = float(np.mean(parsed_as_truth[parsed] == truth))
    if empty_truth:
        # An empty ground truth template is missing in the CSV: it never equals a parse.
        pa = float(np.mean((parsed_as_truth[parsed] == truth) & ~np.isin(truth, list(empty_truth))))

    keep = ~np.isin(truth, list(empty_truth)) if empty_truth else np.ones(total, dtype=bool)
    t = truth[keep]
    p = parsed[keep]
    if t.size == 0:
        raise ValueError("no message has a ground truth template")

    # Distinct (truth, parsed) pairs with their message counts.
    width = int(np.max(p)) + 1
    keys, pair_counts = np.unique(t * width + p, return_counts=True)
    pairs = np.stack([keys // width, keys % width], axis=1)
    t_ids, t_sizes = np.unique(t, return_counts=True)
    p_ids, p_sizes = np.unique(p, return_counts=True)
    t_size = dict(zip(t_ids.tolist(), t_sizes.tolist(), strict=True))
    p_size = dict(zip(p_ids.tolist(), p_sizes.tolist(), strict=True))
    parts_per_truth = np.bincount(np.searchsorted(t_ids, pairs[:, 0]), minlength=t_ids.size)
    parts_per_parsed = np.bincount(np.searchsorted(p_ids, pairs[:, 1]), minlength=p_ids.size)

    accurate_lines = 0
    accurate_groups = 0
    correct_templates = 0
    for (tc, pc), count in zip(pairs.tolist(), pair_counts.tolist(), strict=True):
        single_truth_part = parts_per_truth[np.searchsorted(t_ids, tc)] == 1
        single_parsed_part = parts_per_parsed[np.searchsorted(p_ids, pc)] == 1
        if single_truth_part and count == t_size[tc] and p_size[pc] == count:
            accurate_lines += count
            accurate_groups += 1
        if single_parsed_part and parsed_strings[pc] == truth_strings[tc]:
            correct_templates += 1

    n_truth = int(t_ids.size)
    n_parsed = int(p_ids.size)
    ga = accurate_lines / int(t.size)
    fga = _f1(accurate_groups / n_parsed, accurate_groups / n_truth)
    fta = _f1(correct_templates / n_parsed, correct_templates / n_truth)
    return ParsingScores(
        lines=total,
        parsed_templates=n_parsed,
        truth_templates=n_truth,
        ga=ga,
        fga=fga,
        pa=pa,
        fta=fta,
    )


def evaluate_templates(truth: Sequence[str], parsed: Sequence[str]) -> ParsingScores:
    """Score per-message template strings (convenience wrapper around :func:`evaluate_codes`)."""

    t_codes = TemplateCodes()
    p_codes = TemplateCodes()
    return evaluate_codes(
        t_codes.encode(truth), t_codes.strings, p_codes.encode(parsed), p_codes.strings
    )


def evaluate_clusters(
    truth: NDArray[np.int64],
    truth_strings: Sequence[str],
    cluster_ids: NDArray[np.integer],
    cluster_templates: Sequence[str],
) -> ParsingScores:
    """Score Drain cluster ids, merging clusters whose template strings are identical."""

    codes = TemplateCodes()
    merged = np.array([codes.code(s) for s in cluster_templates], dtype=np.int64)
    return evaluate_codes(truth, truth_strings, merged[np.asarray(cluster_ids)], codes.strings)


def _open_text(path: Path, member: str | None) -> io.TextIOBase:
    if member is None:
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
        return open(path, encoding="utf-8", errors="replace", newline="")
    archive = zipfile.ZipFile(path)
    raw = archive.open(member)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")


def read_truth(path: str | Path, member: str | None = None) -> tuple[NDArray[np.int64], list[str]]:
    """Stream the ``EventTemplate`` column of a Loghub structured CSV (plain, gzip, or in a zip).

    Returns one integer code per row and the code-to-template table.
    """

    codes = TemplateCodes()
    values: list[int] = []
    with _open_text(Path(path), member) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column = header.index("EventTemplate")
        for row in reader:
            values.append(codes.code(row[column] if column < len(row) else ""))
    return np.asarray(values, dtype=np.int64), codes.strings
