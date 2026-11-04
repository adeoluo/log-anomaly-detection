# ADR-0003: Follow the LogPAI Drain and the Loghub-2.0 evaluator, with one fix

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Published Drain numbers (Zhu et al. 2019; Jiang et al. 2024) come from the LogPAI implementation
with per-dataset settings and from the Loghub-2.0 evaluation scripts. Small differences, such as
grouping by template string or by cluster id, or how messages with an empty ground truth template
are counted, move the metrics by several points, which would make any comparison meaningless.

## Decision

* Use the published per-dataset settings (log format, masking regexes, depth, similarity) and the
  same parse-tree semantics as the LogPAI Drain.
* Compute GA, FGA, PA and FTA exactly as the Loghub-2.0 evaluator does: groups keyed by template
  string, empty ground truth rows left out of the grouping metrics, PA as exact string equality.
  The metric code was checked against the reference evaluator on all 14 Loghub-2k datasets and
  agrees to every printed digit; `tests/test_loghub.py` pins those values.
* Fix one defect: in the LogPAI code a message with fewer tokens than the tree depth is never
  stored in a leaf, so every repeat of a short message starts a new cluster. Here such a message
  is stored at the node where its tokens run out (drain3 also handles this case).

## Consequences

* On Loghub-2k the templates are identical to the LogPAI code's on all 14 datasets. On
  Loghub-2.0, 12 of 14 datasets match the published numbers to three decimals. Linux and Spark
  differ slightly (Linux FGA 0.786 against 0.778, Spark FGA 0.870 against 0.861). Rerunning
  the LogPAI code on Linux reproduces the published figures, so that difference comes from this
  fix; Spark was not rerun with the LogPAI code (it needs more memory than the test machine has
  for 16 million lines), and drain3 agrees with this package there.
* Accuracy numbers can be compared directly with the published tables, and the comparison with
  the drain3 library uses the same masks, depth and threshold.
* The parser inherits Drain's known weaknesses (for example low PA, because masked tokens and
  punctuation differ from the annotated templates); they are reported, not tuned away.
