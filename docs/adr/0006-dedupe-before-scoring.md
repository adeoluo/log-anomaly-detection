# ADR-0006: Score distinct sequences and windows, then expand

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Logs repeat themselves. The 575,061 HDFS sessions contain far fewer distinct event-count vectors
and far fewer distinct (history, next event) windows. Scoring every session, or training a
next-event model on every window, wastes most of the compute.

## Decision

* Count detectors score the distinct rows of the test count matrix once and expand the scores
  through the inverse index (`numpy.unique(..., return_inverse=True)`).
* Invariant mining and LogCluster train on distinct rows with weights, which is exact for both.
* DeepLog and the transformer build distinct sequences, then distinct (history, next) windows
  with counts; training samples windows in proportion to their counts (the same distribution as
  the full data), and scoring ranks each distinct window once.

## Consequences

* Results are identical to scoring every row; only the cost changes.
* Isolation forest training subsamples rows as in the original algorithm, so it does not need
  deduplication.
