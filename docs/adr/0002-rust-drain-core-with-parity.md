# ADR-0002: A Rust Drain core that reproduces the Python reference exactly

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

The HDFS log has 11.2 million lines and Loghub-2.0 has 50 million annotated lines in total. A
Drain parser in Python parses roughly 100,000 to 150,000 lines per second, so one pass over the
larger logs takes minutes, and every anomaly-detection experiment starts with a full parse. Most
of the time goes into header splitting and masking regexes, not the tree itself.

## Decision

Implement Drain in Rust (`rust/src/drain.rs`), compiled into the package with maturin and used by
default, and keep the Python implementation (`src/log_anomaly_detection/drain.py`) as the
reference:

* Both use the same whitespace set (Python's `str.isspace`, which adds U+001C to U+001F to
  Unicode White_Space), the same ASCII digit rule, the same tree routing and `max_children`
  overflow, the same similarity tie-break (more parameters wins) and the same template update.
* File parsing (`rust/src/pipeline.rs`) reads chunks of 65,536 lines; header extraction, masking
  and tokenizing run in parallel with rayon, then messages enter the tree one at a time in input
  order. The tree update is the only sequential part, so the result is identical to a sequential
  run.
* Session keys, labels and timestamps are extracted in the same pass and returned as packed
  arrays, so Python never holds millions of line strings.

## Consequences

* `tests/test_rust_parity.py` checks identical cluster ids and templates on all 14 Loghub-2k
  datasets, on random lines with odd whitespace, and on file parsing with sessions, labels and
  times, in CI with the extension required.
* The packaged wheel needs a Rust toolchain to build from source.
* The parser remains single-tree and sequential at its core; sharding the tree across threads
  would change results, so it is not done.
