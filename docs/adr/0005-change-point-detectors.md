# ADR-0005: CUSUM and BOCPD for performance-baseline regressions

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Besides anomalous log sequences, operators need to know when a numeric metric (latency, error
rate, throughput, new-template rate) shifts to a new level, for example after a deployment. The
detector has to work online on streaming metrics and offline on a stored history.

## Decision

* Provide two detectors with online `update` and offline `detect` APIs:
  - two-sided tabular CUSUM with a baseline window and re-baselining after each change: cheap,
    interpretable (thresholds in standard deviations), the standard tool for level shifts;
  - Bayesian online change-point detection (Adams and MacKay, 2007) with a Normal-Gamma model and
    a constant hazard of 1/100, keeping the full run-length posterior; offline change points
    come from walking the most probable run lengths back from the end of the series.
* Evaluate on the Turing Change Point Dataset with the benchmark's own F1 and covering
  definitions and default parameters, and on a seeded synthetic regression benchmark, against
  ruptures (Pelt and binary segmentation) run on the same data.

## Consequences

* Offline Pelt is more accurate on both benchmarks; CUSUM's advantage is that it is online and
  cheap, and BOCPD's is a calibrated change probability. The README reports this as it is.
* Both detectors assume piecewise-constant means; trends and seasonality cause false alarms.
