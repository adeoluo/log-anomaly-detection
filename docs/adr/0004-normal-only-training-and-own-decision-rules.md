# ADR-0004: Chronological split, normal-only training, each detector's own decision rule

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Log anomaly detection results are notoriously sensitive to protocol: random splits leak future
templates into training, thresholds tuned on test labels inflate F1, and detectors trained on
data containing anomalies behave differently from those trained on normal data (Le and Zhang,
ICSE 2022). Published numbers mix all of these.

## Decision

One protocol for all six detectors (`log_anomaly_detection.evaluation`):

* sequences (HDFS block sessions, BGL time windows) are ordered by time and split 80/20;
* every detector is fitted on the normal training sequences only (labels are used solely to drop
  anomalies from training, the semi-supervised setting of DeepLog);
* each detector decides with its own rule: the PCA Q-statistic limit at alpha 0.001, any violated
  invariant, a LogCluster distance above 0.3, an isolation-forest score above 0.5, an event
  outside DeepLog's top 9 predictions. No threshold is tuned on test labels;
* precision, recall and F1 use those decisions; ROC AUC and average precision use the raw scores,
  so a detector with a poorly calibrated rule but a good ranking is visible.

## Consequences

* Numbers are lower than many published ones that tune thresholds or split randomly, and the
  README compares with published figures only as quoted references, with this caveat.
* Stochastic detectors run with three seeds and report mean and standard deviation.
