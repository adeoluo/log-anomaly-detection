# Model / System Card - Log Anomaly Detection

## Overview

- **What it is:** a log analysis pipeline: a Drain template parser (Rust core), six anomaly
  detectors over parsed event sequences, a streaming monitor with alerts, and change-point
  detectors for metric regressions.
- **Models:** PCA (Q statistic), invariant mining, LogCluster, isolation forest (all numpy), and
  two small next-event networks trained from scratch with PyTorch: DeepLog (2-layer LSTM, hidden
  size 64) and a 2-layer transformer encoder (width 64, 4 heads). No pretrained weights.
- **Version and date:** 0.1.0, November 2026.

## Intended use

- Offline evaluation of log parsers and detectors on the Loghub benchmarks.
- Monitoring a service's logs after training on a period of normal operation; flagging
  regressions in performance metrics.

## Out-of-scope use

- Security intrusion detection against adaptive adversaries.
- Automated remediation without a human reviewing alerts.
- Logs whose normal behavior was never seen in training (expect many alerts).

## Performance

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) and the README results: parsing accuracy on
Loghub-2k and Loghub-2.0 against published Drain and the drain3 library, anomaly detection on
full HDFS and BGL, change-point detection on TCPD and a synthetic regression benchmark.
Performance differs widely between datasets: detectors that are strong on HDFS sessions are weak
on BGL time windows, where new templates appear after the training period.

## Risks and limitations

- Distribution shift: the chronological split exposes it; retrain when the software changes.
- Decision rules are not tuned on labels, so recall and precision trade off differently per
  detector; ROC AUC and average precision are reported alongside.
- Stochastic detectors vary with the seed (three seeds reported).

## Safety mitigations

Alerts carry masked templates and scores, not raw lines. The service binds to localhost by
default and caps request sizes.

## Environmental and cost footprint

All results were produced on one Apple M1 laptop (8 cores, 8 GB); the neural detectors train in
minutes on the M1 GPU through PyTorch MPS.
