# Evaluation Report - Log Anomaly Detection

> Every number here comes from a script in `benchmarks/` and its JSON in `benchmarks/results/`.
> Hardware: Apple M1, 8 cores, 8 GB RAM, shared with other work; PyTorch MPS for the neural
> detectors. Data: `scripts/download_data.sh`.

## 1. Log parsing

**Task.** Assign each log line to a template; compare with the Loghub-2.0 annotations.

**Data.** Loghub-2.0: 14 datasets, 50,416,623 annotated lines, streamed from the Zenodo zips.
Loghub-2k: the same 14 datasets, 2,000 lines each.

**Metrics.** GA, FGA, PA, FTA, computed as the Loghub-2.0 evaluator computes them. The
implementation was compared with the reference evaluator on all 14 Loghub-2k datasets and agrees
to every printed digit (pinned in `tests/test_loghub.py`).

**Baselines.** Published Drain (Jiang et al., ISSTA 2024; quoted, not rerun), the drain3 library
(0.9.11, same masks, depth and threshold, run here), and the LogPAI Drain code (rerun on
Loghub-2k and on Loghub-2.0 Linux).

**Results.** Average over 14 datasets:

| | GA | FGA | PA | FTA |
|---|---:|---:|---:|---:|
| Loghub-2.0, this package | 0.843 | 0.555 | 0.468 | 0.282 |
| Loghub-2.0, published Drain | 0.840 | 0.550 | 0.470 | 0.280 |
| Loghub-2.0, drain3 | 0.843 | 0.555 | 0.468 | 0.282 |
| Loghub-2k, this package (= drain3 = LogPAI code rerun) | 0.853 | 0.735 | 0.311 | 0.271 |
| Loghub-2k, published Drain | 0.850 | 0.740 | 0.320 | 0.280 |

Per-dataset tables are in the README. Throughput over the 50.4 million Loghub-2.0 lines, end to
end from the zips: 155.7 s (323,771 lines per second) for this package against 1,703.8 s for
drain3 (10.9x).

**Error analysis.** Drain's weak spots are datasets with highly variable messages: HealthApp
(27,997 parsed groups for 156 true templates, FGA 0.010) and OpenStack (FGA 0.007). PA is low on
most datasets because the parser keeps punctuation attached to tokens and masks whole tokens,
while the annotations mark finer-grained parameters. These are Drain's properties under the
published settings; they are reported, not tuned.

## 2. Anomaly detection

**Task.** Flag anomalous HDFS block sessions and BGL 5-minute windows.

**Data.** HDFS_v1: 11,175,629 lines, 575,061 blocks, 16,838 anomalous. BGL: 4,747,963 lines,
348,460 anomalous lines, 14,495 non-empty windows, 1,054 anomalous.

**Protocol.** Chronological: first 70 percent of sequences to fit (normal only), next 10 percent
validation, last 20 percent test. Own-rule decisions: PCA Q-statistic limit at alpha 0.001, any
violated invariant, LogCluster distance above 0.3, isolation-forest score above 0.5, an event
outside DeepLog's or the transformer's top 9. Validation-threshold decisions: the threshold with
the best F1 on the validation part. Seeds 0, 1, 2 for the stochastic detectors (mean and
standard deviation). Precision and recall columns are for the own rule.

**HDFS: 11,175,629 lines, 575,061 sequences, test 115,012 (1,679 anomalous)**

| Detector | Precision | Recall | F1, own rule | F1, validation threshold | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 0.996 | 0.689 | 0.814 | 0.047 | 0.913 | 0.726 |
| Invariant mining | 0.775 | 1.000 | 0.873 | 0.873 | 0.996 | 0.654 |
| LogCluster | 1.000 | 0.921 | 0.959 | 0.976 | 0.999 | 0.975 |
| Isolation forest | 0.018 ± 0.001 | 0.769 ± 0.031 | 0.036 ± 0.001 | 0.043 ± 0.000 | 0.595 ± 0.001 | 0.018 ± 0.000 |
| DeepLog | 0.720 ± 0.016 | 0.610 ± 0.025 | 0.660 ± 0.021 | 0.386 ± 0.295 | 0.993 ± 0.006 | 0.822 ± 0.037 |
| Transformer | 0.812 ± 0.024 | 0.680 ± 0.129 | 0.735 ± 0.082 | 0.606 ± 0.268 | 0.992 ± 0.006 | 0.832 ± 0.012 |

**BGL: 4,747,963 lines, 14,495 sequences, test 2,899 (225 anomalous)**

| Detector | Precision | Recall | F1, own rule | F1, validation threshold | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 0.385 | 0.462 | 0.420 | 0.218 | 0.820 | 0.303 |
| Invariant mining | 0.094 | 1.000 | 0.171 | 0.346 | 0.761 | 0.489 |
| LogCluster | 0.119 | 0.969 | 0.212 | 0.212 | 0.603 | 0.088 |
| Isolation forest | 0.111 ± 0.157 | 0.002 ± 0.002 | 0.003 ± 0.004 | 0.172 ± 0.027 | 0.690 ± 0.009 | 0.237 ± 0.016 |
| DeepLog | 0.092 ± 0.000 | 1.000 ± 0.000 | 0.168 ± 0.000 | 0.236 ± 0.016 | 0.797 ± 0.065 | 0.344 ± 0.142 |
| Transformer | 0.093 ± 0.001 | 1.000 ± 0.000 | 0.170 ± 0.001 | 0.218 ± 0.015 | 0.788 ± 0.059 | 0.316 ± 0.115 |

**Findings.**

* HDFS: LogCluster (F1 0.959) and invariant mining (0.873) are strongest with their own rules;
  PCA is precise (0.996) but misses 31 percent. DeepLog and the transformer rank almost perfectly
  (ROC AUC 0.99) but the top-9 rule from the DeepLog paper is loose for a vocabulary of 48
  templates plus the end token, and misses about a third of the failed blocks.
* Thresholds tuned on the validation period transfer badly on HDFS. The validation period has
  3.6 percent anomalous sessions against 1.5 percent in the test period, and of different kinds:
  PCA's tuned threshold (0.063) sits far below its Q-statistic limit and floods the test set with
  alerts, and the neural detectors' tuned thresholds vary strongly by seed.
* BGL: every detector suffers from distribution shift. 65 percent of the test windows (62
  percent of the normal ones) contain a template never seen in the normal training windows, so
  DeepLog and the transformer flag every test window (recall 1.0, precision 0.09). This is the effect Le and Zhang (ICSE 2022) describe for chronological splits;
  random splits, which many published BGL results use, hide it.
* Isolation forest on event counts is weak on both datasets. It agrees with scikit-learn's
  IsolationForest on HDFS (ROC AUC 0.596 against 0.596 and 0.619 for two seeds), so this is the
  method under normal-only training rather than an implementation fault.

**Published references (not rerun, different protocols).** HDFS F1: PCA 0.79, invariant mining
0.91, LogCluster 0.80 (He et al., ISSRE 2016), DeepLog 0.96 (Du et al., CCS 2017).

## 3. Change-point detection

**TCPD** (38 univariate series, default settings, F1 with margin 5 and covering as defined by the
benchmark): BOCPD 0.692 F1 and 0.624 cover against the published R ocp BOCPD at 0.693 and 0.630
on the same series; CUSUM 0.535 and 0.495; ruptures Pelt 0.708 and 0.665; no change points 0.658
and 0.573 (identical to the published value, which checks the metric code).

**Synthetic regressions** (300 seeded series, 458 shifts of 3 to 30 percent): CUSUM F1 0.736,
BOCPD 0.453 raw and 0.758 after a 3-point median, ruptures Pelt 0.900.

Offline Pelt is the most accurate on both. CUSUM is the online, constant-memory option; BOCPD
needs spike filtering, which the regression API applies by dropping segments shorter than three
points.

## 4. Reproduction

```bash
scripts/download_data.sh
D=../.bench-data/log-anomaly-detection
uv sync --extra deep --group validation
uv run --no-sync python benchmarks/parse_loghub.py --loghub2k $D/loghub2/2k_dataset --loghub2 $D/loghub2_full --drain3 --python
uv run --no-sync python benchmarks/anomaly.py hdfs --log $D/hdfs_v1/HDFS.log --labels $D/hdfs_v1/preprocessed/anomaly_label.csv
uv run --no-sync python benchmarks/anomaly.py bgl --log $D/bgl_v1/BGL.log
uv run --no-sync python benchmarks/changepoint_tcpd.py --tcpd $D/tcpd
uv run --no-sync python benchmarks/changepoint_regressions.py
uv run --no-sync python benchmarks/throughput.py --hdfs $D/hdfs_v1/HDFS.log --bgl $D/bgl_v1/BGL.log
uv run --no-sync python benchmarks/report.py
```
