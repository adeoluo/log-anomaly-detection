# Data Card - Log Anomaly Detection

## Sources and provenance

| Dataset | Source | Size used | Labels |
|---|---|---|---|
| HDFS_v1 (Loghub) | Zenodo 8196385; Xu et al., SOSP 2009 | 11,175,629 lines (1.58 GB), 575,061 blocks | 16,838 anomalous blocks (`anomaly_label.csv`) |
| BGL (Loghub) | Zenodo 8196385; Oliner and Stearley, DSN 2007 | 4,747,963 lines (743 MB) | per line (348,460 lines not `-`) |
| Loghub-2.0, 14 datasets | Zenodo 8275861; Jiang et al., ISSTA 2024 | 50.4 million annotated lines (965 MB of zips, streamed) | template per line |
| Loghub-2k, 14 datasets | github.com/logpai/loghub-2.0 `2k_dataset` | 28,000 lines | template per line |
| TCPD | github.com/alan-turing-institute/TCPD; van den Burg and Williams, 2020 | 38 univariate series | change points from 5 annotators |

The bundled test fixtures (`tests/fixtures/loghub_2k/`) are gzipped Loghub-2k logs with only the
`LineId` and `EventTemplate` ground truth columns, about 590 KB.

## Licensing

Loghub and Loghub-2.0 are free for research and academic work, provided the Loghub repository is
referenced and the Loghub paper cited; the license notice ships with the fixtures. TCPD is MIT
licensed (series keep their original licenses). Nothing except the 2k fixtures is redistributed;
`scripts/download_data.sh` fetches the rest.

## Composition

Raw text logs from distributed systems (HDFS, Hadoop, Spark, Zookeeper, OpenStack),
supercomputers (BGL, HPC, Thunderbird), operating systems (Linux, Mac), server software (Apache,
OpenSSH) and standalone software (HealthApp, Proxifier). TCPD holds economic, environmental and
industrial time series.

## Preprocessing

Lines are stripped and split with the published LogPAI log formats; lines that do not match are
skipped for parsing evaluation (none did on the 2k data). Variables are masked with the published
regexes. HDFS sessions are block ids; BGL is cut into 5-minute windows. TCPD missing values are
linearly interpolated.

## Contamination checks

Anomaly detectors train on the first 80 percent of sequences by time and are tested on the rest;
no test labels set any threshold. Parsing is unsupervised; ground truth templates are only read
by the scorer.

## Known biases and gaps

HDFS anomaly labels are per block and come from the original 2009 study's heuristics; some
labels are noisy. BGL labels are alert categories assigned by administrators. The logs are 10 to
20 years old; modern structured logs differ. Loghub-2.0 templates were annotated with heuristic
rules plus manual review.

## Ethical considerations

The logs are public research datasets released by their owners; they contain machine
identifiers and internal IP addresses but no personal data beyond user names on some systems.
