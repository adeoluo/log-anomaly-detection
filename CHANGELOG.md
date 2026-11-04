# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Features

- Drain log parser with a Rust core and Loghub parsing evaluation (#1)
- Event features and PCA, invariant mining, LogCluster and isolation forest detectors (#2)
- DeepLog LSTM and transformer next-event detectors (#3)
- CUSUM and Bayesian online change-point detection for metric regressions (#4)
- Streaming monitor with alerting and an HTTP service (#5)
- Validation-tuned thresholds and an end-of-session token for next-event detectors (#8)

### Bug fixes

- Treat isolated spikes as spikes, not level shifts, in regression detection (#7)

### Performance

- Full-scale benchmarks on Loghub-2.0, HDFS, BGL and TCPD with recorded results (#9)

### Documentation

- ADRs, data, model and threat cards, architecture diagram, data download script (#6)
- State the MIT license in the README (#10)

### Maintenance

- Scaffold project
