# AGENTS.md

Working notes for coding agents and contributors in **Log Anomaly Detection**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Log parsing and anomaly detection for large clusters: a Drain parser with a Rust core, six detectors from PCA to a transformer, streaming alerts, and change-point detection for metric regressions, measured on Loghub, Loghub-2.0 and TCPD.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `examples/`: runnable examples and sample configuration
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/drain.rs`: Drain: the fixed-depth parse-tree log parser, identical to the Python reference in `src/log_anomaly_detection/drain.py`.
- `rust/src/lib.rs`: Rust core of log-anomaly-detection.
- `rust/src/pipeline.rs`: Log formats and batch or file parsing: header extraction, masking and tokenizing run in parallel chunks, then messages enter the (sequential) Drain tree in input order.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `log_anomaly_rs` extension module.
- `src/log_anomaly_detection/__init__.py`: Log parsing and anomaly detection for large clusters.
- `src/log_anomaly_detection/backend.py`: The Rust backend, compiled into the package on install and used by default.
- `src/log_anomaly_detection/changepoint.py`: Change-point detection for performance-baseline regressions: CUSUM and Bayesian online CPD.
- `src/log_anomaly_detection/cli.py`: Command line interface: ``python -m log_anomaly_detection <command>``.
- `src/log_anomaly_detection/configuration.py`: Typed service settings loaded from the environment (prefix ``LAD_``) or a ``.env`` file.
- `src/log_anomaly_detection/datasets.py`: Anomaly-detection datasets: HDFS sessions and BGL windows, parsed straight from the raw logs.
- `src/log_anomaly_detection/detectors/__init__.py`: Anomaly detectors over event counts (PCA, invariant mining, LogCluster, isolation forest).
- `src/log_anomaly_detection/detectors/base.py`: The detector interface shared by every model.
- `src/log_anomaly_detection/detectors/iforest.py`: Isolation forest (Liu, Ting and Zhou, ICDM 2008), implemented with numpy.
- `src/log_anomaly_detection/detectors/invariants.py`: Invariant mining (Lou et al., USENIX ATC 2010): sparse integer relations among event counts.
- `src/log_anomaly_detection/detectors/logcluster.py`: LogCluster (Lin et al., ICSE 2016): cluster normal sequences, flag rows far from every cluster.
- `src/log_anomaly_detection/detectors/pca.py`: PCA anomaly detection (Xu et al., SOSP 2009) with the squared prediction error statistic.
- `src/log_anomaly_detection/detectors/sequence.py`: Next-event sequence detectors: DeepLog (Du et al., CCS 2017) and a transformer variant.
- `src/log_anomaly_detection/drain.py`: Drain, the fixed-depth parse-tree log parser (He et al., ICWS 2017): reference implementation.
- `src/log_anomaly_detection/evaluation.py`: Anomaly-detection evaluation: chronological split, normal-only training, test scoring.
- `src/log_anomaly_detection/features.py`: Feature extraction: event sequences per session or time window, and event count matrices.
- `src/log_anomaly_detection/formats.py`: Log line formats and the per-dataset parser settings used by the Loghub benchmarks.
- `src/log_anomaly_detection/loghub.py`: Loghub parsing benchmarks: locate datasets, parse them, score them, time them.
- `src/log_anomaly_detection/metrics.py`: Detection metrics: precision, recall, F1, ROC AUC and average precision, vectorized.
- `src/log_anomaly_detection/observability.py`: Thread-safe request and pipeline metrics, exported in the Prometheus text format.
- `src/log_anomaly_detection/parser.py`: High-level log parsing: one API over the Rust core and the Python reference.
- `src/log_anomaly_detection/parsing_metrics.py`: Log parsing accuracy metrics, computed exactly as the Loghub-2.0 benchmark computes them.
- `src/log_anomaly_detection/service.py`: HTTP service (standard library): parse lines, ingest a stream, score sessions, find regressions.
- `src/log_anomaly_detection/streaming.py`: Streaming detection: parse lines as they arrive, close sessions, score them, raise alerts.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs Python parity tests |
| `make parse-eval` | Parsing accuracy on the bundled Loghub-2k samples |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
