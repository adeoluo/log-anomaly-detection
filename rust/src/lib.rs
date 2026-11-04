//! Rust core of log-anomaly-detection.
//!
//! The Drain parser here reproduces the Python reference in `src/log_anomaly_detection/drain.py`
//! exactly: the same whitespace set, the same tree routing, the same similarity tie-breaking and
//! the same template updates, so both assign every line to the same cluster with the same
//! template (`tests/test_rust_parity.py`). On top of the parser, `pipeline` streams files in
//! chunks: header extraction, masking and tokenizing run in parallel with rayon, and messages then
//! enter the tree in input order, which keeps the result identical to a sequential run.

pub mod drain;
pub mod pipeline;

#[cfg(feature = "python")]
mod python;

pub use drain::{split_tokens, Drain, DrainConfig, Masker, WILDCARD};
pub use pipeline::{Extract, LogFormat, ParsedFile, Pipeline, Unmatched, UNMATCHED};
