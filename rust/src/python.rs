//! PyO3 bindings, built by maturin into the `log_anomaly_rs` extension module.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use regex::Regex;

use crate::drain::{Drain, DrainConfig, Masker};
use crate::pipeline::{Extract, LogFormat, Pipeline, Unmatched};

fn u32_bytes<'py>(py: Python<'py>, values: &[u32]) -> Bound<'py, PyBytes> {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    PyBytes::new(py, &raw)
}

fn i64_bytes<'py>(py: Python<'py>, values: &[i64]) -> Bound<'py, PyBytes> {
    let mut raw = Vec::with_capacity(values.len() * 8);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    PyBytes::new(py, &raw)
}

/// A Drain parser with an optional log format; mirrors the Python `Drain` plus line handling.
#[pyclass(module = "log_anomaly_rs")]
struct RustDrain {
    inner: Pipeline,
}

#[pymethods]
impl RustDrain {
    #[new]
    #[pyo3(signature = (depth=4, similarity=0.5, max_children=100, masks=Vec::new(), log_format=None, skip_unmatched=false))]
    fn new(
        depth: usize,
        similarity: f64,
        max_children: usize,
        masks: Vec<String>,
        log_format: Option<String>,
        skip_unmatched: bool,
    ) -> PyResult<Self> {
        let masker = Masker::new(&masks).map_err(PyValueError::new_err)?;
        let drain = Drain::new(DrainConfig {
            depth,
            similarity,
            max_children,
            masks,
        })
        .map_err(PyValueError::new_err)?;
        let format = match log_format {
            Some(f) => Some(LogFormat::new(&f).map_err(PyValueError::new_err)?),
            None => None,
        };
        let unmatched = if skip_unmatched {
            Unmatched::Skip
        } else {
            Unmatched::WholeLine
        };
        Ok(Self {
            inner: Pipeline {
                drain,
                masker,
                format,
                unmatched,
            },
        })
    }

    /// Mask and tokenize one message content (no format), as the parser sees it.
    fn tokenize(&self, content: &str) -> Vec<String> {
        self.inner.masker.tokenize(content)
    }

    /// Add one message content (already extracted from its line); return its cluster id.
    fn add(&mut self, content: &str) -> u32 {
        let tokens = self.inner.masker.tokenize(content);
        self.inner.drain.add_tokens(tokens)
    }

    /// Add raw lines (stripped, format-matched, masked) in order; little-endian u32 ids.
    fn add_lines<'py>(&mut self, py: Python<'py>, lines: Vec<String>) -> Bound<'py, PyBytes> {
        let ids = py.detach(|| self.inner.parse_lines(&lines));
        u32_bytes(py, &ids)
    }

    /// Stream a file; returns a dict of little-endian arrays and session keys.
    #[pyo3(signature = (path, session_regex=None, label_field=None, normal_label="-".to_string(), time_field=None, max_lines=None, chunk=65536))]
    #[allow(clippy::too_many_arguments)]
    fn parse_file<'py>(
        &mut self,
        py: Python<'py>,
        path: String,
        session_regex: Option<String>,
        label_field: Option<String>,
        normal_label: String,
        time_field: Option<String>,
        max_lines: Option<usize>,
        chunk: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if chunk == 0 {
            return Err(PyValueError::new_err("chunk must be positive"));
        }
        let session = match session_regex {
            Some(s) => Some(Regex::new(&s).map_err(|e| PyValueError::new_err(e.to_string()))?),
            None => None,
        };
        let extract = Extract {
            session,
            label_field,
            normal_label,
            time_field,
        };
        let out = py
            .detach(|| self.inner.parse_file(&path, &extract, max_lines, chunk))
            .map_err(PyRuntimeError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("event_ids", u32_bytes(py, &out.event_ids))?;
        d.set_item("session_lines", u32_bytes(py, &out.session_lines))?;
        d.set_item("session_ids", u32_bytes(py, &out.session_ids))?;
        d.set_item("session_keys", out.session_keys)?;
        d.set_item("labels", PyBytes::new(py, &out.labels))?;
        d.set_item("times", i64_bytes(py, &out.times))?;
        d.set_item("unmatched", out.unmatched)?;
        Ok(d)
    }

    #[getter]
    fn num_clusters(&self) -> usize {
        self.inner.drain.num_clusters()
    }

    fn template(&self, cid: usize) -> PyResult<String> {
        if cid >= self.inner.drain.num_clusters() {
            return Err(PyValueError::new_err("unknown cluster id"));
        }
        Ok(self.inner.drain.template(cid))
    }

    fn templates(&self) -> Vec<String> {
        self.inner.drain.templates()
    }

    fn sizes(&self) -> Vec<u64> {
        self.inner.drain.sizes()
    }
}

#[pymodule]
fn log_anomaly_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustDrain>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
