//! Log formats and batch or file parsing: header extraction, masking and tokenizing run in
//! parallel chunks, then messages enter the (sequential) Drain tree in input order.

use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

use flate2::read::MultiGzDecoder;

use rayon::prelude::*;
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::drain::{py_strip, Drain, Masker};

/// Marks a line that did not match the log format when unmatched lines are skipped.
pub const UNMATCHED: u32 = u32::MAX;

/// A log format such as `<Date> <Time> <Level>: <Content>`, compiled as in the Python package.
#[derive(Clone, Debug)]
pub struct LogFormat {
    regex: Regex,
    content: usize,
}

impl LogFormat {
    pub fn new(format: &str) -> Result<Self, String> {
        let field = Regex::new(r"(<[^<>]+>)").expect("static regex");
        let spaces = Regex::new(" +").expect("static regex");
        // Python's `\s` also matches U+001C to U+001F; Rust's does not, so spell the class out.
        let ws = r"[\s\x1C-\x1F]+";
        let mut source = String::from("^");
        let mut last = 0;
        let mut names: Vec<String> = Vec::new();
        for m in field.find_iter(format) {
            source.push_str(&spaces.replace_all(&format[last..m.start()], ws));
            let name = &format[m.start() + 1..m.end() - 1];
            let valid = name
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            if !valid {
                return Err(format!("invalid field name {name:?} in log format"));
            }
            if names.iter().any(|n| n == name) {
                return Err(format!("duplicate field {name:?} in log format"));
            }
            names.push(name.to_owned());
            source.push_str(&format!("(?P<{name}>.*?)"));
            last = m.end();
        }
        source.push_str(&spaces.replace_all(&format[last..], ws));
        source.push('$');
        if !names.iter().any(|n| n == "Content") {
            return Err("a log format must contain a <Content> field".into());
        }
        let regex = Regex::new(&source).map_err(|e| format!("invalid log format: {e}"))?;
        let content = group_index(&regex, "Content").expect("Content group exists");
        Ok(Self { regex, content })
    }

    /// The `Content` field of an already stripped line, or None if the line does not match.
    pub fn content<'a>(&self, line: &'a str) -> Option<&'a str> {
        self.regex
            .captures(line)
            .and_then(|c| c.get(self.content))
            .map(|m| m.as_str())
    }

    pub fn group(&self, name: &str) -> Option<usize> {
        group_index(&self.regex, name)
    }

    pub fn regex(&self) -> &Regex {
        &self.regex
    }
}

fn group_index(regex: &Regex, name: &str) -> Option<usize> {
    regex
        .capture_names()
        .position(|n| n.is_some_and(|n| n == name))
}

/// What to do with lines that do not match the log format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unmatched {
    /// Parse the whole stripped line as the content.
    WholeLine,
    /// Leave the line out of the parser and give it the id `UNMATCHED`.
    Skip,
}

/// Optional per-line extraction while parsing a file.
#[derive(Clone, Debug, Default)]
pub struct Extract {
    /// Every distinct match of this regex in the content is a session key (HDFS block ids).
    pub session: Option<Regex>,
    /// Header field whose value is a label; any value other than `normal_label` is anomalous.
    pub label_field: Option<String>,
    pub normal_label: String,
    /// Header field holding an integer timestamp (seconds).
    pub time_field: Option<String>,
}

/// Columnar output of parsing a file.
#[derive(Debug, Default)]
pub struct ParsedFile {
    pub event_ids: Vec<u32>,
    /// `(line, session)` pairs, in line order.
    pub session_lines: Vec<u32>,
    pub session_ids: Vec<u32>,
    pub session_keys: Vec<String>,
    pub labels: Vec<u8>,
    pub times: Vec<i64>,
    pub unmatched: u64,
}

struct Prepared {
    tokens: Option<Vec<String>>,
    sessions: Vec<String>,
    label: u8,
    time: i64,
}

/// Parses lines through a format, a masker and a Drain tree.
pub struct Pipeline {
    pub drain: Drain,
    pub masker: Masker,
    pub format: Option<LogFormat>,
    pub unmatched: Unmatched,
}

impl Pipeline {
    fn prepare(
        &self,
        line: &str,
        extract: &Extract,
        groups: (Option<usize>, Option<usize>),
    ) -> Prepared {
        let line = py_strip(line);
        let mut out = Prepared {
            tokens: None,
            sessions: Vec::new(),
            label: 0,
            time: 0,
        };
        let content = match &self.format {
            None => Some(line),
            Some(fmt) => match fmt.regex().captures(line) {
                Some(caps) => {
                    if let Some(g) = groups.0 {
                        let value = caps.get(g).map_or("", |m| m.as_str());
                        out.label = u8::from(value != extract.normal_label);
                    }
                    if let Some(g) = groups.1 {
                        out.time = caps
                            .get(g)
                            .and_then(|m| m.as_str().trim().parse::<i64>().ok())
                            .unwrap_or(0);
                    }
                    caps.get(fmt.content).map(|m| m.as_str())
                }
                None => match self.unmatched {
                    Unmatched::WholeLine => Some(line),
                    Unmatched::Skip => None,
                },
            },
        };
        if let Some(content) = content {
            if let Some(re) = &extract.session {
                for m in re.find_iter(content) {
                    if !out.sessions.iter().any(|s| s == m.as_str()) {
                        out.sessions.push(m.as_str().to_owned());
                    }
                }
            }
            out.tokens = Some(self.masker.tokenize(content));
        }
        out
    }

    fn groups(&self, extract: &Extract) -> Result<(Option<usize>, Option<usize>), String> {
        let lookup = |name: &Option<String>| -> Result<Option<usize>, String> {
            match (name, &self.format) {
                (None, _) => Ok(None),
                (Some(n), Some(fmt)) => fmt
                    .group(n)
                    .map(Some)
                    .ok_or_else(|| format!("log format has no field {n:?}")),
                (Some(n), None) => Err(format!("field {n:?} needs a log format")),
            }
        };
        Ok((lookup(&extract.label_field)?, lookup(&extract.time_field)?))
    }

    /// Parse a batch of raw lines in order; returns one cluster id per line.
    pub fn parse_lines(&mut self, lines: &[String]) -> Vec<u32> {
        let extract = Extract::default();
        let prepared: Vec<Prepared> = lines
            .par_iter()
            .map(|l| self.prepare(l, &extract, (None, None)))
            .collect();
        prepared
            .into_iter()
            .map(|p| match p.tokens {
                Some(t) => self.drain.add_tokens(t),
                None => UNMATCHED,
            })
            .collect()
    }

    /// Stream a file through the parser in chunks of `chunk` lines.
    pub fn parse_file(
        &mut self,
        path: &str,
        extract: &Extract,
        max_lines: Option<usize>,
        chunk: usize,
    ) -> Result<ParsedFile, String> {
        let groups = self.groups(extract)?;
        let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let source: Box<dyn Read> = if path.ends_with(".gz") {
            Box::new(MultiGzDecoder::new(file))
        } else {
            Box::new(file)
        };
        let mut reader = BufReader::with_capacity(1 << 20, source);
        let mut out = ParsedFile::default();
        let mut keys: FxHashMap<String, u32> = FxHashMap::default();
        let mut buf: Vec<u8> = Vec::new();
        let mut batch: Vec<String> = Vec::with_capacity(chunk);
        let limit = max_lines.unwrap_or(usize::MAX);
        let mut seen = 0usize;
        let mut done = false;
        while !done {
            batch.clear();
            while batch.len() < chunk && seen < limit {
                buf.clear();
                let n = reader
                    .read_until(b'\n', &mut buf)
                    .map_err(|e| format!("read error in {path}: {e}"))?;
                if n == 0 {
                    done = true;
                    break;
                }
                if buf.last() == Some(&b'\n') {
                    buf.pop();
                }
                batch.push(String::from_utf8_lossy(&buf).into_owned());
                seen += 1;
            }
            if seen >= limit {
                done = true;
            }
            if batch.is_empty() {
                break;
            }
            let prepared: Vec<Prepared> = batch
                .par_iter()
                .map(|l| self.prepare(l, extract, groups))
                .collect();
            for p in prepared {
                let line = out.event_ids.len() as u32;
                let id = match p.tokens {
                    Some(t) => self.drain.add_tokens(t),
                    None => {
                        out.unmatched += 1;
                        UNMATCHED
                    }
                };
                out.event_ids.push(id);
                if groups.0.is_some() {
                    out.labels.push(p.label);
                }
                if groups.1.is_some() {
                    out.times.push(p.time);
                }
                for key in p.sessions {
                    let next = keys.len() as u32;
                    let sid = match keys.entry(key) {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(e) => {
                            out.session_keys.push(e.key().clone());
                            *e.insert(next)
                        }
                    };
                    out.session_lines.push(line);
                    out.session_ids.push(sid);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drain::DrainConfig;
    use std::io::Write;

    fn pipeline(format: Option<&str>, unmatched: Unmatched) -> Pipeline {
        let masks = vec![r"blk_-?\d+".to_string()];
        Pipeline {
            drain: Drain::new(DrainConfig {
                masks: masks.clone(),
                ..DrainConfig::default()
            })
            .unwrap(),
            masker: Masker::new(&masks).unwrap(),
            format: format.map(|f| LogFormat::new(f).unwrap()),
            unmatched,
        }
    }

    #[test]
    fn format_extracts_content() {
        let f = LogFormat::new(r"<Date> <Time>  <Level> \[<Comp>\]: <Content>").unwrap();
        assert_eq!(
            f.content("081109 2035 INFO [dfs]: hello world"),
            Some("hello world")
        );
        assert_eq!(f.content("garbage"), None);
        assert!(LogFormat::new("<Date> <Time>").is_err());
        assert!(LogFormat::new("<A> <A> <Content>").is_err());
        assert!(LogFormat::new("<1x> <Content>").is_err());
    }

    #[test]
    fn parse_lines_skips_or_keeps_unmatched() {
        let lines: Vec<String> = ["INFO open blk_1", "junk", "INFO open blk_2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut p = pipeline(Some("<Level> <Content>"), Unmatched::Skip);
        assert_eq!(p.parse_lines(&lines), vec![0, UNMATCHED, 0]);
        let mut p = pipeline(Some("<Level> <Content> x"), Unmatched::Skip);
        assert_eq!(p.parse_lines(&lines), vec![UNMATCHED; 3]);
        let mut p = pipeline(Some("<Level> <Content> x"), Unmatched::WholeLine);
        assert_eq!(p.parse_lines(&lines), vec![0, 1, 0]);
        let mut p = pipeline(None, Unmatched::Skip);
        assert_eq!(p.parse_lines(&lines), vec![0, 1, 0]);
    }

    #[test]
    fn parse_file_extracts_sessions_labels_and_times() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("test-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.log");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "- 100 copy blk_1 to blk_2 and blk_1").unwrap();
        writeln!(f, "FATAL 160 copy blk_2 to blk_3 and blk_3\r").unwrap();
        writeln!(f, "nomatch").unwrap();
        write!(f, "- 200 delete blk_1").unwrap();
        drop(f);
        let mut p = pipeline(Some("<Label> <Timestamp> <Content>"), Unmatched::Skip);
        let extract = Extract {
            session: Some(Regex::new(r"blk_-?\d+").unwrap()),
            label_field: Some("Label".into()),
            normal_label: "-".into(),
            time_field: Some("Timestamp".into()),
        };
        let out = p
            .parse_file(path.to_str().unwrap(), &extract, None, 2)
            .unwrap();
        assert_eq!(out.event_ids, vec![0, 0, UNMATCHED, 1]);
        assert_eq!(out.labels, vec![0, 1, 0, 0]);
        assert_eq!(out.times, vec![100, 160, 0, 200]);
        assert_eq!(out.session_keys, vec!["blk_1", "blk_2", "blk_3"]);
        assert_eq!(out.session_lines, vec![0, 0, 1, 1, 3]);
        assert_eq!(out.session_ids, vec![0, 1, 1, 2, 0]);
        assert_eq!(out.unmatched, 1);
        let head = p
            .parse_file(path.to_str().unwrap(), &extract, Some(1), 2)
            .unwrap();
        assert_eq!(head.event_ids.len(), 1);
        let bad = Extract {
            label_field: Some("Nope".into()),
            ..Extract::default()
        };
        assert!(p.parse_file(path.to_str().unwrap(), &bad, None, 2).is_err());
        assert!(p
            .parse_file("/nonexistent/file.log", &extract, None, 2)
            .is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
