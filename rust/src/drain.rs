//! Drain: the fixed-depth parse-tree log parser, identical to the Python reference in
//! `src/log_anomaly_detection/drain.py`.

use regex::{NoExpand, Regex};
use rustc_hash::FxHashMap;

pub const WILDCARD: &str = "<*>";

/// Python's `str.isspace()`: Unicode White_Space plus the separators U+001C to U+001F.
#[inline]
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()` with the same whitespace set as Python.
#[inline]
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `str.split()` with the same whitespace set as Python.
pub fn split_tokens(s: &str) -> Vec<String> {
    s.split(is_py_space)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

#[inline]
fn has_digit(token: &str) -> bool {
    token.bytes().any(|b| b.is_ascii_digit())
}

/// Drain hyperparameters; see `DrainConfig` in the Python package.
#[derive(Clone, Debug)]
pub struct DrainConfig {
    pub depth: usize,
    pub similarity: f64,
    pub max_children: usize,
    pub masks: Vec<String>,
}

impl Default for DrainConfig {
    fn default() -> Self {
        Self {
            depth: 4,
            similarity: 0.5,
            max_children: 100,
            masks: Vec::new(),
        }
    }
}

/// Compiled masking regexes, applied in order; each match becomes `<*>`.
#[derive(Clone, Debug)]
pub struct Masker {
    regexes: Vec<Regex>,
}

impl Masker {
    pub fn new(patterns: &[String]) -> Result<Self, String> {
        let regexes = patterns
            .iter()
            .map(|p| Regex::new(p).map_err(|e| format!("invalid mask {p:?}: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { regexes })
    }

    pub fn mask(&self, content: &str) -> String {
        let mut out = content.to_owned();
        for re in &self.regexes {
            if re.is_match(&out) {
                out = re.replace_all(&out, NoExpand(WILDCARD)).into_owned();
            }
        }
        out
    }

    /// Mask and split `content` into tokens.
    pub fn tokenize(&self, content: &str) -> Vec<String> {
        split_tokens(&self.mask(content))
    }
}

#[derive(Default, Debug)]
struct Node {
    children: FxHashMap<String, usize>,
    clusters: Vec<u32>,
}

/// The online Drain parser.
#[derive(Debug)]
pub struct Drain {
    pub config: DrainConfig,
    levels: usize,
    nodes: Vec<Node>,
    by_length: FxHashMap<usize, usize>,
    templates: Vec<Vec<String>>,
    sizes: Vec<u64>,
}

impl Drain {
    pub fn new(config: DrainConfig) -> Result<Self, String> {
        if config.depth < 3 {
            return Err("depth must be at least 3".into());
        }
        if !(0.0..=1.0).contains(&config.similarity) {
            return Err("similarity must be in [0, 1]".into());
        }
        if config.max_children < 2 {
            return Err("max_children must be at least 2".into());
        }
        let levels = config.depth - 2;
        Ok(Self {
            config,
            levels,
            nodes: Vec::new(),
            by_length: FxHashMap::default(),
            templates: Vec::new(),
            sizes: Vec::new(),
        })
    }

    fn new_node(&mut self) -> usize {
        self.nodes.push(Node::default());
        self.nodes.len() - 1
    }

    fn search(&self, tokens: &[String]) -> Option<u32> {
        let mut node = *self.by_length.get(&tokens.len())?;
        for token in tokens.iter().take(self.levels - 1) {
            let children = &self.nodes[node].children;
            node = match children.get(token) {
                Some(&c) => c,
                None => *children.get(WILDCARD)?,
            };
        }
        self.fast_match(&self.nodes[node].clusters, tokens)
    }

    fn fast_match(&self, candidates: &[u32], tokens: &[String]) -> Option<u32> {
        let mut best: Option<u32> = None;
        let mut best_sim = -1.0f64;
        let mut best_params: i64 = -1;
        let length = tokens.len();
        for &cid in candidates {
            let template = &self.templates[cid as usize];
            let mut same = 0usize;
            let mut params: i64 = 0;
            for (left, right) in template.iter().zip(tokens) {
                if left == WILDCARD {
                    params += 1;
                } else if left == right {
                    same += 1;
                }
            }
            let sim = if length == 0 {
                1.0
            } else {
                same as f64 / length as f64
            };
            if sim > best_sim || (sim == best_sim && params > best_params) {
                best = Some(cid);
                best_sim = sim;
                best_params = params;
            }
        }
        match best {
            Some(cid) if best_sim >= self.config.similarity => Some(cid),
            _ => None,
        }
    }

    fn insert(&mut self, cid: u32) {
        let length = self.templates[cid as usize].len();
        let mut node = match self.by_length.get(&length) {
            Some(&n) => n,
            None => {
                let n = self.new_node();
                self.by_length.insert(length, n);
                n
            }
        };
        let limit = self.config.max_children;
        for i in 0..length.min(self.levels - 1) {
            let token = self.templates[cid as usize][i].clone();
            let existing = self.nodes[node].children.get(&token).copied();
            let next = match existing {
                Some(c) => c,
                None => {
                    let wild = self.nodes[node].children.get(WILDCARD).copied();
                    let count = self.nodes[node].children.len();
                    if !has_digit(&token) {
                        match wild {
                            Some(w) => {
                                if count < limit {
                                    let c = self.new_node();
                                    self.nodes[node].children.insert(token, c);
                                    c
                                } else {
                                    w
                                }
                            }
                            None => {
                                let c = self.new_node();
                                if count + 1 < limit {
                                    self.nodes[node].children.insert(token, c);
                                } else {
                                    self.nodes[node].children.insert(WILDCARD.to_owned(), c);
                                }
                                c
                            }
                        }
                    } else {
                        match wild {
                            Some(w) => w,
                            None => {
                                let c = self.new_node();
                                self.nodes[node].children.insert(WILDCARD.to_owned(), c);
                                c
                            }
                        }
                    }
                }
            };
            node = next;
        }
        self.nodes[node].clusters.push(cid);
    }

    /// Add one tokenized message and return its cluster id.
    pub fn add_tokens(&mut self, tokens: Vec<String>) -> u32 {
        match self.search(&tokens) {
            None => {
                let cid = self.templates.len() as u32;
                self.templates.push(tokens);
                self.sizes.push(1);
                self.insert(cid);
                cid
            }
            Some(cid) => {
                let template = &mut self.templates[cid as usize];
                for (left, right) in template.iter_mut().zip(tokens.iter()) {
                    if left != right && left != WILDCARD {
                        *left = WILDCARD.to_owned();
                    }
                }
                self.sizes[cid as usize] += 1;
                cid
            }
        }
    }

    pub fn num_clusters(&self) -> usize {
        self.templates.len()
    }

    pub fn template(&self, cid: usize) -> String {
        self.templates[cid].join(" ")
    }

    pub fn templates(&self) -> Vec<String> {
        self.templates.iter().map(|t| t.join(" ")).collect()
    }

    pub fn sizes(&self) -> Vec<u64> {
        self.sizes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(depth: usize, st: f64, masks: &[&str]) -> (Drain, Masker) {
        let masks: Vec<String> = masks.iter().map(|s| s.to_string()).collect();
        let cfg = DrainConfig {
            depth,
            similarity: st,
            max_children: 100,
            masks: masks.clone(),
        };
        (Drain::new(cfg).unwrap(), Masker::new(&masks).unwrap())
    }

    #[test]
    fn groups_messages_and_generalizes_templates() {
        let (mut d, m) = drain(4, 0.5, &[r"blk_-?\d+"]);
        let a = d.add_tokens(m.tokenize("Receiving block blk_1 src: /10.0.0.1"));
        let b = d.add_tokens(m.tokenize("Receiving block blk_-2 src: /10.0.0.2"));
        let c = d.add_tokens(m.tokenize("Deleting block blk_3 file /tmp/x"));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(d.template(a as usize), "Receiving block <*> src: <*>");
        assert_eq!(d.sizes(), vec![2, 1]);
    }

    #[test]
    fn short_messages_share_a_cluster() {
        let (mut d, m) = drain(6, 0.5, &[]);
        let a = d.add_tokens(m.tokenize("restart"));
        let b = d.add_tokens(m.tokenize("restart"));
        let e1 = d.add_tokens(m.tokenize(""));
        let e2 = d.add_tokens(m.tokenize("   "));
        assert_eq!(a, b);
        assert_eq!(e1, e2);
        assert_eq!(d.num_clusters(), 2);
    }

    #[test]
    fn digit_tokens_route_to_wildcard_and_max_children_overflow() {
        let cfg = DrainConfig {
            depth: 4,
            similarity: 0.9,
            max_children: 3,
            masks: vec![],
        };
        let mut d = Drain::new(cfg).unwrap();
        for word in ["alpha", "beta", "gamma", "delta", "x1", "y2"] {
            d.add_tokens(split_tokens(&format!("{word} done")));
        }
        assert_eq!(d.num_clusters(), 6);
        assert_eq!(d.add_tokens(split_tokens("alpha done")), 0);
    }

    #[test]
    fn whitespace_matches_python() {
        assert_eq!(split_tokens("a\u{1c}b\u{a0}c\td"), vec!["a", "b", "c", "d"]);
        assert_eq!(py_strip("\u{1f} x \u{3000}"), "x");
    }

    #[test]
    fn rejects_bad_config() {
        let bad = DrainConfig {
            depth: 2,
            ..DrainConfig::default()
        };
        assert!(Drain::new(bad).is_err());
        let bad = DrainConfig {
            similarity: 1.5,
            ..DrainConfig::default()
        };
        assert!(Drain::new(bad).is_err());
        let bad = DrainConfig {
            max_children: 1,
            ..DrainConfig::default()
        };
        assert!(Drain::new(bad).is_err());
        assert!(Masker::new(&["(".to_string()]).is_err());
    }
}
