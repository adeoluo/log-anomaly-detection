"""Drain, the fixed-depth parse-tree log parser (He et al., ICWS 2017): reference implementation.

Each message is masked (every masking regex match becomes ``<*>``), split on whitespace and routed
through a tree: first by token count, then by its first ``depth - 3`` tokens (a token containing a
digit routes to the ``<*>`` child, and a node with ``max_children`` children routes new tokens to
``<*>``). At the leaf, the message joins the most similar cluster if the fraction of matching
non-wildcard positions reaches ``similarity``; positions that differ become ``<*>`` in the
template. Otherwise it starts a new cluster.

This module follows the LogPAI implementation used for the published benchmark numbers, with one
documented fix (ADR-0003): a message shorter than the tree depth is stored at the node where its
tokens run out, so repeated short messages share a cluster instead of each starting a new one.

The Rust core (``rust/src/drain.rs``) reproduces this implementation exactly and is the default;
``tests/test_rust_parity.py`` checks that both assign every line to the same cluster with the same
template.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

WILDCARD = "<*>"


def split_tokens(text: str) -> list[str]:
    """Split ``text`` on runs of whitespace, as ``str.split()`` does.

    ``str.isspace()`` covers Unicode White_Space plus the ASCII separators U+001C to U+001F; the
    Rust core uses the same set, so both implementations tokenize any input identically.
    """

    return text.split()


def has_digit(token: str) -> bool:
    """Return whether ``token`` contains an ASCII digit."""

    return any("0" <= ch <= "9" for ch in token)


@dataclass(frozen=True)
class DrainConfig:
    """Drain hyperparameters.

    ``depth`` counts every tree layer as in the LogPAI implementation (root, length layer, token
    layers, leaf), so a message is routed by its first ``depth - 3`` tokens: ``depth=4`` routes
    on the first token and ``depth=3`` on the token count alone.
    """

    depth: int = 4
    similarity: float = 0.5
    max_children: int = 100
    masks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.depth < 3:
            raise ValueError("depth must be at least 3")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must be in [0, 1]")
        if self.max_children < 2:
            raise ValueError("max_children must be at least 2")
        for pattern in self.masks:
            re.compile(pattern)


@dataclass
class _Node:
    children: dict[str, _Node] = field(default_factory=dict)
    clusters: list[int] = field(default_factory=list)


class Drain:
    """Online Drain parser: add messages one at a time, read clusters and templates."""

    def __init__(self, config: DrainConfig | None = None) -> None:
        self.config = config or DrainConfig()
        self._masks = [re.compile(p) for p in self.config.masks]
        self._levels = self.config.depth - 2
        self._root: dict[int, _Node] = {}
        self._templates: list[list[str]] = []
        self._sizes: list[int] = []

    # ----------------------------------------------------------------- preprocessing
    def mask(self, content: str) -> str:
        """Replace every masking-regex match with ``<*>``, applying the regexes in order."""

        for regex in self._masks:
            content = regex.sub(WILDCARD, content)
        return content

    def tokenize(self, content: str) -> list[str]:
        """Mask and split ``content`` into tokens."""

        return split_tokens(self.mask(content))

    # ----------------------------------------------------------------- tree search
    def _search(self, tokens: Sequence[str]) -> int | None:
        node = self._root.get(len(tokens))
        if node is None:
            return None
        depth = 1
        for token in tokens:
            if depth >= self._levels:
                break
            child = node.children.get(token)
            if child is None:
                child = node.children.get(WILDCARD)
                if child is None:
                    return None
            node = child
            depth += 1
        return self._fast_match(node.clusters, tokens)

    def _fast_match(self, candidates: Sequence[int], tokens: Sequence[str]) -> int | None:
        best: int | None = None
        best_sim = -1.0
        best_params = -1
        length = len(tokens)
        for cid in candidates:
            template = self._templates[cid]
            same = 0
            params = 0
            for left, right in zip(template, tokens, strict=True):
                if left == WILDCARD:
                    params += 1
                elif left == right:
                    same += 1
            sim = same / length if length else 1.0
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = cid, sim, params
        if best is not None and best_sim >= self.config.similarity:
            return best
        return None

    def _insert(self, cid: int) -> None:
        template = self._templates[cid]
        length = len(template)
        node = self._root.get(length)
        if node is None:
            node = _Node()
            self._root[length] = node
        depth = 1
        limit = self.config.max_children
        for token in template:
            if depth >= self._levels:
                break
            child = node.children.get(token)
            if child is None:
                wild = node.children.get(WILDCARD)
                if not has_digit(token):
                    if wild is not None:
                        if len(node.children) < limit:
                            child = _Node()
                            node.children[token] = child
                        else:
                            child = wild
                    elif len(node.children) + 1 < limit:
                        child = _Node()
                        node.children[token] = child
                    elif len(node.children) + 1 == limit:
                        child = _Node()
                        node.children[WILDCARD] = child
                    else:  # pragma: no cover - unreachable: the wildcard child exists by then
                        raise AssertionError("tree node is full without a wildcard child")
                elif wild is None:
                    child = _Node()
                    node.children[WILDCARD] = child
                else:
                    child = wild
            node = child
            depth += 1
        node.clusters.append(cid)

    # ----------------------------------------------------------------- public API
    def add_tokens(self, tokens: list[str]) -> int:
        """Add one tokenized message and return its cluster id."""

        cid = self._search(tokens)
        if cid is None:
            cid = len(self._templates)
            self._templates.append(list(tokens))
            self._sizes.append(1)
            self._insert(cid)
            return cid
        template = self._templates[cid]
        for i, (left, right) in enumerate(zip(template, tokens, strict=True)):
            if left != right:
                template[i] = WILDCARD
        self._sizes[cid] += 1
        return cid

    def add(self, content: str) -> int:
        """Mask, tokenize and add one message content; return its cluster id."""

        return self.add_tokens(self.tokenize(content))

    def add_many(self, contents: Iterable[str]) -> list[int]:
        """Add many message contents in order; return their cluster ids."""

        return [self.add(content) for content in contents]

    @property
    def num_clusters(self) -> int:
        """The number of clusters (templates) found so far."""

        return len(self._templates)

    def template(self, cid: int) -> str:
        """The current template of cluster ``cid``, tokens joined by single spaces."""

        return " ".join(self._templates[cid])

    def templates(self) -> list[str]:
        """Every cluster's current template, indexed by cluster id."""

        return [" ".join(t) for t in self._templates]

    def sizes(self) -> list[int]:
        """The number of messages in each cluster, indexed by cluster id."""

        return list(self._sizes)
