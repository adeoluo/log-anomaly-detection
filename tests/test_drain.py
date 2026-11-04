"""Unit and property tests for the Drain parser (both backends through LogParser)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from log_anomaly_detection.drain import WILDCARD, Drain, DrainConfig, has_digit, split_tokens
from log_anomaly_detection.parser import LogParser


def test_groups_similar_messages_and_generalizes() -> None:
    drain = Drain(DrainConfig(masks=(r"blk_-?\d+",)))
    a = drain.add("Receiving block blk_1 src: /10.0.0.1")
    b = drain.add("Receiving block blk_-2 src: /10.0.0.2")
    c = drain.add("Deleting block blk_3 file /tmp/x")
    assert a == b != c
    assert drain.template(a) == "Receiving block <*> src: <*>"
    assert drain.sizes() == [2, 1]
    assert drain.num_clusters == 2
    assert drain.templates()[c] == "Deleting block <*> file /tmp/x"


def test_similarity_threshold_splits_dissimilar_messages() -> None:
    strict = Drain(DrainConfig(similarity=0.9))
    loose = Drain(DrainConfig(similarity=0.3))
    msgs = ["job alpha finished ok now", "job beta failed badly now"]
    assert len(set(strict.add_many(msgs))) == 2
    assert len(set(loose.add_many(msgs))) == 1


def test_short_messages_share_a_cluster() -> None:
    drain = Drain(DrainConfig(depth=6))
    assert drain.add("restart") == drain.add("restart")
    assert drain.add("") == drain.add("   ")
    assert drain.num_clusters == 2


def test_depth_three_routes_on_length_only() -> None:
    drain = Drain(DrainConfig(depth=3, similarity=0.5))
    assert drain.add("x1 opened file") == drain.add("y2 opened file")


def test_max_children_overflow_routes_to_wildcard() -> None:
    drain = Drain(DrainConfig(similarity=0.9, max_children=3))
    for word in ["alpha", "beta", "gamma", "delta", "x1", "y2"]:
        drain.add(f"{word} done")
    assert drain.num_clusters == 6
    assert drain.add("alpha done") == 0


def test_ties_prefer_the_template_with_more_parameters() -> None:
    drain = Drain(DrainConfig(similarity=0.5))
    first = drain.add("a b c d")
    drain.add("a b x y")  # generalizes cluster 0 to "a b <*> <*>"
    assert drain.template(first) == "a b <*> <*>"
    assert drain.add("a b z w") == first


@pytest.mark.parametrize(
    "config",
    [
        {"depth": 2},
        {"similarity": 1.5},
        {"similarity": -0.1},
        {"max_children": 1},
        {"masks": ("(",)},
    ],
)
def test_invalid_config(config: dict[str, object]) -> None:
    with pytest.raises(Exception):  # noqa: B017 - ValueError or re.error
        DrainConfig(**config)  # type: ignore[arg-type]


def test_tokenizer_helpers() -> None:
    assert split_tokens(" a\x1cb\u00a0c\td ") == ["a", "b", "c", "d"]
    assert has_digit("abc1") and not has_digit("abc") and not has_digit("\u00b2")


def test_masking_applies_in_order() -> None:
    drain = Drain(DrainConfig(masks=(r"\d+\.\d+\.\d+\.\d+", r"\d+")))
    assert drain.tokenize("from 10.0.0.1 port 22") == ["from", WILDCARD, "port", WILDCARD]


def test_parser_facade(backend: str) -> None:
    parser = LogParser(log_format="<Level> <Content>", masks=[r"\d+"], backend=backend)
    ids = parser.add_lines(["INFO open 1", "INFO open 2", "WARN close"])
    assert ids.tolist() == [0, 0, 1]
    assert parser.add("open 3") == 0
    assert parser.template(0) == "open <*>"
    assert parser.sizes() == [3, 1]
    assert parser.num_clusters == 2
    assert parser.tokenize("open 77") == ["open", "<*>"]
    assert parser.add_lines([]).tolist() == []
    with pytest.raises(ValueError):
        parser.template(9)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        LogParser(backend="java")


WORDS = st.sampled_from(["open", "close", "blk_7", "user", "x9", "id=3", "<*>", "ok", "é", "42"])
MESSAGES = st.lists(st.lists(WORDS, max_size=7).map(" ".join), min_size=1, max_size=60)


@settings(max_examples=150, deadline=None)
@given(MESSAGES, st.integers(3, 6), st.sampled_from([0.2, 0.5, 0.8, 1.0]))
def test_templates_cover_their_messages(messages: list[str], depth: int, sim: float) -> None:
    drain = Drain(DrainConfig(depth=depth, similarity=sim, max_children=4))
    ids = drain.add_many(messages)
    assert sum(drain.sizes()) == len(messages)
    for message, cid in zip(messages, ids, strict=True):
        tokens = drain.tokenize(message)
        template = drain.template(cid).split(" ") if tokens else []
        assert len(template) == len(tokens)
        for t, m in zip(template, tokens, strict=True):
            assert t in (WILDCARD, m)


@settings(max_examples=60, deadline=None)
@given(MESSAGES)
def test_parsing_is_deterministic(messages: list[str]) -> None:
    one, two = Drain(), Drain()
    assert one.add_many(messages) == two.add_many(messages)
    assert one.templates() == two.templates()
