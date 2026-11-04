from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from log_anomaly_detection.formats import settings_for
from log_anomaly_detection.observability import Metrics
from log_anomaly_detection.service import App, make_handler
from log_anomaly_detection.streaming import StreamMonitor
from tests.synthetic import hdfs_like_lines

BLK = r"blk_-?\d+"


def trained(detector: str = "pca", **kwargs: object) -> StreamMonitor:
    monitor = StreamMonitor.for_dataset(
        settings_for("HDFS"),
        session_regex=BLK,
        idle_lines=12,
        detector=detector,
        **kwargs,  # type: ignore[arg-type]
    )
    info = monitor.train(hdfs_like_lines(300, set(), seed=1))
    assert info == {"lines": 900, "sessions": 300, "templates": monitor.n_events}
    return monitor


def test_stream_alerts_on_failed_sessions() -> None:
    monitor = trained("invariants")
    bad = {5, 17, 40}
    lines = hdfs_like_lines(60, bad, seed=2)
    alerts = list(monitor.process_many(lines)) + monitor.flush()
    flagged = {a.key for a in alerts if a.kind == "session"}
    assert flagged == {f"blk_{1000 + b}" for b in bad}
    assert monitor.stats["sessions_scored"] == 60 and monitor.open_sessions == 0
    assert monitor.stats["unknown_lines"] == 3  # the failure template is new
    alert = next(a for a in alerts if a.kind == "session")
    assert any(t.endswith(" [new]") for t in alert.templates)
    assert alert.as_dict()["events"] == 3


def test_sessions_close_when_idle() -> None:
    monitor = trained()
    lines = hdfs_like_lines(40, set(), seed=3)
    for line in lines:
        monitor.process(line)
    assert 0 < monitor.stats["sessions_scored"] < 40
    assert monitor.open_sessions > 0
    monitor.flush()
    assert monitor.stats["sessions_scored"] == 40


def test_window_mode_and_rate_alerts() -> None:
    monitor = StreamMonitor.for_dataset(
        settings_for("HDFS"), window_lines=30, rate_window=30, detector="pca"
    )
    monitor.train(hdfs_like_lines(200, set(), seed=4))
    normal = hdfs_like_lines(150, set(), seed=5)
    broken = hdfs_like_lines(150, set(range(150)), seed=6)
    alerts = list(monitor.process_many(normal + broken)) + monitor.flush()
    kinds = {a.kind for a in alerts}
    assert "window" in kinds and "rate" in kinds
    assert all(a.line > 450 for a in alerts if a.kind == "rate")


def test_monitor_errors() -> None:
    with pytest.raises(ValueError):
        StreamMonitor.for_dataset(settings_for("HDFS"), detector="svm")
    with pytest.raises(ValueError):
        StreamMonitor.for_dataset(settings_for("HDFS"), idle_lines=0)
    monitor = StreamMonitor.for_dataset(settings_for("HDFS"), session_regex=BLK)
    assert not monitor.ready and monitor.flush() == []
    with pytest.raises(RuntimeError):
        monitor.process("x")
    with pytest.raises(RuntimeError):
        monitor.score_sessions([["x"]])
    with pytest.raises(ValueError):
        monitor.train(["no block ids here"])


@pytest.fixture
def app() -> App:
    return App(trained(), Metrics())


def call(app: App, method: str, path: str, payload: object | None = None) -> tuple[int, dict]:
    body = b"" if payload is None else json.dumps(payload).encode()
    status, ctype, out = app.handle(method, path, body)
    return status, (json.loads(out) if ctype == "application/json" else {"text": out})


def test_routes(app: App) -> None:
    assert call(app, "GET", "/health")[1]["status"] == "ok"
    assert call(app, "GET", "/ready") == (200, {"ready": True})
    status, out = call(
        app, "POST", "/parse", {"lines": ["081109 203518 143 INFO dfs.X: hello 1", "?"]}
    )
    assert status == 200 and out["results"][0]["template"] == "hello 1"
    status, out = call(
        app, "POST", "/ingest", {"lines": hdfs_like_lines(20, {3}, seed=9), "flush": True}
    )
    assert status == 200 and [a["key"] for a in out["alerts"] if a["kind"] == "session"] == [
        "blk_1003"
    ]
    sessions = [hdfs_like_lines(1, set(), seed=1), hdfs_like_lines(1, {0}, seed=1)]
    status, out = call(app, "POST", "/score", {"sessions": sessions})
    assert [r["anomalous"] for r in out["results"]] == [False, True]
    status, out = call(app, "POST", "/regress", {"values": [10.0] * 40 + [14.0] * 40})
    assert status == 200 and len(out["regressions"]) == 1
    status, out = call(app, "GET", "/templates")
    assert status == 200 and out["templates"][0]["count"] > 0
    status, out = call(app, "GET", "/metrics")
    text = out["text"]
    assert 'lad_requests_total{route="/parse",status="200"} 1' in text
    assert "lad_templates" in text and "lad_alerts_total" in text


@pytest.mark.parametrize(
    ("method", "path", "payload", "status"),
    [
        ("GET", "/nope", None, 404),
        ("PUT", "/parse", None, 405),
        ("POST", "/parse", {"lines": "x"}, 400),
        ("POST", "/parse", [1], 400),
        ("POST", "/score", {"sessions": [[1]]}, 400),
        ("POST", "/regress", {"values": [1]}, 400),
        ("POST", "/regress", {"values": ["a", "b", "c"]}, 400),
        ("POST", "/regress", {"values": [1, 2, 3], "method": "x"}, 400),
        ("POST", "/nope", {}, 404),
    ],
)
def test_bad_requests(app: App, method: str, path: str, payload: object, status: int) -> None:
    assert call(app, method, path, payload)[0] == status


def test_invalid_json_and_untrained(app: App) -> None:
    assert app.handle("POST", "/parse", b"{nope")[0] == 400
    blank = App(StreamMonitor.for_dataset(settings_for("HDFS")))
    assert call(blank, "GET", "/ready")[0] == 503
    assert call(blank, "POST", "/ingest", {"lines": []})[0] == 503
    assert call(blank, "POST", "/score", {"sessions": []})[0] == 503
    assert call(blank, "POST", "/parse", {"lines": ["a b"]})[0] == 200


def test_over_a_real_socket(app: App) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, max_body=2048))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health") as r:
            assert json.loads(r.read())["status"] == "ok"
        req = urllib.request.Request(base + "/parse", data=json.dumps({"lines": ["a"]}).encode())
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
        big = urllib.request.Request(base + "/parse", data=b"x" * 4096)
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(big)
        assert info.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
