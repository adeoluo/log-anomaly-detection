"""HTTP service (standard library): parse lines, ingest a stream, score sessions, find regressions.

Routes:

* ``GET /health``: liveness; ``GET /ready``: whether a detector is trained;
  ``GET /metrics``: Prometheus text; ``GET /templates``: the templates learned so far.
* ``POST /parse`` ``{"lines": [...]}``: event id and template per line.
* ``POST /ingest`` ``{"lines": [...], "flush": false}``: feed the streaming monitor; returns
  alerts raised by these lines.
* ``POST /score`` ``{"sessions": [[line, ...], ...]}``: anomaly score per independent session.
* ``POST /regress`` ``{"values": [...], "method": "cusum", "direction": "up"}``: level shifts.

The monitor is single-writer: requests that touch the parser are serialized with a lock.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from log_anomaly_detection import __version__
from log_anomaly_detection.changepoint import detect_regressions
from log_anomaly_detection.observability import Metrics
from log_anomaly_detection.parser import UNMATCHED
from log_anomaly_detection.streaming import StreamMonitor

MAX_LINES = 100_000


class HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _strings(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise HTTPError(400, f"'{key}' must be a list of strings")
    if len(value) > MAX_LINES:
        raise HTTPError(413, f"at most {MAX_LINES} lines per request")
    return value


class App:
    """Request handling, independent of the socket layer (easy to test)."""

    def __init__(self, monitor: StreamMonitor, metrics: Metrics | None = None) -> None:
        self.monitor = monitor
        self.metrics = metrics or Metrics()
        self.lock = threading.Lock()

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, str]:
        """Return (status, content type, body)."""

        start = time.perf_counter()
        route = path.split("?", 1)[0]
        try:
            status, ctype, out = self._dispatch(method, route, body)
        except HTTPError as exc:
            status, ctype, out = exc.status, "application/json", json.dumps({"error": str(exc)})
        except (ValueError, RuntimeError) as exc:
            status, ctype, out = 400, "application/json", json.dumps({"error": str(exc)})
        known = route if route in self.ROUTES else "other"
        self.metrics.observe_request(known, status, time.perf_counter() - start)
        return status, ctype, out

    ROUTES = (
        "/health",
        "/ready",
        "/metrics",
        "/templates",
        "/parse",
        "/ingest",
        "/score",
        "/regress",
    )

    def _json(self, body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise HTTPError(400, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise HTTPError(400, "the request body must be a JSON object")
        return payload

    def _dispatch(self, method: str, route: str, body: bytes) -> tuple[int, str, str]:
        ok = "application/json"
        if method == "GET":
            if route == "/health":
                return 200, ok, json.dumps({"status": "ok", "version": __version__})
            if route == "/ready":
                ready = self.monitor.ready
                return (200 if ready else 503), ok, json.dumps({"ready": ready})
            if route == "/metrics":
                self.metrics.set_gauge("templates", self.monitor.parser.num_clusters)
                self.metrics.set_gauge("open_sessions", self.monitor.open_sessions)
                for name, value in self.monitor.stats.items():
                    self.metrics.set_gauge(f"monitor_{name}", value)
                return 200, "text/plain; version=0.0.4", self.metrics.render()
            if route == "/templates":
                with self.lock:
                    sizes = self.monitor.parser.sizes()
                    templates = self.monitor.parser.templates()
                rows = [
                    {"id": i, "count": n, "template": t}
                    for i, (n, t) in enumerate(zip(sizes, templates, strict=True))
                ]
                return 200, ok, json.dumps({"templates": rows})
            raise HTTPError(404, "not found")
        if method != "POST":
            raise HTTPError(405, "method not allowed")
        payload = self._json(body)
        if route == "/parse":
            lines = _strings(payload, "lines")
            with self.lock:
                ids = self.monitor.parser.add_lines(lines).tolist()
                out = [
                    {"event_id": None, "template": None}
                    if e == int(UNMATCHED)
                    else {"event_id": e, "template": self.monitor.parser.template(e)}
                    for e in ids
                ]
            self.metrics.inc("lines_parsed", len(lines))
            return 200, ok, json.dumps({"results": out})
        if route == "/ingest":
            lines = _strings(payload, "lines")
            if not self.monitor.ready:
                raise HTTPError(503, "no detector is trained; start the service with --train")
            with self.lock:
                alerts = [a.as_dict() for a in self.monitor.process_many(lines)]
                if payload.get("flush"):
                    alerts += [a.as_dict() for a in self.monitor.flush()]
            self.metrics.inc("lines_ingested", len(lines))
            self.metrics.inc("alerts", len(alerts))
            return 200, ok, json.dumps({"alerts": alerts})
        if route == "/score":
            sessions = payload.get("sessions")
            if not isinstance(sessions, list) or not all(
                isinstance(s, list) and all(isinstance(x, str) for x in s) for s in sessions
            ):
                raise HTTPError(400, "'sessions' must be a list of lists of strings")
            if not self.monitor.ready:
                raise HTTPError(503, "no detector is trained; start the service with --train")
            with self.lock:
                results = self.monitor.score_sessions(sessions)
            self.metrics.inc("sessions_scored", len(results))
            return 200, ok, json.dumps({"results": results})
        if route == "/regress":
            values = payload.get("values")
            if not isinstance(values, list) or len(values) < 3:
                raise HTTPError(400, "'values' must be a list of at least 3 numbers")
            try:
                series = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise HTTPError(400, "'values' must be numbers") from exc
            found = detect_regressions(
                series,
                method=str(payload.get("method", "cusum")),
                direction=str(payload.get("direction", "up")),
                min_relative=float(payload.get("min_relative", 0.05)),
            )
            return 200, ok, json.dumps({"regressions": [r.as_dict() for r in found]})
        raise HTTPError(404, "not found")


def make_handler(app: App, max_body: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"log-anomaly-detection/{__version__}"

        def _send(self, status: int, ctype: str, body: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._send(*app.handle("GET", self.path, b""))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                self._send(413, "application/json", json.dumps({"error": "body too large"}))
                return
            self._send(*app.handle("POST", self.path, self.rfile.read(length)))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(app: App, host: str, port: int, max_body: int) -> None:  # pragma: no cover - blocking
    server = ThreadingHTTPServer((host, port), make_handler(app, max_body))
    print(f"serving on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
