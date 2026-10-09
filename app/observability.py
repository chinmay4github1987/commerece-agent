"""Structured logging, per-request tracing and in-process metrics.

- Logs are single-line JSON so they can be shipped to CloudWatch / Loki as-is.
- A Tracer collects spans for one request (plan steps, tool attempts) and is
  returned in the API response, which makes failures debuggable without a
  tracing backend. Swapping in OpenTelemetry later only touches this module.
- Metrics are exposed in Prometheus text format at /metrics.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


logger = logging.getLogger("commerce_agent")


def log_event(msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


class Tracer:
    """Collects spans for a single request."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        span: dict[str, Any] = {"name": name, "attrs": attrs, "status": "ok"}
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span["status"] = "error"
            span["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            with self._lock:
                self.spans.append(span)
            log_event("span", trace_id=self.trace_id, **span)


class Metrics:
    """Thread-safe counters and a latency histogram."""

    BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.runs_total: dict[str, int] = defaultdict(int)       # by status
        self.tool_calls: dict[tuple[str, str], int] = defaultdict(int)  # (tool, outcome)
        self.tokens_total = 0
        self.latency_buckets = [0] * len(self.BUCKETS_MS)
        self.latency_sum_ms = 0.0
        self.latency_count = 0

    def record_run(self, status: str, latency_ms: float, tokens: int) -> None:
        with self._lock:
            self.runs_total[status] += 1
            self.tokens_total += tokens
            self.latency_sum_ms += latency_ms
            self.latency_count += 1
            for i, bound in enumerate(self.BUCKETS_MS):
                if latency_ms <= bound:
                    self.latency_buckets[i] += 1

    def record_tool(self, tool: str, outcome: str) -> None:
        with self._lock:
            self.tool_calls[(tool, outcome)] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = ["# TYPE agent_runs_total counter"]
            for status, n in sorted(self.runs_total.items()):
                lines.append(f'agent_runs_total{{status="{status}"}} {n}')
            lines.append("# TYPE agent_tool_calls_total counter")
            for (tool, outcome), n in sorted(self.tool_calls.items()):
                lines.append(f'agent_tool_calls_total{{tool="{tool}",outcome="{outcome}"}} {n}')
            lines.append("# TYPE agent_llm_tokens_total counter")
            lines.append(f"agent_llm_tokens_total {self.tokens_total}")
            lines.append("# TYPE agent_run_latency_ms histogram")
            for bound, n in zip(self.BUCKETS_MS, self.latency_buckets):
                lines.append(f'agent_run_latency_ms_bucket{{le="{bound}"}} {n}')
            lines.append(f'agent_run_latency_ms_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"agent_run_latency_ms_sum {round(self.latency_sum_ms, 2)}")
            lines.append(f"agent_run_latency_ms_count {self.latency_count}")
            return "\n".join(lines) + "\n"


METRICS = Metrics()
