"""Controlled tool execution: validation, permissions, timeouts, bounded retries.

Every tool call from the agent goes through execute_tool(). Nothing in the
graph calls a tool function directly, so these guarantees hold for every path.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.observability import METRICS, Tracer
from app.tools import (
    TOOLS,
    PermanentToolError,
    ToolError,
    ToolPermissionError,
    ToolTimeoutError,
    TransientToolError,
)

# Tools run on a worker pool so a hung upstream can't block the request past
# its timeout. Note: Python can't kill a running thread; a timed-out call keeps
# its worker until it returns. Real upstream clients should also set their own
# socket timeouts (documented in README "Known limitations").
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")

BASE_BACKOFF_S = 0.05


def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    tracer: Tracer,
    approved: bool = False,
) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        METRICS.record_tool(name, "unknown_tool")
        raise PermanentToolError(f"unknown tool: {name}")
    if spec.requires_approval and not approved:
        METRICS.record_tool(name, "denied")
        raise ToolPermissionError(f"tool '{name}' requires human approval")

    try:
        inp = spec.input_model.model_validate(args)
    except ValidationError as exc:
        METRICS.record_tool(name, "invalid_args")
        raise PermanentToolError(
            f"invalid arguments for {name}: {exc.errors(include_url=False)}"
        ) from exc

    attempts = settings.tool_max_retries + 1
    last_error: ToolError | None = None
    for attempt in range(1, attempts + 1):
        with tracer.span(f"tool.{name}", attempt=attempt) as span:
            future = _POOL.submit(spec.fn, inp, settings)
            try:
                result = future.result(timeout=settings.tool_timeout_s)
                METRICS.record_tool(name, "ok")
                return result
            except FutureTimeout:
                future.cancel()
                last_error = ToolTimeoutError(f"{name} timed out after {settings.tool_timeout_s}s")
            except TransientToolError as exc:
                last_error = exc
            except ToolError:
                METRICS.record_tool(name, "error")
                raise
            except Exception as exc:  # bug inside a tool: don't retry, don't crash the graph
                METRICS.record_tool(name, "error")
                raise PermanentToolError(f"{name} failed: {type(exc).__name__}: {exc}") from exc
            span["status"] = "error"
            span["error"] = str(last_error)
        if attempt < attempts:
            # exponential backoff with jitter
            time.sleep(BASE_BACKOFF_S * (2 ** (attempt - 1)) * (1 + random.random()))

    METRICS.record_tool(name, "exhausted")
    assert last_error is not None
    raise last_error
