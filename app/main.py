"""FastAPI entrypoint.

Endpoints
  GET  /health                     liveness probe (process is up)
  GET  /ready                      readiness probe (checkpoint store reachable)
  GET  /metrics                    Prometheus metrics
  POST /v1/search                  direct product search (no agent)
  POST /v1/agent/run               run the agent for a query, optionally in a session
  GET  /v1/agent/sessions/{id}     persisted session history
"""
from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader

from app.agent.service import AgentService
from app.config import get_settings
from app.executor import execute_tool
from app.observability import METRICS, Tracer, configure_logging, log_event
from app.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    SearchRequest,
    SearchResponse,
    SessionResponse,
)
from app.tools import PermanentToolError, ToolError


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.agent = AgentService(settings)
    log_event("startup", version=settings.version, planner=app.state.agent.planner.name)
    yield
    app.state.agent.close()


app = FastAPI(title="AI Commerce Operations Agent", version="0.1.0", lifespan=lifespan)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, key: str | None = Security(_api_key_header)) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:  # auth disabled (local dev)
        return
    if key is None or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_agent(request: Request) -> AgentService:
    return request.app.state.agent


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    if request.url.path not in ("/health", "/ready", "/metrics"):
        log_event("http_request", request_id=request_id, method=request.method,
                  path=request.url.path, status=response.status_code, latency_ms=latency_ms)
    return response


@app.exception_handler(ToolError)
async def tool_error_handler(request: Request, exc: ToolError):
    code = 400 if isinstance(exc, PermanentToolError) else 503
    return JSONResponse(status_code=code, content={"error": type(exc).__name__, "detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log_event("unhandled_error", level=40, path=request.url.path, error=f"{type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"error": "internal_error"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(agent: AgentService = Depends(get_agent)):
    if not agent.healthy():
        raise HTTPException(status_code=503, detail="checkpoint store unavailable")
    return {"status": "ready", "planner": agent.planner.name,
            "checkpoints": agent.checkpoint_backend, "version": agent.settings.version}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return METRICS.render_prometheus()


@app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search(req: SearchRequest, agent: AgentService = Depends(get_agent)):
    tracer = Tracer()
    # Tool code is synchronous; run it off the event loop.
    result = await run_in_threadpool(
        execute_tool, "search_products", req.model_dump(), settings=agent.settings, tracer=tracer
    )
    return SearchResponse(**result, trace_id=tracer.trace_id)


@app.post("/v1/agent/run", response_model=AgentRunResponse, dependencies=[Depends(require_api_key)])
async def run_agent(req: AgentRunRequest, agent: AgentService = Depends(get_agent)):
    # Graph execution + SQLite checkpointing are blocking: keep them off the event loop.
    result = await run_in_threadpool(agent.run, req.query, req.session_id)
    return AgentRunResponse(**result)


@app.get("/v1/agent/sessions/{session_id}", response_model=SessionResponse,
         dependencies=[Depends(require_api_key)])
async def get_session(session_id: str, agent: AgentService = Depends(get_agent)):
    session = await run_in_threadpool(agent.get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionResponse(**session)
