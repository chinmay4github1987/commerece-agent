"""AgentService: owns the compiled graph, the checkpointer and per-session locks."""
from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from app.agent.checkpoints import open_checkpoint_store
from app.agent.graph import AgentContext, build_graph
from app.agent.planner import Planner, build_planner
from app.config import Settings
from app.observability import METRICS, Tracer, log_event


class AgentService:
    def __init__(self, settings: Settings, planner: Planner | None = None) -> None:
        self.settings = settings
        self.planner = planner or build_planner(settings.llm_provider, settings.openai_model)
        self._store = open_checkpoint_store(settings)
        self.checkpoint_backend = self._store.backend
        self.graph = build_graph(self._store.saver)
        # One run at a time per session: two concurrent runs on the same
        # thread_id would interleave checkpoints.
        self._session_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._session_locks[session_id]

    def run(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or uuid.uuid4().hex
        tracer = Tracer()
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
        start = time.perf_counter()
        initial = {"query": query, "steps": [], "iterations": 0, "decision": None,
                   "answer": None, "recommended_product_id": None, "status": None, "tokens": 0}

        with self._lock_for(session_id):
            try:
                with tracer.span("agent.run", session_id=session_id):
                    final = self.graph.invoke(
                        initial, config,
                        context=AgentContext(settings=self.settings, planner=self.planner, tracer=tracer),
                    )
                status = final["status"]
            except Exception:
                latency_ms = (time.perf_counter() - start) * 1000
                METRICS.record_run("error", latency_ms, 0)
                log_event("agent_run_failed", level=40, trace_id=tracer.trace_id, session_id=session_id)
                raise

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        METRICS.record_run(status, latency_ms, final.get("tokens", 0))
        log_event("agent_run", trace_id=tracer.trace_id, session_id=session_id, status=status,
                  iterations=final.get("iterations"), tokens=final.get("tokens", 0),
                  latency_ms=latency_ms, planner=self.planner.name)
        return {
            "session_id": session_id,
            "trace_id": tracer.trace_id,
            "status": status,
            "answer": final["answer"],
            "recommended_product_id": final.get("recommended_product_id"),
            "steps": final.get("steps", []),
            "iterations": final.get("iterations", 0),
            "tokens": final.get("tokens", 0),
            "latency_ms": latency_ms,
            "spans": tracer.spans,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        snapshot = self.graph.get_state({"configurable": {"thread_id": session_id}})
        if not snapshot.values:
            return None
        v = snapshot.values
        return {"session_id": session_id, "history": v.get("history", []),
                "last_status": v.get("status"), "last_answer": v.get("answer")}

    def healthy(self) -> bool:
        try:
            self._store.ping()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._store.close()
