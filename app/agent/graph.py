"""LangGraph workflow: plan -> act -> plan ... -> finalize.

            ┌──────────┐ call_tool ┌───────┐
  START ──▶ │   plan   │ ────────▶ │  act  │
            └──────────┘ ◀──────── └───────┘
                 │ finish / max_iterations / loop_detected
                 ▼
            ┌──────────┐
            │ finalize │ ──▶ END   (grounding check, status, history)
            └──────────┘

Why LangGraph rather than a plain while-loop: explicit nodes and routing make
the control flow testable, the checkpointer persists state per session
(thread_id) after every node, and runtime context lets us inject per-request
dependencies (planner, tracer) without putting them in persisted state.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agent.planner import Planner
from app.config import Settings
from app.executor import execute_tool
from app.observability import Tracer
from app.tools import AGENT_TOOLS, ToolError

TERMINAL = {"max_iterations", "loop_detected"}


class AgentState(TypedDict, total=False):
    query: str
    steps: list[dict[str, Any]]                    # tool calls for the current run
    iterations: int
    decision: dict[str, Any] | None
    answer: str | None
    recommended_product_id: str | None
    status: str | None
    tokens: int
    history: Annotated[list[dict[str, Any]], operator.add]  # one entry per finished run in this session


@dataclass
class AgentContext:
    settings: Settings
    planner: Planner
    tracer: Tracer


def _signature(tool: str | None, args: dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


def plan(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    iteration = state.get("iterations", 0)
    if iteration >= ctx.settings.max_iterations:
        return {"status": "max_iterations",
                "answer": "I couldn't complete this request within the step limit."}

    with ctx.tracer.span("agent.plan", iteration=iteration, planner=ctx.planner.name) as span:
        decision, tokens = ctx.planner.decide(state["query"], state.get("steps", []))
        span["attrs"].update(action=decision.action, tool=decision.tool, tokens=tokens)

    update: dict[str, Any] = {
        "decision": decision.model_dump(),
        "iterations": iteration + 1,
        "tokens": state.get("tokens", 0) + tokens,
    }

    if decision.action == "call_tool":
        if decision.tool not in AGENT_TOOLS:
            # The planner may not invoke write tools or invent tools. Record it as
            # a failed step so the planner sees the error on its next turn.
            update["decision"] = {**update["decision"], "blocked": True}
        # Loop guard: the same call with the same arguments twice is never progress.
        sig = _signature(decision.tool, decision.args)
        if any(_signature(s["tool"], s["args"]) == sig for s in state.get("steps", [])):
            update["status"] = "loop_detected"
            update["answer"] = "I stopped because the agent repeated the same action without progress."
    return update


def route_after_plan(state: AgentState) -> str:
    if state.get("status") in TERMINAL:
        return "finalize"
    decision = state.get("decision") or {}
    return "act" if decision.get("action") == "call_tool" else "finalize"


def act(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    d = state["decision"] or {}
    step: dict[str, Any] = {"tool": d.get("tool"), "args": d.get("args", {})}
    if d.get("blocked"):
        step.update(ok=False, error=f"tool '{d.get('tool')}' is not available to the agent", retryable=False)
    else:
        try:
            step.update(ok=True, output=execute_tool(step["tool"], step["args"],
                                                     settings=ctx.settings, tracer=ctx.tracer))
        except ToolError as exc:
            step.update(ok=False, error=str(exc), retryable=exc.retryable)
    return {"steps": [*state.get("steps", []), step]}


def _grounded(product_id: str, steps: list[dict[str, Any]]) -> bool:
    """A recommendation is grounded if search returned it and, when inventory
    was checked, it was reported in stock."""
    seen = {p["id"] for s in steps if s["tool"] == "search_products" and s.get("ok")
            for p in s["output"]["products"]}
    if product_id not in seen:
        return False
    for s in steps:
        if s["tool"] == "check_inventory" and s.get("ok"):
            item = s["output"]["items"].get(product_id)
            if item is not None and not item["in_stock"]:
                return False
    return True


def finalize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    steps = state.get("steps", [])
    decision = state.get("decision") or {}
    status = state.get("status")
    answer = state.get("answer") or decision.get("answer") or ""
    rec = decision.get("recommended_product_id") if status not in TERMINAL else None

    with ctx.tracer.span("agent.finalize") as span:
        if rec and not _grounded(rec, steps):
            status = "ungrounded"
            answer = "I couldn't produce a recommendation backed by catalog and stock data."
            rec = None
        if status is None:
            status = "degraded" if any(not s.get("ok") for s in steps) else "completed"
        span["attrs"].update(status=status, recommended=rec)

    return {
        "status": status,
        "answer": answer,
        "recommended_product_id": rec,
        "history": [{"query": state["query"], "answer": answer, "status": status,
                     "recommended_product_id": rec}],
    }


def build_graph(checkpointer=None):
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "finalize": "finalize"})
    g.add_edge("act", "plan")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
