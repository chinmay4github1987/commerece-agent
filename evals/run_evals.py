"""Agent evaluation harness.

Runs every case in evals/dataset.json through the real agent (same graph,
executor and tools as the API) and scores:

  task_success      status, recommended product and forbidden products all correct
  tool_correctness  tool sequence equals the expected sequence
  grounded          recommended product appeared in search results and was in stock
  latency / tokens  p50 / p95 latency, total tokens

Usage:
  python -m evals.run_evals                     # deterministic rules planner
  LLM_PROVIDER=openai python -m evals.run_evals # evaluate the LLM planner
  python -m evals.run_evals --min-success 0.9   # fail CI below a threshold

Comparing the JSON report before and after a change (prompt, model, tool
schema) is how regressions are caught before deploy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from unittest import mock

from app.agent.graph import _grounded
from app.agent.service import AgentService
from app.config import get_settings

HERE = Path(__file__).parent


def run_case(case: dict) -> dict:
    env = {"CHECKPOINT_DB": os.path.join(tempfile.mkdtemp(), "eval.sqlite"), **case.get("env", {})}
    with mock.patch.dict(os.environ, env):
        svc = AgentService(get_settings())
        try:
            r = svc.run(case["query"])
        finally:
            svc.close()

    tools = [s["tool"] for s in r["steps"]]
    rec = r["recommended_product_id"]
    status_ok = r["status"] == case["expected_status"]
    product_ok = rec == case["expected_product"]
    forbidden_ok = rec not in case.get("forbidden_products", [])
    grounded = rec is None or _grounded(rec, r["steps"])
    return {
        "id": case["id"],
        "query": case["query"],
        "task_success": status_ok and product_ok and forbidden_ok,
        "tool_correctness": tools == case["expected_tools"],
        "grounded": grounded,
        "status": r["status"],
        "recommended": rec,
        "expected": case["expected_product"],
        "tools": tools,
        "latency_ms": r["latency_ms"],
        "tokens": r["tokens"],
        "trace_id": r["trace_id"],
    }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset.json"))
    ap.add_argument("--out", default=str(HERE / "report.json"))
    ap.add_argument("--min-success", type=float, default=0.0)
    args = ap.parse_args()

    cases = json.loads(Path(args.dataset).read_text())
    results = [run_case(c) for c in cases]
    n = len(results)
    lat = [r["latency_ms"] for r in results]
    summary = {
        "planner": get_settings().llm_provider,
        "cases": n,
        "task_success_rate": round(sum(r["task_success"] for r in results) / n, 3),
        "tool_correctness_rate": round(sum(r["tool_correctness"] for r in results) / n, 3),
        "grounded_rate": round(sum(r["grounded"] for r in results) / n, 3),
        "latency_p50_ms": round(statistics.median(lat), 2),
        "latency_p95_ms": round(pct(lat, 0.95), 2),
        "total_tokens": sum(r["tokens"] for r in results),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(f"{'case':<22}{'success':<9}{'tools':<7}{'grounded':<10}{'status':<11}{'got':<9}{'expected':<9}ms")
    for r in results:
        print(f"{r['id']:<22}{str(r['task_success']):<9}{str(r['tool_correctness']):<7}"
              f"{str(r['grounded']):<10}{r['status']:<11}{str(r['recommended']):<9}"
              f"{str(r['expected']):<9}{r['latency_ms']}")
    print("\n" + json.dumps(summary, indent=2))

    if summary["task_success_rate"] < args.min_success:
        print(f"FAIL: task_success_rate {summary['task_success_rate']} < {args.min_success}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
