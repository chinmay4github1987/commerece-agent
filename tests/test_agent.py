"""Agent workflow tests: control flow, guards, grounding, persistence."""
from __future__ import annotations

from app.agent.planner import Decision
from app.agent.service import AgentService
from app.config import get_settings
from tests.conftest import ScriptedPlanner


def tools_called(result):
    return [s["tool"] for s in result["steps"]]


def test_happy_path_search_inventory_compare(service):
    r = service.run("wireless headphones under 10000")
    assert r["status"] == "completed"
    assert tools_called(r) == ["search_products", "check_inventory", "compare_products"]
    assert r["recommended_product_id"] == "AUD-001"


def test_out_of_stock_products_are_never_recommended(service):
    r = service.run("noise cancelling headphones")
    assert r["recommended_product_id"] != "AUD-003"  # JBL is out of stock


def test_nothing_in_stock(service):
    r = service.run("running shoes")
    assert r["status"] == "completed"
    assert r["recommended_product_id"] is None
    assert "none are currently in stock" in r["answer"]


def test_no_results_finishes_after_one_tool(service):
    r = service.run("submarine")
    assert tools_called(r) == ["search_products"]
    assert r["recommended_product_id"] is None


def test_inventory_outage_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("INVENTORY_FAILURE_RATE", "1.0")
    svc = AgentService(get_settings())
    r = svc.run("wireless headphones under 10000")
    assert r["status"] == "degraded"
    assert "couldn't verify stock" in r["answer"]
    # retries happened, but the agent did not call the failing tool again
    assert tools_called(r).count("check_inventory") == 1
    svc.close()


def test_loop_guard_stops_repeated_identical_calls(settings):
    same = Decision(action="call_tool", tool="search_products", args={"query": "headphones"})
    planner = ScriptedPlanner([same, same, same])
    svc = AgentService(settings, planner=planner)
    r = svc.run("headphones")
    assert r["status"] == "loop_detected"
    assert len(r["steps"]) == 1
    svc.close()


def test_iteration_cap(settings, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "3")
    decisions = [Decision(action="call_tool", tool="search_products", args={"query": f"q{i}"})
                 for i in range(10)]
    svc = AgentService(get_settings(), planner=ScriptedPlanner(decisions))
    r = svc.run("anything")
    assert r["status"] == "max_iterations"
    assert r["iterations"] == 3
    svc.close()


def test_planner_cannot_call_write_tool(settings):
    planner = ScriptedPlanner([
        Decision(action="call_tool", tool="reserve_stock", args={"product_id": "AUD-002", "quantity": 1}),
        Decision(action="finish", answer="done"),
    ])
    svc = AgentService(settings, planner=planner)
    r = svc.run("reserve one")
    assert r["steps"][0]["ok"] is False
    assert "not available" in r["steps"][0]["error"]
    assert r["status"] == "degraded"
    svc.close()


def test_hallucinated_recommendation_is_rejected(settings):
    planner = ScriptedPlanner([
        Decision(action="call_tool", tool="search_products", args={"query": "headphones"}),
        Decision(action="finish", answer="Buy the AirPods Max!", recommended_product_id="APL-999"),
    ])
    svc = AgentService(settings, planner=planner)
    r = svc.run("headphones")
    assert r["status"] == "ungrounded"
    assert r["recommended_product_id"] is None
    svc.close()


def test_recommending_out_of_stock_is_rejected(settings):
    planner = ScriptedPlanner([
        Decision(action="call_tool", tool="search_products", args={"query": "noise cancelling headphones"}),
        Decision(action="call_tool", tool="check_inventory", args={"product_ids": ["AUD-003"]}),
        Decision(action="finish", answer="JBL!", recommended_product_id="AUD-003"),
    ])
    svc = AgentService(settings, planner=planner)
    assert svc.run("nc headphones")["status"] == "ungrounded"
    svc.close()


def test_token_usage_is_accumulated(settings):
    planner = ScriptedPlanner([
        Decision(action="call_tool", tool="search_products", args={"query": "laptop"}),
        Decision(action="finish", answer="ok"),
    ], tokens_per_call=120)
    svc = AgentService(settings, planner=planner)
    assert svc.run("laptop")["tokens"] == 240
    svc.close()


def test_session_history_persists_across_service_restarts(settings):
    svc = AgentService(settings)
    svc.run("best laptop", session_id="s-1")
    svc.run("coffee machine", session_id="s-1")
    svc.run("phone", session_id="s-2")
    svc.close()

    restarted = AgentService(settings)  # same SQLite file, new process state
    s1 = restarted.get_session("s-1")
    assert [h["query"] for h in s1["history"]] == ["best laptop", "coffee machine"]
    assert len(restarted.get_session("s-2")["history"]) == 1   # sessions are isolated
    assert restarted.get_session("unknown") is None
    restarted.close()


def test_each_run_starts_with_fresh_steps(service):
    service.run("best laptop", session_id="s")
    r = service.run("coffee machine", session_id="s")
    assert r["steps"][0]["args"]["query"] == "coffee machine"
    assert r["iterations"] <= 4
