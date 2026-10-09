"""Unit tests: each tool and the executor in isolation (no agent)."""
from __future__ import annotations

import time

import pytest

from app import tools
from app.agent.planner import parse_category, parse_max_price
from app.config import get_settings
from app.executor import execute_tool
from app.observability import Tracer
from app.tools import (
    CheckInventoryInput,
    CompareProductsInput,
    PermanentToolError,
    SearchProductsInput,
    ToolPermissionError,
    ToolTimeoutError,
    TransientToolError,
)


def test_search_ranks_relevant_products(settings):
    out = tools.search_products(SearchProductsInput(query="wireless headphones"), settings)
    ids = [p["id"] for p in out["products"]]
    assert ids and all(i.startswith("AUD") for i in ids)


def test_search_respects_price_and_category(settings):
    out = tools.search_products(
        SearchProductsInput(query="laptop", max_price=60000, category="laptops"), settings)
    assert out["products"]
    assert all(p["price"] <= 60000 and p["category"] == "laptops" for p in out["products"])


def test_search_no_match_returns_empty(settings):
    out = tools.search_products(SearchProductsInput(query="submarine"), settings)
    assert out["products"] == []


def test_inventory_reports_stock_and_unknown(settings):
    out = tools.check_inventory(CheckInventoryInput(product_ids=["AUD-001", "AUD-003", "NOPE"]), settings)
    assert out["items"]["AUD-001"]["in_stock"] is True
    assert out["items"]["AUD-003"]["in_stock"] is False
    assert out["items"]["NOPE"]["found"] is False


def test_compare_picks_value_and_rating(settings):
    out = tools.compare_products(CompareProductsInput(product_ids=["AUD-001", "AUD-005"]), settings)
    assert out["top_rated"] == "AUD-005"
    assert out["best_value"] == "AUD-001"
    assert out["cheapest"] == "AUD-001"


def test_compare_unknown_id_is_permanent_error(settings):
    with pytest.raises(PermanentToolError):
        tools.compare_products(CompareProductsInput(product_ids=["AUD-001", "BAD-1"]), settings)


@pytest.mark.parametrize("query,expected", [
    ("headphones under 10000", 10000),
    ("laptop below ₹55,000", 55000),
    ("phone under 20k", 20000),
    ("best laptop", None),
])
def test_parse_max_price(query, expected):
    assert parse_max_price(query) == expected


def test_parse_category():
    assert parse_category("cheap earbuds") == "audio"
    assert parse_category("espresso maker") == "kitchen"
    assert parse_category("something nice") is None


# ---------- executor ----------

def test_executor_rejects_invalid_args(settings):
    with pytest.raises(PermanentToolError, match="invalid arguments"):
        execute_tool("compare_products", {"product_ids": ["AUD-001"]}, settings=settings, tracer=Tracer())


def test_executor_rejects_unknown_tool(settings):
    with pytest.raises(PermanentToolError, match="unknown tool"):
        execute_tool("delete_database", {}, settings=settings, tracer=Tracer())


def test_executor_requires_approval_for_write_tool(settings):
    with pytest.raises(ToolPermissionError):
        execute_tool("reserve_stock", {"product_id": "AUD-002", "quantity": 1},
                     settings=settings, tracer=Tracer())


def test_executor_retries_transient_failures(monkeypatch, settings):
    calls = {"n": 0}

    def flaky(inp, s):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError("503")
        return {"items": {}}

    monkeypatch.setitem(tools.TOOLS, "check_inventory",
                        tools.ToolSpec("check_inventory", "", tools.CheckInventoryInput, flaky))
    tracer = Tracer()
    out = execute_tool("check_inventory", {"product_ids": ["AUD-001"]}, settings=settings, tracer=tracer)
    assert out == {"items": {}}
    assert calls["n"] == 3
    assert [s["status"] for s in tracer.spans] == ["error", "error", "ok"]


def test_executor_gives_up_after_bounded_retries(monkeypatch):
    monkeypatch.setenv("INVENTORY_FAILURE_RATE", "1.0")
    s = get_settings()
    tracer = Tracer()
    with pytest.raises(TransientToolError):
        execute_tool("check_inventory", {"product_ids": ["AUD-001"]}, settings=s, tracer=tracer)
    assert len(tracer.spans) == s.tool_max_retries + 1


def test_executor_times_out_slow_tool(monkeypatch):
    monkeypatch.setenv("INVENTORY_LATENCY_S", "0.3")
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.05")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "0")
    s = get_settings()
    start = time.perf_counter()
    with pytest.raises(ToolTimeoutError):
        execute_tool("check_inventory", {"product_ids": ["AUD-001"]}, settings=s, tracer=Tracer())
    assert time.perf_counter() - start < 0.25  # returned at the timeout, not after the slow call


def test_executor_wraps_unexpected_tool_bug(monkeypatch, settings):
    def buggy(inp, s):
        raise KeyError("oops")

    monkeypatch.setitem(tools.TOOLS, "search_products",
                        tools.ToolSpec("search_products", "", tools.SearchProductsInput, buggy))
    with pytest.raises(PermanentToolError, match="KeyError"):
        execute_tool("search_products", {"query": "x"}, settings=settings, tracer=Tracer())
