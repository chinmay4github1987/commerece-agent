"""Integration tests through the HTTP layer."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["planner"] == "rules"


def test_search_endpoint(client):
    r = client.post("/v1/search", json={"query": "laptop", "max_price": 60000})
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] and all(p["price"] <= 60000 for p in body["products"])


def test_search_validation_error(client):
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "limit": 50}).status_code == 422


def test_agent_run_and_session(client):
    r = client.post("/v1/agent/run", json={"query": "best laptop", "session_id": "api-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["recommended_product_id"] == "LAP-003"
    assert any(s["name"] == "tool.check_inventory" for s in body["spans"])
    assert r.headers["X-Request-ID"]

    s = client.get("/v1/agent/sessions/api-1")
    assert s.status_code == 200 and s.json()["history"][0]["query"] == "best laptop"
    assert client.get("/v1/agent/sessions/missing").status_code == 404


def test_session_id_is_validated(client):
    r = client.post("/v1/agent/run", json={"query": "x", "session_id": "../../etc"})
    assert r.status_code == 422


def test_metrics_exposed(client):
    client.post("/v1/agent/run", json={"query": "coffee machine"})
    text = client.get("/metrics").text
    assert 'agent_runs_total{status="completed"}' in text
    assert "agent_run_latency_ms_count" in text


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    from app.main import app
    with TestClient(app) as c:
        assert c.post("/v1/search", json={"query": "laptop"}).status_code == 401
        assert c.post("/v1/search", json={"query": "laptop"},
                      headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/search", json={"query": "laptop"},
                      headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200  # probes stay unauthenticated
