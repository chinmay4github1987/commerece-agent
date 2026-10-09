"""Postgres checkpointer: run only when TEST_POSTGRES_URL is set.

    TEST_POSTGRES_URL=postgresql://agent@localhost:5433/agentdb pytest tests/test_postgres.py
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.agent.service import AgentService
from app.config import get_settings

PG_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="TEST_POSTGRES_URL not set")


def test_session_resumes_on_another_replica(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_URL", PG_URL)
    sid = f"pg-{uuid.uuid4().hex[:8]}"

    replica_a = AgentService(get_settings())
    replica_b = AgentService(get_settings())   # separate pool, like a second pod
    assert replica_a.checkpoint_backend == "postgres" and replica_a.healthy()

    replica_a.run("best laptop", session_id=sid)
    replica_b.run("coffee machine", session_id=sid)

    hist = replica_a.get_session(sid)["history"]
    assert [h["query"] for h in hist] == ["best laptop", "coffee machine"]
    replica_a.close()
    replica_b.close()
