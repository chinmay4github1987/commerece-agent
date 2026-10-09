from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.planner import Decision
from app.agent.service import AgentService
from app.config import get_settings


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("CHECKPOINT_URL", raising=False)
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.5")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "2")


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def service(settings):
    svc = AgentService(settings)
    yield svc
    svc.close()


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


class ScriptedPlanner:
    """Replays a fixed list of decisions; used to force specific agent paths
    (misbehaving LLM, loops, hallucinated ids) deterministically."""

    name = "scripted"

    def __init__(self, decisions: list[Decision], tokens_per_call: int = 0) -> None:
        self._decisions = decisions
        self._tokens = tokens_per_call
        self.calls = 0

    def decide(self, query: str, steps: list[dict[str, Any]]) -> tuple[Decision, int]:
        d = self._decisions[min(self.calls, len(self._decisions) - 1)]
        self.calls += 1
        return d, self._tokens
