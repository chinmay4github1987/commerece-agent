"""Runtime configuration, read from environment variables.

Settings are rebuilt on each call to get_settings() so tests (and k8s ConfigMaps)
can change behaviour through the environment without module reloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    app_name: str
    version: str
    api_key: str | None
    checkpoint_db: str         # SQLite file, used when checkpoint_url is not set
    checkpoint_url: str | None # postgresql://... -> shared checkpoints for multi-replica deploys
    llm_provider: str          # "rules" (deterministic, offline) | "openai"
    openai_model: str
    max_iterations: int        # hard cap on plan->act cycles per request
    tool_timeout_s: float      # per-attempt timeout for a tool call
    tool_max_retries: int      # retries for transient tool failures
    inventory_failure_rate: float  # fault injection for the inventory upstream
    inventory_latency_s: float     # simulated upstream latency


def get_settings() -> Settings:
    return Settings(
        app_name="commerce-ops-agent",
        version=os.getenv("APP_VERSION", "0.1.0"),
        api_key=os.getenv("API_KEY") or None,
        checkpoint_db=os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite"),
        checkpoint_url=os.getenv("CHECKPOINT_URL") or None,
        llm_provider=os.getenv("LLM_PROVIDER", "rules"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_iterations=_int("AGENT_MAX_ITERATIONS", "6"),
        tool_timeout_s=_float("TOOL_TIMEOUT_S", "2.0"),
        tool_max_retries=_int("TOOL_MAX_RETRIES", "2"),
        inventory_failure_rate=_float("INVENTORY_FAILURE_RATE", "0"),
        inventory_latency_s=_float("INVENTORY_LATENCY_S", "0"),
    )
