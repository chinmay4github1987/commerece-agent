"""API request/response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200, examples=["wireless headphones under 10000"])
    max_price: float | None = Field(default=None, gt=0)
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=10)


class ProductHit(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    price: float
    rating: float
    score: float


class SearchResponse(BaseModel):
    products: list[ProductHit]
    source: str
    trace_id: str


class AgentRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["compare noise cancelling headphones under 10k"])
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


RunStatus = Literal["completed", "degraded", "max_iterations", "loop_detected", "ungrounded"]


class AgentRunResponse(BaseModel):
    session_id: str
    trace_id: str
    status: RunStatus
    answer: str
    recommended_product_id: str | None
    steps: list[dict[str, Any]]
    iterations: int
    tokens: int
    latency_ms: float
    spans: list[dict[str, Any]]


class SessionResponse(BaseModel):
    session_id: str
    history: list[dict[str, Any]]
    last_status: str | None
    last_answer: str | None
