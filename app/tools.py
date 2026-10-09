"""Agent tools: typed inputs, pure functions, one registry.

Each tool:
- declares a Pydantic input schema (validated before execution, and used to
  describe the tool to an LLM planner),
- raises TransientToolError for failures worth retrying and
  PermanentToolError for failures that are not,
- is independently unit-testable without the agent.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.data import INVENTORY, PRODUCTS, PRODUCTS_BY_ID


# ---------- errors ----------

class ToolError(Exception):
    retryable = False


class TransientToolError(ToolError):
    retryable = True


class ToolTimeoutError(TransientToolError):
    pass


class PermanentToolError(ToolError):
    pass


class ToolPermissionError(PermanentToolError):
    pass


# ---------- search_products ----------

STOPWORDS = {
    "a", "an", "the", "for", "with", "and", "or", "to", "of", "in", "on", "me", "my", "i",
    "under", "below", "less", "than", "budget", "rs", "inr", "best", "good", "show", "find",
    "want", "need", "buy", "is", "are", "which", "that", "some", "any", "please", "compare",
    "vs", "versus", "recommend", "available", "stock", "top", "rated", "cheap", "cheapest",
    "cheaper", "lowest", "price", "highest", "k",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9']*", text.lower())
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # naive singularisation: headphones -> headphone
        out.add(w)
    return out


class SearchProductsInput(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="Natural-language product query")
    max_price: float | None = Field(default=None, gt=0, description="Maximum price in INR")
    category: str | None = Field(default=None, description="audio | laptops | phones | footwear | kitchen")
    limit: int = Field(default=5, ge=1, le=10)
    sort: Literal["relevance", "price_asc", "rating_desc"] = Field(
        default="relevance",
        description="Tie-break order among equally relevant results. Use price_asc for "
                    "'cheap/cheapest' requests and rating_desc for 'best/top rated'.",
    )


_SORT_KEYS = {
    # Relevance first in every mode; the mode decides the order within a relevance tier.
    "relevance": lambda sp: (-sp[0], -sp[1].rating, sp[1].price),
    "price_asc": lambda sp: (-sp[0], sp[1].price, -sp[1].rating),
    "rating_desc": lambda sp: (-sp[0], -sp[1].rating, sp[1].price),
}


def search_products(inp: SearchProductsInput, settings: Settings) -> dict[str, Any]:
    q_tokens = tokenize(inp.query)
    scored = []
    for p in PRODUCTS:
        if inp.category and p.category != inp.category:
            continue
        if inp.max_price is not None and p.price > inp.max_price:
            continue
        doc = tokenize(f"{p.name} {p.brand} {p.category} {' '.join(p.tags)} {p.description}")
        if not q_tokens:
            continue
        score = len(q_tokens & doc) / len(q_tokens)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=_SORT_KEYS[inp.sort])
    return {
        "products": [
            {**p.model_dump(include={"id", "name", "brand", "category", "price", "rating"}),
             "score": round(s, 3)}
            for s, p in scored[: inp.limit]
        ],
        "source": "catalog-v1",
    }


# ---------- check_inventory ----------

class CheckInventoryInput(BaseModel):
    product_ids: list[str] = Field(min_length=1, max_length=10)


def check_inventory(inp: CheckInventoryInput, settings: Settings) -> dict[str, Any]:
    # Simulated upstream behaviour, controlled by config for fault-injection tests.
    if settings.inventory_latency_s:
        time.sleep(settings.inventory_latency_s)
    if settings.inventory_failure_rate and random.random() < settings.inventory_failure_rate:
        raise TransientToolError("inventory service returned 503")
    items = {}
    for pid in inp.product_ids:
        if pid not in INVENTORY:
            items[pid] = {"found": False, "in_stock": False, "quantity": 0, "warehouse": None}
            continue
        qty, wh = INVENTORY[pid]
        items[pid] = {"found": True, "in_stock": qty > 0, "quantity": qty, "warehouse": wh}
    return {"items": items}


# ---------- compare_products ----------

class CompareProductsInput(BaseModel):
    product_ids: list[str] = Field(min_length=2, max_length=4)


def compare_products(inp: CompareProductsInput, settings: Settings) -> dict[str, Any]:
    unknown = [pid for pid in inp.product_ids if pid not in PRODUCTS_BY_ID]
    if unknown:
        raise PermanentToolError(f"unknown product ids: {unknown}")
    rows = []
    for pid in inp.product_ids:
        p = PRODUCTS_BY_ID[pid]
        rows.append({
            "id": p.id, "name": p.name, "price": p.price, "rating": p.rating,
            # rating points per ₹10,000: a simple, explainable value metric
            "value_score": round(p.rating / p.price * 10_000, 3),
        })
    best_value = max(rows, key=lambda r: r["value_score"])["id"]
    top_rated = max(rows, key=lambda r: (r["rating"], -r["price"]))["id"]
    cheapest = min(rows, key=lambda r: r["price"])["id"]
    return {"rows": rows, "best_value": best_value, "top_rated": top_rated, "cheapest": cheapest}


# ---------- reserve_stock (write action, needs human approval) ----------

class ReserveStockInput(BaseModel):
    product_id: str
    quantity: int = Field(ge=1, le=5)


def reserve_stock(inp: ReserveStockInput, settings: Settings) -> dict[str, Any]:
    if inp.product_id not in INVENTORY:
        raise PermanentToolError(f"unknown product id: {inp.product_id}")
    qty, wh = INVENTORY[inp.product_id]
    if qty < inp.quantity:
        raise PermanentToolError("insufficient stock")
    INVENTORY[inp.product_id] = (qty - inp.quantity, wh)
    return {"reserved": inp.quantity, "remaining": qty - inp.quantity}


# ---------- registry ----------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, Settings], dict[str, Any]]
    requires_approval: bool = False


TOOLS: dict[str, ToolSpec] = {
    t.name: t for t in [
        ToolSpec("search_products", "Search the catalog by text, optional max_price and category.",
                 SearchProductsInput, search_products),
        ToolSpec("check_inventory", "Check stock level and warehouse for product ids.",
                 CheckInventoryInput, check_inventory),
        ToolSpec("compare_products", "Compare 2-4 products on price, rating and value.",
                 CompareProductsInput, compare_products),
        ToolSpec("reserve_stock", "Reserve units of a product. Consequential: needs approval.",
                 ReserveStockInput, reserve_stock, requires_approval=True),
    ]
}

# Tools the planner may choose on its own. Write tools are excluded.
AGENT_TOOLS = [n for n, t in TOOLS.items() if not t.requires_approval]
