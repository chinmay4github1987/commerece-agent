"""Planners decide the agent's next action.

The graph only depends on the Planner protocol, so the deterministic
RulePlanner (offline, used by tests and CI evals) and the LLM-backed
OpenAIPlanner are interchangeable. Each returns a Decision: call one tool
with arguments, or finish with an answer.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.data import PRODUCTS_BY_ID
from app.tools import AGENT_TOOLS, TOOLS


class Decision(BaseModel):
    action: Literal["call_tool", "finish"]
    tool: str | None = Field(default=None, description="Tool name when action=call_tool")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    answer: str | None = Field(default=None, description="Final answer when action=finish")
    recommended_product_id: str | None = Field(
        default=None, description="Product id being recommended, if any (must come from tool results)"
    )
    reasoning: str = Field(default="", description="One short sentence on why")


class Planner(Protocol):
    name: str

    def decide(self, query: str, steps: list[dict[str, Any]]) -> tuple[Decision, int]:
        """Return the next decision and the LLM tokens it consumed."""


# ---------- query parsing helpers ----------

_PRICE_RE = re.compile(
    r"(?:under|below|less than|upto|up to|max|within|budget(?: of)?)\s*"
    r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS = {
    "audio": ["headphone", "earbud", "earphone", "audio"],
    "laptops": ["laptop", "macbook", "notebook"],
    "phones": ["phone", "smartphone", "mobile"],
    "footwear": ["shoe", "sneaker", "footwear"],
    "kitchen": ["coffee", "espresso", "kitchen"],
}


def parse_max_price(query: str) -> float | None:
    m = _PRICE_RE.search(query)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    if m.group(2):
        value *= 1000
    return value


def parse_category(query: str) -> str | None:
    q = query.lower()
    for category, words in _CATEGORY_KEYWORDS.items():
        if any(w in q for w in words):
            return category
    return None


_CHEAP_WORDS = ("cheap", "cheapest", "lowest price", "least expensive")
_BEST_WORDS = ("best", "top rated", "premium", "highest rated")


def parse_sort_intent(query: str) -> str:
    """One intent for the whole run: it must shape retrieval as well as the
    final pick, or the right product can be cut before it is ever compared."""
    q = query.lower()
    if any(w in q for w in _CHEAP_WORDS):
        return "price_asc"
    if any(w in q for w in _BEST_WORDS):
        return "rating_desc"
    return "relevance"


def _last_ok(steps: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    for s in reversed(steps):
        if s["tool"] == tool and s.get("ok"):
            return s
    return None


def _failed(steps: list[dict[str, Any]], tool: str) -> bool:
    return any(s["tool"] == tool and not s.get("ok") for s in steps)


def _fmt(pid: str) -> str:
    p = PRODUCTS_BY_ID[pid]
    return f"{p.name} (₹{p.price:,.0f}, rated {p.rating})"


# ---------- deterministic planner ----------

class RulePlanner:
    """Fixed search -> inventory -> (compare) -> recommend policy.

    Deterministic, so evals and regression tests are repeatable in CI without
    an API key. It is also the fallback policy when no LLM is configured.
    """

    name = "rules"

    def decide(self, query: str, steps: list[dict[str, Any]]) -> tuple[Decision, int]:
        q = query.lower()
        intent = parse_sort_intent(q)

        # 1. search
        search = _last_ok(steps, "search_products")
        if search is None:
            if _failed(steps, "search_products"):
                return Decision(action="finish", answer="Product search is unavailable right now. Please try again shortly.",
                                reasoning="search failed"), 0
            return Decision(
                action="call_tool", tool="search_products",
                args={"query": query, "max_price": parse_max_price(q),
                      "category": parse_category(q), "limit": 5, "sort": intent},
                reasoning="find candidate products",
            ), 0

        products = search["output"]["products"]
        if not products:
            return Decision(action="finish", answer="I couldn't find any products matching that request.",
                            reasoning="no search results"), 0
        candidates = [p["id"] for p in products[:3]]

        # 2. inventory
        inventory = _last_ok(steps, "check_inventory")
        if inventory is None:
            if _failed(steps, "check_inventory"):
                top = candidates[0]
                return Decision(
                    action="finish", recommended_product_id=top,
                    answer=f"Best match is {_fmt(top)}, but I couldn't verify stock because the "
                           f"inventory service is unavailable. Please confirm availability before ordering.",
                    reasoning="inventory failed; degrade gracefully",
                ), 0
            return Decision(action="call_tool", tool="check_inventory",
                            args={"product_ids": candidates}, reasoning="verify availability"), 0

        items = inventory["output"]["items"]
        in_stock = [pid for pid in candidates if items.get(pid, {}).get("in_stock")]
        if not in_stock:
            names = ", ".join(PRODUCTS_BY_ID[c].name for c in candidates)
            return Decision(action="finish",
                            answer=f"I found matches ({names}) but none are currently in stock.",
                            reasoning="nothing in stock"), 0

        # 3. compare when there is a real choice to make
        compare = _last_ok(steps, "compare_products")
        if len(in_stock) >= 2 and compare is None and not _failed(steps, "compare_products"):
            return Decision(action="call_tool", tool="compare_products",
                            args={"product_ids": in_stock}, reasoning="compare in-stock options"), 0

        # 4. recommend
        if compare is not None:
            out = compare["output"]
            if intent == "rating_desc":
                pick, why = out["top_rated"], "it is the highest rated in-stock option"
            elif intent == "price_asc":
                pick, why = out["cheapest"], "it is the cheapest in-stock option"
            else:
                pick, why = out["best_value"], "it gives the best rating for the price"
            others = [r["id"] for r in out["rows"] if r["id"] != pick]
            alt = f" Alternatives in stock: {', '.join(_fmt(o) for o in others)}." if others else ""
        else:
            pick, why, alt = in_stock[0], "it is the best in-stock match", ""

        stock = items[pick]
        answer = (f"I recommend {_fmt(pick)} because {why}. "
                  f"{stock['quantity']} units in stock at {stock['warehouse']}.{alt}")
        return Decision(action="finish", answer=answer, recommended_product_id=pick,
                        reasoning="recommendation ready"), 0


# ---------- LLM planner ----------

SYSTEM_PROMPT = """You are a commerce operations agent. Decide the single next action.

Tools (call one at a time, arguments must match the JSON schema):
{tools}

Rules:
- Always search before recommending. Only recommend product ids that appear in tool results.
- Check inventory before recommending; never recommend an out-of-stock product.
- Compare when two or more in-stock options fit the request.
- If a tool failed, do not call it again with the same arguments; finish with an honest answer.
- When finishing, set recommended_product_id to the product you recommend (or null).
"""


class OpenAIPlanner:
    """LLM planner using structured output (function calling) via LangChain."""

    name = "openai"

    def __init__(self, model: str) -> None:
        from langchain_openai import ChatOpenAI  # imported lazily: optional dependency

        tool_docs = "\n".join(
            f"- {n}: {TOOLS[n].description} schema={json.dumps(TOOLS[n].input_model.model_json_schema())}"
            for n in AGENT_TOOLS
        )
        self._system = SYSTEM_PROMPT.format(tools=tool_docs)
        llm = ChatOpenAI(model=model, temperature=0, timeout=20, max_retries=2)
        self._llm = llm.with_structured_output(Decision, method="function_calling", include_raw=True)

    def decide(self, query: str, steps: list[dict[str, Any]]) -> tuple[Decision, int]:
        # Keep the observation compact to control token cost.
        history = json.dumps(steps, default=str)[-6000:]
        messages = [
            ("system", self._system),
            ("human", f"User request: {query}\n\nSteps so far (JSON): {history}"),
        ]
        out = self._llm.invoke(messages)
        raw, parsed = out["raw"], out["parsed"]
        usage = getattr(raw, "usage_metadata", None) or {}
        tokens = int(usage.get("total_tokens", 0))
        if parsed is None:
            return Decision(action="finish", answer="I couldn't plan the next step reliably.",
                            reasoning=f"unparseable LLM output: {out.get('parsing_error')}"), tokens
        return parsed, tokens


def build_planner(provider: str, model: str) -> Planner:
    if provider == "openai":
        return OpenAIPlanner(model)
    if provider == "rules":
        return RulePlanner()
    raise ValueError(f"unknown LLM_PROVIDER: {provider}")
