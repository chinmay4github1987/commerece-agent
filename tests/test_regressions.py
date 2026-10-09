"""Regression tests for bugs found in this project. One test per incident;
see README "Debugging incident" for the write-up."""
from __future__ import annotations

from app import tools
from app.tools import SearchProductsInput


def test_incident_001_cheapest_product_reaches_candidates(service):
    # Before the fix, ties in search were always broken by rating, the agent only
    # inventory-checks the top 3, so the cheapest product (boAt, ₹1,499) was cut
    # before comparison and the agent recommended a ₹8,990 pair as "cheapest".
    r = service.run("cheapest bluetooth headphones")
    assert r["steps"][0]["args"]["sort"] == "price_asc"
    assert r["recommended_product_id"] == "AUD-002"


def test_incident_001_search_price_sort(settings):
    out = tools.search_products(
        SearchProductsInput(query="bluetooth headphones", category="audio", sort="price_asc"), settings)
    prices = [p["price"] for p in out["products"]]
    assert prices == sorted(prices)


def test_incident_001_relevance_still_beats_price(settings):
    # price_asc must only reorder within a relevance tier, never surface less relevant items first
    out = tools.search_products(SearchProductsInput(query="noise cancelling earbuds", sort="price_asc"), settings)
    assert out["products"][0]["id"] == "AUD-004"
