"""Concurrent runs across sessions sharing one checkpointer must not corrupt state."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

QUERIES = ["best laptop", "coffee machine", "wireless headphones under 10000", "5g phone under 20k"]


def test_parallel_sessions_are_isolated(service):
    def work(i: int):
        q = QUERIES[i % len(QUERIES)]
        return i, q, service.run(q, session_id=f"c-{i}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(40)))

    for i, q, r in results:
        assert r["status"] == "completed", (i, r["status"])
        assert r["steps"][0]["args"]["query"] == q
        hist = service.get_session(f"c-{i}")["history"]
        assert [h["query"] for h in hist] == [q]


def test_parallel_runs_same_session_are_serialized(service):
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda q: service.run(q, session_id="shared"), QUERIES * 3))
    hist = service.get_session("shared")["history"]
    assert len(hist) == len(QUERIES) * 3
    assert all(h["status"] == "completed" for h in hist)
