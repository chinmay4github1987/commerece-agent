"""Checkpoint store selection.

- SQLite (default): zero setup, fine for local dev and a single replica.
- PostgreSQL (CHECKPOINT_URL=postgresql://...): required when running more
  than one replica, so any pod can resume any session.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings


@dataclass
class CheckpointStore:
    saver: Any
    ping: Callable[[], None]
    close: Callable[[], None]
    backend: str


def open_checkpoint_store(settings: Settings) -> CheckpointStore:
    if settings.checkpoint_url:
        return _open_postgres(settings.checkpoint_url)
    return _open_sqlite(settings.checkpoint_db)


def _open_sqlite(path: str) -> CheckpointStore:
    from langgraph.checkpoint.sqlite import SqliteSaver

    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return CheckpointStore(saver, lambda: conn.execute("SELECT 1"), conn.close, "sqlite")


def _open_postgres(url: str) -> CheckpointStore:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=url, min_size=1, max_size=10, open=True, timeout=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent migrations

    def ping() -> None:
        with pool.connection() as conn:
            conn.execute("SELECT 1")

    return CheckpointStore(saver, ping, pool.close, "postgres")
