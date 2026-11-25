"""Postgres connections with the schema applied and pgvector types registered."""

from __future__ import annotations

from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from filinglens.db.repositories import schema_sql


def connect(database_url: str, *, apply_schema: bool = True) -> psycopg.Connection[Any]:
    """Open an autocommit connection; numpy arrays then bind as ``vector`` parameters."""
    conn = psycopg.connect(database_url, autocommit=True)
    if apply_schema:
        with conn.transaction():
            conn.execute(schema_sql())
    register_vector(conn)
    return conn
