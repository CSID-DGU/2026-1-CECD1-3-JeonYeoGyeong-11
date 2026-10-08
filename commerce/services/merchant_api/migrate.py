"""Additive migrations for orders.sqlite (A-only).

ensure_schema() functions use CREATE TABLE IF NOT EXISTS, which never touches
a table that already exists, so a column added later would be missing from
every DB created before it. add_columns() closes that gap for the additive
case: it adds each missing column with its declared default. Anything beyond
adding a nullable or defaulted column (renames, new CHECK constraints) is out
of scope on purpose -- that needs a real migration step and a team decision.
"""
from __future__ import annotations

import sqlite3
from typing import Mapping


def add_columns(conn: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    """columns: name -> SQL declaration after the name, e.g. {"media_json": "TEXT"}."""
    present = {row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)}
    for name, declaration in columns.items():
        if name not in present:
            conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, declaration))
