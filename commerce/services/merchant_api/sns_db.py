"""Store SNS tables in orders.sqlite (A-only, no cross-role contract).

Posts reuse social_db's `posts` table (kind 'article' = photo/text post,
'short_video' = reel) and gain columns for the media list, hashtags, edits and
soft deletion. New tables hold product tags, likes, comments and views.
Everything is seller-local: a post's audience is this store's customers, so
nothing here leaves the seller's server (architecture.md §2).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Optional

from commerce.services.merchant_api import social_db
from commerce.services.merchant_api.migrate import add_columns


def ensure_schema(conn: sqlite3.Connection) -> None:
    social_db.ensure_schema(conn)  # posts lives there
    add_columns(conn, "posts", {"media_json": "TEXT", "hashtags_json": "TEXT", "updated_at": "TEXT",
                                "deleted_at": "TEXT"})
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS post_products (
            seller_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            PRIMARY KEY (seller_id, post_id, item_id_local)
        );
        CREATE INDEX IF NOT EXISTS idx_post_products_item ON post_products(seller_id, item_id_local);

        CREATE TABLE IF NOT EXISTS post_likes (
            seller_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, post_id, customer_id_local)
        );

        CREATE TABLE IF NOT EXISTS post_comments (
            seller_id TEXT NOT NULL,
            comment_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            author TEXT NOT NULL CHECK (author IN ('customer', 'seller')),
            customer_id_local TEXT,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, comment_id)
        );
        CREATE INDEX IF NOT EXISTS idx_post_comments_post ON post_comments(seller_id, post_id, created_at);

        -- One row per customer and post: how often and when they last saw it (ranking avoids repeats).
        CREATE TABLE IF NOT EXISTS post_views (
            seller_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            viewer TEXT NOT NULL,
            views INTEGER NOT NULL DEFAULT 0,
            last_viewed_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, post_id, viewer)
        );
        """
    )


def _post(row: sqlite3.Row) -> dict[str, Any]:
    post = dict(row)
    post["media"] = json.loads(post.get("media_json") or "[]")
    post["hashtags"] = json.loads(post.get("hashtags_json") or "[]")
    if not post["media"] and post.get("media_path"):  # rows written before media_json
        post["media"] = [{"type": "image", "name": post["media_path"]}]
    return post


def insert_post(conn: sqlite3.Connection, seller_id: str, post_id: str, kind: str, title: str, body: Optional[str],
                media: list[dict], hashtags: list[str], created_at: str) -> None:
    conn.execute(
        "INSERT INTO posts (seller_id, post_id, kind, title, body, media_path, created_at, media_json, hashtags_json) "
        "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
        (seller_id, post_id, kind, title, body, created_at, json.dumps(media), json.dumps(hashtags, ensure_ascii=False)),
    )


def update_post(conn: sqlite3.Connection, seller_id: str, post_id: str, title: str, body: Optional[str],
                hashtags: list[str], updated_at: str) -> None:
    conn.execute(
        "UPDATE posts SET title = ?, body = ?, hashtags_json = ?, updated_at = ? WHERE seller_id = ? AND post_id = ?",
        (title, body, json.dumps(hashtags, ensure_ascii=False), updated_at, seller_id, post_id),
    )


def soft_delete_post(conn: sqlite3.Connection, seller_id: str, post_id: str, deleted_at: str) -> bool:
    cur = conn.execute("UPDATE posts SET deleted_at = ? WHERE seller_id = ? AND post_id = ? AND deleted_at IS NULL",
                       (deleted_at, seller_id, post_id))
    return cur.rowcount == 1


def fetch_post(conn: sqlite3.Connection, seller_id: str, post_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM posts WHERE seller_id = ? AND post_id = ? AND deleted_at IS NULL",
                       (seller_id, post_id)).fetchone()
    return _post(row) if row else None


def list_posts(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM posts WHERE seller_id = ? AND deleted_at IS NULL "
                        "ORDER BY created_at DESC, rowid DESC", (seller_id,)).fetchall()
    return [_post(r) for r in rows]


def set_post_products(conn: sqlite3.Connection, seller_id: str, post_id: str, item_ids: Iterable[str]) -> None:
    conn.execute("DELETE FROM post_products WHERE seller_id = ? AND post_id = ?", (seller_id, post_id))
    conn.executemany("INSERT OR IGNORE INTO post_products (seller_id, post_id, item_id_local) VALUES (?, ?, ?)",
                     [(seller_id, post_id, i) for i in item_ids])


def post_products(conn: sqlite3.Connection, seller_id: str) -> dict[str, list[str]]:
    """post_id -> tagged item ids (one query for the whole feed)."""
    out: dict[str, list[str]] = {}
    for r in conn.execute("SELECT post_id, item_id_local FROM post_products WHERE seller_id = ? ORDER BY item_id_local",
                          (seller_id,)):
        out.setdefault(r["post_id"], []).append(r["item_id_local"])
    return out


def toggle_like(conn: sqlite3.Connection, seller_id: str, post_id: str, customer_id_local: str, now: str) -> bool:
    """True if the post is liked after the call."""
    cur = conn.execute("DELETE FROM post_likes WHERE seller_id = ? AND post_id = ? AND customer_id_local = ?",
                       (seller_id, post_id, customer_id_local))
    if cur.rowcount:
        return False
    conn.execute("INSERT INTO post_likes (seller_id, post_id, customer_id_local, created_at) VALUES (?, ?, ?, ?)",
                 (seller_id, post_id, customer_id_local, now))
    return True


def like_counts(conn: sqlite3.Connection, seller_id: str) -> dict[str, int]:
    return {r["post_id"]: r["n"] for r in conn.execute(
        "SELECT post_id, COUNT(*) AS n FROM post_likes WHERE seller_id = ? GROUP BY post_id", (seller_id,))}


def liked_by(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> set[str]:
    return {r["post_id"] for r in conn.execute(
        "SELECT post_id FROM post_likes WHERE seller_id = ? AND customer_id_local = ?", (seller_id, customer_id_local))}


def likes_with_time(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT post_id, customer_id_local, created_at FROM post_likes WHERE seller_id = ?", (seller_id,))]


def insert_comment(conn: sqlite3.Connection, seller_id: str, comment_id: str, post_id: str, author: str,
                   customer_id_local: Optional[str], body: str, now: str) -> None:
    conn.execute("INSERT INTO post_comments (seller_id, comment_id, post_id, author, customer_id_local, body, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)", (seller_id, comment_id, post_id, author, customer_id_local, body, now))


def delete_comment(conn: sqlite3.Connection, seller_id: str, comment_id: str,
                   customer_id_local: Optional[str] = None) -> bool:
    """The seller may delete any comment; a customer only their own (customer_id_local given)."""
    if customer_id_local is None:
        cur = conn.execute("DELETE FROM post_comments WHERE seller_id = ? AND comment_id = ?", (seller_id, comment_id))
    else:
        cur = conn.execute("DELETE FROM post_comments WHERE seller_id = ? AND comment_id = ? AND customer_id_local = ?",
                           (seller_id, comment_id, customer_id_local))
    return cur.rowcount == 1


def list_comments(conn: sqlite3.Connection, seller_id: str, post_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM post_comments WHERE seller_id = ? AND post_id = ? ORDER BY created_at, rowid",
        (seller_id, post_id))]


def comment_counts(conn: sqlite3.Connection, seller_id: str) -> dict[str, int]:
    return {r["post_id"]: r["n"] for r in conn.execute(
        "SELECT post_id, COUNT(*) AS n FROM post_comments WHERE seller_id = ? GROUP BY post_id", (seller_id,))}


def record_view(conn: sqlite3.Connection, seller_id: str, post_id: str, viewer: str, now: str) -> None:
    conn.execute(
        """
        INSERT INTO post_views (seller_id, post_id, viewer, views, last_viewed_at) VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(seller_id, post_id, viewer) DO UPDATE SET views = views + 1, last_viewed_at = excluded.last_viewed_at
        """,
        (seller_id, post_id, viewer, now),
    )


def view_counts(conn: sqlite3.Connection, seller_id: str) -> dict[str, int]:
    return {r["post_id"]: r["n"] for r in conn.execute(
        "SELECT post_id, SUM(views) AS n FROM post_views WHERE seller_id = ? GROUP BY post_id", (seller_id,))}


def viewed_by(conn: sqlite3.Connection, seller_id: str, viewer: str) -> dict[str, str]:
    """post_id -> last time this viewer opened it."""
    return {r["post_id"]: r["last_viewed_at"] for r in conn.execute(
        "SELECT post_id, last_viewed_at FROM post_views WHERE seller_id = ? AND viewer = ?", (seller_id, viewer))}
