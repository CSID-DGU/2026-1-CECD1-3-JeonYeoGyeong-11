"""The seller's feature store: B's own SQLite ledger (model.md §4·§7, interfaces.md §2·§3).

One store holds one seller's purchase events, catalog versions and feature_epoch.
A purchase event or a catalog change commits together with its epoch bump, so a
normal return means the change is durable (A then marks its outbox row delivered).

- Events are immutable and keyed by purchase_event_id. The same ID with the same
  canonical body is a success; with another body it is DUPLICATE_EVENT.
- Catalog items keep every source_seq they arrived with. An older source_seq than
  the item's latest is ignored; the same source_seq with another body is rejected.
- A snapshot at epoch E is every event and the latest catalog version committed at
  or before E. Nothing is deleted, so any past epoch can be read again; that is what
  local_data_ref pins for a training round.

Each call opens its own connection: A's request threads and the seller's job
thread share the store, and sqlite3 connections are per thread. Nothing at an
epoch ever changes, so the latest snapshot read is kept and reused until the
epoch moves.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
from typing import Iterator

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.ids import canonical_json
from commerce.packages.contracts.types import Payload
from commerce.packages.data_adapters.baskets import LocalBasket, basket_from_event
from commerce.packages.data_adapters.validation import check_payload

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS events (purchase_event_id TEXT PRIMARY KEY, body TEXT NOT NULL, "
    "epoch INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS catalog_versions (item_id_local TEXT NOT NULL, source_seq INTEGER NOT NULL, "
    "body TEXT NOT NULL, epoch INTEGER NOT NULL, PRIMARY KEY (item_id_local, source_seq))",
    "CREATE INDEX IF NOT EXISTS catalog_by_epoch ON catalog_versions (item_id_local, epoch)",
)


@dataclass(frozen=True)
class Snapshot:
    epoch: int
    baskets: tuple[LocalBasket, ...]  # every event committed at or before epoch, in commit order
    catalog: dict[str, Payload]  # item_id_local -> latest catalog_item.v1 at epoch, active or not


class FeatureStore:
    def __init__(self, path: str | Path, seller_id: str):
        self.path = Path(path)
        self.seller_id = seller_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last: Snapshot | None = None
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            for statement in SCHEMA:
                db.execute(statement)
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT value FROM meta WHERE key = 'seller_id'").fetchone()
            if owner is None:
                db.execute("INSERT INTO meta VALUES ('seller_id', ?)", (seller_id,))
                db.execute("INSERT INTO meta VALUES ('feature_epoch', '0')")
            elif owner[0] != seller_id:
                db.execute("ROLLBACK")
                raise ValueError("this feature store belongs to another seller")
            db.execute("COMMIT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _epoch(db: sqlite3.Connection) -> int:
        return int(db.execute("SELECT value FROM meta WHERE key = 'feature_epoch'").fetchone()[0])

    @staticmethod
    def _bump(db: sqlite3.Connection) -> int:
        epoch = FeatureStore._epoch(db) + 1
        db.execute("UPDATE meta SET value = ? WHERE key = 'feature_epoch'", (str(epoch),))
        return epoch

    @property
    def feature_epoch(self) -> int:
        with self._connect() as db:
            return self._epoch(db)

    def ingest_event(self, event: Payload) -> bool:
        """Store one purchase_event.v1. True if it was new, False for an identical replay."""
        basket = basket_from_event(event)  # schema, ID rule and per-source rules
        if basket.seller_id != self.seller_id:
            raise ContractError("FORBIDDEN", "/seller_id")
        body = canonical_json(event).decode("utf-8")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT body FROM events WHERE purchase_event_id = ?",
                                 (basket.purchase_event_id,)).fetchone()
                if row is not None:
                    if row[0] != body:
                        raise ContractError("DUPLICATE_EVENT", "/purchase_event_id")
                    db.execute("ROLLBACK")
                    return False
                epoch = self._bump(db)
                db.execute("INSERT INTO events VALUES (?, ?, ?)", (basket.purchase_event_id, body, epoch))
                db.execute("COMMIT")
                return True
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def upsert_item(self, item: Payload, source_seq: int) -> bool:
        """Store one catalog_item.v1 at A's outbox sequence. True if it became the latest version."""
        check_payload("catalog_item.v1", item)
        if item["seller_id"] != self.seller_id:
            raise ContractError("FORBIDDEN", "/seller_id")
        if isinstance(source_seq, bool) or not isinstance(source_seq, int) or source_seq < 0:
            raise ValueError("source_seq must be a non-negative integer")
        body = canonical_json(item).decode("utf-8")
        item_id = item["item_id_local"]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                same = db.execute("SELECT body FROM catalog_versions WHERE item_id_local = ? AND source_seq = ?",
                                  (item_id, source_seq)).fetchone()
                if same is not None:
                    if same[0] != body:
                        raise ContractError("DUPLICATE_EVENT", "/item_id_local")
                    db.execute("ROLLBACK")
                    return False
                latest = db.execute("SELECT MAX(source_seq) FROM catalog_versions WHERE item_id_local = ?",
                                    (item_id,)).fetchone()[0]
                if latest is not None and source_seq < latest:
                    db.execute("ROLLBACK")  # a late delivery of an older version: never overwrite
                    return False
                epoch = self._bump(db)
                db.execute("INSERT INTO catalog_versions VALUES (?, ?, ?, ?)", (item_id, source_seq, body, epoch))
                db.execute("COMMIT")
                return True
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def snapshot(self, epoch: int | None = None) -> Snapshot:
        """Events and the latest catalog versions at epoch (default: now), read in one transaction."""
        with self._connect() as db:
            db.execute("BEGIN")
            current = self._epoch(db)
            if epoch is None:
                epoch = current
            elif not 0 <= epoch <= current:
                db.execute("ROLLBACK")
                raise ContractError("NOT_FOUND")
            with self._lock:
                last = self._last
            if last is not None and last.epoch == epoch:
                db.execute("COMMIT")
                return last
            events = db.execute("SELECT body FROM events WHERE epoch <= ? ORDER BY epoch", (epoch,)).fetchall()
            rows = db.execute(
                "SELECT c.item_id_local, c.body FROM catalog_versions c JOIN ("
                " SELECT item_id_local, MAX(source_seq) AS seq FROM catalog_versions WHERE epoch <= ?"
                " GROUP BY item_id_local) m ON c.item_id_local = m.item_id_local AND c.source_seq = m.seq",
                (epoch,)).fetchall()
            db.execute("COMMIT")
        baskets = tuple(basket_from_event(json.loads(body)) for (body,) in events)
        snap = Snapshot(epoch, baskets, {item_id: json.loads(body) for item_id, body in rows})
        with self._lock:
            if self._last is None or self._last.epoch <= epoch:
                self._last = snap
        return snap
