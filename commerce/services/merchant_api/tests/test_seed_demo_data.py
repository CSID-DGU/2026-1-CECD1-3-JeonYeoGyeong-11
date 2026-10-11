"""Guards the demo seed script: idempotent re-run, no duplicate idempotency
keys, screens-relevant tables all populated."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from commerce.services.merchant_api import orders_db, seed_demo_data

SELLER = "seller-1"


class SeedDemoDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = orders_db.connect(Path(self.tmp.name) / "orders.sqlite")
        self.addCleanup(self.conn.close)

    def _counts(self) -> dict[str, int]:
        tables = ["orders", "customers", "seller_accounts", "messages", "posts", "group_buys"]
        return {t: self.conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0] for t in tables}

    def test_seed_populates_every_screen_relevant_table(self):
        seed_demo_data.seed(self.conn, SELLER)
        counts = self._counts()
        for table, count in counts.items():
            self.assertGreater(count, 0, "%s should be non-empty after seeding" % table)

    def test_seed_is_idempotent(self):
        seed_demo_data.seed(self.conn, SELLER)
        first = self._counts()
        seed_demo_data.seed(self.conn, SELLER)
        second = self._counts()
        self.assertEqual(first, second)

    def test_main_follows_the_app_environment(self):
        # run_local.py passes only MERCHANT_ID and FEATURE_DB_PATH; the seed must
        # land where the app looks (orders.sqlite next to it) under that seller_id.
        root = Path(self.tmp.name) / "merchant_1"
        environ = {"MERCHANT_ID": "merchant-1", "FEATURE_DB_PATH": str(root / "features.sqlite")}
        with mock.patch.dict(os.environ, environ, clear=True), contextlib.redirect_stdout(io.StringIO()):
            seed_demo_data.main([])
        conn = orders_db.connect(root / "orders.sqlite")
        self.addCleanup(conn.close)
        sellers = {r[0] for r in conn.execute("SELECT DISTINCT seller_id FROM orders")}
        self.assertEqual(sellers, {"merchant-1"})

    def test_main_refuses_without_seller_id_or_db_path(self):
        db_path = str(Path(self.tmp.name) / "x" / "orders.sqlite")
        for environ in ({"MERCHANT_DB_PATH": db_path}, {"MERCHANT_ID": "merchant-1"}):
            with mock.patch.dict(os.environ, environ, clear=True), \
                    self.assertRaises(SystemExit, msg=str(environ)), contextlib.redirect_stderr(io.StringIO()):
                seed_demo_data.main([])
        self.assertFalse((Path(self.tmp.name) / "x").exists(), "nothing is created on refusal")

    def test_main_refuses_a_fresh_orders_db_next_to_an_old_feature_ledger(self):
        root = Path(self.tmp.name) / "merchant_9"
        root.mkdir()
        (root / "features.sqlite").write_bytes(b"")  # B's ledger left behind by a half reset
        environ = {"MERCHANT_ID": "merchant-9", "FEATURE_DB_PATH": str(root / "features.sqlite")}
        with mock.patch.dict(os.environ, environ, clear=True), self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            seed_demo_data.main([])
        self.assertFalse((root / "orders.sqlite").exists())

    def test_store_seed_is_its_own_shop_and_idempotent(self):
        with contextlib.redirect_stdout(io.StringIO()):
            seed_demo_data.seed_store(self.conn, "merchant-2", "seafood", customers=8)
            first = self._counts()
            seed_demo_data.seed_store(self.conn, "merchant-2", "seafood", customers=8)
        self.assertEqual(self._counts(), first)
        items = {r[0] for r in self.conn.execute("SELECT item_id_local FROM catalog_items WHERE seller_id = 'merchant-2'")}
        self.assertEqual(items, set(seed_demo_data.STORES["seafood"][2]))
        name = self.conn.execute("SELECT display_name FROM seller_accounts WHERE seller_id = 'merchant-2'").fetchone()[0]
        self.assertEqual(name, "제주 바다 수산")
        self.assertGreater(first["orders"], 20)

    def test_store_mapping_follows_the_launcher_and_shares_some_items(self):
        self.assertIsNone(seed_demo_data.store_for(1))
        self.assertEqual([seed_demo_data.store_for(i) for i in (2, 3, 4, 5)], ["seafood", "bakery", "produce", "seafood"])
        catalogs = [set(seed_demo_data.STORES[s][2]) for s in seed_demo_data.STORE_ORDER]
        self.assertTrue(catalogs[0] & catalogs[2], "a staple sold by more than one store")

    def _bulk(self, customers=10):
        with contextlib.redirect_stdout(io.StringIO()):
            seed_demo_data.seed(self.conn, SELLER)
            seed_demo_data.seed_bulk(self.conn, SELLER, customers=customers)

    def test_bulk_runs_on_top_of_the_base_seed_and_is_idempotent(self):
        self._bulk()
        first = self._counts()
        self._bulk()
        self.assertEqual(self._counts(), first)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM catalog_items").fetchone()[0],
                         6 + len(seed_demo_data._BULK_PRODUCTS))
        self.assertGreater(first["orders"], 50)

    def test_bulk_customers_lean_towards_their_pattern(self):
        self._bulk(customers=15)
        pattern_items = {i: set(p[1]) for i, p in enumerate(seed_demo_data._BULK_PATTERNS)}
        for customer_id, _, pattern in seed_demo_data.bulk_customers(15):
            bought = [item["item_id_local"] for order in orders_db.list_orders(self.conn, SELLER)
                      if order["customer_id_local"] == customer_id for item in order["items"]]
            in_pattern = sum(1 for i in bought if i in pattern_items[pattern])
            self.assertGreater(in_pattern, len(bought) / 2, customer_id)

    def test_bulk_events_carry_the_backdated_completion_time(self):
        self._bulk()
        rows = self.conn.execute(
            "SELECT o.completed_at, pe.payload_json FROM orders o JOIN purchase_events pe "
            "ON pe.seller_id = o.seller_id AND pe.order_id = o.order_id WHERE o.idempotency_key LIKE 'bulk-%'"
        ).fetchall()
        self.assertTrue(rows)
        import json
        for completed_at, payload in rows:
            self.assertEqual(json.loads(payload)["time"]["value"], completed_at)

    def test_no_duplicate_idempotency_keys(self):
        seed_demo_data.seed(self.conn, SELLER)
        rows = self.conn.execute("SELECT idempotency_key, COUNT(*) c FROM orders GROUP BY idempotency_key HAVING c > 1").fetchall()
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
