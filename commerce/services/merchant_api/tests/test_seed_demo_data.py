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
            seed_demo_data.main()
        conn = orders_db.connect(root / "orders.sqlite")
        self.addCleanup(conn.close)
        sellers = {r[0] for r in conn.execute("SELECT DISTINCT seller_id FROM orders")}
        self.assertEqual(sellers, {"merchant-1"})

    def test_main_refuses_without_seller_id_or_db_path(self):
        db_path = str(Path(self.tmp.name) / "x" / "orders.sqlite")
        for environ in ({"MERCHANT_DB_PATH": db_path}, {"MERCHANT_ID": "merchant-1"}):
            with mock.patch.dict(os.environ, environ, clear=True), \
                    self.assertRaises(SystemExit, msg=str(environ)), contextlib.redirect_stderr(io.StringIO()):
                seed_demo_data.main()
        self.assertFalse((Path(self.tmp.name) / "x").exists(), "nothing is created on refusal")

    def test_no_duplicate_idempotency_keys(self):
        seed_demo_data.seed(self.conn, SELLER)
        rows = self.conn.execute("SELECT idempotency_key, COUNT(*) c FROM orders GROUP BY idempotency_key HAVING c > 1").fetchall()
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
