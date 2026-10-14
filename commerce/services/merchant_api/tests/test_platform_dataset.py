"""The synthetic platform dataset: a small store fills every table, a re-run adds nothing,
and the export is B's SellerInput shape with backdated, ordered events."""
from __future__ import annotations

import dataclasses
import datetime as dt
import tempfile
import unittest
from pathlib import Path

from commerce.services.merchant_api import orders_db, platform_dataset as pd

NOW = dt.datetime(2026, 10, 8, 12, 0, tzinfo=dt.timezone.utc)


class PlatformDatasetTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.conn = orders_db.connect(Path(tmp.name) / "orders.sqlite")
        self.addCleanup(self.conn.close)
        self.store = dataclasses.replace(pd.STORES[0], customers=12, history_days=30)

    def test_store_fills_every_part_and_rerun_adds_nothing(self):
        first = pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)
        stats = pd.summary(self.conn, "merchant-1")
        for key in ("products", "customers", "orders", "purchase_events", "reviews", "posts", "likes", "views",
                    "group_buys", "wishlist", "messages"):
            self.assertGreater(stats[key], 0, key)
        self.assertGreater(first["orders"], 0)
        again = pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)
        self.assertEqual(again["orders"], 0)
        self.assertEqual(pd.summary(self.conn, "merchant-1"), stats)

    def test_export_is_b_seller_input_shaped(self):
        pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)
        data = pd.export_seller_input(self.conn, "merchant-1")
        self.assertEqual(set(data), {"seller_id", "catalog", "events", "customers"})
        self.assertTrue(all(c["schema_version"] == "catalog_item.v1" for c in data["catalog"]))
        times = [e["time"]["value"] for e in data["events"]]
        self.assertEqual(times, sorted(times))
        self.assertLess(times[0], (NOW - dt.timedelta(days=7)).isoformat())  # backdated, not "now"
        self.assertTrue({e["customer_id_local"] for e in data["events"]} <= set(data["customers"]))
        event_ids = [e["purchase_event_id"] for e in data["events"]]
        self.assertEqual(len(event_ids), len(set(event_ids)))  # what an FL input attestation requires

    def test_two_fresh_databases_get_the_same_history(self):
        # B's #49 check: the same command elsewhere must give the same events and times,
        # or each machine's FL input attestation (ids.snapshot_digest) differs.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other = orders_db.connect(Path(tmp.name) / "orders.sqlite")
        self.addCleanup(other.close)
        pd.seed_store(self.conn, "merchant-1", self.store)
        pd.seed_store(other, "merchant-1", self.store)
        self.assertEqual(pd.export_seller_input(self.conn, "merchant-1"), pd.export_seller_input(other, "merchant-1"))

    def test_master_account_logs_in_both_ways_and_has_a_history(self):
        from commerce.services.merchant_api import accounts_service, fulfillment, notifications
        pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)
        pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)  # again: nothing doubles
        accounts_service.authenticate_customer(self.conn, seller_id="merchant-1", customer_id_local=pd.MASTER_ID,
                                               password=pd.MASTER_PASSWORD)
        accounts_service.authenticate_seller(self.conn, seller_id="merchant-1", username=pd.MASTER_ID,
                                             password=pd.MASTER_PASSWORD)
        mine = orders_db.list_orders_by_customer(self.conn, "merchant-1", pd.MASTER_ID)
        self.assertEqual(sorted(o["status"] for o in mine).count("completed"), 9)
        self.assertEqual({o["status"] for o in mine} - {"completed"}, {"accepted", "requested"})
        self.assertTrue(all(fulfillment.of_order(self.conn, "merchant-1", o["order_id"]) for o in mine))
        self.assertTrue(notifications.is_following(self.conn, "merchant-1", pd.MASTER_ID))
        self.assertGreater(notifications.unread_count(self.conn, "merchant-1", pd.MASTER_ID), 0)

    def test_new_item_is_listed_late(self):
        pd.seed_store(self.conn, "merchant-1", self.store, now=NOW)
        catalog = {c["item_id_local"]: c for c in pd.export_seller_input(self.conn, "merchant-1")["catalog"]}
        self.assertGreater(catalog[self.store.new_item]["first_listed_at"], catalog["sku-milk"]["first_listed_at"])


if __name__ == "__main__":
    unittest.main()
