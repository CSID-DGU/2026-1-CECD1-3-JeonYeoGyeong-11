"""The live-demo simulator drives the real screens: customers order (directly
and through the cart) and the seller's completions reach the runtime."""
from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.services.merchant_api import orders_db, seed_demo_data, simulate_activity
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import FakeRecommenderRuntime

SELLER = "merchant-1"


class SimulateActivityTest(unittest.TestCase):
    def test_rounds_create_orders_and_completions_reach_the_runtime(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        conn = orders_db.connect(root / "orders.sqlite")
        with contextlib.redirect_stdout(io.StringIO()):
            seed_demo_data.seed(conn, SELLER)
        before = len(orders_db.list_orders(conn, SELLER))
        conn.close()

        runtime = FakeRecommenderRuntime(SELLER)
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "features.sqlite",
                                    model_dir=root / "models", merchant_db_path=root / "orders.sqlite")
        app = create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: runtime))
        with TestClient(app):  # runs the lifespan once; the simulator's clients share the app
            delivered_at_start = len(runtime.ingested_events)
            sim = simulate_activity.Simulator(
                "http://testserver", SELLER, random.Random(1), follow_recommendations=0.6, bulk_customers=0,
                client_factory=lambda: TestClient(app, follow_redirects=False))
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(8):
                    sim.shop_once()
                    sim.work_orders(limit=5)

        conn = orders_db.connect(root / "orders.sqlite")
        self.addCleanup(conn.close)
        self.assertGreater(len(orders_db.list_orders(conn, SELLER)), before)
        self.assertGreater(len(runtime.ingested_events), delivered_at_start, "completed orders were handed to B")


if __name__ == "__main__":
    unittest.main()
