"""a1 scope: state transitions, idempotency, durable delivery, catalog upsert.

Uses commerce/services/merchant_api/tests/fakes.py as the B double (development.md
"상대 모듈을 대체하는 방법"). Never imports commerce.packages.recommender.runtime's
real implementation, which does not exist yet.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_db as db
from commerce.services.merchant_api import orders_service as svc
from commerce.services.merchant_api.tests.fakes import AlwaysFailingRuntime, FakeRecommenderRuntime

SELLER = "synthetic-seller-1"
OTHER_SELLER = "synthetic-seller-2"


def _items(*pairs: tuple[str, int, int]) -> list[dict]:
    return [{"item_id_local": item_id, "quantity": qty, "unit_price_minor": price} for item_id, qty, price in pairs]


class OrdersServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = db.connect(Path(self.tmp.name) / "orders.sqlite")
        self.addCleanup(self.conn.close)
        self.runtime = FakeRecommenderRuntime(SELLER)

    def _place(self, key="idem-1", customer="cust-1", items=None):
        return svc.place_order(
            self.conn, seller_id=SELLER, customer_id_local=customer, idempotency_key=key,
            items=items or _items(("sku-milk", 2, 3200)), currency="KRW",
        )

    # --- creation & idempotency ------------------------------------------------

    def test_place_order_creates_requested_with_version_one(self):
        order = self._place()
        self.assertEqual(order["status"], "requested")
        self.assertEqual(order["status_version"], 1)
        self.assertIsNone(order["completed_at"])
        self.assertEqual(order["items"], _items(("sku-milk", 2, 3200)))

    def test_duplicate_item_id_in_request_is_merged_by_quantity(self):
        order = self._place(items=_items(("sku-milk", 1, 3200), ("sku-milk", 2, 3200)))
        self.assertEqual(order["items"], [{"item_id_local": "sku-milk", "quantity": 3, "unit_price_minor": 3200}])

    def test_identical_retry_returns_original_order(self):
        first = self._place()
        second = self._place()
        self.assertEqual(first["order_id"], second["order_id"])
        self.assertEqual(first["status_version"], second["status_version"])

    def test_same_key_different_body_is_rejected(self):
        self._place()
        with self.assertRaises(ContractError) as ctx:
            self._place(items=_items(("sku-bread", 1, 1500)))
        self.assertEqual(ctx.exception.code, "DUPLICATE_IDEMPOTENCY_KEY")

    # --- state machine ----------------------------------------------------------

    def test_accept_then_complete_delivers_purchase_event(self):
        order = self._place()
        accepted = svc.transition_order(
            self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept",
            expected_status_version=1,
        )
        self.assertEqual(accepted["status"], "accepted")
        completed = svc.transition_order(
            self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete",
            expected_status_version=2, runtime=self.runtime,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertIsNotNone(completed["completed_at"])
        self.assertEqual(self.runtime.ingest_calls, 1)
        self.assertEqual(len(self.runtime.ingested_events), 1)
        pending = db.fetch_pending_outbox(self.conn, SELLER, kind="purchase_event")
        self.assertEqual(pending, [])  # delivered, not left pending

    def test_complete_without_accept_is_illegal_transition(self):
        order = self._place()
        with self.assertRaises(ContractError) as ctx:
            svc.transition_order(
                self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete",
                expected_status_version=1,
            )
        self.assertEqual(ctx.exception.code, "ILLEGAL_STATE_TRANSITION")

    def test_stale_status_version_is_rejected(self):
        order = self._place()
        svc.transition_order(
            self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept",
            expected_status_version=1,
        )
        with self.assertRaises(ContractError) as ctx:
            svc.transition_order(
                self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept",
                expected_status_version=1,  # already moved to 2
            )
        self.assertEqual(ctx.exception.code, "STALE_STATUS_VERSION")

    def test_cancel_allowed_from_requested_and_accepted(self):
        requested = self._place(key="idem-cancel-1")
        cancelled = svc.transition_order(
            self.conn, seller_id=SELLER, order_id=requested["order_id"], action="cancel",
            expected_status_version=1,
        )
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNotNone(cancelled["cancelled_at"])

        accepted_first = self._place(key="idem-cancel-2")
        svc.transition_order(
            self.conn, seller_id=SELLER, order_id=accepted_first["order_id"], action="accept",
            expected_status_version=1,
        )
        cancelled2 = svc.transition_order(
            self.conn, seller_id=SELLER, order_id=accepted_first["order_id"], action="cancel",
            expected_status_version=2,
        )
        self.assertEqual(cancelled2["status"], "cancelled")

    def test_completed_order_has_no_further_transition(self):
        order = self._place()
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2, runtime=self.runtime)
        with self.assertRaises(ContractError) as ctx:
            svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="cancel", expected_status_version=3)
        self.assertEqual(ctx.exception.code, "ILLEGAL_STATE_TRANSITION")

    # --- cross-seller permission -------------------------------------------------

    def test_other_seller_cannot_see_or_transition_order(self):
        order = self._place()
        with self.assertRaises(ContractError) as ctx:
            svc.get_order(self.conn, seller_id=OTHER_SELLER, order_id=order["order_id"])
        self.assertEqual(ctx.exception.code, "NOT_FOUND")
        with self.assertRaises(ContractError) as ctx:
            svc.transition_order(self.conn, seller_id=OTHER_SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        self.assertEqual(ctx.exception.code, "NOT_FOUND")

    # --- durable delivery ---------------------------------------------------------

    def test_delivery_failure_stays_pending_for_retry(self):
        failing = AlwaysFailingRuntime(SELLER)
        order = self._place()
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2, runtime=failing)
        pending = db.fetch_pending_outbox(self.conn, SELLER, kind="purchase_event")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["attempts"], 1)

        # B comes back: a later retry with a working runtime delivers it.
        svc.deliver_pending_purchase_events(self.conn, seller_id=SELLER, runtime=self.runtime)
        self.assertEqual(db.fetch_pending_outbox(self.conn, SELLER, kind="purchase_event"), [])
        self.assertEqual(self.runtime.ingest_calls, 1)

    def test_conflicting_redelivery_is_quarantined_not_retried(self):
        order = self._place()
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2, runtime=self.runtime)
        delivered_event = next(iter(self.runtime.ingested_events.values()))

        # Simulate a corrupted re-queue: same purchase_event_id, different items.
        conflicting = dict(delivered_event)
        conflicting["items"] = [{"item_id_local": "sku-other", "quantity_observed": 9}]
        with self.conn:
            db.insert_outbox(self.conn, SELLER, "purchase_event", conflicting["purchase_event_id"], conflicting, "2026-01-01T00:00:00.000000Z")
        svc.deliver_pending_purchase_events(self.conn, seller_id=SELLER, runtime=self.runtime)

        pending_after = db.fetch_pending_outbox(self.conn, SELLER, kind="purchase_event")
        self.assertEqual(pending_after, [])  # not left pending: quarantined, not retried
        row = self.conn.execute(
            "SELECT status FROM outbox WHERE seller_id = ? AND kind = 'purchase_event' AND ref_id = ?",
            (SELLER, conflicting["purchase_event_id"]),
        ).fetchone()
        self.assertEqual(row["status"], "quarantined")

    # --- catalog -------------------------------------------------------------------

    def test_catalog_upsert_increments_source_seq_and_forwards(self):
        first = svc.register_catalog_item(
            self.conn, seller_id=SELLER, item_id_local="sku-milk", title_text="우유 1L", runtime=self.runtime,
        )
        second = svc.register_catalog_item(
            self.conn, seller_id=SELLER, item_id_local="sku-milk", title_text="우유 1L (개정)", runtime=self.runtime,
        )
        self.assertEqual(first["first_listed_at"], second["first_listed_at"])  # unchanged on update
        stored_seq, stored_item = self.runtime.catalog["sku-milk"]
        self.assertEqual(stored_seq, 2)
        self.assertEqual(stored_item["title_text"], "우유 1L (개정)")
        self.assertEqual(db.fetch_pending_outbox(self.conn, SELLER, kind="catalog_item"), [])

    # --- recommendation display (mock fallback while B is unimplemented) -----------

    def test_recommendation_falls_back_to_p_topfreq_when_runtime_not_implemented(self):
        svc.register_catalog_item(self.conn, seller_id=SELLER, item_id_local="sku-milk", title_text="우유")
        svc.register_catalog_item(self.conn, seller_id=SELLER, item_id_local="sku-bread", title_text="식빵")
        for key in ("o1", "o2"):
            order = self._place(key=key, items=_items(("sku-milk", 1, 2500)))
            svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
            svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2)
        order = self._place(key="o3", items=_items(("sku-bread", 1, 3800)))
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        svc.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2)

        result = svc.get_recommendations_for_display(
            self.conn, seller_id=SELLER, customer_id_local="cust-1", runtime=self.runtime,
        )
        self.assertEqual(result["model_version"], svc.MOCK_MODEL_VERSION)
        self.assertTrue(result["is_cold_start"])
        self.assertEqual([i["item_id_local"] for i in result["items"]], ["sku-milk", "sku-bread"])  # milk bought more

    def test_recommendation_passes_through_a_real_runtime_unchanged(self):
        canned = {
            "schema_version": "recommendation.v1", "seller_id": SELLER, "customer_id_local": "cust-1",
            "as_of": "2026-01-01T00:00:00.000000Z", "model_version": "real-model-v1",
            "score_semantics": "next_purchase", "horizon_days": None, "is_cold_start": False,
            "fallback_reason": None, "items": [{"item_id_local": "sku-milk", "score": 0.9}],
        }

        class RealRuntime(FakeRecommenderRuntime):
            def predict_local(self, request, *, model_variant="text_relation", mode="auto"):
                return canned

        result = svc.get_recommendations_for_display(
            self.conn, seller_id=SELLER, customer_id_local="cust-1", runtime=RealRuntime(SELLER),
        )
        self.assertIs(result, canned)  # B's real response passes through unchanged


if __name__ == "__main__":
    unittest.main()
