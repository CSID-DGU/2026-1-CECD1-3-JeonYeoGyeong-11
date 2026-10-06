"""Layer 6 social/add-on coverage: DM threads, feed, group-buy settlement, price history."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_db as db
from commerce.services.merchant_api import social_db
from commerce.services.merchant_api import social_service as svc

SELLER = "synthetic-seller-1"


class SocialServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = db.connect(Path(self.tmp.name) / "orders.sqlite")
        social_db.ensure_schema(self.conn)
        self.addCleanup(self.conn.close)
        from commerce.services.merchant_api import orders_service
        for item_id, title in (("sku-rice", "쌀 10kg"), ("sku-fish", "은갈치")):
            orders_service.register_catalog_item(self.conn, seller_id=SELLER, item_id_local=item_id, title_text=title)

    # --- messages ------------------------------------------------------------

    def test_message_thread_round_trips_in_order(self):
        svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-1", sender="customer", body="재고 있나요?")
        svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-1", sender="seller", body="네, 있습니다!")
        thread = svc.list_thread_messages(self.conn, seller_id=SELLER, customer_id_local="cust-1")
        self.assertEqual([m["sender"] for m in thread], ["customer", "seller"])

    def test_threads_list_shows_latest_message_per_customer(self):
        svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-1", sender="customer", body="first")
        svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-2", sender="customer", body="other customer")
        svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-1", sender="seller", body="latest")
        threads = svc.list_threads(self.conn, seller_id=SELLER)
        by_customer = {t["customer_id_local"]: t["last_body"] for t in threads}
        self.assertEqual(by_customer["cust-1"], "latest")
        self.assertEqual(by_customer["cust-2"], "other customer")

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ContractError):
            svc.send_message(self.conn, seller_id=SELLER, customer_id_local="cust-1", sender="customer", body="   ")

    # --- feed ------------------------------------------------------------------

    def test_feed_lists_newest_post_first(self):
        svc.create_post(self.conn, seller_id=SELLER, kind="article", title="첫 소식")
        svc.create_post(self.conn, seller_id=SELLER, kind="short_video", title="수확 영상")
        feed = svc.list_feed(self.conn, seller_id=SELLER)
        self.assertEqual(feed[0]["title"], "수확 영상")
        self.assertEqual(feed[0]["kind"], "short_video")

    # --- group buys --------------------------------------------------------------

    def _group_buy(self, target_quantity: int) -> dict:
        return svc.create_group_buy(
            self.conn, seller_id=SELLER, item_id_local="sku-rice", target_quantity=target_quantity,
            unit_price_minor=10000, deadline_at="2999-01-01T00:00:00.000000Z",
        )

    def _expire(self, group_buy_id: str) -> None:
        """The deadline passes (a group buy can only be created with a future one)."""
        with self.conn:
            self.conn.execute("UPDATE group_buys SET deadline_at = ? WHERE seller_id = ? AND group_buy_id = ?",
                              ("2000-01-01T00:00:00.000000Z", SELLER, group_buy_id))

    def test_group_buy_succeeds_immediately_when_target_met_before_deadline(self):
        gb = self._group_buy(target_quantity=5)
        first = svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                                   customer_id_local="cust-1", quantity=3)
        self.assertEqual(first["status"], "open")
        result = svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                                    customer_id_local="cust-2", quantity=2)
        self.assertEqual(result["status"], "succeeded")

        from commerce.services.merchant_api import orders_service
        for customer, quantity in (("cust-1", 3), ("cust-2", 2)):
            (order,) = orders_service.list_orders_by_customer(self.conn, seller_id=SELLER, customer_id_local=customer)
            self.assertEqual(order["items"], [{"item_id_local": "sku-rice", "quantity": quantity, "unit_price_minor": 10000}])

    def test_group_buy_fails_when_under_target_past_deadline(self):
        gb = self._group_buy(target_quantity=10)
        svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                            customer_id_local="cust-1", quantity=1)
        self._expire(gb["group_buy_id"])
        result = svc.list_group_buys(self.conn, seller_id=SELLER)
        self.assertEqual(result[0]["status"], "failed")

    def test_join_after_deadline_is_rejected_even_before_the_sweep(self):
        gb = self._group_buy(target_quantity=1)
        self._expire(gb["group_buy_id"])
        with self.assertRaises(ContractError) as ctx:  # no list call has swept it yet
            svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                                customer_id_local="cust-1", quantity=1)
        self.assertEqual(ctx.exception.code, "ILLEGAL_STATE_TRANSITION")
        self.assertEqual(svc.list_group_buys(self.conn, seller_id=SELLER)[0]["status"], "failed")

    def test_double_join_is_rejected(self):
        gb = self._group_buy(target_quantity=10)
        svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                            customer_id_local="cust-1", quantity=1)
        with self.assertRaises(ContractError):
            svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                                customer_id_local="cust-1", quantity=1)

    def test_join_after_settlement_is_illegal(self):
        gb = self._group_buy(target_quantity=1)
        svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                            customer_id_local="cust-1", quantity=1)  # reaches target -> succeeded
        with self.assertRaises(ContractError):
            svc.join_group_buy(self.conn, seller_id=SELLER, group_buy_id=gb["group_buy_id"],
                                customer_id_local="cust-2", quantity=1)

    def test_group_buy_rejects_unknown_item_past_deadline_and_negative_price(self):
        for kwargs in ({"item_id_local": "sku-none"}, {"deadline_at": "2000-01-01T00:00:00.000000Z"},
                       {"unit_price_minor": -1}):
            args = {"item_id_local": "sku-rice", "target_quantity": 2, "unit_price_minor": 10000,
                    "deadline_at": "2999-01-01T00:00:00.000000Z", **kwargs}
            with self.assertRaises(ContractError, msg=str(kwargs)):
                svc.create_group_buy(self.conn, seller_id=SELLER, **args)

    def test_unknown_post_kind_is_a_contract_error_not_a_crash(self):
        with self.assertRaises(ContractError):
            svc.create_post(self.conn, seller_id=SELLER, kind="podcast", title="x")

    # --- price history -------------------------------------------------------------

    def test_price_for_unknown_item_or_negative_is_rejected(self):
        with self.assertRaises(ContractError):
            svc.record_price(self.conn, seller_id=SELLER, item_id_local="sku-none", price_minor=1000)
        with self.assertRaises(ContractError):
            svc.record_price(self.conn, seller_id=SELLER, item_id_local="sku-fish", price_minor=-1)

    def test_price_upsert_overwrites_same_day(self):
        svc.record_price(self.conn, seller_id=SELLER, item_id_local="sku-fish", price_minor=9000, price_date="2026-09-30")
        svc.record_price(self.conn, seller_id=SELLER, item_id_local="sku-fish", price_minor=9500, price_date="2026-09-30")
        svc.record_price(self.conn, seller_id=SELLER, item_id_local="sku-fish", price_minor=8800, price_date="2026-10-01")
        history = svc.get_price_history(self.conn, seller_id=SELLER, item_id_local="sku-fish")
        self.assertEqual([(h["price_date"], h["price_minor"]) for h in history],
                          [("2026-09-30", 9500), ("2026-10-01", 8800)])


if __name__ == "__main__":
    unittest.main()
