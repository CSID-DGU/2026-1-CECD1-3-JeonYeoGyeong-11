"""Direct-trade checkout over HTTP: pickup or delivery with fee and address checks, stock that
orders take and cancels give back, the order timeline and who may see it, buyer notifications
(accept, tracking, group buy, review reply, a followed store's post), product photos, and the
sales numbers on the seller overview."""
from __future__ import annotations

import datetime as dt
import re
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.services.merchant_api import fulfillment, notifications, orders_db, orders_service, sales, session
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import ScriptedRuntime

SELLER = "seller-1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class FulfillmentScreensTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.db_path = root / "orders.sqlite"
        self.runtime = ScriptedRuntime(SELLER)
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "features.sqlite",
                                    model_dir=root / "models", merchant_db_path=self.db_path)
        app = create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: self.runtime))
        self.client = TestClient(app, follow_redirects=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        r = self.client.post(f"/seller/{SELLER}/signup", data={
            "username": "owner", "display_name": "제주 농장", "password": "pw-1234",
            "business_reg_no": "1234567890", "business_open_date": "20200101", "business_rep_name": "홍길동"})
        self.assertEqual(r.status_code, 303, r.text)
        for item, title, price in (("sku-fish", "제주 은갈치", 15000), ("sku-egg", "유정란 10구", 6900)):
            self.client.post(f"/seller/{SELLER}/products", data={
                "title_text": title, "display_price_minor": str(price), "item_id_local": item, "csrf_token": self._seller_csrf()})
        self._customer("cust-1")

    # --- helpers ------------------------------------------------------------------

    def _seller_csrf(self):
        return session.csrf_token(self.client.cookies.get(session.seller_cookie(SELLER)))

    def _csrf(self):
        return session.csrf_token(self.client.cookies.get(session.customer_cookie(SELLER)))

    def _customer(self, customer_id):
        self.client.cookies.delete(session.customer_cookie(SELLER))
        r = self.client.post(f"/buyer/{SELLER}/signup", data={
            "customer_id_local": customer_id, "display_name": "김하준", "password": "pw-1234"})
        self.assertEqual(r.status_code, 303, r.text)

    def _login(self, customer_id):
        self.client.cookies.delete(session.customer_cookie(SELLER))
        self.client.post(f"/buyer/{SELLER}/login", data={"customer_id_local": customer_id, "password": "pw-1234"})

    def _db(self):
        conn = orders_db.connect(self.db_path)
        self.addCleanup(conn.close)
        return conn

    def _order(self, item="sku-fish", quantity=1, **fields):
        return self.client.post(f"/buyer/{SELLER}/orders", data={
            "item_id_local": item, "quantity": str(quantity), "csrf_token": self._csrf(), **fields})

    def _last_order(self):
        return orders_db.list_orders(self._db(), SELLER)[0]

    def _seller_action(self, order, action, version):
        return self.client.post(f"/seller/{SELLER}/orders/{order['order_id']}/{action}",
                                data={"expected_status_version": str(version), "csrf_token": self._seller_csrf()})

    # --- checkout ------------------------------------------------------------------

    def test_an_order_without_details_is_a_pickup(self):
        # C's rehearsal and the simulator post only item and quantity.
        self.assertEqual(self._order().status_code, 200)
        f = fulfillment.of_order(self._db(), SELLER, self._last_order()["order_id"])
        self.assertEqual((f["method"], f["shipping_fee"], f["pickup_slot"]), ("pickup", 0, fulfillment.PICKUP_SLOTS[0]))

    def test_delivery_needs_recipient_phone_and_address_and_charges_the_fee(self):
        r = self._order(method="delivery", recipient="김하준", phone="010-1234-5678", address="")
        self.assertIn("e=address", r.headers["location"])
        r = self._order(method="delivery", recipient="김하준", phone="12345", address="제주시 첨단로 1")
        self.assertIn("e=phone", r.headers["location"])
        self.assertEqual(orders_db.list_orders(self._db(), SELLER), [])
        page = self.client.get(r.headers["location"]).text
        self.assertIn("010-1234-5678 형식", page)

        r = self._order(method="delivery", recipient="김하준", phone="010-1234-5678", address="제주시 첨단로 1, 101호", memo="문 앞")
        self.assertEqual(r.status_code, 200)
        self.assertIn("3,000원", r.text)          # 15,000 < 30,000 free-delivery line
        self.assertIn("18,000원", r.text)
        f = fulfillment.of_order(self._db(), SELLER, self._last_order()["order_id"])
        self.assertEqual((f["method"], f["shipping_fee"], f["memo"]), ("delivery", 3000, "문 앞"))
        # the next checkout is pre-filled from the saved profile
        self.assertIn('value="제주시 첨단로 1, 101호"', self.client.get(f"/buyer/{SELLER}/items/sku-egg").text)

    def test_contact_details_never_reach_b(self):
        self._order(method="delivery", recipient="김하준", phone="010-1234-5678", address="제주시 첨단로 1")
        order = self._last_order()
        self._seller_action(order, "accept", 1)
        self._seller_action(order, "complete", 2)
        events = list(self.runtime.ingested_events.values())
        self.assertEqual(len(events), 1)
        self.assertNotIn("010-1234-5678", repr(events[0]))
        self.assertNotIn("첨단로", repr(events[0]))

    def test_free_delivery_over_the_line_and_cart_checkout(self):
        self.client.post(f"/buyer/{SELLER}/cart", data={"item_id_local": "sku-fish", "quantity": "2", "csrf_token": self._csrf()})
        cart = self.client.get(f"/buyer/{SELLER}/cart").text
        key = re.search(r'name="checkout_key" value="([^"]+)"', cart).group(1)
        r = self.client.post(f"/buyer/{SELLER}/checkout", data={
            "checkout_key": key, "method": "delivery", "recipient": "김하준", "phone": "01012345678",
            "address": "제주시 첨단로 1", "csrf_token": self._csrf()})
        self.assertEqual(r.status_code, 200, r.text)
        f = fulfillment.of_order(self._db(), SELLER, self._last_order()["order_id"])
        self.assertEqual(f["shipping_fee"], 0)  # 30,000 >= 30,000

    # --- stock ---------------------------------------------------------------------

    def test_stock_limits_orders_and_cancels_give_it_back(self):
        r = self.client.post(f"/seller/{SELLER}/products/sku-egg/stock", data={"stock": "2", "csrf_token": self._seller_csrf()})
        self.assertEqual(r.status_code, 303)
        self.assertIn("e=stock", self._order("sku-egg", 3).headers["location"])
        self.assertEqual(self._order("sku-egg", 2).status_code, 200)
        self.assertEqual(fulfillment.stock_levels(self._db(), SELLER)["sku-egg"], 0)
        self.assertIn("품절", self.client.get(f"/buyer/{SELLER}/").text)
        self.assertIn("품절됐어요", self.client.get(f"/buyer/{SELLER}/items/sku-egg").text)
        order = self._last_order()
        self._seller_action(order, "cancel", 1)
        self.assertEqual(fulfillment.stock_levels(self._db(), SELLER)["sku-egg"], 2)
        self._seller_action(order, "cancel", 2)  # a second cancel is refused and changes nothing
        self.assertEqual(fulfillment.stock_levels(self._db(), SELLER)["sku-egg"], 2)

    def test_buyer_cancel_returns_stock(self):
        self.client.post(f"/seller/{SELLER}/products/sku-egg/stock", data={"stock": "5", "csrf_token": self._seller_csrf()})
        self._order("sku-egg", 2)
        order = self._last_order()
        self.client.post(f"/buyer/{SELLER}/orders/{order['order_id']}/cancel", data={"csrf_token": self._csrf()})
        self.assertEqual(fulfillment.stock_levels(self._db(), SELLER)["sku-egg"], 5)

    # --- order detail, notifications ---------------------------------------------------

    def test_timeline_and_notifications_follow_the_seller(self):
        self._order(method="delivery", recipient="김하준", phone="010-1234-5678", address="제주시 첨단로 1")
        order = self._last_order()
        self._seller_action(order, "accept", 1)
        detail = self.client.get(f"/buyer/{SELLER}/orders/{order['order_id']}").text
        self.assertIn("판매자 확인·준비 중", detail)
        self.assertEqual(notifications.unread_count(self._db(), SELLER, "cust-1"), 1)
        self.assertIn('class="count-dot">1<', self.client.get(f"/buyer/{SELLER}/").text)
        self.client.post(f"/seller/{SELLER}/orders/{order['order_id']}/tracking",
                         data={"tracking_no": "CJ-1234567", "csrf_token": self._seller_csrf()})
        self.assertIn("CJ-1234567", self.client.get(f"/buyer/{SELLER}/orders/{order['order_id']}").text)
        page = self.client.get(f"/buyer/{SELLER}/notifications").text
        self.assertIn("송장번호 CJ-1234567", page)
        self.assertIn("주문을 확인했어요", page)
        self.assertEqual(notifications.unread_count(self._db(), SELLER, "cust-1"), 0)  # opened = read
        seller_view = self.client.get(f"/seller/{SELLER}/orders/{order['order_id']}").text
        self.assertIn("제주시 첨단로 1", seller_view)

    def test_another_customer_cannot_open_the_order(self):
        self._order()
        order = self._last_order()
        self._customer("cust-2")
        self.assertEqual(self.client.get(f"/buyer/{SELLER}/orders/{order['order_id']}").status_code, 404)

    def test_group_buy_success_and_review_reply_notify(self):
        self.client.post(f"/buyer/{SELLER}/group-buys", data={
            "item_id_local": "sku-fish", "target_quantity": "3", "quantity": "1", "days": "7", "csrf_token": self._csrf()})
        gid = self._db().execute("SELECT group_buy_id FROM group_buys").fetchone()[0]
        self._customer("cust-2")
        self.client.post(f"/buyer/{SELLER}/group-buys/{gid}/join", data={"quantity": "2", "csrf_token": self._csrf()})
        conn = self._db()
        self.assertEqual(notifications.unread_count(conn, SELLER, "cust-1"), 1)
        self.assertEqual(notifications.unread_count(conn, SELLER, "cust-2"), 1)

        order = [o for o in orders_db.list_orders(conn, SELLER) if o["customer_id_local"] == "cust-2"][0]
        self._seller_action(order, "accept", 1)
        self._seller_action(order, "complete", 2)
        self.client.post(f"/buyer/{SELLER}/items/sku-fish/reviews", data={"rating": "5", "body": "좋아요", "csrf_token": self._csrf()})
        self.client.post(f"/seller/{SELLER}/products/sku-fish/reviews/cust-2/reply", data={"reply": "감사합니다", "csrf_token": self._seller_csrf()})
        titles = [n["title"] for n in notifications.recent(self._db(), SELLER, "cust-2")]
        self.assertIn("판매자가 내 리뷰에 답글을 남겼어요.", titles)
        self.assertIn("김*준", self.client.get(f"/buyer/{SELLER}/items/sku-fish").text)  # reviewer name, masked

    def test_followers_hear_about_new_posts(self):
        page = self.client.get(f"/buyer/{SELLER}/store").text
        self.assertIn("제주 농장", page)
        self.client.post(f"/buyer/{SELLER}/store/follow", data={"csrf_token": self._csrf()})
        self.assertIn("단골 ✓", self.client.get(f"/buyer/{SELLER}/store").text)
        self.client.post(f"/seller/{SELLER}/feed", data={"kind": "article", "caption": "갈치 들어왔어요", "item_ids": ["sku-fish"],
                                                       "csrf_token": self._seller_csrf()})
        titles = [n["title"] for n in notifications.recent(self._db(), SELLER, "cust-1")]
        self.assertTrue(any("갈치 들어왔어요" in t for t in titles), titles)

    def test_my_page_saves_and_checks_details(self):
        r = self.client.post(f"/buyer/{SELLER}/me", data={"method": "delivery", "recipient": "김하준", "phone": "02-12",
                                                          "address": "x", "csrf_token": self._csrf()})
        self.assertIn("e=phone", r.headers["location"])
        r = self.client.post(f"/buyer/{SELLER}/me", data={"method": "delivery", "recipient": "김하준", "phone": "010-2222-3333",
                                                          "address": "서귀포시 1", "csrf_token": self._csrf()})
        self.assertIn("saved=1", r.headers["location"])
        self.assertEqual(fulfillment.profile(self._db(), SELLER, "cust-1")["phone"], "010-2222-3333")

    # --- photos, overview --------------------------------------------------------------

    def test_product_photos_show_and_svg_is_refused(self):
        r = self.client.post(f"/seller/{SELLER}/products/sku-fish/photos", data={"csrf_token": self._seller_csrf()},
                             files=[("files", ("a.png", PNG, "image/png"))])
        self.assertIn("photo=ok", r.headers["location"])
        name = re.search(r'/media/seller-1/(up-[0-9a-f]+\.png)', self.client.get(f"/buyer/{SELLER}/").text).group(1)
        self.assertEqual(self.client.get(f"/media/{SELLER}/{name}").status_code, 200)
        r = self.client.post(f"/seller/{SELLER}/products/sku-fish/photos", data={"csrf_token": self._seller_csrf()},
                             files=[("files", ("x.svg", b"<svg/>", "image/svg+xml"))])
        self.assertIn("photo=bad", r.headers["location"])

    def test_overview_shows_sales(self):
        self._order(quantity=2)
        order = self._last_order()
        self._seller_action(order, "accept", 1)
        self._seller_action(order, "complete", 2)
        page = self.client.get(f"/seller/{SELLER}/overview").text
        self.assertIn("30,000원", page)
        self.assertIn('class="bar-chart"', page)
        self.assertIn("제주 은갈치", page)


class SalesSummaryTest(unittest.TestCase):
    def _order(self, customer, day, total, status="completed"):
        when = dt.datetime(2026, 10, day, 3, 0, tzinfo=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"status": status, "customer_id_local": customer, "completed_at": when if status == "completed" else None,
                "items": [{"item_id_local": "a", "quantity": 1, "unit_price_minor": total}]}

    def test_windows_change_and_repeat_share(self):
        orders = [self._order("c1", 10, 1000), self._order("c1", 9, 2000), self._order("c2", 2, 4000),
                  self._order("c3", 10, 9999, status="cancelled")]
        s = sales.summary(orders, {"a": "상품 A"}, today=dt.date(2026, 10, 10))
        self.assertEqual((s["revenue_today"], s["revenue_7d"], s["revenue_prev_7d"]), (1000, 3000, 4000))
        self.assertEqual(s["change_7d"], -25)
        self.assertEqual((s["buyers"], s["repeat_buyers"], s["repeat_share"]), (2, 1, 50))
        self.assertEqual(s["top_items"][0]["revenue"], 7000)
        chart = sales.bar_chart(s["series"], s["peak"])
        self.assertEqual(len(chart["bars"]), 14)
        self.assertTrue(chart["bars"][-1]["d"].startswith("M"))


if __name__ == "__main__":
    unittest.main()
