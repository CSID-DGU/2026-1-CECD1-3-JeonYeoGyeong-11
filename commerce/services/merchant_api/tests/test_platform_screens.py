"""The platform screens added for the demo, over HTTP: the seller's post studio and the
buyer's explore feed, buyer-proposed group buys, reviews only after a purchase,
wishlist, buyer cancel, product edit reaching B, media serving, and the chatbot
(rules mode, an injected LLM, CSRF, guests not stored)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from commerce.services.merchant_api import chatbot, orders_db, orders_service, session, sns_db, social_db
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import ScriptedRuntime

SELLER = "seller-1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class PlatformScreensTest(unittest.TestCase):
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
        env = mock.patch.dict("os.environ", {"CHATBOT_MODE": "rules"})
        env.start()
        self.addCleanup(env.stop)
        self._seller()
        for item, title, price in (("sku-fish", "제주 은갈치", 15000), ("sku-egg", "유정란 10구", 6900)):
            r = self.client.post(f"/seller/{SELLER}/products", data={
                "title_text": title, "display_price_minor": str(price), "item_id_local": item, "csrf_token": self._seller_csrf()})
            self.assertEqual(r.status_code, 303, r.text)

    # --- helpers -------------------------------------------------------------------

    def _seller(self):
        r = self.client.post(f"/seller/{SELLER}/signup", data={
            "username": "owner", "display_name": "매장", "password": "pw-1234",
            "business_reg_no": "1234567890", "business_open_date": "20200101", "business_rep_name": "홍길동"})
        self.assertEqual(r.status_code, 303, r.text)

    def _seller_csrf(self):
        return session.csrf_token(self.client.cookies.get(session.seller_cookie(SELLER)))

    def _customer(self, customer_id="cust-1"):
        self.client.cookies.delete(session.customer_cookie(SELLER))
        r = self.client.post(f"/buyer/{SELLER}/signup", data={
            "customer_id_local": customer_id, "display_name": "고객", "password": "pw-1234"})
        self.assertEqual(r.status_code, 303, r.text)

    def _csrf(self):
        return session.csrf_token(self.client.cookies.get(session.customer_cookie(SELLER)))

    def _db(self):
        conn = orders_db.connect(self.db_path)
        self.addCleanup(conn.close)
        return conn

    def _completed_order(self, customer_id, item="sku-fish", key="k"):
        conn = self._db()
        order = orders_service.place_order(conn, seller_id=SELLER, customer_id_local=customer_id, idempotency_key=key,
                                           items=[{"item_id_local": item, "quantity": 1, "unit_price_minor": 15000}], currency="KRW")
        orders_service.transition_order(conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        orders_service.transition_order(conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2)

    def _post(self, caption="오늘 갈치 #제철", files=None):
        data = {"kind": "article", "caption": caption, "item_ids": ["sku-fish"], "csrf_token": self._seller_csrf()}
        return self.client.post(f"/seller/{SELLER}/feed", data=data, files=files or [])

    # --- SNS -------------------------------------------------------------------------

    def test_seller_posts_and_buyer_sees_likes_and_comments(self):
        r = self._post(files=[("files", ("photo.png", PNG, "image/png"))])
        self.assertEqual(r.status_code, 303, r.text)
        post = sns_db.list_posts(self._db(), SELLER)[0]
        image = self.client.get(f"/media/{SELLER}/{post['media'][0]['name']}")
        self.assertEqual((image.status_code, image.headers["content-type"]), (200, "image/png"))

        self._customer()
        feed = self.client.get(f"/buyer/{SELLER}/feed")
        self.assertEqual(feed.status_code, 200)
        self.assertIn(post["post_id"], feed.text)
        self.assertIn(f"/buyer/{SELLER}/feed?tag=%EC%A0%9C%EC%B2%A0", self.client.get(f"/buyer/{SELLER}/posts/{post['post_id']}").text)

        base = f"/buyer/{SELLER}/posts/{post['post_id']}"
        self.assertEqual(self.client.post(base + "/like", data={"csrf_token": self._csrf()}).status_code, 303)
        self.assertEqual(self.client.post(base + "/comments", data={"body": "맛있겠다", "csrf_token": self._csrf()}).status_code, 303)
        detail = self.client.get(base).text
        self.assertIn("맛있겠다", detail)
        conn = self._db()
        self.assertEqual(sns_db.like_counts(conn, SELLER)[post["post_id"]], 1)
        self.assertEqual(sns_db.view_counts(conn, SELLER)[post["post_id"]], 2)

    def test_like_without_csrf_is_refused(self):
        self._post()
        pid = sns_db.list_posts(self._db(), SELLER)[0]["post_id"]
        self._customer()
        r = self.client.post(f"/buyer/{SELLER}/posts/{pid}/like", data={"csrf_token": "forged"})
        self.assertEqual(r.status_code, 403)

    def test_bad_upload_rerenders_the_studio_with_a_message(self):
        r = self._post(files=[("files", ("x.svg", b"<svg/>", "image/svg+xml"))])
        self.assertEqual(r.status_code, 400)
        self.assertIn("사진", r.text)
        self.assertEqual(sns_db.list_posts(self._db(), SELLER), [])

    def test_media_route_refuses_paths_and_other_sellers(self):
        self.assertEqual(self.client.get(f"/media/{SELLER}/..%2Forders.sqlite").status_code, 404)
        self.assertEqual(self.client.get("/media/other-seller/gen-abc.svg").status_code, 404)

    def test_seller_can_edit_and_delete_a_post(self):
        self._post()
        pid = sns_db.list_posts(self._db(), SELLER)[0]["post_id"]
        r = self.client.post(f"/seller/{SELLER}/posts/{pid}", data={"caption": "수정했어요 #new", "item_ids": ["sku-egg"],
                                                                  "csrf_token": self._seller_csrf()})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(sns_db.fetch_post(self._db(), SELLER, pid)["hashtags"], ["new"])
        self.client.post(f"/seller/{SELLER}/posts/{pid}/delete", data={"csrf_token": self._seller_csrf()})
        self.assertIsNone(sns_db.fetch_post(self._db(), SELLER, pid))

    # --- group buys (buyer proposes) ---------------------------------------------------------

    def test_buyer_proposes_others_join_and_it_settles_into_orders(self):
        self._customer("cust-1")
        r = self.client.post(f"/buyer/{SELLER}/group-buys", data={
            "item_id_local": "sku-fish", "target_quantity": "3", "quantity": "1", "days": "7", "message": "같이 사요",
            "csrf_token": self._csrf()})
        self.assertEqual(r.headers["location"], f"/buyer/{SELLER}/group-buys?ok=proposed")
        gb = social_db.list_group_buys(self._db(), SELLER)[0]
        self.assertEqual((gb["proposer_customer_id"], gb["unit_price_minor"]), ("cust-1", 13500))  # 10% off

        again = self.client.post(f"/buyer/{SELLER}/group-buys", data={
            "item_id_local": "sku-fish", "target_quantity": "3", "quantity": "1", "days": "7", "csrf_token": self._csrf()})
        self.assertIn("e=DUPLICATE_EVENT", again.headers["location"])

        self._customer("cust-2")
        r = self.client.post(f"/buyer/{SELLER}/group-buys/{gb['group_buy_id']}/join", data={"quantity": "2", "csrf_token": self._csrf()})
        self.assertIn("ok=succeeded", r.headers["location"])
        orders = orders_db.list_orders(self._db(), SELLER)
        self.assertEqual(sorted(o["customer_id_local"] for o in orders), ["cust-1", "cust-2"])
        self.assertTrue(all(o["items"][0]["unit_price_minor"] == 13500 for o in orders))

    def test_seller_settings_and_close(self):
        r = self.client.post(f"/seller/{SELLER}/store-settings", data={
            "group_discount_pct": "20", "group_min_target": "5", "store_intro": "안녕하세요", "csrf_token": self._seller_csrf()})
        self.assertEqual(r.status_code, 303)
        self._customer()
        low = self.client.post(f"/buyer/{SELLER}/group-buys", data={
            "item_id_local": "sku-egg", "target_quantity": "3", "quantity": "1", "days": "7", "csrf_token": self._csrf()})
        self.assertIn("e=INVALID_TYPE", low.headers["location"])  # below the seller's minimum
        self.client.post(f"/buyer/{SELLER}/group-buys", data={
            "item_id_local": "sku-egg", "target_quantity": "5", "quantity": "1", "days": "7", "csrf_token": self._csrf()})
        gb = social_db.list_group_buys(self._db(), SELLER)[0]
        self.assertEqual(gb["unit_price_minor"], 5520)
        self.client.post(f"/seller/{SELLER}/group-buys/{gb['group_buy_id']}/close", data={"csrf_token": self._seller_csrf()})
        closed = social_db.fetch_group_buy(self._db(), SELLER, gb["group_buy_id"])
        self.assertEqual((closed["status"], closed["closed_reason"]), ("failed", "closed_by_seller"))

    # --- shop extras ---------------------------------------------------------------------------

    def test_review_needs_a_completed_purchase(self):
        self._customer()
        denied = self.client.post(f"/buyer/{SELLER}/items/sku-fish/reviews", data={"rating": "5", "body": "최고", "csrf_token": self._csrf()})
        self.assertIn("review=denied", denied.headers["location"])
        self._completed_order("cust-1")
        ok = self.client.post(f"/buyer/{SELLER}/items/sku-fish/reviews", data={"rating": "4", "body": "신선해요", "csrf_token": self._csrf()})
        self.assertNotIn("denied", ok.headers["location"])
        self.assertIn("신선해요", self.client.get(f"/buyer/{SELLER}/items/sku-fish").text)
        r = self.client.post(f"/seller/{SELLER}/products/sku-fish/reviews/cust-1/reply",
                             data={"reply": "감사합니다", "csrf_token": self._seller_csrf()})
        self.assertEqual(r.status_code, 303)
        self.assertIn("감사합니다", self.client.get(f"/buyer/{SELLER}/items/sku-fish").text)

    def test_wishlist_toggles(self):
        self._customer()
        self.client.post(f"/buyer/{SELLER}/wishlist/sku-egg", data={"csrf_token": self._csrf(), "back": "wishlist"})
        self.assertIn("유정란", self.client.get(f"/buyer/{SELLER}/wishlist").text)
        self.client.post(f"/buyer/{SELLER}/wishlist/sku-egg", data={"csrf_token": self._csrf(), "back": "wishlist"})
        self.assertNotIn("유정란 10구</", self.client.get(f"/buyer/{SELLER}/wishlist").text)

    def test_buyer_cancels_only_own_unaccepted_order(self):
        self._customer("cust-1")
        conn = self._db()
        order = orders_service.place_order(conn, seller_id=SELLER, customer_id_local="cust-1", idempotency_key="c1",
                                           items=[{"item_id_local": "sku-egg", "quantity": 1, "unit_price_minor": 6900}], currency="KRW")
        self._customer("cust-2")
        r = self.client.post(f"/buyer/{SELLER}/orders/{order['order_id']}/cancel", data={"csrf_token": self._csrf()})
        self.assertIn("e=NOT_FOUND", r.headers["location"])
        self.client.cookies.delete(session.customer_cookie(SELLER))
        self.client.post(f"/buyer/{SELLER}/login", data={"customer_id_local": "cust-1", "password": "pw-1234"})
        r = self.client.post(f"/buyer/{SELLER}/orders/{order['order_id']}/cancel", data={"csrf_token": self._csrf()})
        self.assertIn("ok=cancelled", r.headers["location"])
        self.assertEqual(orders_db.fetch_order(self._db(), SELLER, order["order_id"])["status"], "cancelled")

    def test_product_edit_reaches_b_as_a_newer_catalog_version(self):
        r = self.client.post(f"/seller/{SELLER}/products/sku-egg", data={
            "title_text": "동물복지 유정란 10구", "display_price_minor": "7200", "category": "식품 > 계란",
            "description_text": "방사 사육", "listing_status": "active", "csrf_token": self._seller_csrf()})
        self.assertEqual(r.status_code, 303)
        seq, item = self.runtime.catalog["sku-egg"]
        self.assertEqual((seq, item["title_text"], item["category_path"], item["description_text"]),
                         (2, "동물복지 유정란 10구", ["식품", "계란"], "방사 사육"))

    def test_search_and_category_filter_on_home(self):
        self._customer()
        page = self.client.get(f"/buyer/{SELLER}/", params={"q": "갈치"}).text
        self.assertIn("제주 은갈치", page)
        browse = page.split("상품 둘러보기")[1]
        self.assertIn("검색 결과 1개", browse)
        self.assertNotIn("유정란 10구", browse)

    # --- chatbot -------------------------------------------------------------------------------

    def test_rules_chatbot_answers_and_adds_to_cart_for_a_member(self):
        self._customer()
        r = self.client.post(f"/buyer/{SELLER}/chat", data={"question": "갈치 있어요?", "csrf_token": self._csrf()},
                             headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("15,000원", r.json()["reply"])
        self.assertEqual(r.json()["mode"], "rules")
        r = self.client.post(f"/buyer/{SELLER}/chat", data={"question": "유정란 담아 줘", "csrf_token": self._csrf()},
                             headers={"Accept": "application/json"})
        self.assertTrue(r.json()["actions"])
        cart = orders_service.get_cart(self._db(), seller_id=SELLER, customer_id_local="cust-1")
        self.assertEqual([i["item_id_local"] for i in cart["lines"]], ["sku-egg"])
        self.assertEqual(len(chatbot.history(self._db(), SELLER, "cust-1")), 4)

    def test_member_chat_needs_csrf_and_guest_chat_is_not_stored(self):
        self._customer()
        self.assertEqual(self.client.post(f"/buyer/{SELLER}/chat", data={"question": "안녕", "csrf_token": "x"}).status_code, 403)
        self.client.cookies.delete(session.customer_cookie(SELLER))
        r = self.client.post(f"/buyer/{SELLER}/chat", data={"question": "공동구매 있어요?"}, headers={"Accept": "application/json"})
        self.assertIn("공동구매", r.json()["reply"])
        self.assertEqual(self._db().execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_llm_answer_is_used_and_a_failure_falls_back_to_rules(self):
        conn = self._db()
        seen = {}

        def fake_llm(tools, question, past, brand):
            seen["products"] = tools.search_products("갈치")["count"]
            return "갈치는 15,000원이에요."
        reply = chatbot.ask(conn, seller_id=SELLER, customer_id=None, question="갈치?", brand="오이", llm=fake_llm)
        self.assertEqual((reply.text, reply.mode, seen["products"]), ("갈치는 15,000원이에요.", "llm", 1))

        def broken(*a):
            raise RuntimeError("network down")
        reply = chatbot.ask(conn, seller_id=SELLER, customer_id=None, question="갈치 있어요?", brand="오이", llm=broken)
        self.assertEqual(reply.mode, "rules-fallback")
        self.assertIn("은갈치", reply.text)

    def test_customer_data_tools_are_off_by_default(self):
        names = {t["name"] for t in chatbot.tool_definitions(chatbot.customer_data_allowed())}
        self.assertNotIn("my_recent_orders", names)
        self.assertTrue(all(t["strict"] and t["input_schema"]["additionalProperties"] is False
                            for t in chatbot.tool_definitions(True)))


if __name__ == "__main__":
    unittest.main()
