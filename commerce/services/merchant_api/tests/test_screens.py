"""Screen routes over HTTP: CSRF on logged-in forms, server-side prices, the
recommendation badge for both A's baseline and B's own fallback, a recommender
failure not breaking the buyer page, and the four-arm comparison screen."""
from __future__ import annotations

import importlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import ComparisonArm, ComparisonResult
from commerce.services.merchant_api import orders_db, session
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import ScriptedRuntime

SELLER = "seller-1"
_CSRF_RE = re.compile(r'name="csrf_token" value="([^"]*)"')


def _recommendation(items, *, fallback_reason=None, model_version="harex.R_lm.v1-r500"):
    return {
        "schema_version": "recommendation.v1", "seller_id": SELLER, "customer_id_local": "cust-1",
        "as_of": "2026-10-06T00:00:00.000000Z", "model_version": model_version,
        "score_semantics": "next_purchase", "horizon_days": None,
        "is_cold_start": fallback_reason is not None, "fallback_reason": fallback_reason,
        "items": [{"item_id_local": i, "score": 1.0 - n / 10} for n, i in enumerate(items)],
    }


class ScreenTest(unittest.TestCase):
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

    def _csrf(self, path: str) -> str:
        page = self.client.get(path)
        self.assertEqual(page.status_code, 200, page.text)
        match = _CSRF_RE.search(page.text)
        self.assertIsNotNone(match, "no csrf field on %s" % path)
        return match.group(1)

    def _seller_with_product(self):
        r = self.client.post(f"/seller/{SELLER}/signup", data={
            "username": "owner", "display_name": "매장", "password": "pw-1234",
            "business_reg_no": "1234567890", "business_open_date": "20200101", "business_rep_name": "홍길동",
        })
        self.assertEqual(r.status_code, 303, r.text)
        token = self._csrf(f"/seller/{SELLER}/products")
        r = self.client.post(f"/seller/{SELLER}/products", data={
            "title_text": "은갈치", "display_price_minor": "15000", "item_id_local": "sku-fish", "csrf_token": token,
        })
        self.assertEqual(r.status_code, 303, r.text)

    def _customer(self, customer_id="cust-1"):
        r = self.client.post(f"/buyer/{SELLER}/signup", data={
            "customer_id_local": customer_id, "display_name": "고객", "password": "pw-1234",
        })
        self.assertEqual(r.status_code, 303, r.text)

    # --- branding ----------------------------------------------------------------

    def test_brand_name_comes_from_one_constant_on_both_apps(self):
        from commerce.services.merchant_api import main
        self._seller_with_product()
        self.assertIn(main.BRAND_NAME, self.client.get(f"/seller/{SELLER}/overview").text)
        self.assertIn(main.BRAND_NAME, self.client.get(f"/buyer/{SELLER}/").text)

    def test_thumbnail_emoji_follows_the_most_specific_category(self):
        from commerce.services.merchant_api.main import _product_emoji
        self.assertEqual(_product_emoji(["식품", "유제품", "우유"]), "🥛")
        self.assertEqual(_product_emoji(["수산", "생선"]), "🐟")
        self.assertIsNone(_product_emoji(None))
        self.assertIsNone(_product_emoji(["기타"]))

    # --- CSRF ------------------------------------------------------------------

    def test_seller_form_without_csrf_token_is_rejected(self):
        self._seller_with_product()
        r = self.client.post(f"/seller/{SELLER}/products", data={
            "title_text": "x", "display_price_minor": "1", "item_id_local": "sku-x",
        })
        self.assertEqual(r.status_code, 403)
        r = self.client.post(f"/seller/{SELLER}/products", data={
            "title_text": "x", "display_price_minor": "1", "item_id_local": "sku-x", "csrf_token": "forged",
        })
        self.assertEqual(r.status_code, 403)

    def test_buyer_form_needs_the_token_of_its_own_session(self):
        self._seller_with_product()
        self._customer()
        stale = self._csrf(f"/buyer/{SELLER}/items/sku-fish")
        self.client.get(f"/buyer/{SELLER}/logout")
        r = self.client.post(f"/buyer/{SELLER}/login", data={"customer_id_local": "cust-1", "password": "pw-1234"})
        self.assertEqual(r.status_code, 303)
        r = self.client.post(f"/buyer/{SELLER}/orders", data={"item_id_local": "sku-fish", "quantity": "1", "csrf_token": stale})
        self.assertEqual(r.status_code, 403, "a token from the previous session must not work")
        fresh = self._csrf(f"/buyer/{SELLER}/items/sku-fish")
        r = self.client.post(f"/buyer/{SELLER}/orders", data={"item_id_local": "sku-fish", "quantity": "1", "csrf_token": fresh})
        self.assertEqual(r.status_code, 200, r.text)

    # --- prices are the server's -------------------------------------------------

    def test_order_price_comes_from_catalog_not_form(self):
        self._seller_with_product()
        self._customer()
        token = self._csrf(f"/buyer/{SELLER}/items/sku-fish")
        r = self.client.post(f"/buyer/{SELLER}/orders", data={
            "item_id_local": "sku-fish", "quantity": "2", "unit_price_minor": "1", "csrf_token": token,
        })
        self.assertEqual(r.status_code, 200, r.text)
        conn = orders_db.connect(self.db_path)
        self.addCleanup(conn.close)
        (order,) = orders_db.list_orders(conn, SELLER)
        self.assertEqual(order["items"], [{"item_id_local": "sku-fish", "quantity": 2, "unit_price_minor": 15000}])

    def test_order_for_unknown_item_or_bad_quantity_is_rejected(self):
        self._seller_with_product()
        self._customer()
        token = self._csrf(f"/buyer/{SELLER}/items/sku-fish")
        r = self.client.post(f"/buyer/{SELLER}/orders", data={"item_id_local": "sku-none", "quantity": "1", "csrf_token": token})
        self.assertEqual(r.status_code, 404)
        r = self.client.post(f"/buyer/{SELLER}/orders", data={"item_id_local": "sku-fish", "quantity": "0", "csrf_token": token})
        self.assertEqual(r.status_code, 422)

    # --- recommendation slot -----------------------------------------------------

    def test_badge_for_a_baseline_while_b_is_a_stub(self):
        self._seller_with_product()
        page = self.client.get(f"/buyer/{SELLER}/")
        self.assertIn("임시 · 추천 모델 미연결", page.text)

    def test_badge_for_b_fallback_and_none_for_a_model_ranking(self):
        self._seller_with_product()
        self.runtime.recommendation = _recommendation(["sku-fish"], fallback_reason="no_shared_model",
                                                      model_version="popularity.local")
        self.assertIn("공유 모델 준비 전", self.client.get(f"/buyer/{SELLER}/").text)
        self.runtime.recommendation = _recommendation(["sku-fish"])
        page = self.client.get(f"/buyer/{SELLER}/").text
        self.assertIn("추천 상품", page)
        self.assertNotIn('class="badge muted"', page.split("상품 둘러보기")[0])

    def test_recommender_failure_does_not_break_buyer_home(self):
        self._seller_with_product()
        self.runtime.recommendation = ContractError("NOT_FOUND", "/candidate_item_ids/0")
        page = self.client.get(f"/buyer/{SELLER}/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("임시 · 추천 모델 미연결", page.text)

    def test_recommendation_request_leaves_candidates_to_b_and_drops_unknown_items(self):
        self._seller_with_product()
        self.runtime.recommendation = _recommendation(["sku-gone", "sku-fish"])
        page = self.client.get(f"/buyer/{SELLER}/").text
        self.assertIsNone(self.runtime.requests[-1]["candidate_item_ids"])
        self.assertNotIn("sku-gone", page)

    # --- comparison screen -------------------------------------------------------

    def test_compare_screen_says_so_while_b_cannot_compare(self):
        self._seller_with_product()
        page = self.client.get(f"/seller/{SELLER}/compare", params={"customer_id_local": "cust-1"})
        self.assertEqual(page.status_code, 200)
        self.assertIn("비교할 수 없습니다", page.text)

    def test_compare_screen_shows_four_arms_and_never_fills_p_with_g(self):
        self._seller_with_product()
        g = _recommendation(["sku-fish"])
        self.runtime.comparison = ComparisonResult(
            comparison_id="c1", as_of="2026-10-06T00:00:00.000000Z", feature_snapshot_id="fe-3",
            candidate_set_hash="ab" * 32, arms=(
                ComparisonArm("T-G", "text_only", "global", True, None, "t-r500", None, g),
                ComparisonArm("R-G", "text_relation", "global", True, None, "r-r500", None, g),
                ComparisonArm("T-P", "text_only", "personalized", False, "insufficient_data", "t-r500", None, None),
                ComparisonArm("R-P", "text_relation", "personalized", False, "personalization_not_ready", "r-r500", None, None),
            ))
        page = self.client.get(f"/seller/{SELLER}/compare", params={"customer_id_local": "cust-1"}).text
        for arm in ("T-G", "R-G", "T-P", "R-P"):
            self.assertIn(arm, page)
        self.assertEqual(page.count("<li>은갈치</li>"), 2, "only the two G arms list items")
        self.assertIn("개인화에 필요한 이력이 부족함", page)
        self.assertIn("fe-3", page)
        self.assertEqual(len([r for r in self.runtime.requests if "candidate_item_ids" in r]), 1,
                         "one compare_local call, not one request per arm")

    def test_compare_screen_requires_seller_login(self):
        r = self.client.get(f"/seller/{SELLER}/compare")
        self.assertEqual(r.status_code, 303)
        self.assertTrue(r.headers["location"].endswith("/login"))


    # --- dashboard and cancel ----------------------------------------------------

    def test_overview_says_whether_a_real_model_answers(self):
        self._seller_with_product()
        self.assertIn("미연결", self.client.get(f"/seller/{SELLER}/overview").text)
        self.runtime.recommendation = _recommendation([], fallback_reason="no_shared_model", model_version="popularity.local")
        self.assertIn("연결됨 · 모델 준비 중", self.client.get(f"/seller/{SELLER}/overview").text)
        self.runtime.recommendation = _recommendation(["sku-fish"])
        self.assertIn("실제 모델로 추천 중", self.client.get(f"/seller/{SELLER}/overview").text)

    def test_seller_can_cancel_a_requested_order_with_csrf(self):
        self._seller_with_product()
        conn = orders_db.connect(self.db_path)
        self.addCleanup(conn.close)
        from commerce.services.merchant_api import orders_service
        order = orders_service.place_order(conn, seller_id=SELLER, customer_id_local="cust-9", idempotency_key="k1",
                                           items=[{"item_id_local": "sku-fish", "quantity": 1, "unit_price_minor": 15000}],
                                           currency="KRW")
        path = f"/seller/{SELLER}/orders/{order['order_id']}/cancel"
        self.assertEqual(self.client.post(path, data={"expected_status_version": "1"}).status_code, 403)
        token = self._csrf(f"/seller/{SELLER}/orders")
        r = self.client.post(path, data={"expected_status_version": "1", "csrf_token": token})
        self.assertEqual(r.status_code, 303, r.text)
        self.assertEqual(orders_service.get_order(conn, seller_id=SELLER, order_id=order["order_id"])["status"], "cancelled")


class StartupRedeliveryTest(unittest.TestCase):
    """Orders written while no runtime was attached (the demo seed) reach B when the app starts."""

    def _app(self, root: Path, runtime):
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "features.sqlite",
                                    model_dir=root / "models", merchant_db_path=root / "orders.sqlite")
        return create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: runtime))

    def _seeded_root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        from commerce.services.merchant_api import seed_demo_data
        import contextlib, io
        conn = orders_db.connect(root / "orders.sqlite")
        with contextlib.redirect_stdout(io.StringIO()):
            seed_demo_data.seed(conn, SELLER)
        conn.close()
        return root

    def test_seeded_orders_and_catalog_are_delivered_at_startup(self):
        root = self._seeded_root()
        runtime = ScriptedRuntime(SELLER)
        with TestClient(self._app(root, runtime)):
            pass
        conn = orders_db.connect(root / "orders.sqlite")
        self.addCleanup(conn.close)
        completed = conn.execute("SELECT COUNT(*) FROM orders WHERE status = 'completed'").fetchone()[0]
        self.assertGreater(completed, 0)
        self.assertEqual(len(runtime.ingested_events), completed)
        self.assertEqual(len(runtime.catalog), 6)
        pending = conn.execute("SELECT COUNT(*) FROM outbox WHERE status = 'pending'").fetchone()[0]
        self.assertEqual(pending, 0)

    def test_app_still_starts_when_the_recommender_is_down(self):
        from commerce.services.merchant_api.tests.fakes import AlwaysFailingRuntime
        root = self._seeded_root()
        with TestClient(self._app(root, AlwaysFailingRuntime(SELLER))) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
        conn = orders_db.connect(root / "orders.sqlite")
        self.addCleanup(conn.close)
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM outbox WHERE status = 'pending'").fetchone()[0], 0,
                           "nothing is marked delivered when B never acknowledged")


class SessionSecretTest(unittest.TestCase):
    def tearDown(self):
        importlib.reload(session)  # back to a random per-process key

    def test_merchant_secret_keeps_sessions_valid_across_restart(self):
        with mock.patch.dict(os.environ, {"MERCHANT_SECRET": "stable-secret"}):
            importlib.reload(session)
            token = session.sign({"role": "customer"})
            csrf = session.csrf_token(token)
            importlib.reload(session)  # a "restart" with the same secret
            self.assertEqual(session.unsign(token)["role"], "customer")
            self.assertTrue(session.check_csrf(token, csrf))

    def test_without_merchant_secret_a_restart_drops_sessions(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERCHANT_SECRET", None)
            importlib.reload(session)
            token = session.sign({"role": "customer"})
            importlib.reload(session)
            self.assertIsNone(session.unsign(token))


if __name__ == "__main__":
    unittest.main()
