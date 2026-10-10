"""g2: order -> features -> real recommendation through A's merchant app and B's real runtime.

One seller in process: A's create_app (orders, screens, accounts) over B's SellerRuntime at
CI size (TINY_ARCHITECTURES, FakeText). Products and orders go through A's JSON routes, the
customer signs up and looks at the buyer home, staff open the comparison screen. A base is
installed the way the FL client does, with a release descriptor from C's registry. FL stays
off. Generated synthetic values only (IDs and names made up).
"""
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.packages.recommender import scenario as sc
from commerce.packages.recommender.seller_runtime import SellerRuntime
from commerce.services.fl_coordinator.round_core import ModelRegistry
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app

SELLER = "g2-seller-1"
VARIANT = "text_relation"
ITEMS = [("g2-item-%d" % n, title, price) for n, (title, price) in enumerate([
    ("시험용 우유 1L", 2500), ("시험용 시리얼 500g", 4800), ("시험용 바나나 1송이", 3900), ("시험용 요거트 450g", 4200),
    ("시험용 식빵", 3500), ("시험용 계란 10입", 6500), ("시험용 커피 원두 200g", 11000), ("시험용 감귤 2kg", 9000),
], start=1)]
FALLBACK_NOTE = "아직 모델 추천을 낼 수 없어"


class G2Flow(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        text = sc.FakeText()
        factory = lambda seller_id, features, models: SellerRuntime(  # noqa: E731
            seller_id, features, models, text=text, architectures=sc.TINY_ARCHITECTURES)
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "features.sqlite", model_dir=root / "models")
        self.app = create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=factory))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.runtime = self.app.state.merchant.runtime

    def complete_order(self, customer, item_ids):
        prices = {item_id: price for item_id, _title, price in ITEMS}
        order = self.client.post("/sellers/%s/orders" % SELLER, json={
            "customer_id_local": customer, "idempotency_key": "g2-%s" % uuid.uuid4().hex, "currency": "KRW",
            "items": [{"item_id_local": i, "quantity": 1, "unit_price_minor": prices[i]} for i in item_ids]})
        self.assertEqual(order.status_code, 201, order.text)
        state = order.json()
        for action in ("accept", "complete"):
            step = self.client.post("/sellers/%s/orders/%s/%s" % (SELLER, state["order_id"], action),
                                    json={"expected_status_version": state["status_version"]})
            self.assertEqual(step.status_code, 200, step.text)
            state = step.json()
        self.assertEqual(state["status"], "completed")

    def buyer_home(self):
        page = self.client.get("/buyer/%s/" % SELLER)
        self.assertEqual(page.status_code, 200)
        return page.text

    def install_base(self):
        """What the FL client does with a release: descriptor from the registry, then B install_release."""
        manifest, tensors = sc.random_init_release(self.runtime, VARIANT)
        registry = ModelRegistry()
        descriptor = registry.register("g2-base-0", manifest, tensors)
        self.runtime.install_release(descriptor, registry.manifest("g2-base-0"), registry.tensors("g2-base-0"),
                                     model_variant=VARIANT)

    def test_order_to_features_to_a_model_ranking_and_the_comparison_screen(self):
        for item_id, title, price in ITEMS:
            created = self.client.post("/sellers/%s/catalog-items" % SELLER, json={
                "item_id_local": item_id, "title_text": title, "category_path": ["시험", "식품"],
                "display_price_minor": price})
            self.assertEqual(created.status_code, 201, created.text)
        signup = self.client.post("/buyer/%s/signup" % SELLER, data={
            "customer_id_local": "g2-cust-a", "display_name": "시험 고객", "password": "g2-pass-1234"},
            follow_redirects=False)
        self.assertEqual(signup.status_code, 303)
        for basket in (["g2-item-1", "g2-item-2"], ["g2-item-1", "g2-item-4"], ["g2-item-3", "g2-item-1"]):
            self.complete_order("g2-cust-a", basket)
        for basket in (["g2-item-5", "g2-item-6"], ["g2-item-7"], ["g2-item-5", "g2-item-8"]):
            self.complete_order("g2-cust-b", basket)

        # no shared model yet: B answers with its own fallback, which A labels
        before = self.buyer_home()
        self.assertIn("공유 모델 준비 전", before)

        self.install_base()
        after = self.buyer_home()
        self.assertIn("추천 상품", after)
        self.assertNotIn(FALLBACK_NOTE, after)  # a model ranking: no badge, no fallback note

        staff = self.client.post("/seller/%s/signup" % SELLER, data={
            "username": "g2-owner", "display_name": "시험 매장", "password": "g2-pass-1234",
            "business_reg_no": "123-45-67890", "business_open_date": "20200101", "business_rep_name": "시험"},
            follow_redirects=False)
        self.assertEqual(staff.status_code, 303, staff.text[:200])
        compare = self.client.get("/seller/%s/compare" % SELLER, params={"customer_id_local": "g2-cust-a"})
        self.assertEqual(compare.status_code, 200)
        for arm in ("T-G", "R-G", "T-P", "R-P"):
            self.assertIn(arm, compare.text)
        self.assertIn("기준 모델 g2-base-0", compare.text)  # R-G served by the installed base
        self.assertIn("공유 모델이 아직 설치되지 않음", compare.text)  # text_only has no base here
        self.assertIn("이 매장의 개인화를 아직 실행하지 않음", compare.text)  # P is not filled with G


if __name__ == "__main__":
    unittest.main()
