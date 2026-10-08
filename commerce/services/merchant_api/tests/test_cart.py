"""Cart: adding merges quantities, prices come from the catalog at checkout,
checkout is one multi-item order and idempotent per rendered cart, items no
longer on sale are left out, and carts are per customer."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import cart_db, orders_db
from commerce.services.merchant_api import orders_service as svc
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import FakeRecommenderRuntime

SELLER = "seller-1"


class CartServiceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.conn = orders_db.connect(Path(tmp.name) / "orders.sqlite")
        cart_db.ensure_schema(self.conn)
        self.addCleanup(self.conn.close)
        for item_id, title, price in (("sku-milk", "우유", 2500), ("sku-egg", "계란", 6900)):
            svc.register_catalog_item(self.conn, seller_id=SELLER, item_id_local=item_id, title_text=title,
                                      display_price_minor=price)

    def _add(self, item, qty, customer="cust-1"):
        svc.add_to_cart(self.conn, seller_id=SELLER, customer_id_local=customer, item_id_local=item, quantity=qty)

    def _cart(self, customer="cust-1"):
        return svc.get_cart(self.conn, seller_id=SELLER, customer_id_local=customer)

    def test_adding_the_same_item_merges_and_totals_use_catalog_price(self):
        self._add("sku-milk", 2)
        self._add("sku-milk", 1)
        self._add("sku-egg", 1)
        cart = self._cart()
        self.assertEqual([(l["item_id_local"], l["quantity"]) for l in cart["lines"]], [("sku-milk", 3), ("sku-egg", 1)])
        self.assertEqual(cart["total"], 3 * 2500 + 6900)

    def test_quantity_zero_removes_and_out_of_range_is_rejected(self):
        self._add("sku-milk", 2)
        with self.assertRaises(ContractError):
            self._add("sku-milk", 0)
        with self.assertRaises(ContractError):
            svc.set_cart_quantity(self.conn, seller_id=SELLER, customer_id_local="cust-1", item_id_local="sku-milk", quantity=100)
        svc.set_cart_quantity(self.conn, seller_id=SELLER, customer_id_local="cust-1", item_id_local="sku-milk", quantity=0)
        self.assertEqual(self._cart()["lines"], [])

    def test_merged_quantity_is_capped(self):
        self._add("sku-milk", 60)
        self._add("sku-milk", 60)
        self.assertEqual(self._cart()["lines"][0]["quantity"], svc.MAX_CART_QUANTITY)

    def test_unknown_item_cannot_be_added(self):
        with self.assertRaises(ContractError):
            self._add("sku-none", 1)

    def test_checkout_is_one_order_at_catalog_price_and_empties_the_cart(self):
        self._add("sku-milk", 2)
        self._add("sku-egg", 1)
        # the price changes after the item was added; checkout uses today's price
        svc.register_catalog_item(self.conn, seller_id=SELLER, item_id_local="sku-milk", title_text="우유",
                                  display_price_minor=2700)
        order = svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")
        self.assertEqual(order["items"], [
            {"item_id_local": "sku-egg", "quantity": 1, "unit_price_minor": 6900},
            {"item_id_local": "sku-milk", "quantity": 2, "unit_price_minor": 2700},
        ])
        self.assertEqual(self._cart()["lines"], [])

    def test_resubmitting_the_same_checkout_returns_the_same_order(self):
        self._add("sku-milk", 1)
        first = svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")
        again = svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")
        self.assertEqual(first["order_id"], again["order_id"])
        self.assertEqual(len(orders_db.list_orders(self.conn, SELLER)), 1)

    def test_another_customer_cannot_reuse_a_checkout_key(self):
        self._add("sku-milk", 1)
        svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")
        self._add("sku-egg", 1, customer="cust-2")
        with self.assertRaises(ContractError):
            svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-2", checkout_key="cart-a")

    def test_empty_cart_cannot_check_out(self):
        with self.assertRaises(ContractError):
            svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")

    def test_item_taken_off_sale_is_left_out(self):
        self._add("sku-milk", 1)
        self._add("sku-egg", 1)
        svc.register_catalog_item(self.conn, seller_id=SELLER, item_id_local="sku-egg", title_text="계란",
                                  display_price_minor=6900, listing_status="inactive")
        cart = self._cart()
        self.assertEqual([l["item_id_local"] for l in cart["lines"]], ["sku-milk"])
        self.assertEqual([r["item_id_local"] for r in cart["unavailable"]], ["sku-egg"])
        order = svc.checkout_cart(self.conn, seller_id=SELLER, customer_id_local="cust-1", checkout_key="cart-a")
        self.assertEqual([i["item_id_local"] for i in order["items"]], ["sku-milk"])

    def test_carts_are_per_customer(self):
        self._add("sku-milk", 1, customer="cust-1")
        self.assertEqual(self._cart("cust-2")["lines"], [])


class CartScreenTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "features.sqlite",
                                    model_dir=root / "models", merchant_db_path=root / "orders.sqlite")
        app = create_app(settings, context_factory=lambda s: build_context(
            s, runtime_factory=lambda *a: FakeRecommenderRuntime(SELLER)))
        self.client = TestClient(app, follow_redirects=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        conn = orders_db.connect(root / "orders.sqlite")
        svc.register_catalog_item(conn, seller_id=SELLER, item_id_local="sku-milk", title_text="우유", display_price_minor=2500)
        conn.close()
        self.client.post(f"/buyer/{SELLER}/signup", data={"customer_id_local": "cust-1", "display_name": "고객", "password": "pw-1234"})

    def _token(self, path):
        return re.search(r'name="csrf_token" value="([^"]+)"', self.client.get(path).text).group(1)

    def test_add_from_product_page_then_checkout_from_cart(self):
        token = self._token(f"/buyer/{SELLER}/items/sku-milk")
        self.assertEqual(self.client.post(f"/buyer/{SELLER}/cart", data={"item_id_local": "sku-milk", "quantity": "1"}).status_code,
                         403, "CSRF is required")
        r = self.client.post(f"/buyer/{SELLER}/cart", data={"item_id_local": "sku-milk", "quantity": "2", "csrf_token": token})
        self.assertEqual(r.status_code, 303)
        page = self.client.get(f"/buyer/{SELLER}/cart").text
        self.assertIn("5,000원", page)
        key = re.search(r'name="checkout_key" value="([^"]+)"', page).group(1)
        token = self._token(f"/buyer/{SELLER}/cart")
        r = self.client.post(f"/buyer/{SELLER}/checkout", data={"checkout_key": key, "csrf_token": token})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("주문이 접수되었습니다", r.text)
        r = self.client.post(f"/buyer/{SELLER}/checkout", data={"checkout_key": key, "csrf_token": token})
        self.assertEqual(r.status_code, 200, "a double submit shows the same order, it does not fail")
        self.assertEqual(self.client.get(f"/buyer/{SELLER}/orders").text.count("우유 × 2"), 1)

    def test_cart_requires_login(self):
        self.client.get(f"/buyer/{SELLER}/logout")
        r = self.client.get(f"/buyer/{SELLER}/cart")
        self.assertEqual(r.status_code, 303)


if __name__ == "__main__":
    unittest.main()
