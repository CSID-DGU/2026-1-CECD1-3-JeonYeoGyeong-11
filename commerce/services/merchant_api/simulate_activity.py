"""Live demo traffic against a running merchant app, through its real screens.

Seeded customers log in, look at their recommendations, and buy -- sometimes
something recommended, sometimes something else, directly or through the
cart -- while the seller accepts and completes orders, which is what hands
purchase events to B. Everything goes over HTTP with the same forms and CSRF
tokens a browser uses, so it exercises the app exactly as people would. A
dev/demo tool only; it needs the demo seed's accounts (seed_demo_data.py).

Usage (app already running, e.g. on 8101 as run_local.py starts merchant-1):
    python -m commerce.services.merchant_api.simulate_activity --base http://127.0.0.1:8101 \\
        --seller merchant-1 --interval 3 --count 20
Stop with Ctrl+C. --count 0 runs until stopped.
"""
from __future__ import annotations

import argparse
import html
import random
import re
import sys
import time
from typing import Optional

import httpx

from commerce.services.merchant_api import seed_demo_data

_CSRF = re.compile(r'name="csrf_token" value="([^"]*)"')
_CHECKOUT_KEY = re.compile(r'name="checkout_key" value="([^"]+)"')
_ITEM_LINK = re.compile(r'href="/buyer/[^/]+/items/([^"]+)"')
_TITLE = re.compile(r'<h1 class="page-title">([^<]+)</h1>')
_ORDER_FORM = re.compile(
    r'action="/seller/[^/]+/orders/([0-9a-f]+)/(accept|complete)".*?name="expected_status_version" value="(\d+)"', re.S)


def _log(who: str, message: str) -> None:
    print("[%s] %s: %s" % (time.strftime("%H:%M:%S"), who, message), flush=True)


class Simulator:
    def __init__(self, base: str, seller_id: str, rng: random.Random, follow_recommendations: float,
                 bulk_customers: int, client_factory=None):
        self.base, self.seller_id, self.rng = base.rstrip("/"), seller_id, rng
        # One cookie jar per customer; tests pass a factory that returns TestClients.
        self._new_client = client_factory or (lambda: httpx.Client(base_url=self.base, follow_redirects=False, timeout=30))
        self.follow = follow_recommendations
        self.customers = [(cid, name) for cid, name in seed_demo_data._CUSTOMERS]
        self.customers += [(cid, name) for cid, name, _ in seed_demo_data.bulk_customers(bulk_customers)]
        self.clients: dict[str, httpx.Client] = {}
        self.seller = self._client()
        r = self.seller.post(f"/seller/{seller_id}/login", data={
            "username": seed_demo_data._SELLER_ACCOUNT["username"], "password": seed_demo_data._SELLER_ACCOUNT["password"]})
        if r.status_code != 303:
            raise SystemExit("seller login failed (%d) -- run seed_demo_data first" % r.status_code)

    def _client(self) -> httpx.Client:
        return self._new_client()

    def _customer(self, customer_id: str) -> Optional[httpx.Client]:
        if customer_id not in self.clients:
            client = self._client()
            r = client.post(f"/buyer/{self.seller_id}/login",
                            data={"customer_id_local": customer_id, "password": seed_demo_data._DEMO_PASSWORD})
            if r.status_code != 303:
                return None  # e.g. bulk customers when the DB was seeded without --bulk
            self.clients[customer_id] = client
        return self.clients[customer_id]

    def shop_once(self) -> None:
        customer_id, name = self.rng.choice(self.customers)
        client = self._customer(customer_id)
        if client is None:
            return
        home = client.get(f"/buyer/{self.seller_id}/").text
        recommended_part, _, catalog_part = home.partition("상품 둘러보기")
        recommended = _ITEM_LINK.findall(recommended_part)
        catalog = _ITEM_LINK.findall(catalog_part) or recommended
        if not catalog:
            _log(name, "살 상품이 없음")
            return
        from_recs = bool(recommended) and self.rng.random() < self.follow
        item = self.rng.choice(recommended if from_recs else catalog)
        page = client.get(f"/buyer/{self.seller_id}/items/{item}").text
        title = html.unescape(_TITLE.search(page).group(1)) if _TITLE.search(page) else item
        token = _CSRF.search(page).group(1)
        quantity = str(self.rng.randint(1, 2))
        source = "추천에서" if from_recs else "둘러보다가"
        if self.rng.random() < 0.5:
            r = client.post(f"/buyer/{self.seller_id}/orders",
                            data={"item_id_local": item, "quantity": quantity, "csrf_token": token})
            _log(name, "%s '%s' %s개 바로 주문 (%d)" % (source, title, quantity, r.status_code))
            return
        client.post(f"/buyer/{self.seller_id}/cart", data={"item_id_local": item, "quantity": quantity, "csrf_token": token})
        extra = self.rng.choice(catalog)
        if extra != item and self.rng.random() < 0.6:
            client.post(f"/buyer/{self.seller_id}/cart", data={"item_id_local": extra, "quantity": "1", "csrf_token": token})
        cart = client.get(f"/buyer/{self.seller_id}/cart").text
        key = _CHECKOUT_KEY.search(cart)
        if key is None:
            return
        r = client.post(f"/buyer/{self.seller_id}/checkout",
                        data={"checkout_key": key.group(1), "csrf_token": _CSRF.search(cart).group(1)})
        _log(name, "%s '%s' 포함 장바구니 주문 (%d)" % (source, title, r.status_code))

    def work_orders(self, limit: int = 3) -> None:
        page = self.seller.get(f"/seller/{self.seller_id}/orders").text
        done = {"accept": 0, "complete": 0}
        for order_id, action, version in _ORDER_FORM.findall(page)[:limit]:
            token = _CSRF.search(page).group(1)
            r = self.seller.post(f"/seller/{self.seller_id}/orders/{order_id}/{action}",
                                 data={"expected_status_version": version, "csrf_token": token})
            if r.status_code == 303:
                done[action] += 1
        if any(done.values()):
            _log("판매자", "주문 수락 %d건, 완료 %d건 (완료된 주문은 추천 모델로 전달)" % (done["accept"], done["complete"]))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Live synthetic buyer/seller traffic against a running merchant app.")
    parser.add_argument("--base", default="http://127.0.0.1:8101")
    parser.add_argument("--seller", default="merchant-1", help="seller_id in the URL (the app's MERCHANT_ID)")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between rounds")
    parser.add_argument("--count", type=int, default=0, help="rounds to run; 0 = until Ctrl+C")
    parser.add_argument("--follow", type=float, default=0.6, help="chance a customer buys something recommended")
    parser.add_argument("--bulk", type=int, default=40, help="how many bulk customers the DB was seeded with")
    parser.add_argument("--seed", type=int, default=None, help="random seed for a repeatable run")
    args = parser.parse_args(argv)
    sim = Simulator(args.base, args.seller, random.Random(args.seed), args.follow, args.bulk)
    rounds = 0
    try:
        while args.count == 0 or rounds < args.count:
            sim.shop_once()
            sim.work_orders()
            rounds += 1
            if args.count == 0 or rounds < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except httpx.HTTPError as exc:
        print("stopped: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
    print("rounds: %d" % rounds)


if __name__ == "__main__":
    main()
