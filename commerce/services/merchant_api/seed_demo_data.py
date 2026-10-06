"""Fills one seller's orders.sqlite with a few weeks of synthetic buyer/seller
activity so the screens look like an operating store instead of an empty
shell on first open. Not a wire contract, not test fixtures -- a dev/demo-only
tool (approach "①" from the team's own discussion of how to fake interaction
on a first-time platform: generate a realistic timeline of synthetic activity
rather than needing two live people to click through it).

Usage:
    MERCHANT_DB_PATH=/path/to/orders.sqlite python -m commerce.services.merchant_api.seed_demo_data

Safe to re-run: existing accounts/products are skipped, not duplicated.
Dates are backdated by directly setting orders.created_at/completed_at after
the normal service calls (place_order/transition_order always stamp "now";
there's no override param, and adding one just for a demo seed script isn't
worth it) -- everything else (messages, posts, group-buy timestamps, price
history) takes an explicit date already, so those go through the normal
service layer untouched.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import sys
from pathlib import Path

from commerce.services.merchant_api import accounts_db, accounts_service, orders_db, orders_service, social_db, social_service
from commerce.services.merchant_api.accounts_service import AccountError

_SELLER_ACCOUNT = {
    "username": "owner-1", "display_name": "제주 유기농 농장", "password": "demo-pass-1234",
    "business_reg_no": "123-45-67890", "business_open_date": "20190301", "business_rep_name": "김농부",
}

_PRODUCTS = [
    ("sku-milk", "제주 목장 유기농 우유 1L", 2500, ["식품", "유제품", "우유"]),
    ("sku-egg", "동물복지 유정란 10입", 6900, ["식품", "축산", "계란"]),
    ("sku-fish", "제주 은갈치 (1마리)", 15000, ["수산", "생선"]),
    ("sku-bread", "우리밀 식빵", 3800, ["식품", "베이커리", "빵"]),
    ("sku-coffee", "한라산 핸드드립 원두 200g", 12000, ["식품", "음료", "커피"]),
    ("sku-orange", "제주 노지 감귤 3kg", 18000, ["농산", "과일"]),
]

_CUSTOMERS = [
    ("cust-eunji", "김은지"), ("cust-minjun", "박민준"), ("cust-seoyeon", "이서연"),
    ("cust-jihoon", "최지훈"), ("cust-hana", "정하나"), ("cust-doyoon", "강도윤"),
    ("cust-yuna", "윤유나"), ("cust-sian", "조시안"),
]
_DEMO_PASSWORD = "demo-pass-1234"

_CUSTOMER_MESSAGES = [
    "혹시 오늘 들어온 물건 신선한가요?",
    "다음 주에도 재입고 되나요?",
    "포장 꼼꼼하게 해주셔서 감사했어요!",
    "택배 말고 직접 픽업도 가능한가요?",
    "양이 생각보다 많아서 좋았어요.",
]
_SELLER_REPLIES = [
    "네! 오늘 아침에 들어온 물건이라 아주 신선합니다 :)",
    "네, 매주 화요일마다 재입고되니 참고해주세요!",
    "감사합니다, 다음에도 좋은 상품으로 찾아뵐게요.",
    "픽업 가능합니다! 방문 전에 쪽지 한 번 더 주세요.",
    "넉넉하게 담아드리려고 늘 신경 쓰고 있어요, 감사합니다!",
]

_FEED_POSTS = [
    ("article", "이번 주 감귤 수확 시작했어요", "날씨가 좋아서 당도가 평년보다 높습니다. 많이 주문해주세요!"),
    ("article", "유정란 공급처가 늘었습니다", "동물복지 인증 농장이 하나 더 추가되어 물량이 넉넉해졌어요."),
    ("short_video", "은갈치 손질 과정 소개", "손질부터 포장까지, 믹서 없이 손으로 직접 작업하는 모습입니다."),
    ("article", "명절 선물세트 예약 받습니다", "감귤+유정란 세트로 준비 중이니 쪽지로 문의해주세요."),
]


def _days_ago(n: int, hour: int = 12) -> str:
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=n)
    return when.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def _backdate_order(conn, seller_id: str, order_id: str, created_days_ago: int, completed_days_ago: int | None) -> None:
    created_at = _days_ago(created_days_ago)
    completed_at = _days_ago(completed_days_ago) if completed_days_ago is not None else None
    conn.execute(
        "UPDATE orders SET created_at = ?, completed_at = COALESCE(?, completed_at) WHERE seller_id = ? AND order_id = ?",
        (created_at, completed_at, seller_id, order_id),
    )


def seed(conn, seller_id: str) -> None:
    # Local, not module-level: a fresh Random(42) every call keeps the whole
    # function deterministic even when seed() runs twice in one process
    # (e.g. a test calling it twice) -- a module-level RNG would keep
    # advancing across calls and desync the second call's draws from the
    # first's, breaking the idempotency-key matching below.
    rng = random.Random(42)
    orders_db.ensure_schema(conn)
    social_db.ensure_schema(conn)
    accounts_db.ensure_schema(conn)

    # --- seller account ------------------------------------------------------
    if accounts_db.fetch_seller_account(conn, seller_id, _SELLER_ACCOUNT["username"]) is None:
        accounts_service.signup_seller(conn, seller_id=seller_id, **_SELLER_ACCOUNT)
        print("seller account: %s / %s" % (_SELLER_ACCOUNT["username"], _SELLER_ACCOUNT["password"]))

    # --- catalog --------------------------------------------------------------
    for item_id, title, price, category_path in _PRODUCTS:
        if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id) is None:
            orders_service.register_catalog_item(
                conn, seller_id=seller_id, item_id_local=item_id, title_text=title,
                category_path=category_path, display_price_minor=price,
            )

    # --- customers --------------------------------------------------------------
    for customer_id, display_name in _CUSTOMERS:
        if accounts_db.fetch_customer(conn, seller_id, customer_id) is None:
            accounts_service.signup_customer(
                conn, seller_id=seller_id, customer_id_local=customer_id,
                display_name=display_name, password=_DEMO_PASSWORD,
            )
    print("customer accounts: %d (password for all: %s)" % (len(_CUSTOMERS), _DEMO_PASSWORD))

    # --- orders spread over the last ~30 days --------------------------------
    order_count = 0
    for customer_id, _ in _CUSTOMERS:
        for _ in range(rng.randint(1, 4)):
            item_id, _, price, _ = rng.choice(_PRODUCTS)
            quantity = rng.randint(1, 3)
            created_days_ago = rng.randint(1, 30)
            order = orders_service.place_order(
                conn, seller_id=seller_id, customer_id_local=customer_id,
                idempotency_key="seed-%s-%d" % (customer_id, order_count),
                items=[{"item_id_local": item_id, "quantity": quantity, "unit_price_minor": price}],
                currency="KRW",
            )
            # Draw every random value unconditionally, every run, in the same
            # order -- otherwise a re-run (where place_order returns an
            # already-progressed order instead of a fresh "requested" one)
            # skips some draws and desyncs the RNG sequence from run 1, which
            # then changes a *later* order's randomly-picked item/quantity and
            # breaks its idempotency_key's request_hash match.
            stage = rng.choices(["requested", "accepted", "completed"], weights=[1, 1, 5])[0]
            completed_offset = rng.randint(0, 2)

            if order["status"] == "requested":
                completed_days_ago = None
                if stage in ("accepted", "completed"):
                    orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"],
                                                     action="accept", expected_status_version=1)
                if stage == "completed":
                    orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"],
                                                      action="complete", expected_status_version=2)
                    completed_days_ago = max(created_days_ago - completed_offset, 0)
                _backdate_order(conn, seller_id, order["order_id"], created_days_ago, completed_days_ago)
            order_count += 1
    print("orders: %d" % order_count)

    # --- DM threads -----------------------------------------------------------
    dm_count = 0
    for customer_id, _ in rng.sample(_CUSTOMERS, k=min(5, len(_CUSTOMERS))):
        days_ago = rng.randint(1, 20)
        if social_db.list_thread_messages(conn, seller_id, customer_id):
            continue
        social_db.insert_message(conn, seller_id, "seed-msg-%s-1" % customer_id, customer_id, "customer",
                                  rng.choice(_CUSTOMER_MESSAGES), _days_ago(days_ago, hour=10))
        social_db.insert_message(conn, seller_id, "seed-msg-%s-2" % customer_id, customer_id, "seller",
                                  rng.choice(_SELLER_REPLIES), _days_ago(days_ago, hour=14))
        dm_count += 1
    print("DM threads: %d" % dm_count)

    # --- feed posts -------------------------------------------------------------
    if not social_db.list_posts(conn, seller_id):
        for i, (kind, title, body) in enumerate(_FEED_POSTS):
            social_db.insert_post(conn, seller_id, "seed-post-%d" % i, kind, title, body, None,
                                   _days_ago(len(_FEED_POSTS) * 3 - i * 3))
    print("feed posts: %d" % len(_FEED_POSTS))

    # --- group buys -------------------------------------------------------------
    if not social_db.list_group_buys(conn, seller_id):
        # one still running, short of target
        open_gb_id = "seed-gb-open"
        social_db.insert_group_buy(conn, seller_id, open_gb_id, "sku-orange", 10, 16000,
                                    _days_ago(-7), _days_ago(3))  # deadline 7 days in the future
        for customer_id, _ in _CUSTOMERS[:3]:
            social_db.insert_group_buy_participant(conn, seller_id, open_gb_id, customer_id,
                                                     rng.randint(1, 2), _days_ago(2))
        # one already succeeded, backdated
        done_gb_id = "seed-gb-done"
        social_db.insert_group_buy(conn, seller_id, done_gb_id, "sku-coffee", 5, 11000,
                                    _days_ago(10), _days_ago(15))
        for customer_id, _ in _CUSTOMERS[3:7]:
            social_db.insert_group_buy_participant(conn, seller_id, done_gb_id, customer_id, 2, _days_ago(14))
        social_service.settle_due_group_buys(conn, seller_id=seller_id)
        # the orders settle_due_group_buys just placed default to "now"; backdate
        # them too so they don't look newer than the group-buy they came from.
        for customer_id, _ in _CUSTOMERS[3:7]:
            gb_order = orders_db.fetch_order_by_idempotency_key(conn, seller_id, "group-buy-%s-%s" % (done_gb_id, customer_id))
            if gb_order:
                _backdate_order(conn, seller_id, gb_order["order_id"], 14, 14)
    print("group buys: 2 (one open, one settled)")

    # --- price history (small random walk, last 14 days) ------------------------
    for item_id, _, base_price, _ in _PRODUCTS[:3]:
        if social_db.list_price_history(conn, seller_id, item_id):
            continue
        price = base_price
        for days_ago in range(14, -1, -1):
            price = max(int(price * (1 + rng.uniform(-0.05, 0.05))), int(base_price * 0.7))
            date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).strftime("%Y-%m-%d")
            social_db.upsert_price(conn, seller_id, item_id, date, price)
    print("price history: 14 days for %d items" % len(_PRODUCTS[:3]))


def main() -> None:
    seller_id = os.environ.get("MERCHANT_ID", "seller-1")
    db_path = os.environ.get("MERCHANT_DB_PATH")
    if not db_path:
        print("Set MERCHANT_DB_PATH (and optionally MERCHANT_ID) before running this.", file=sys.stderr)
        raise SystemExit(1)
    conn = orders_db.connect(Path(db_path))
    try:
        with conn:
            seed(conn, seller_id)
    except AccountError as exc:
        print("seed aborted: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
    finally:
        conn.close()
    print("done: %s" % db_path)


if __name__ == "__main__":
    main()
