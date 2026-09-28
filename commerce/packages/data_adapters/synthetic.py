"""Hand-invented inputs shaped like Instacart tables and live payloads.

Nothing is copied from the raw data: product names, aisles and departments are
invented, and every numeric ID is 990000 or above, outside the raw ID ranges.
The explicit cases pin rules the adapters must keep (first-order gap missing,
capped 30-day gap, duplicate line, non-prior and unassigned orders, the
'missing' placeholder, size-only and sweetened/unsweetened pairs, identical
texts, Korean, whitespace). A seeded filler from config.json adds volume.

    python -m commerce.packages.data_adapters.synthetic commerce/packages/data_adapters/tests/fixtures
"""
import csv
import json
from pathlib import Path
import random
import sys

from commerce.packages.contracts.ids import purchase_event_id

DEPARTMENTS = [(990001, "chilled goods"), (990002, "dry goods shelf"), (990003, "missing")]
AISLES = [(990001, "plant drinks"), (990002, "cultured dairy"), (990003, "dairy drinks"),
          (990004, "tea biscuits"), (990005, "fizzy drinks"), (990006, "missing")]
# (product_id, product_name, aisle_id, department_id)
PRODUCTS = [
    (990001, "Fixture Farm Oat Drink Original 1 L", 990001, 990001),
    (990002, "Fixture Farm Oat Drink Original 2 L", 990001, 990001),  # size only
    (990003, "Fixture Farm Almond Drink Unsweetened 946 ml", 990001, 990001),
    (990004, "Fixture Farm Almond Drink Sweetened 946 ml", 990001, 990001),
    (990005, "Fixture Farm Greek Style Yogurt 1.5 kg", 990002, 990001),
    (990006, "Fixture Farm Whole Milk 1/2 gal", 990003, 990001),
    (990007, "Fixture Bakehouse Café Crème Biscuits 200 g", 990004, 990002),
    (990008, "  Fixture   Fizz  Lime\tSparkling Water 12 x 355 ml ", 990005, 990002),
    (990009, "Fixture Mystery Item", 990006, 990003),  # placeholder aisle and department
    (990010, "Fixture Farm Oat Drink Original 1 L", 990001, 990001),  # same text as 990001
    (990011, "Fixture Pantry Rice Crackers", 990004, 990002),  # only an unassigned user buys it
]
CLIENTS = [990101, 990102, 990103]
# user_id -> client_id; user 990004 is left out on purpose.
ASSIGNED = {990001: 990101, 990002: 990101, 990003: 990102}
# (order_id, user_id, eval_set, order_number, days_since_prior_order, product lines)
ORDERS = [
    (990001001, 990001, "prior", 1, "", [990001, 990005]),
    (990001002, 990001, "prior", 2, "7.0", [990002, 990002, 990007]),  # duplicate line
    (990001003, 990001, "prior", 3, "30.0", [990003, 990008]),  # capped gap
    (990001004, 990001, "prior", 4, "3.0", [990001, 990009]),
    (990001005, 990001, "train", 5, "5.0", [990004]),  # official train order, not read
    (990002001, 990002, "prior", 1, "", [990004]),
    (990002002, 990002, "prior", 2, "0.0", [990004, 990006]),  # same-day reorder
    (990002003, 990002, "prior", 3, "12.0", [990010]),
    (990002004, 990002, "test", 4, "6.0", []),  # official test order, no lines
    (990003001, 990003, "prior", 1, "", [990001]),
    (990003002, 990003, "prior", 2, "30.0", [990007]),
    (990004001, 990004, "prior", 1, "", [990011]),  # unassigned user
]

LIVE_SELLER = "seller-a"
# (item_id_local, title_text, description_text, category_path, listing_status, first_listed_at)
LIVE_ITEMS = [
    ("item-1", "유기농 우유 1L", "무항생제 인증 목장의 원유만 사용", ["식품", "유제품", "우유"],
     "active", "2026-09-01T00:00:00Z"),
    ("item-2", "일반 우유 1L", None, ["식품", "유제품", "우유"], "active", "2026-09-01T00:00:00Z"),
    ("item-3", "무가당 두유 950ml", None, ["식품", "음료", "두유"], "active", "2026-09-01T00:00:00Z"),
    ("item-4", "가당 두유 950ml", None, ["식품", "음료", "두유"], "active", "2026-09-01T00:00:00Z"),
    ("item-5", "통밀 식빵 450g", "국산 통밀 100%", ["식품", "베이커리", "식빵"], "active",
     "2026-09-02T00:00:00Z"),
    ("item-6", "통밀 식빵 900g", "국산 통밀 100%", ["식품", "베이커리", "식빵"], "active",
     "2026-09-02T00:00:00Z"),
    ("item-7", "그릭 요거트 Greek Yogurt 0.5kg", "", None, "active", "2026-09-03T00:00:00Z"),
    ("item-8", "제주  감귤 주스 1/2 L", None, ["식품", "음료"], "inactive", "2026-09-20T00:00:00Z"),
]
# (order_id, customer, completed_at, [(item, quantity)])
LIVE_ORDERS = [
    ("order-1001", "buyer-1", "2026-09-10T01:00:00Z", [("item-1", 2), ("item-5", 1)]),
    ("order-1003", "buyer-1", "2026-09-12T09:30:00Z", [("item-2", 1)]),  # same time as 1002
    ("order-1002", "buyer-1", "2026-09-12T09:30:00Z", [("item-3", 1)]),
    ("order-1004", "buyer-2", "2026-09-15T12:00:00Z", [("item-6", 3), ("item-7", 1)]),
]


def _filler(config: dict) -> tuple[dict[int, int], list[tuple]]:
    rng = random.Random(config["seed"])
    pool = [p for p, *_ in PRODUCTS if p != 990011]
    assigned, orders = {}, []
    for k in range(config["filler_users"]):
        user = 990011 + k
        assigned[user] = rng.choice(CLIENTS)
        for number in range(1, rng.randint(1, config["filler_orders_max"]) + 1):
            gap = "" if number == 1 else "%.1f" % rng.randint(0, 30)
            basket = rng.sample(pool, rng.randint(1, config["filler_basket_max"]))
            orders.append((user * 1000 + number, user, "prior", number, gap, basket))
    return assigned, orders


def _write_csv(path: Path, header: tuple, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    newline="\n")


def write_fixtures(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    rng = random.Random(config["seed"] + 1)  # dow/hour only; the filler has its own stream
    filler_assigned, filler_orders = _filler(config)
    orders = ORDERS + filler_orders

    ic = out_dir / "instacart_small"
    ic.mkdir(parents=True, exist_ok=True)
    _write_csv(ic / "departments.csv", ("department_id", "department"), DEPARTMENTS)
    _write_csv(ic / "aisles.csv", ("aisle_id", "aisle"), AISLES)
    _write_csv(ic / "products.csv", ("product_id", "product_name", "aisle_id", "department_id"),
               PRODUCTS)
    _write_csv(ic / "assignment.csv", ("user_id", "client_id"),
               sorted({**ASSIGNED, **filler_assigned}.items()))
    _write_csv(ic / "orders.csv",
               ("order_id", "user_id", "eval_set", "order_number", "order_dow",
                "order_hour_of_day", "days_since_prior_order"),
               [(o, u, s, n, rng.randint(0, 6), rng.randint(0, 23), g) for o, u, s, n, g, _ in orders])
    lines, bought = [], set()
    for order, user, eval_set, _, _, products in orders:
        if eval_set != "prior":
            continue
        for position, product in enumerate(products, start=1):
            lines.append((order, product, position, int((user, product) in bought)))
            bought.add((user, product))
    _write_csv(ic / "order_products__prior.csv",
               ("order_id", "product_id", "add_to_cart_order", "reordered"), lines)

    live = out_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    _write_json(live / "catalog_items.json", [
        {"schema_version": "catalog_item.v1", "seller_id": LIVE_SELLER, "item_id_local": item,
         "source": "live", "title_text": title, "description_text": description,
         "category_path": path, "listing_status": status, "first_listed_at": listed}
        for item, title, description, path, status, listed in LIVE_ITEMS])
    _write_json(live / "purchase_events.json", [
        {"schema_version": "purchase_event.v1", "seller_id": LIVE_SELLER, "source": "live",
         "seller_partition": "platform_seller", "customer_id_local": customer,
         "basket_id_local": order,
         "purchase_event_id": purchase_event_id(LIVE_SELLER, "live", order),
         "time": {"kind": "absolute", "value": completed_at}, "order_rank": None,
         "items": [{"item_id_local": i, "quantity_observed": q} for i, q in items]}
        for order, customer, completed_at, items in LIVE_ORDERS])


if __name__ == "__main__":
    write_fixtures(sys.argv[1])
