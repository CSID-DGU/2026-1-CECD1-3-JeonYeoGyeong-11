"""Instacart prior orders -> purchase_event.v1 and catalog_item.v1 (data.md §2-3).

Only eval_set=prior is read; the official train/test orders are not this
project's split. Time is each customer's own cumulative days_since_prior_order
from 0, which becomes a lower bound once a capped 30 appears (baskets.py
derives that bit). Quantity is never observed, so it stays None.

The seller assignment (user_id -> client_id) is an input. Building it from the
train period only (data.md §2) is a separate step.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
import unicodedata

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.contracts.types import Payload
from commerce.packages.data_adapters.baskets import INSTACART_GAP_CAP_DAYS
from commerce.packages.data_adapters.text import normalize_field
from commerce.packages.data_adapters.validation import check_payload

SOURCE = "instacart"
COLUMNS = {
    "orders.csv": ("order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"),
    "order_products__prior.csv": ("order_id", "product_id"),
    "products.csv": ("product_id", "product_name", "aisle_id", "department_id"),
    "aisles.csv": ("aisle_id", "aisle"),
    "departments.csv": ("department_id", "department"),
}
# Instacart's own placeholder for an unknown aisle/department. It is not text.
MISSING_LABEL = "missing"


def seller_id(client_id: int) -> str:
    return "ic-client-%d" % client_id


def customer_id(user_id: int) -> str:
    return "ic-user-%d" % user_id


def item_id(product_id: int) -> str:
    return "ic-p-%d" % product_id


def basket_id(order_id: int) -> str:
    return "ic-o-%d" % order_id


@dataclass(frozen=True)
class InstacartSample:
    events: list[Payload]  # sorted by (client, user, order_number)
    catalog_items: list[Payload]  # one per (client, product seen in that client's orders)
    report: dict[str, int]  # row counts and exclusions for the reproduction record


def load_assignment(path: str | Path) -> dict[int, int]:
    """user_id,client_id CSV. One customer's orders all go to one client."""
    assignment: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            user = int(r["user_id"])
            if user in assignment:
                raise ValueError("a user is assigned twice")
            assignment[user] = int(r["client_id"])
    return assignment


def _rows(data_dir: Path, name: str) -> Iterator[dict[str, str]]:
    with open(data_dir / name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        absent = [c for c in COLUMNS[name] if c not in (reader.fieldnames or ())]
        if absent:
            raise ValueError("%s lacks columns %s" % (name, ", ".join(absent)))
        yield from reader


def _label(value: str) -> str | None:
    text = normalize_field(value)
    return None if text is None or text == MISSING_LABEL else text


def _category_path(department: str | None, aisle: str | None) -> list[str] | None:
    if department is None and aisle is not None:
        raise ValueError("an aisle without a department cannot be written as category_path")
    return [p for p in (department, aisle) if p is not None] or None


def load_instacart(data_dir: str | Path, assignment: Mapping[int, int]) -> InstacartSample:
    data_dir = Path(data_dir)
    report = dict.fromkeys((
        "order_rows", "orders_not_prior", "orders_unassigned", "prior_orders",
        "line_rows", "lines_kept", "duplicate_lines_merged",
    ), 0)

    aisles = {int(r["aisle_id"]): _label(r["aisle"]) for r in _rows(data_dir, "aisles.csv")}
    departments = {int(r["department_id"]): _label(r["department"])
                   for r in _rows(data_dir, "departments.csv")}
    products = {int(r["product_id"]): (r["product_name"], aisles[int(r["aisle_id"])],
                                       departments[int(r["department_id"])])
                for r in _rows(data_dir, "products.csv")}

    # order_id -> (user_id, order_number, raw days_since_prior_order)
    orders: dict[int, tuple[int, int, str]] = {}
    for r in _rows(data_dir, "orders.csv"):
        report["order_rows"] += 1
        if r["eval_set"] != "prior":
            report["orders_not_prior"] += 1
        elif int(r["user_id"]) not in assignment:
            report["orders_unassigned"] += 1
        else:
            orders[int(r["order_id"])] = (int(r["user_id"]), int(r["order_number"]),
                                          r["days_since_prior_order"])
    report["prior_orders"] = len(orders)

    lines: dict[int, set[int]] = {order: set() for order in orders}
    for r in _rows(data_dir, "order_products__prior.csv"):
        report["line_rows"] += 1
        order, product = int(r["order_id"]), int(r["product_id"])
        if order not in lines:
            continue
        if product not in products:
            raise ValueError("an order line names a product missing from products.csv")
        if product in lines[order]:
            report["duplicate_lines_merged"] += 1
        else:
            lines[order].add(product)
            report["lines_kept"] += 1

    day = _cumulative_days(orders)
    events, seen = [], set()
    for order in sorted(orders, key=lambda o: (assignment[orders[o][0]], orders[o][0], orders[o][1])):
        user, number, _ = orders[order]
        if not lines[order]:
            raise ValueError("a prior order has no product lines")
        client = assignment[user]
        seen.update((client, product) for product in lines[order])
        event = {
            "schema_version": "purchase_event.v1",
            "seller_id": seller_id(client), "source": SOURCE,
            "seller_partition": "synthetic_partition",
            "customer_id_local": customer_id(user), "basket_id_local": basket_id(order),
            "purchase_event_id": purchase_event_id(seller_id(client), SOURCE, basket_id(order)),
            "time": {"kind": "relative_day", "value": day[order]},
            "order_rank": number,
            "items": [{"item_id_local": i, "quantity_observed": None}
                      for i in sorted(item_id(p) for p in lines[order])],
        }
        check_payload("purchase_event.v1", event)
        events.append(event)

    catalog = []
    for client, product in sorted(seen):
        name, aisle, department = products[product]
        if normalize_field(name) is None:
            raise ValueError("a purchased product has no product_name")
        item = {
            "schema_version": "catalog_item.v1",
            "seller_id": seller_id(client), "item_id_local": item_id(product), "source": SOURCE,
            # The display title keeps the raw name (NFC only); text.py builds the encoder input.
            "title_text": unicodedata.normalize("NFC", name), "description_text": None,
            "category_path": _category_path(department, aisle),
            # No listing period in the source: every observed product is listed from the start.
            "listing_status": "active", "first_listed_at": None,
        }
        check_payload("catalog_item.v1", item)
        catalog.append(item)

    report.update(users=len({u for u, _, _ in orders.values()}),
                  sellers=len({assignment[u] for u, _, _ in orders.values()}),
                  catalog_items=len(catalog))
    return InstacartSample(events, catalog, report)


def _cumulative_days(orders: Mapping[int, tuple[int, int, str]]) -> dict[int, float]:
    by_user: dict[int, list[tuple[int, str, int]]] = {}
    for order, (user, number, gap) in orders.items():
        by_user.setdefault(user, []).append((number, gap, order))
    day = {}
    for history in by_user.values():
        history.sort()
        if [number for number, _, _ in history] != list(range(1, len(history) + 1)):
            raise ValueError("a user's prior order_number is not 1..n")
        total = 0.0
        for number, gap, order in history:
            if number == 1:
                if gap.strip():
                    raise ValueError("a first order has days_since_prior_order")
            else:
                if not gap.strip():
                    raise ValueError("a later order lacks days_since_prior_order")
                value = float(gap)
                if not 0 <= value <= INSTACART_GAP_CAP_DAYS:
                    raise ValueError("days_since_prior_order outside 0..30")
                total += value
            day[order] = total
    return day
