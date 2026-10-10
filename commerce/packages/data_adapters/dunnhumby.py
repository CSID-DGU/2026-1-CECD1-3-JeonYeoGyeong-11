"""Dunnhumby Complete Journey -> purchase_event.v1 and catalog_item.v1 (data.md §2-3).

Input is the completejourney R distribution (transactions.rds, products.rda),
read by rds.py without R or any package outside the lock (OQ12).

Cleaning, in data.md §2 order, each step counted in the report:
product_category present (a product missing from products has none) ->
quantity > 0 -> sales_value >= 0 -> product_category != COUPON/MISC ITEMS.

Sellers come from the train weeks only (data.md §2). A household belongs to
the store where it has the most cleaned baskets in weeks 2-39 (ties: smallest
numeric store_id); a household without a train-week basket gets no store.
Only its baskets at that store are kept (OQ06: a seller sees its own sales;
the rest are counted and dropped), so no customer is in two sellers. The
roster is every store with at least 300 kept train-week baskets.

Time: transaction_timestamp is a POSIXct whose tzone attribute is
America/New_York, so the zone comes from the file, not a guess (OQ12). Events
carry the UTC instant. Weeks start Monday 00:00 New York time and week 2 starts
2017-01-02; the file's own week column is checked against that rule. New York
time follows the US rule in force since 2007 (daylight time from the second
Sunday of March to the first Sunday of November, both at 02:00 local) instead
of zoneinfo, which has no zone data on Windows without tzdata (not in the
lock). Split by week (evaluation.md §2): warmup before 2, train 2-39,
validation 40-43, test 44-52, after past 52.

Quantity is the integer sum of a basket's rows for one product.

IDs are R character columns. A few were written in R's exponent form when the
distribution turned numbers into text (a product "1e+05"); canonical_id turns
them back into the exact integer. Two spellings of one product in products are
an error; in transactions they are the same ID and join as one.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.contracts.types import Payload
from commerce.packages.data_adapters.rds import data_frame, read_rda, read_rds
from commerce.packages.data_adapters.text import normalize_field
from commerce.packages.data_adapters.validation import check_payload

SOURCE = "dunnhumby"
TIMEZONE = "America/New_York"  # transaction_timestamp's tzone attribute
WEEK2_START = date(2017, 1, 2)  # a Monday, local time; earlier days are week 1
TRAIN_WEEKS = (2, 39)
VALIDATION_WEEKS = (40, 43)
TEST_WEEKS = (44, 52)
MIN_STORE_BASKETS = 300
EXCLUDED_CATEGORY = "COUPON/MISC ITEMS"
TRANSACTION_COLUMNS = ("household_id", "store_id", "basket_id", "product_id", "quantity", "sales_value",
                       "week", "transaction_timestamp")
PRODUCT_COLUMNS = ("product_id", "department", "product_category", "product_type", "package_size")
ROSTER = Path(__file__).resolve().parent / "rosters" / "dunnhumby_rb.csv"


_DIGITS = re.compile(r"[0-9]+")
_EXPONENT = re.compile(r"[0-9](\.[0-9]+)?e\+[0-9]+")


def canonical_id(value: str) -> str:
    """A raw ID as plain digits: R's exponent form ("1e+05") becomes the integer it encodes."""
    if _DIGITS.fullmatch(value):
        return value
    if _EXPONENT.fullmatch(value):
        try:
            number = Decimal(value)
        except InvalidOperation:
            raise ValueError("unreadable ID %r" % value) from None
        if number == number.to_integral_value():
            return str(int(number))
    raise ValueError("an ID that is not an integer: %r" % value)


def _canonical_column(values: list[str], name: str, report: dict, *, key: bool = False) -> list[str]:
    distinct = set(values)
    mapping = {v: canonical_id(v) for v in distinct}
    if key and len(set(mapping.values())) != len(distinct):
        raise ValueError("canonical %s IDs collide" % name)
    report["ids_rewritten_" + name] = sum(1 for v, c in mapping.items() if v != c)
    return [mapping[v] for v in values]


def split_role(week: int) -> str:
    if week < TRAIN_WEEKS[0]:
        return "warmup"
    if week <= TRAIN_WEEKS[1]:
        return "train"
    if week <= VALIDATION_WEEKS[1]:
        return "validation"
    return "test" if week <= TEST_WEEKS[1] else "after"


def _sunday_on_or_after(day: date) -> date:
    return day + timedelta(days=(6 - day.weekday()) % 7)


def new_york_time(instant: datetime) -> datetime:
    """Wall-clock New York time (naive) of an aware instant."""
    utc = instant.astimezone(timezone.utc)
    if utc.year < 2007:
        raise ValueError("the US daylight-time rule used here starts in 2007")
    start = datetime.combine(_sunday_on_or_after(date(utc.year, 3, 8)), time(7), tzinfo=timezone.utc)  # 02:00 EST
    end = datetime.combine(_sunday_on_or_after(date(utc.year, 11, 1)), time(6), tzinfo=timezone.utc)  # 02:00 EDT
    hours = -4 if start <= utc < end else -5
    return (utc + timedelta(hours=hours)).replace(tzinfo=None)


def week_of(instant: datetime) -> int:
    """The source's week number of an instant: Monday-start weeks in New York time."""
    local = new_york_time(instant).date()
    if local < WEEK2_START:
        return 1
    return 2 + (local - WEEK2_START).days // 7


def utc_text(seconds: float) -> str:
    instant = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return instant.strftime("%Y-%m-%dT%H:%M:%S") + (".%06dZ" % instant.microsecond if instant.microsecond else "Z")


def seller_id(store: str) -> str:
    return "dh-store-%s" % store


def customer_id(household: str) -> str:
    return "dh-hh-%s" % household


def item_id(product: str) -> str:
    return "dh-p-%s" % product


def basket_id(basket: str) -> str:
    return "dh-b-%s" % basket


def event_week(event: Payload) -> int:
    value = event["time"]["value"]
    return week_of(datetime.fromisoformat(value.replace("Z", "+00:00")))


@dataclass(frozen=True)
class Product:
    department: str | None
    category: str | None
    type: str | None
    size: str | None


@dataclass(frozen=True)
class DunnhumbySample:
    events: list[Payload]  # sorted by (store, household, time, basket)
    catalog_items: list[Payload]  # one per (store, product seen in that store's kept baskets)
    store_train_baskets: dict[int, int]  # every store with a primary household -> kept train-week baskets
    roster: dict[int, int]  # stores with >= min_baskets of them; also the Instacart client sizes (assignment.py)
    report: dict  # counts for the Git-excluded reproduction record


def _text(value) -> str | None:
    return normalize_field(value) if isinstance(value, str) else None


def load_products(path: str | Path, report: dict | None = None) -> dict[str, Product]:
    columns, _, n = data_frame(read_rda(path)["products"])
    absent = [c for c in PRODUCT_COLUMNS if c not in columns]
    if absent:
        raise ValueError("products lacks columns %s" % ", ".join(absent))
    ids = _canonical_column(columns["product_id"], "products", {} if report is None else report, key=True)
    return {ids[i]: Product(_text(columns["department"][i]), _text(columns["product_category"][i]),
                            _text(columns["product_type"][i]), _text(columns["package_size"][i]))
            for i in range(n)}


def load_baseline_roster(path: str | Path = ROSTER) -> set[int]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["store_id"]) for r in csv.DictReader(f)}


def _tzone(attrs: Mapping) -> str | None:
    tz = attrs.get("tzone")
    return tz.data[0] if tz is not None and len(tz.data) else None


def load_dunnhumby(data_dir: str | Path, *, stores: Iterable[int] | None = None,
                   min_baskets: int = MIN_STORE_BASKETS) -> DunnhumbySample:
    """Events and catalog of `stores` (default: the roster this run computes)."""
    data_dir = Path(data_dir)
    report: dict = {"timezone": TIMEZONE}
    products = load_products(data_dir / "products.rda", report)
    tx, attrs, n = data_frame(read_rds(data_dir / "transactions.rds"))
    absent = [c for c in TRANSACTION_COLUMNS if c not in tx]
    if absent:
        raise ValueError("transactions lacks columns %s" % ", ".join(absent))
    if _tzone(attrs["transaction_timestamp"]) != TIMEZONE:
        raise ValueError("transaction_timestamp is not in %s" % TIMEZONE)

    household, store, basket, product = (_canonical_column(tx[c], c.split("_")[0], report)
                                         for c in ("household_id", "store_id", "basket_id", "product_id"))
    quantity, sales, week, stamp = tx["quantity"], tx["sales_value"], tx["week"], tx["transaction_timestamp"]
    report.update(rows=n, products=len(products))

    # Cleaning in data.md order; a row leaves at its first failing step.
    category = [products[p].category if p in products else None for p in product]
    report["rows_product_unknown"] = sum(1 for p in product if p not in products)
    keep = np.array([c is not None for c in category])
    steps = (("no_category", keep),
             ("quantity_not_positive", quantity > 0),
             ("sales_value_negative", sales >= 0),
             ("coupon_misc", np.array([c != EXCLUDED_CATEGORY for c in category])))
    alive = np.ones(n, dtype=bool)
    for name, ok in steps:
        report["rows_dropped_" + name] = int((alive & ~ok).sum())
        alive &= ok
    if (quantity[alive] != np.floor(quantity[alive])).any():
        raise ValueError("a kept quantity is not a whole number")
    report["rows_clean"] = int(alive.sum())

    # The file's week column against the fixed rule, on distinct timestamps.
    weeks_by_stamp = {}
    for i in np.flatnonzero(alive):
        weeks_by_stamp.setdefault(float(stamp[i]), set()).add(int(week[i]))
    mismatched = sum(1 for s, ws in weeks_by_stamp.items()
                     if ws != {week_of(datetime.fromtimestamp(s, tz=timezone.utc))})
    if mismatched:
        raise ValueError("%d timestamps disagree with the week rule" % mismatched)

    clean = np.flatnonzero(alive)
    baskets_of = Counter()  # (household, store) -> train-week baskets
    seen_basket = set()
    for i in clean:
        if TRAIN_WEEKS[0] <= week[i] <= TRAIN_WEEKS[1] and basket[i] not in seen_basket:
            seen_basket.add(basket[i])
            baskets_of[(household[i], store[i])] += 1
    best: dict[str, tuple[int, int]] = {}  # household -> (count, -store)
    for (h, s), count in baskets_of.items():
        key = (count, -int(s))
        if h not in best or key > best[h]:
            best[h] = key
    primary = {h: str(-k[1]) for h, k in best.items()}
    households = {household[i] for i in clean}
    report["households_clean"] = len(households)
    report["households_without_train_basket"] = len(households - set(primary))

    has_store = np.array([household[i] in primary for i in clean], dtype=bool)
    at_primary = np.array([primary.get(household[i]) == store[i] for i in clean], dtype=bool)
    kept = clean[at_primary]
    other = clean[has_store & ~at_primary]
    report["rows_no_primary_store"] = int((~has_store).sum())
    report["rows_other_store"] = len(other)
    report["baskets_other_store"] = len({basket[i] for i in other})
    report["rows_kept"] = len(kept)
    report["baskets_kept"] = len({basket[i] for i in kept})

    train_baskets: dict[int, set] = defaultdict(set)
    for i in kept:
        if TRAIN_WEEKS[0] <= week[i] <= TRAIN_WEEKS[1]:
            train_baskets[int(store[i])].add(basket[i])
    store_train_baskets = {s: len(b) for s, b in sorted(train_baskets.items())}
    roster = {s: c for s, c in store_train_baskets.items() if c >= min_baskets}
    baseline = load_baseline_roster()
    report.update(stores_with_households=len(store_train_baskets), roster_stores=len(roster),
                  roster_min_baskets=min_baskets, roster_train_baskets=sum(roster.values()),
                  baseline_roster_stores=len(baseline), roster_also_in_baseline=len(set(roster) & baseline),
                  roster_not_in_baseline=len(set(roster) - baseline),
                  baseline_not_in_roster=len(baseline - set(roster)))

    chosen = set(roster) if stores is None else {int(s) for s in stores}
    lines: dict[str, dict[str, int]] = defaultdict(dict)  # basket -> product -> quantity
    head: dict[str, tuple[str, str, float]] = {}  # basket -> (store, household, timestamp)
    merged = 0
    for i in kept:
        if int(store[i]) not in chosen:
            continue
        b = basket[i]
        head.setdefault(b, (store[i], household[i], float(stamp[i])))
        if product[i] in lines[b]:
            merged += 1
        lines[b][product[i]] = lines[b].get(product[i], 0) + int(quantity[i])
    report["duplicate_lines_merged"] = merged

    events, seen, roles = [], set(), Counter()
    for b in sorted(head, key=lambda b: (int(head[b][0]), int(head[b][1]), head[b][2], b)):
        s, h, t = head[b]
        seller = seller_id(s)
        seen.update((s, p) for p in lines[b])
        event = {
            "schema_version": "purchase_event.v1", "seller_id": seller, "source": SOURCE,
            "seller_partition": "simulated_independent_store",
            "customer_id_local": customer_id(h), "basket_id_local": basket_id(b),
            "purchase_event_id": purchase_event_id(seller, SOURCE, basket_id(b)),
            "time": {"kind": "absolute", "value": utc_text(t)}, "order_rank": None,
            "items": [{"item_id_local": item_id(p), "quantity_observed": q} for p, q in
                      sorted(lines[b].items(), key=lambda pq: item_id(pq[0]))],
        }
        check_payload("purchase_event.v1", event)
        events.append(event)
        roles[split_role(week_of(datetime.fromtimestamp(t, tz=timezone.utc)))] += 1

    catalog = []
    for s, p in sorted(seen, key=lambda sp: (int(sp[0]), sp[1])):
        catalog.append(catalog_item(seller_id(s), p, products[p]))
    report.update(sample_stores=len(chosen), sample_households=len({e["customer_id_local"] for e in events}),
                  sample_baskets=len(events), sample_catalog_items=len(catalog),
                  **{"sample_baskets_" + r: roles[r] for r in ("warmup", "train", "validation", "test", "after")})
    return DunnhumbySample(events, catalog, store_train_baskets, roster, report)


def catalog_item(seller: str, product: str, info: Product) -> Payload:
    """Dunnhumby has no product name. The display title is product_type (product_category when
    that is missing): a stand-in, never a real unique name (data.md §3). package_size goes in
    description_text and the path is [department, product_category]; text.py rebuilds
    [CAT] [TYPE] [SIZE] from these fields."""
    if info.category is None:
        raise ValueError("a kept product has no product_category")
    item = {
        "schema_version": "catalog_item.v1", "seller_id": seller, "item_id_local": item_id(product),
        "source": SOURCE, "title_text": info.type or info.category, "description_text": info.size,
        "category_path": [c for c in (info.department, info.category) if c is not None],
        # No listing period in the source: every observed product is listed from the start.
        "listing_status": "active", "first_listed_at": None,
    }
    check_payload("catalog_item.v1", item)
    return item
