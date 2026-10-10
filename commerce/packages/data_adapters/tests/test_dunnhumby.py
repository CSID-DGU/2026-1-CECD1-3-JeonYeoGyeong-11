"""The Dunnhumby adapter on made-up completejourney-shaped files (rds_writer.py; IDs from 990000)."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.data_adapters.dunnhumby import (
    WEEK2_START, canonical_id, event_week, load_dunnhumby, new_york_time, split_role, utc_text, week_of,
)
from commerce.packages.data_adapters.tests.rds_writer import write_dunnhumby
from commerce.packages.data_adapters.text import catalog_item_text

EST, EDT = timezone(timedelta(hours=-5)), timezone(timedelta(hours=-4))
S1, S2, S3 = "990001", "990002", "990003"
P1, P2, P_NO_CATEGORY, P_COUPON, P_UNKNOWN = "990101", "990102", "990103", "990104", "990199"
PRODUCTS = [
    {"product_id": P1, "department": "TEST DEPT", "product_category": "TEST CAT A", "product_type": "TYPE A",
     "package_size": "1 LB"},
    {"product_id": P2, "department": "TEST DEPT", "product_category": "TEST CAT B"},
    {"product_id": P_NO_CATEGORY, "department": "TEST DEPT"},
    {"product_id": P_COUPON, "department": "TEST DEPT", "product_category": "COUPON/MISC ITEMS"},
]


def new_york(d: date, hour: int = 10, minute: int = 0) -> datetime:
    """2017 New York wall time, daylight time from 2017-03-12 to 2017-11-05 (not at 0-3 a.m. on those two days)."""
    zone = EDT if date(2017, 3, 12) <= d < date(2017, 11, 5) else EST
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=zone)


def in_week(week: int, day: int = 0, hour: int = 10, minute: int = 0) -> datetime:
    """A New York time in a source week: week 1 is Sunday 2017-01-01, week k >= 2 starts on a Monday."""
    start = date(2017, 1, 1) if week == 1 else WEEK2_START + timedelta(days=7 * (week - 2))
    return new_york(start + timedelta(days=day), hour, minute)


def row(household, store, basket, product, when, quantity=1, sales=1.0):
    return {"household": household, "store": store, "basket": basket, "product": product, "quantity": quantity,
            "sales_value": sales, "when": when}


def scenario() -> list[dict]:
    h1, h2, h3, h4, h5 = "990201", "990202", "990203", "990204", "990205"
    return [
        # h1: two train baskets at S1, one at S2 -> S1. Its S2 baskets leave as another store's sales.
        row(h1, S1, "990301", P1, in_week(2)), row(h1, S1, "990301", P1, in_week(2), quantity=2),
        row(h1, S1, "990301", P2, in_week(2)),
        row(h1, S1, "990301", P_NO_CATEGORY, in_week(2), quantity=0),  # fails two steps, counted at the first
        row(h1, S1, "990301", P_COUPON, in_week(2)), row(h1, S1, "990301", P_UNKNOWN, in_week(2)),
        row(h1, S1, "990301", P2, in_week(2), quantity=0),
        row(h1, S1, "990302", P1, in_week(3)), row(h1, S1, "990302", P2, in_week(3), sales=-1.0),
        row(h1, S2, "990303", P2, in_week(4)), row(h1, S2, "990304", P1, in_week(45)),
        row(h1, S1, "990305", P1, in_week(1)), row(h1, S1, "990306", P1, in_week(53, day=1)),
        # h2: one train basket at S3 and one at S2 -> the smaller store ID.
        row(h2, S3, "990311", P1, in_week(5)), row(h2, S2, "990312", P2, in_week(6)),
        # h3: no train-week basket, so no store.
        row(h3, S1, "990321", P1, in_week(1)), row(h3, S1, "990322", P1, in_week(44)),
        # h4: S1 in the train weeks; more baskets at S2 later do not move it.
        row(h4, S1, "990331", P1, in_week(10)), row(h4, S2, "990332", P1, in_week(44)),
        row(h4, S2, "990333", P1, in_week(45)), row(h4, S2, "990334", P1, in_week(46)),
        # h5: the week edge in winter and the first Monday after the clocks moved on 2017-03-12.
        row(h5, S1, "990341", P1, in_week(2, day=6, hour=23, minute=59)),
        row(h5, S1, "990342", P1, in_week(3, hour=0)),
        row(h5, S1, "990343", P1, new_york(date(2017, 3, 13), 0, 30)),
    ]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, rows=None, products=PRODUCTS, **kwargs):
        write_dunnhumby(self.root, scenario() if rows is None else rows, products,
                        **{k: v for k, v in kwargs.items() if k in ("tzone", "week_override")})
        return load_dunnhumby(self.root, **{k: v for k, v in kwargs.items() if k in ("stores", "min_baskets")})


class Weeks(unittest.TestCase):
    def test_monday_weeks_in_new_york_time(self):
        self.assertEqual(week_of(in_week(1, hour=6)), 1)
        self.assertEqual(week_of(in_week(2, day=6, hour=23, minute=59)), 2)
        self.assertEqual(week_of(in_week(3, hour=0)), 3)
        # 00:30 New York on 2017-03-13 is 04:30 UTC after the change, 05:30 UTC would be before it.
        self.assertEqual(week_of(datetime(2017, 3, 13, 4, 30, tzinfo=timezone.utc)), 12)
        self.assertEqual(week_of(datetime(2017, 3, 13, 3, 59, tzinfo=timezone.utc)), 11)
        # 00:30 on 2017-11-06 is 05:30 UTC once standard time is back; 04:59 UTC is still Sunday.
        self.assertEqual(week_of(datetime(2017, 11, 6, 5, 30, tzinfo=timezone.utc)), 46)
        self.assertEqual(week_of(datetime(2017, 11, 6, 4, 59, tzinfo=timezone.utc)), 45)
        self.assertEqual(new_york_time(datetime(2017, 7, 4, 16, 0, tzinfo=timezone.utc)), datetime(2017, 7, 4, 12, 0))

    def test_split_roles(self):
        self.assertEqual([split_role(w) for w in (1, 2, 39, 40, 43, 44, 52, 53)],
                         ["warmup", "train", "train", "validation", "validation", "test", "test", "after"])

    def test_utc_text(self):
        self.assertEqual(utc_text(in_week(2).timestamp()), "2017-01-02T15:00:00Z")

    def test_r_exponent_ids_become_their_integer(self):
        self.assertEqual(canonical_id("990101"), "990101")
        self.assertEqual(canonical_id("9.90101e+05"), "990101")
        with self.assertRaises(ValueError):
            canonical_id("9.901015e+05")
        with self.assertRaises(ValueError):
            canonical_id("99-01")


class Cleaning(Base):
    def test_each_dropped_row_is_counted_at_its_first_failing_step(self):
        r = self.load(min_baskets=2).report
        self.assertEqual(r["rows"], 24)
        self.assertEqual((r["rows_product_unknown"], r["rows_dropped_no_category"]), (1, 2))
        self.assertEqual(r["rows_dropped_quantity_not_positive"], 1)
        self.assertEqual(r["rows_dropped_sales_value_negative"], 1)
        self.assertEqual(r["rows_dropped_coupon_misc"], 1)
        self.assertEqual(r["rows_clean"], 19)

    def test_households_get_their_train_week_primary_store_only(self):
        sample = self.load(min_baskets=2)
        r = sample.report
        self.assertEqual((r["households_clean"], r["households_without_train_basket"]), (5, 1))
        self.assertEqual(r["rows_no_primary_store"], 2)
        self.assertEqual((r["rows_other_store"], r["baskets_other_store"]), (6, 6))
        self.assertEqual((r["rows_kept"], r["baskets_kept"]), (11, 9))
        self.assertEqual(sample.store_train_baskets, {int(S1): 6, int(S2): 1})
        self.assertEqual(sample.roster, {int(S1): 6})
        # No customer is in two sellers.
        sellers_of = {}
        for e in self.load(min_baskets=1).events:
            sellers_of.setdefault(e["customer_id_local"], set()).add(e["seller_id"])
        self.assertTrue(all(len(s) == 1 for s in sellers_of.values()))
        self.assertEqual(sellers_of["dh-hh-990202"], {"dh-store-990002"})

    def test_the_roster_is_compared_with_the_baseline(self):
        r = self.load(min_baskets=2).report
        self.assertEqual((r["roster_stores"], r["roster_not_in_baseline"], r["roster_also_in_baseline"]), (1, 1, 0))
        self.assertEqual(r["baseline_not_in_roster"], r["baseline_roster_stores"])

    def test_the_file_must_be_new_york_time_with_matching_weeks(self):
        with self.assertRaises(ValueError):
            self.load(tzone="UTC")
        with self.assertRaises(ValueError):
            self.load(week_override={0: 3})

    def test_exponent_ids_join_their_product_and_a_collision_stops(self):
        rows = scenario()
        rows[0] = dict(rows[0], product="9.90101e+05")
        sample = self.load(rows, min_baskets=2)
        self.assertEqual(sample.report["ids_rewritten_product"], 1)
        self.assertEqual(sample.report["rows_clean"], 19)
        with self.assertRaises(ValueError):
            self.load(products=PRODUCTS + [dict(PRODUCTS[0], product_id="9.90101e+05")])


class Events(Base):
    def test_baskets_become_purchase_events_of_the_roster(self):
        sample = self.load(min_baskets=2)
        self.assertEqual({e["seller_id"] for e in sample.events}, {"dh-store-990001"})
        self.assertEqual(len(sample.events), 8)
        first = next(e for e in sample.events if e["basket_id_local"] == "dh-b-990301")
        self.assertEqual(first["items"], [{"item_id_local": "dh-p-990101", "quantity_observed": 3},
                                          {"item_id_local": "dh-p-990102", "quantity_observed": 1}])
        self.assertEqual(first["time"], {"kind": "absolute", "value": "2017-01-02T15:00:00Z"})
        self.assertEqual((first["source"], first["seller_partition"], first["order_rank"]),
                         ("dunnhumby", "simulated_independent_store", None))
        self.assertEqual(first["purchase_event_id"], purchase_event_id("dh-store-990001", "dunnhumby", "dh-b-990301"))
        self.assertEqual(sample.report["duplicate_lines_merged"], 1)
        self.assertEqual([sample.report["sample_baskets_" + r] for r in ("warmup", "train", "validation", "test", "after")],
                         [1, 6, 0, 0, 1])
        self.assertEqual([event_week(e) for e in sample.events if e["customer_id_local"] == "dh-hh-990205"], [2, 3, 12])
        keys = [(e["customer_id_local"], e["time"]["value"]) for e in sample.events]
        self.assertEqual(keys, sorted(keys))

    def test_chosen_stores_and_their_catalog_texts(self):
        sample = self.load(stores=[int(S2)], min_baskets=2)
        self.assertEqual([e["basket_id_local"] for e in sample.events], ["dh-b-990312"])
        sample = self.load(min_baskets=2)
        texts = {i["item_id_local"]: catalog_item_text(i) for i in sample.catalog_items}
        self.assertEqual(texts, {"dh-p-990101": "[CAT] TEST CAT A [TYPE] TYPE A [SIZE] 1 LB",
                                 "dh-p-990102": "[CAT] TEST CAT B"})
        self.assertTrue(all(i["seller_id"] == "dh-store-990001" and i["first_listed_at"] is None
                            for i in sample.catalog_items))


if __name__ == "__main__":
    unittest.main()
