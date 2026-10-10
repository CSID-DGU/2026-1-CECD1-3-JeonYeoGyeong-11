import csv
from pathlib import Path
import shutil
import tempfile
import unittest

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.instacart import cart_orders, first_in_cart, load_assignment, load_instacart
from commerce.packages.data_adapters.tests import INSTACART_SMALL
from commerce.packages.data_adapters.text import (
    audit_texts, build_product_text, catalog_item_fields, instacart_fields,
)


def load(data_dir=INSTACART_SMALL):
    return load_instacart(data_dir, load_assignment(INSTACART_SMALL / "assignment.csv"))


class SmallSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = load()
        cls.events = {e["basket_id_local"]: e for e in cls.sample.events}

    def test_report(self):
        report = self.sample.report
        self.assertEqual(report["orders_not_prior"], 2)  # one train, one test order
        self.assertEqual(report["orders_unassigned"], 1)
        self.assertEqual(report["duplicate_lines_merged"], 1)
        self.assertEqual(report["prior_orders"], len(self.sample.events))
        self.assertEqual(report["sellers"], 3)

    def test_only_prior_orders_of_assigned_users(self):
        self.assertNotIn("ic-o-990001005", self.events)  # official train
        self.assertNotIn("ic-o-990002004", self.events)  # official test
        self.assertNotIn("ic-o-990004001", self.events)  # unassigned user
        self.assertNotIn("ic-p-990011", {c["item_id_local"] for c in self.sample.catalog_items})

    def test_ids_follow_the_source_prefixes(self):
        event = self.events["ic-o-990001002"]
        self.assertEqual((event["seller_id"], event["customer_id_local"]),
                         ("ic-client-990101", "ic-user-990001"))
        for e in self.sample.events:
            self.assertEqual(e["purchase_event_id"],
                             purchase_event_id(e["seller_id"], "instacart", e["basket_id_local"]))

    def test_duplicate_line_merged_and_quantity_unobserved(self):
        self.assertEqual(self.events["ic-o-990001002"]["items"],
                         [{"item_id_local": "ic-p-990002", "quantity_observed": None},
                          {"item_id_local": "ic-p-990007", "quantity_observed": None}])
        self.assertEqual({i["quantity_observed"] for e in self.sample.events for i in e["items"]},
                         {None})

    def test_each_customer_stays_at_one_seller(self):
        sellers = {}
        for e in self.sample.events:
            sellers.setdefault(e["customer_id_local"], set()).add(e["seller_id"])
        self.assertTrue(all(len(s) == 1 for s in sellers.values()))

    def test_relative_time_and_censoring_round_trip(self):
        baskets = [basket_from_event(e) for e in self.sample.events
                   if e["customer_id_local"] == "ic-user-990001"]
        visits = customer_visits(baskets)
        self.assertEqual([v.basket.time_value for v in visits], [0.0, 7.0, 37.0, 40.0])
        self.assertEqual([v.gap_censored for v in visits], [False, False, True, False])
        self.assertEqual([v.time_lower_bound for v in visits], [False, False, True, True])
        same_day = [basket_from_event(e) for e in self.sample.events
                    if e["customer_id_local"] == "ic-user-990002"]
        self.assertEqual([v.gap_days for v in customer_visits(same_day)], [None, 0.0, 12.0])

    def test_every_customer_history_is_whole(self):
        by_customer = {}
        for e in self.sample.events:
            by_customer.setdefault(e["customer_id_local"], []).append(basket_from_event(e))
        for baskets in by_customer.values():
            customer_visits(baskets)


class CartOrder(unittest.TestCase):
    def test_first_product_added_to_each_cart(self):
        first = first_in_cart(INSTACART_SMALL, [990001002, 990002002, 990001005])
        # 990001002 lists 990002 twice then 990007; 990001005 is an official train order with no prior lines.
        self.assertEqual(first, {990001002: 990002, 990002002: 990004})

    def test_whole_cart_in_order_with_duplicates_once(self):
        carts = cart_orders(INSTACART_SMALL, [990001002])
        self.assertEqual(carts, {990001002: [990002, 990007]})


class CatalogAndText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = load()
        with open(INSTACART_SMALL / "products.csv", newline="", encoding="utf-8") as f:
            cls.products = {"ic-p-%s" % r["product_id"]: r for r in csv.DictReader(f)}
        with open(INSTACART_SMALL / "aisles.csv", newline="", encoding="utf-8") as f:
            cls.aisles = {r["aisle_id"]: r["aisle"] for r in csv.DictReader(f)}
        with open(INSTACART_SMALL / "departments.csv", newline="", encoding="utf-8") as f:
            cls.departments = {r["department_id"]: r["department"] for r in csv.DictReader(f)}

    def raw_fields(self, item_id):
        p = self.products[item_id]
        aisle, department = self.aisles[p["aisle_id"]], self.departments[p["department_id"]]
        return instacart_fields(p["product_name"], None if aisle == "missing" else aisle,
                                None if department == "missing" else department)

    def test_catalog_is_what_each_seller_sold(self):
        sold = {(e["seller_id"], i["item_id_local"]) for e in self.sample.events for i in e["items"]}
        self.assertEqual({(c["seller_id"], c["item_id_local"]) for c in self.sample.catalog_items},
                         sold)

    def test_catalog_text_equals_raw_text(self):
        # OQ03 on the lab side: raw -> catalog_item -> text gives the raw-source text.
        for item in self.sample.catalog_items:
            self.assertEqual(build_product_text(catalog_item_fields(item)),
                             build_product_text(self.raw_fields(item["item_id_local"])))

    def test_placeholder_labels_are_missing_not_text(self):
        item = next(c for c in self.sample.catalog_items if c["item_id_local"] == "ic-p-990009")
        self.assertIsNone(item["category_path"])
        self.assertEqual(build_product_text(catalog_item_fields(item)), "[NAME] Fixture Mystery Item")

    def test_display_title_is_kept_and_text_is_normalized(self):
        item = next(c for c in self.sample.catalog_items if c["item_id_local"] == "ic-p-990008")
        self.assertIn("\t", item["title_text"])
        self.assertTrue(build_product_text(catalog_item_fields(item)).startswith(
            "[NAME] Fixture Fizz Lime Sparkling Water 12 x 355 ml [AISLE]"))

    def test_audit_finds_identical_texts(self):
        seller = [c for c in self.sample.catalog_items if c["seller_id"] == "ic-client-990101"]
        audit = audit_texts({c["item_id_local"]: catalog_item_fields(c) for c in seller})
        self.assertEqual(audit.empty, 0)
        self.assertGreater(audit.duplicate_text, 0)  # 990001 and 990010 cannot be told apart


class RawErrors(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        for name in ("orders.csv", "order_products__prior.csv", "products.csv", "aisles.csv",
                     "departments.csv"):
            shutil.copy(INSTACART_SMALL / name, self.dir / name)

    def edit_orders(self, order_id, **changes):
        path = self.dir / "orders.csv"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["order_id"] == order_id:
                row.update(changes)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def assertRejected(self):
        with self.assertRaises(ValueError):
            load(self.dir)

    def test_later_order_without_gap(self):
        self.edit_orders("990001003", days_since_prior_order="")
        self.assertRejected()

    def test_first_order_with_gap(self):
        self.edit_orders("990001001", days_since_prior_order="4.0")
        self.assertRejected()

    def test_gap_above_the_cap(self):
        self.edit_orders("990001003", days_since_prior_order="31.0")
        self.assertRejected()

    def test_order_number_hole(self):
        self.edit_orders("990001002", eval_set="train")
        self.assertRejected()

    def test_prior_order_without_lines(self):
        self.edit_orders("990002004", eval_set="prior")
        self.assertRejected()

    def test_missing_column(self):
        (self.dir / "aisles.csv").write_text("aisle_id,name\n990001,x\n", encoding="utf-8")
        self.assertRejected()
