import copy
from datetime import datetime, timezone
import json
import unittest

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.tests import CONTRACT_FIXTURES, LIVE


def contract_event(name):
    path = CONTRACT_FIXTURES / "purchase_event.v1" / "valid" / name
    return json.loads(path.read_text(encoding="utf-8"))


def live_events():
    return json.loads((LIVE / "purchase_events.json").read_text(encoding="utf-8"))


class FromEvent(unittest.TestCase):
    def test_every_source_example_converts(self):
        for name, kind, rank in (("live_completed.json", "absolute", None),
                                 ("dunnhumby_absolute.json", "absolute", None),
                                 ("instacart_relative_day_quantity_null.json", "relative_day", 7)):
            basket = basket_from_event(contract_event(name))
            self.assertEqual((basket.time_kind, basket.order_rank), (kind, rank), name)

    def test_live_keeps_quantity_and_utc_time(self):
        basket = basket_from_event(live_events()[0])
        self.assertEqual(basket.time_value, datetime(2026, 9, 10, 1, 0, tzinfo=timezone.utc))
        self.assertEqual([(i.item_id_local, i.quantity_observed) for i in basket.items],
                         [("item-1", 2), ("item-5", 1)])

    def test_instacart_quantity_stays_unobserved(self):
        basket = basket_from_event(contract_event("instacart_relative_day_quantity_null.json"))
        self.assertEqual({i.quantity_observed for i in basket.items}, {None})
        self.assertEqual(basket.time_value, 41.0)

    def _rejects(self, event, field_path):
        with self.assertRaises(ContractError) as caught:
            basket_from_event(event)
        self.assertEqual(caught.exception.field_path, field_path)

    def test_source_rules_the_validator_leaves_to_the_consumer(self):
        live = contract_event("live_completed.json")
        wrong_kind = copy.deepcopy(live)
        wrong_kind["time"] = {"kind": "relative_day", "value": 3}
        self._rejects(wrong_kind, "/time/kind")
        ranked = copy.deepcopy(live)
        ranked["order_rank"] = 2
        self._rejects(ranked, "/order_rank")
        unquantified = copy.deepcopy(live)
        unquantified["items"][0]["quantity_observed"] = None
        self._rejects(unquantified, "/items/0/quantity_observed")
        instacart = contract_event("instacart_relative_day_quantity_null.json")
        faked = copy.deepcopy(instacart)
        faked["items"][1]["quantity_observed"] = 1  # no fake 1 for an unobserved quantity
        self._rejects(faked, "/items/1/quantity_observed")
        partition = copy.deepcopy(instacart)
        partition["seller_partition"] = "platform_seller"
        self._rejects(partition, "/seller_partition")

    def test_calendar_invalid_time(self):
        event = contract_event("live_completed.json")
        event["time"]["value"] = "2026-02-30T00:00:00Z"
        self._rejects(event, "/time/value")

    def test_validator_rejections_pass_through(self):
        event = contract_event("live_completed.json")
        event["purchase_event_id"] = "0" * 32
        self._rejects(event, "/purchase_event_id")
        event = contract_event("live_completed.json")
        event["unit_price_minor"] = 100
        with self.assertRaises(ContractError) as caught:
            basket_from_event(event)
        self.assertEqual(caught.exception.code, "UNKNOWN_FIELD")


def relative(customer, rank, day, basket=None):
    basket = basket or "ic-o-%d" % rank
    return basket_from_event({
        "schema_version": "purchase_event.v1", "seller_id": "ic-client-990101",
        "source": "instacart", "seller_partition": "synthetic_partition",
        "customer_id_local": customer, "basket_id_local": basket,
        "purchase_event_id": purchase_event_id("ic-client-990101", "instacart", basket),
        "time": {"kind": "relative_day", "value": day}, "order_rank": rank,
        "items": [{"item_id_local": "ic-p-990001", "quantity_observed": None}],
    })


class Visits(unittest.TestCase):
    def test_absolute_ties_break_by_basket_id(self):
        baskets = [basket_from_event(e) for e in live_events()
                   if e["customer_id_local"] == "buyer-1"]
        visits = customer_visits(reversed(baskets))
        self.assertEqual([v.basket.basket_id_local for v in visits],
                         ["order-1001", "order-1002", "order-1003"])
        self.assertIsNone(visits[0].gap_days)
        self.assertAlmostEqual(visits[1].gap_days, 56.5 / 24)
        self.assertEqual(visits[2].gap_days, 0.0)
        self.assertFalse(any(v.gap_censored or v.time_lower_bound for v in visits))

    def test_relative_gaps_and_censoring(self):
        visits = customer_visits([relative("ic-user-1", r, d)
                                  for r, d in ((3, 37.0), (1, 0.0), (4, 40.0), (2, 7.0))])
        self.assertEqual([v.gap_days for v in visits], [None, 7.0, 30.0, 3.0])
        self.assertEqual([v.gap_censored for v in visits], [False, False, True, False])
        self.assertEqual([v.time_lower_bound for v in visits], [False, False, True, True])

    def test_relative_history_must_be_whole(self):
        with self.assertRaises(ValueError):  # rank 2 missing: 37 could hide a censored gap
            customer_visits([relative("ic-user-1", 1, 0.0), relative("ic-user-1", 3, 37.0)])
        with self.assertRaises(ValueError):
            customer_visits([relative("ic-user-1", 1, 5.0)])
        with self.assertRaises(ValueError):
            customer_visits([relative("ic-user-1", 1, 0.0), relative("ic-user-1", 2, 31.0)])

    def test_one_customer_only(self):
        with self.assertRaises(ValueError):
            customer_visits([relative("ic-user-1", 1, 0.0), relative("ic-user-2", 2, 1.0)])
