from collections import Counter
from datetime import datetime, timedelta, timezone
import unittest

from commerce.evaluation.dunnhumby_protocol import (
    WeekCounts, before_week, shared_text_rows, visit_week, week_examples,
)
from commerce.packages.data_adapters.baskets import LocalBasket, LocalItem, customer_visits

# Monday 2017-01-02 local time starts week 2 (dunnhumby.WEEK2_START); noon UTC is morning in New York.
WEEK2 = datetime(2017, 1, 2, 17, 0, tzinfo=timezone.utc)
SELLER = "dh-store-990367"


def basket(customer, n, week, items):
    when = WEEK2 + timedelta(weeks=week - 2, hours=n)
    return LocalBasket("dunnhumby", SELLER, "physical_store", customer, "b-%s-%d" % (customer, n),
                       "e-%s-%d" % (customer, n), "absolute", when, None,
                       tuple(LocalItem(i, 1) for i in sorted(items)))


def visits(plan):
    """{customer: [(week, items), ...]} -> {customer: ordered visits}."""
    return {c: customer_visits(basket(c, n, w, items) for n, (w, items) in enumerate(rows))
            for c, rows in plan.items()}


class Weeks(unittest.TestCase):
    def setUp(self):
        self.visits = visits({
            "h1": [(1, {"a"}), (3, {"a", "b"}), (10, {"b"}), (39, {"c"}), (41, {"a"}), (44, {"b", "c"}), (53, {"a"})],
            "h2": [(2, {"c"}), (5, {"c"}), (40, {"a"})],
        })

    def test_the_week_is_the_adapter_week(self):
        self.assertEqual([visit_week(v) for v in self.visits["h1"]], [1, 3, 10, 39, 41, 44, 53])

    def test_a_target_takes_its_week_role_and_warmup_or_later_weeks_are_never_targets(self):
        counts = Counter()
        grouped = week_examples(self.visits, counts)
        roles = {(r, w) for r, w in grouped}
        # h1 has three earlier visits from its 4th visit on (MIN_HISTORY = 3): weeks 39, 41, 44, 53.
        self.assertEqual(roles, {("train", 39), ("validation", 41), ("test", 44)})
        self.assertEqual(counts["targets_outside_weeks"], 1)  # week 53
        # h2 never has three earlier visits.
        self.assertTrue(all(ex.customer_id_local == "h1" for exs in grouped.values() for ex in exs))

    def test_relations_and_popularity_read_only_weeks_before_the_target_week(self):
        visible = before_week(self.visits, 41)
        self.assertEqual([visit_week(v) for v in visible["h1"]], [1, 3, 10, 39])
        self.assertEqual([visit_week(v) for v in visible["h2"]], [2, 5, 40])
        counts = WeekCounts(self.visits)
        self.assertEqual(counts.counts(41), Counter({"a": 3, "b": 2, "c": 3}))
        self.assertEqual(counts.counts(2), Counter({"a": 1}))

    def test_items_sharing_a_text_are_marked(self):
        self.assertEqual(shared_text_rows(["milk", "bread", "milk", "eggs"]), {0, 2})


if __name__ == "__main__":
    unittest.main()
