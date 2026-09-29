from collections import Counter
import unittest

from commerce.evaluation.gci_protocol import relation_visits, seller_units, unit_starts
from commerce.packages.recommender.tests import SELLER, item_id, visits_of

# One customer, ten visits of three items; the cart order is the order written here.
BASKETS = [[3 * v + 1, 3 * v + 2, 3 * v + 3] for v in range(10)]


def customer(baskets=BASKETS):
    visits = {"c": visits_of("990500", baskets)}
    carts = {rank: [990000 + n for n in basket] for rank, basket in enumerate(baskets, start=1)}
    return visits, carts


class Units(unittest.TestCase):
    def test_starts_every_five_and_shifted_by_two(self):
        self.assertEqual(unit_starts(12), [0, 2, 5, 7])
        self.assertEqual(unit_starts(4), [])

    def test_four_items_in_and_the_fifth_as_label(self):
        visits, carts = customer()
        grouped, _, _ = seller_units(SELLER, visits, carts, seed=0)
        sequence = [item_id(n) for basket in BASKETS for n in basket]
        examples = sorted((e for ex in grouped.values() for e in ex), key=lambda e: e.target_position)
        self.assertEqual([e.target_position - 5 for e in examples], unit_starts(len(sequence)))
        for e in examples:
            p = e.target_position - 5
            self.assertEqual([v.items[0] for v in e.history], sequence[p:p + 4])
            self.assertEqual(e.target_items, frozenset({sequence[p + 4]}))
            self.assertEqual(e.prior_counts, dict(Counter(sequence[:p + 4])))

    def test_a_label_can_share_its_order_with_inputs(self):
        visits, carts = customer()
        grouped, _, _ = seller_units(SELLER, visits, carts, seed=0)
        first = next(e for ex in grouped.values() for e in ex if e.target_position == 5)
        # Items 1..4 come from orders 1 and 2; label 5 sits in order 2 with input item 4.
        self.assertEqual(first.target_items, frozenset({item_id(5)}))
        self.assertTrue(first.target_basket_id.endswith("-2"))


class Split(unittest.TestCase):
    def test_fixed_by_seed_and_near_the_shares(self):
        visits, carts = customer([[n] for n in range(1, 1001)])
        roles = lambda seed: {e.target_position: r for (r, _), ex in seller_units(SELLER, visits, carts, seed)[0].items()
                              for e in ex}
        self.assertEqual(roles(0), roles(0))
        self.assertNotEqual(roles(0), roles(1))
        share = Counter(roles(0).values())
        total = sum(share.values())
        self.assertTrue(0.06 < share["test"] / total < 0.14)
        self.assertTrue(0.05 < share["validation"] / total < 0.13)

    def test_relations_never_read_a_held_back_label_order(self):
        visits, carts = customer()
        grouped, held, labels = seller_units(SELLER, visits, carts, seed=0)
        for role in ("validation", "test"):
            for e in grouped[(role, 0)]:
                self.assertIn(e.target_basket_id, held)
        kept = relation_visits(visits, held)
        self.assertFalse({v.basket.basket_id_local for v in kept["c"]} & held)
        self.assertEqual(sum(labels.values()), len(grouped[("train", 0)]))


if __name__ == "__main__":
    unittest.main()
