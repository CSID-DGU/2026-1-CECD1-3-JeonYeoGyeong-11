import random
import unittest

from commerce.packages.data_adapters.instacart import split_role
from commerce.packages.recommender.examples import K_ITEMS, L_VISITS, customer_examples, cut_items
from commerce.packages.recommender.replay import SellerReplay, progress_bucket, visible_prefix
from commerce.packages.recommender.tests import visits_of


class Split(unittest.TestCase):
    def test_roles_follow_floor_seventy_and_eighty(self):
        self.assertEqual([split_role(k, 10) for k in range(1, 11)], ["train"] * 7 + ["validation"] + ["test"] * 2)
        self.assertEqual([split_role(k, 4) for k in range(1, 5)], ["train"] * 2 + ["validation", "test"])
        self.assertEqual([split_role(k, 3) for k in range(1, 4)], ["train"] * 2 + ["test"])
        with self.assertRaises(ValueError):
            split_role(0, 3)


class Examples(unittest.TestCase):
    def setUp(self):
        self.baskets = [[1, 2], [2, 3], [3], [4, 5], [1, 5], [6], [2, 6], [7], [1], [8], [9], [1, 2], [3, 4]]
        self.visits = visits_of("990001", self.baskets, gaps={3: 30.0})

    def test_needs_three_earlier_visits(self):
        self.assertEqual([e.target_position for e in customer_examples(self.visits, range(1, 6))], [4, 5])

    def test_history_is_the_last_visits_before_the_target(self):
        example = customer_examples(self.visits, [13])[0]
        self.assertEqual([h.items for h in example.history],
                         [cut_items(v.basket)[0] for v in self.visits[13 - 1 - L_VISITS:12]])
        self.assertEqual(example.target_items, frozenset({"ic-p-3", "ic-p-4"}))
        # Counts cover all 12 earlier visits: visit 1 holds item 1 but is outside the last L_VISITS.
        self.assertEqual(example.prior_counts["ic-p-1"], 4)
        self.assertEqual(example.prior_counts["ic-p-2"], 4)

    def test_time_bits_come_from_the_visit(self):
        example = customer_examples(self.visits, [5])[0]
        self.assertEqual([h.gap_days for h in example.history], [None, 7.0, 30.0, 7.0])
        self.assertEqual([h.gap_censored for h in example.history], [False, False, True, False])
        self.assertEqual([h.time_lower_bound for h in example.history], [False, False, True, True])

    def test_later_visits_and_the_target_do_not_reach_the_input(self):
        k = 8
        base = customer_examples(self.visits, [k])[0]
        rng = random.Random(0)
        changed = [b if i < k - 1 else [rng.randrange(100, 200) for _ in range(5)]
                   for i, b in enumerate(self.baskets)]
        other = customer_examples(visits_of("990001", changed, gaps={3: 30.0, 9: 2.0}), [k])[0]
        self.assertEqual((base.history, base.prior_counts), (other.history, other.prior_counts))
        self.assertNotEqual(base.target_items, other.target_items)

    def test_the_cut_keeps_k_items_by_a_fixed_hash_order(self):
        big = visits_of("990002", [list(range(1, 51)), [1], [2], [3]])
        items, dropped = cut_items(big[0].basket)
        self.assertEqual((len(items), dropped), (K_ITEMS, 50 - K_ITEMS))
        self.assertEqual(cut_items(big[0].basket), (items, dropped))
        by_id = sorted(i.item_id_local for i in big[0].basket.items)[:K_ITEMS]
        self.assertNotEqual(sorted(items), by_id)  # not simply the lowest IDs
        example = customer_examples(big, [4])[0]
        self.assertEqual(example.history[0].items_dropped, 18)
        self.assertEqual(example.prior_counts["ic-p-40"], 1)  # counts see the whole basket


class ProgressReplay(unittest.TestCase):
    def test_bucket_and_prefix(self):
        self.assertEqual([progress_bucket(k, 10) for k in (1, 2, 10)], [0, 1, 9])
        self.assertEqual([progress_bucket(k, 7) for k in (1, 4, 7)], [0, 4, 8])
        self.assertEqual([visible_prefix(t, 10) for t in (0, 4, 9)], [0, 4, 9])
        self.assertEqual([visible_prefix(t, 7) for t in (0, 5, 9)], [0, 3, 6])

    def test_hand_counted_seller_counts(self):
        replay = SellerReplay({
            "u": visits_of("u", [[1], [1, 2], [2], [3], [1]]),  # n = 5
            "v": visits_of("v", [[2], [2], [4], [1], [1], [1], [1], [1], [1], [5]]),  # n = 10
        })
        # bucket 4: u sees floor(4*5/10) = 2 visits, v sees 4
        self.assertEqual(dict(replay.counts(4)), {"ic-p-1": 3, "ic-p-2": 3, "ic-p-4": 1})
        self.assertEqual(replay.visible_visits[4], 6)
        self.assertEqual(dict(replay.counts(0)), {})

    def test_counts_match_a_brute_force_and_never_include_the_target(self):
        rng = random.Random(3)
        visits = {c: visits_of(c, [[rng.randrange(20) for _ in range(rng.randint(1, 3))]
                                   for _ in range(rng.randint(4, 15))]) for c in "abcdef"}
        replay = SellerReplay(visits)
        for t in range(10):
            expected = {}
            for own in visits.values():
                for visit in own[:t * len(own) // 10]:
                    for item in visit.basket.item_ids:
                        expected[item] = expected.get(item, 0) + 1
            self.assertEqual(dict(replay.counts(t)), expected)
        for own in visits.values():
            for k in range(1, len(own) + 1):
                self.assertLessEqual(visible_prefix(progress_bucket(k, len(own)), len(own)), k - 1)
