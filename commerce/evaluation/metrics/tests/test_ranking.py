"""Hand-computable synthetic examples (A card "a1 selfcheck에서 함께 호출한다"),
plus a cross-check against B's temporary scoring.py while it still exists:
two independent implementations of the same definitions must agree."""
from __future__ import annotations

import importlib.util
import math
import random
import unittest

from commerce.evaluation.metrics.ranking import (
    MacroAverager, classify_relevant_items, expected_metrics, local_popularity_ranking, ndcg_at_k,
    new_item_indices, p_topfreq_ranking, p_topfreq_scores, popularity_scores, recall_at_k,
    repeat_explore_indices, split_recall_by_repeat_explore,
)

# a > {b, c} (tied) > d. relevant = {b, d}, k=2.
# Group (b,c) straddles the k=2 cutoff: each has a 1/2 chance of the one
# remaining slot, so b's expected credit is 0.5, not 0 or 1.
RANKED_WITH_TIE = [("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0)]
RELEVANT = {"b", "d"}


class TieHandlingTest(unittest.TestCase):
    def test_recall_at_k_credits_half_a_hit_for_the_boundary_tie(self):
        # hits = 0 (a) + 1*(1/2) (b, tied with c for the single remaining slot at k=2) = 0.5
        # recall = 0.5 / |{b, d}| = 0.25
        self.assertAlmostEqual(recall_at_k(RANKED_WITH_TIE, RELEVANT, 2), 0.25, places=9)

    def test_ndcg_at_k_averages_the_tied_group_discount(self):
        # dcg: a not relevant (skip). (b,c) group start=2 end=3, only rank 2 is
        # within k=2, so discount_sum = 1/log2(3); split over the 2 tied items:
        expected_discount = (1.0 / math.log2(3)) / 2
        expected_dcg = 1 * expected_discount  # only b is relevant in that group
        expected_idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both relevant items at ranks 1,2
        expected_ndcg = expected_dcg / expected_idcg
        self.assertAlmostEqual(ndcg_at_k(RANKED_WITH_TIE, RELEVANT, 2), expected_ndcg, places=9)
        self.assertAlmostEqual(expected_ndcg, 0.19343, places=5)

    def test_no_ties_matches_textbook_ndcg(self):
        ranked = [("x", 5.0), ("y", 4.0), ("z", 3.0)]
        relevant = {"y"}
        # DCG = 1/log2(3) (y at rank 2). IDCG@3 with 1 relevant item = 1/log2(2) = 1.
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 3), 1.0 / math.log2(3), places=9)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 3), 1.0, places=9)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 1), 0.0, places=9)

    def test_empty_relevant_set_is_zero_not_an_error(self):
        self.assertEqual(recall_at_k(RANKED_WITH_TIE, set(), 2), 0.0)
        self.assertEqual(ndcg_at_k(RANKED_WITH_TIE, set(), 2), 0.0)


class ArrayEntryPointTest(unittest.TestCase):
    def test_expected_metrics_is_the_same_tie_example_by_index(self):
        # candidates a, b, c, d = indices 0..3 with the same scores as RANKED_WITH_TIE
        m = expected_metrics([3.0, 2.0, 2.0, 1.0], relevant=[1, 3], ks=(2,))
        self.assertAlmostEqual(m["recall@2"], 0.25, places=9)
        self.assertAlmostEqual(m["ndcg@2"], 0.19343, places=5)

    def test_candidate_order_does_not_matter_only_scores_do(self):
        a = expected_metrics([3.0, 2.0, 2.0, 1.0], relevant=[1, 3])
        b = expected_metrics([2.0, 1.0, 3.0, 2.0], relevant=[0, 1])  # same scores, shuffled
        self.assertEqual(a, b)

    def test_example_without_relevant_items_is_rejected(self):
        with self.assertRaises(ValueError):
            expected_metrics([1.0, 2.0], relevant=[])


class AggregationTest(unittest.TestCase):
    def test_macro_averages_customers_then_sellers_micro_averages_examples(self):
        avg = MacroAverager()
        # seller s1: customer c1 has two examples (1.0, 0.0) -> 0.5; c2 one example 1.0 -> s1 = 0.75
        avg.add("s1", "c1", {"recall@10": 1.0})
        avg.add("s1", "c1", {"recall@10": 0.0})
        avg.add("s1", "c2", {"recall@10": 1.0})
        # seller s2: one customer, one example 0.0 -> s2 = 0.0
        avg.add("s2", "c9", {"recall@10": 0.0})
        r = avg.result()
        self.assertEqual((r["sellers"], r["examples"]), (2, 4))
        self.assertAlmostEqual(r["per_seller"]["s1"]["recall@10"], 0.75, places=9)
        self.assertAlmostEqual(r["macro"]["recall@10"], (0.75 + 0.0) / 2, places=9)
        self.assertAlmostEqual(r["micro"]["recall@10"], (1 + 0 + 1 + 0) / 4, places=9)

    def test_empty_averager_reports_zero_examples(self):
        self.assertEqual(MacroAverager().result(), {"sellers": 0, "examples": 0})


class BaselineAndCohortTest(unittest.TestCase):
    ITEMS = ["milk", "bread", "egg", "kiwi"]
    SELLER = {"milk": 10, "bread": 10, "egg": 3}
    CUSTOMER = {"egg": 2, "bread": 1}

    def test_local_popularity_is_seller_counts(self):
        self.assertEqual(list(popularity_scores(self.ITEMS, self.SELLER)), [10.0, 10.0, 3.0, 0.0])
        self.assertEqual(local_popularity_ranking(self.SELLER, self.ITEMS),
                         [("bread", 10.0), ("milk", 10.0), ("egg", 3.0), ("kiwi", 0.0)])

    def test_p_topfreq_is_the_customers_own_counts_ties_by_seller_popularity(self):
        # egg: bought twice by this customer -> first, even though the seller sells it least.
        # bread vs milk: bread bought once by the customer -> above milk; milk vs kiwi:
        # both 0 for the customer, milk wins on seller popularity.
        ranking = [item for item, _ in p_topfreq_ranking(self.CUSTOMER, self.SELLER, self.ITEMS)]
        self.assertEqual(ranking, ["egg", "bread", "milk", "kiwi"])
        scores = p_topfreq_scores(self.ITEMS, self.CUSTOMER, self.SELLER)
        self.assertEqual(sorted(range(4), key=lambda i: -scores[i]), [2, 1, 0, 3])

    def test_seller_popularity_never_outweighs_one_own_purchase(self):
        scores = p_topfreq_scores(["hit", "own"], {"own": 1}, {"hit": 10_000, "own": 0})
        self.assertGreater(scores[1], scores[0])

    def test_p_topfreq_without_history_is_local_popularity_order(self):
        self.assertEqual([i for i, _ in p_topfreq_ranking({}, self.SELLER, self.ITEMS)],
                         [i for i, _ in local_popularity_ranking(self.SELLER, self.ITEMS)])

    def test_repeat_explore_and_new_item_subsets(self):
        repeat, explore = repeat_explore_indices(self.ITEMS, [0, 2, 3], self.CUSTOMER)
        self.assertEqual((repeat, explore), ([2], [0, 3]))
        self.assertEqual(new_item_indices(self.ITEMS, [0, 2, 3], self.SELLER), [3])

    def test_classify_relevant_items_flags_repeat_and_new_item_cohort(self):
        labels = classify_relevant_items(
            relevant={"milk", "kiwi"},
            customer_counts_before_cutoff={"milk": 3},
            seller_counts_before_cutoff={"milk": 50},
        )
        self.assertEqual(labels["milk"], {"repeat": True, "new_item_cohort": False})
        self.assertEqual(labels["kiwi"], {"repeat": False, "new_item_cohort": True})

    def test_split_recall_by_repeat_explore(self):
        ranked = [("milk", 2.0), ("kiwi", 1.0)]
        result = split_recall_by_repeat_explore(
            ranked, relevant={"milk", "kiwi"}, k=1, customer_counts_before_cutoff={"milk": 3},
        )
        self.assertEqual(result["repeat"], 1.0)  # milk (repeat) is at rank 1, within k=1
        self.assertEqual(result["explore"], 0.0)  # kiwi (explore) is at rank 2, outside k=1


@unittest.skipUnless(importlib.util.find_spec("commerce.evaluation.scoring"),
                     "B's temporary scoring.py is gone (replaced by this module)")
class CrossCheckWithTemporaryScoringTest(unittest.TestCase):
    """B scored E-G0 with its own copy of these definitions; both must give the same numbers."""

    def test_random_rankings_with_heavy_ties_agree(self):
        import numpy as np
        from commerce.evaluation import scoring
        rng = random.Random(7)
        for _ in range(500):
            n = rng.randint(1, 40)
            scores = np.array([float(rng.randint(0, 5)) for _ in range(n)])  # few distinct values -> many ties
            relevant = rng.sample(range(n), k=rng.randint(1, n))
            ours, theirs = expected_metrics(scores, relevant), scoring.expected_metrics(scores, relevant)
            for name in theirs:
                self.assertAlmostEqual(ours[name], theirs[name], places=12, msg=name)

    def test_baselines_and_averaging_agree(self):
        from commerce.evaluation import scoring
        rng = random.Random(11)
        items = ["i%d" % i for i in range(30)]
        ours, theirs = MacroAverager(), scoring.MacroAverager()
        for example in range(200):
            seller = {i: rng.randint(0, 50) for i in items if rng.random() < 0.7}
            prior = {i: rng.randint(1, 4) for i in items if rng.random() < 0.2}
            self.assertEqual(list(popularity_scores(items, seller)), list(scoring.popularity_scores(items, seller)))
            self.assertEqual(list(p_topfreq_scores(items, prior, seller)), list(scoring.p_topfreq_scores(items, prior, seller)))
            m = expected_metrics(p_topfreq_scores(items, prior, seller), rng.sample(range(30), k=3))
            ours.add("s%d" % (example % 4), "c%d" % (example % 9), m)
            theirs.add("s%d" % (example % 4), "c%d" % (example % 9), m)
        a, b = ours.result(), theirs.result()
        for name in b["macro"]:
            self.assertAlmostEqual(a["macro"][name], b["macro"][name], places=12)
            self.assertAlmostEqual(a["micro"][name], b["micro"][name], places=12)


if __name__ == "__main__":
    unittest.main()
