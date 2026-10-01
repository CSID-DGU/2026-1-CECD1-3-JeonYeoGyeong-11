"""Hand-computable synthetic examples (A card "a1 selfcheck에서 함께 호출한다")."""
from __future__ import annotations

import math
import unittest

from commerce.evaluation.metrics.ranking import (
    classify_relevant_items, macro_ndcg_at_k, macro_recall_at_k, micro_ndcg_at_k,
    micro_recall_at_k, ndcg_at_k, p_topfreq_ranking, recall_at_k,
    split_recall_by_repeat_explore,
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


class AggregationTest(unittest.TestCase):
    def test_macro_is_the_mean_of_per_customer_recall(self):
        per_customer = [
            ([("a", 2.0), ("b", 1.0)], {"a"}),   # recall@1 = 1.0
            ([("a", 2.0), ("b", 1.0)], {"b"}),   # recall@1 = 0.0
        ]
        self.assertAlmostEqual(macro_recall_at_k(per_customer, 1), 0.5, places=9)

    def test_micro_pools_hits_and_relevant_instead_of_averaging_ratios(self):
        per_customer = [
            ([("a", 2.0), ("b", 1.0)], {"a", "b"}),  # 2 relevant, 1 hit at k=1
            ([("a", 2.0), ("b", 1.0)], {"a"}),        # 1 relevant, 1 hit at k=1
        ]
        # pooled: hits=1+1=2, relevant=2+1=3 -> 2/3, which differs from the macro mean.
        self.assertAlmostEqual(micro_recall_at_k(per_customer, 1), 2 / 3, places=9)
        self.assertNotAlmostEqual(micro_recall_at_k(per_customer, 1), macro_recall_at_k(per_customer, 1), places=6)

    def test_macro_and_micro_ndcg_run_without_error_and_agree_on_uniform_input(self):
        per_customer = [([("a", 2.0), ("b", 1.0)], {"a"})] * 3
        self.assertAlmostEqual(macro_ndcg_at_k(per_customer, 1), 1.0, places=9)
        self.assertAlmostEqual(micro_ndcg_at_k(per_customer, 1), 1.0, places=9)

    def test_no_ground_truth_returns_none_not_zero(self):
        self.assertIsNone(macro_recall_at_k([([("a", 1.0)], set())], 1))
        self.assertIsNone(micro_recall_at_k([([("a", 1.0)], set())], 1))


class BaselineAndCohortTest(unittest.TestCase):
    def test_p_topfreq_ranks_by_seller_count_ties_broken_by_item_id(self):
        ranking = p_topfreq_ranking({"milk": 10, "bread": 10, "egg": 3}, ["egg", "bread", "milk", "new-item"])
        self.assertEqual(ranking, [("bread", 10.0), ("milk", 10.0), ("egg", 3.0), ("new-item", 0.0)])

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


if __name__ == "__main__":
    unittest.main()
