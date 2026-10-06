"""Hand-computed checks of the temporary scoring copy used by E-G0 until A's module lands."""
import math
import unittest

import numpy as np

from commerce.evaluation.scoring import MacroAverager, expected_metrics, p_topfreq_scores, popularity_scores


class TiedRanking(unittest.TestCase):
    def test_a_relevant_item_tied_across_the_cut(self):
        # ranks: 0 -> item 0 (3.0); ranks 1-2 -> items 1 and 2 tied at 2.0; only item 1 is relevant.
        m = expected_metrics(np.array([3.0, 2.0, 2.0, 1.0]), {1}, ks=(1, 2))
        self.assertEqual((m["recall@1"], m["ndcg@1"]), (0.0, 0.0))
        self.assertAlmostEqual(m["recall@2"], 0.5)
        self.assertAlmostEqual(m["ndcg@2"], 0.5 / math.log2(3))

    def test_everything_tied_is_the_random_ranking(self):
        m = expected_metrics(np.zeros(4), {0, 1}, ks=(2,))
        self.assertAlmostEqual(m["recall@2"], 0.5)
        self.assertAlmostEqual(m["ndcg@2"], 0.5)

    def test_order_of_ties_in_the_input_does_not_matter(self):
        a = expected_metrics(np.array([1.0, 1.0, 0.0, 1.0]), {3}, ks=(2,))
        b = expected_metrics(np.array([1.0, 1.0, 0.0, 1.0]), {0}, ks=(2,))
        self.assertEqual(a, b)

    def test_no_relevant_item_is_an_error(self):
        with self.assertRaises(ValueError):
            expected_metrics(np.zeros(3), set())


class Baselines(unittest.TestCase):
    def test_p_topfreq_uses_own_counts_then_popularity(self):
        items = ["a", "b", "c", "d"]
        scores = p_topfreq_scores(items, {"c": 2, "d": 2, "a": 1}, {"a": 50, "b": 99, "c": 1, "d": 7})
        self.assertEqual(list(np.argsort(-scores)), [3, 2, 0, 1])  # d, c (2 each, d more popular), a, b
        self.assertEqual(list(popularity_scores(items, {"b": 3})), [0, 3, 0, 0])


class Averages(unittest.TestCase):
    def test_customer_then_seller_then_macro(self):
        avg = MacroAverager()
        avg.add("A", "c1", {"x": 1.0})
        avg.add("A", "c1", {"x": 0.0})
        avg.add("A", "c2", {"x": 1.0})
        avg.add("B", "c3", {"x": 0.0})
        result = avg.result()
        self.assertAlmostEqual(result["macro"]["x"], 0.375)  # A: mean(0.5, 1.0) = 0.75, B: 0
        self.assertAlmostEqual(result["micro"]["x"], 0.5)
        self.assertEqual((result["sellers"], result["examples"]), (2, 4))
