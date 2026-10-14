import copy
import dataclasses
import unittest

import torch

from commerce.evaluation.fl_lab import aggregate, local_round
from commerce.evaluation.harex_compare import shuffled_relations
from commerce.packages.recommender.harex import HarexRecommender
from commerce.packages.recommender.relations import RelationTensors
from commerce.packages.recommender.tests.test_harex import TINY, with_tokens
from commerce.packages.recommender.tests.test_model import seller_data


class Aggregation(unittest.TestCase):
    """The lab runner calls C's core (round_core.aggregate_uniform) through aggregate()."""

    def test_uniform_mean_when_everyone_completed(self):
        mean = aggregate([{"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([3.0, 5.0])}], "cpu")
        torch.testing.assert_close(mean["w"], torch.tensor([2.0, 4.0]))
        self.assertEqual(mean["w"].dtype, torch.float32)

    def test_a_missing_seller_discards_the_round(self):
        self.assertIsNone(aggregate([{"w": torch.ones(2)}, None], "cpu"))
        self.assertIsNone(aggregate([], "cpu"))

    def test_different_tensor_sets_are_refused(self):
        with self.assertRaises(ValueError):
            aggregate([{"w": torch.ones(1)}, {"v": torch.ones(1)}], "cpu")

    def test_the_core_matches_the_former_float32_stand_in(self):
        # Before C's core landed the runner took torch.stack(...).mean(0) in float32. The core
        # averages in float64, so the two agree to float32 rounding, not bit for bit.
        torch.manual_seed(0)
        deltas = [{"a": torch.randn(64, 32) * 1e-3, "b": torch.randn(32)} for _ in range(100)]
        mean = aggregate(deltas, "cpu")
        for key in ("a", "b"):
            torch.testing.assert_close(mean[key], torch.stack([d[key] for d in deltas]).mean(0),
                                       rtol=1e-5, atol=1e-7)


class GlocalRound(unittest.TestCase):
    def setUp(self):
        self.sellers = []
        for seed in (0, 1):
            data, vocab = with_tokens(seller_data(seed=seed))
            self.sellers.append((data, len(vocab)))

    def test_token_tables_stay_local_and_shared_weights_agree_after_a_round(self):
        torch.manual_seed(0)
        models = [HarexRecommender(TINY, vocab_size=v) for _, v in self.sellers]
        global_shared = {k: v.detach().clone() for k, v in models[0].shared_state().items()}
        tokens_before = [m.local_tokens.weight.detach().clone() for m in models]
        deltas = [local_round(m, global_shared, [(0, d)], epochs=1, batch_size=16, lr=3e-3, seed=i)
                  for i, (m, (d, _)) in enumerate(zip(models, self.sellers))]
        self.assertFalse(any(k.startswith("local_tokens") for k in deltas[0]))
        self.assertTrue(all(v.device.type == "cpu" for d in deltas for v in d.values()))
        mean = aggregate(deltas, "cpu")
        global_shared = {k: v + mean[k] for k, v in global_shared.items()}
        for m in models:
            m.load_state_dict(global_shared, strict=False)
        for key, value in models[0].shared_state().items():
            torch.testing.assert_close(value, models[1].shared_state()[key])
        for m, before in zip(models, tokens_before):
            self.assertFalse(torch.equal(m.local_tokens.weight, before))  # trained locally, kept locally

    def test_one_model_with_swapped_tables_equals_a_model_per_seller(self):
        torch.manual_seed(0)
        models = [HarexRecommender(TINY, vocab_size=v) for _, v in self.sellers]
        global_shared = {k: v.detach().clone() for k, v in models[0].shared_state().items()}
        one = copy.deepcopy(models[0])
        tables = [copy.deepcopy(m.local_tokens) for m in models]
        expected = [local_round(m, global_shared, [(0, d)], 1, 16, 3e-3, i)
                    for i, (m, (d, _)) in enumerate(zip(models, self.sellers))]
        for i, (d, _) in enumerate(self.sellers):
            one.local_tokens = tables[i]
            delta = local_round(one, global_shared, [(0, d)], 1, 16, 3e-3, i)
            for key, value in expected[i].items():
                torch.testing.assert_close(delta[key], value)
            torch.testing.assert_close(tables[i].weight, models[i].local_tokens.weight)

    def test_a_seller_without_examples_cannot_complete(self):
        torch.manual_seed(0)
        data, vocab_size = self.sellers[0]
        empty = dataclasses.replace(data, examples=[])
        empty.row_of = data.row_of
        model = HarexRecommender(TINY, vocab_size=vocab_size)
        shared = {k: v.detach().clone() for k, v in model.shared_state().items()}
        self.assertIsNone(local_round(model, shared, [(0, empty)], 1, 16, 1e-3, 0))


class ShuffleControl(unittest.TestCase):
    def relations(self, n=40):
        torch.manual_seed(0)
        neighbor = torch.randint(-1, n, (n, 16))
        return RelationTensors(neighbor=neighbor, features=torch.randn(n, 16, 10),
                               time_forward=torch.rand(n, 16, 32).half(), time_backward=torch.rand(n, 16, 32).half(),
                               has_neighbor=(neighbor >= 0).any(1))

    def test_rows_move_together_and_nothing_is_lost(self):
        original = self.relations()
        shuffled = shuffled_relations(original, "s", seed=0)
        self.assertFalse(torch.equal(shuffled.features, original.features))
        # Every field is permuted by the same rows, so each item's relation row stays whole.
        rows = {tuple(original.features[i].flatten().tolist()): i for i in range(len(original.has_neighbor))}
        for c in range(len(shuffled.has_neighbor)):
            src = rows[tuple(shuffled.features[c].flatten().tolist())]
            self.assertTrue(torch.equal(shuffled.neighbor[c], original.neighbor[src]))
            self.assertTrue(torch.equal(shuffled.time_forward[c], original.time_forward[src]))
            self.assertEqual(bool(shuffled.has_neighbor[c]), bool(original.has_neighbor[src]))

    def test_one_permutation_per_seller_and_seed(self):
        original = self.relations()
        again = shuffled_relations(original, "s", seed=0)
        torch.testing.assert_close(again.features, shuffled_relations(original, "s", seed=0).features)
        self.assertFalse(torch.equal(again.features, shuffled_relations(original, "t", seed=0).features))
        self.assertFalse(torch.equal(again.features, shuffled_relations(original, "s", seed=1).features))

