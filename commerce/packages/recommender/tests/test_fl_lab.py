import dataclasses
import unittest

import torch

from commerce.evaluation.fl_lab import aggregate_uniform, local_round, staged
from commerce.evaluation.harex_compare import shuffled_relations
from commerce.packages.recommender.harex import HarexRecommender
from commerce.packages.recommender.relations import RelationTensors
from commerce.packages.recommender.tests.test_harex import TINY, with_tokens
from commerce.packages.recommender.tests.test_model import seller_data


class Aggregation(unittest.TestCase):
    def test_uniform_mean_when_everyone_completed(self):
        mean = aggregate_uniform([{"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([3.0, 5.0])}])
        torch.testing.assert_close(mean["w"], torch.tensor([2.0, 4.0]))

    def test_a_missing_seller_discards_the_round(self):
        self.assertIsNone(aggregate_uniform([{"w": torch.ones(2)}, None]))
        self.assertIsNone(aggregate_uniform([]))

    def test_different_tensor_sets_are_refused(self):
        with self.assertRaises(ValueError):
            aggregate_uniform([{"w": torch.ones(1)}, {"v": torch.ones(1)}])


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
        mean = aggregate_uniform(deltas)
        global_shared = {k: v + mean[k] for k, v in global_shared.items()}
        for m in models:
            m.load_state_dict(global_shared, strict=False)
        for key, value in models[0].shared_state().items():
            torch.testing.assert_close(value, models[1].shared_state()[key])
        for m, before in zip(models, tokens_before):
            self.assertFalse(torch.equal(m.local_tokens.weight, before))  # trained locally, kept locally

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


class Staging(unittest.TestCase):
    def test_the_model_is_back_on_the_cpu_even_after_an_error(self):
        _, vocab = with_tokens(seller_data())
        model = HarexRecommender(TINY, vocab_size=len(vocab))
        with self.assertRaises(RuntimeError):
            with staged(model, torch.device("cpu")):
                raise RuntimeError("a failing round")
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
