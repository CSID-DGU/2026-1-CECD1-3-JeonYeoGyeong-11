import dataclasses
import unittest

import torch

from commerce.evaluation.fl_lab import aggregate_uniform, local_round
from commerce.packages.recommender.harex import HarexRecommender
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
