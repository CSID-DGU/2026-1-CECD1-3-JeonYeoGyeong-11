from dataclasses import asdict
import hashlib
import math
import random
import unittest

import torch

from commerce.packages.recommender.examples import customer_examples
from commerce.packages.recommender.model import (
    ARCHITECTURES, ModelConfig, TextOnlyRecommender, history_batch, sampled_softmax_loss,
)
from commerce.packages.recommender.tests import item_id, visits_of
from commerce.packages.recommender.training import (
    SellerData, TrainConfig, batch_loss, catalog_scores, train, validation_loss,
)

TINY = ModelConfig("test.tiny", "text_only", d_text=8, d_relation=4, d_model=16, n_layers=1, n_heads=2,
                   d_ffn=32, mlp_hidden=16, dropout=0.0)
N_ITEMS = 30


def seller_data(customers=24, seed=0) -> SellerData:
    """Each customer keeps buying from a personal favourite set, so the history predicts the target."""
    rng = random.Random(seed)
    examples = []
    for c in range(customers):
        favourites = rng.sample(range(1, N_ITEMS + 1), 3)
        baskets = [rng.sample(favourites, 2) + [rng.randint(1, N_ITEMS)] for _ in range(8)]
        visits = visits_of("99%04d" % c, baskets)
        examples += customer_examples(visits, range(1, len(visits) + 1))
    generator = torch.Generator().manual_seed(seed)
    z = torch.nn.functional.normalize(torch.randn(N_ITEMS, TINY.d_text, generator=generator), dim=-1)
    return SellerData("ic-client-990101", tuple(item_id(i) for i in range(1, N_ITEMS + 1)), z, examples)


def digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class Loss(unittest.TestCase):
    """OQ01 by hand: logits [2, 1, 0], the positive first."""

    def test_value(self):
        loss, used = sampled_softmax_loss(torch.tensor([2.0]), torch.tensor([[1.0, 0.0]]),
                                          torch.tensor([[True, True]]))
        self.assertEqual(used, 1)
        self.assertAlmostEqual(float(loss), math.log(1 + math.exp(-1) + math.exp(-2)), places=6)

    def test_gradient_is_softmax_minus_one_hot(self):
        positive = torch.tensor([2.0], requires_grad=True)
        negatives = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss, _ = sampled_softmax_loss(positive, negatives, torch.tensor([[True, True]]))
        loss.backward()
        total = math.exp(2) + math.exp(1) + 1
        self.assertAlmostEqual(float(positive.grad), math.exp(2) / total - 1, places=6)
        self.assertAlmostEqual(float(negatives.grad[0, 0]), math.exp(1) / total, places=6)

    def test_a_target_item_among_the_negatives_is_left_out(self):
        loss, _ = sampled_softmax_loss(torch.tensor([2.0]), torch.tensor([[1.0, 5.0]]),
                                       torch.tensor([[True, False]]))
        self.assertAlmostEqual(float(loss), math.log(1 + math.exp(-1)), places=6)

    def test_mean_over_usable_examples_only(self):
        loss, used = sampled_softmax_loss(torch.tensor([2.0, 0.0]), torch.tensor([[1.0], [3.0]]),
                                          torch.tensor([[True], [False]]))
        self.assertEqual(used, 1)
        self.assertAlmostEqual(float(loss), math.log(1 + math.exp(-1)), places=6)
        _, none = sampled_softmax_loss(torch.tensor([1.0]), torch.tensor([[0.0]]), torch.tensor([[False]]))
        self.assertEqual(none, 0)


class Architecture(unittest.TestCase):
    def test_registered_versions_do_not_drift(self):
        self.assertEqual(asdict(ARCHITECTURES["text_only.v1"]), {
            "architecture_version": "text_only.v1", "model_variant": "text_only", "d_text": 384,
            "d_relation": 64, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ffn": 256, "mlp_hidden": 128,
            "dropout": 0.1, "max_visits": 10, "max_items": 32})

    def test_six_shared_groups_and_no_item_axis(self):
        torch.manual_seed(0)
        model = TextOnlyRecommender(ARCHITECTURES["text_only.v1"])
        groups = {name.split(".")[0] for name, _ in model.named_parameters()}
        self.assertEqual(groups, set(TextOnlyRecommender.GROUPS))
        catalog_sizes = {N_ITEMS, 4380}  # a synthetic and a realistic seller: no parameter may scale with either
        self.assertFalse(any(size in p.shape for p in model.parameters() for size in catalog_sizes))


class Model(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TextOnlyRecommender(TINY).eval()
        self.data = seller_data()

    def test_every_shared_group_gets_a_gradient(self):
        self.model.train()
        loss, used = batch_loss(self.model, self.data, self.data.examples[:16], 10, random.Random(0))
        self.assertGreater(used, 0)
        loss.backward()
        for group in TextOnlyRecommender.GROUPS:
            grads = [p.grad for n, p in self.model.named_parameters() if n.startswith(group + ".")]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), group)
        self.assertFalse(self.data.z.requires_grad)  # z is an input, not a parameter

    def test_zero_relation_columns_get_no_gradient(self):
        self.model.train()
        loss, _ = batch_loss(self.model, self.data, self.data.examples[:16], 10, random.Random(0))
        loss.backward()
        weight = self.model.fusion[0].weight.grad
        self.assertEqual(float(weight[:, TINY.d_text:].abs().sum()), 0.0)

    def test_padding_does_not_move_a_query(self):
        e = self.model.item_repr(self.data.z)
        example = self.data.examples[0]
        alone = self.model.query(e, history_batch([example], self.data.row_of, TINY))
        mixed = self.model.query(e, history_batch([example] + self.data.examples[5:9], self.data.row_of, TINY))
        torch.testing.assert_close(alone[0], mixed[0], atol=1e-5, rtol=0)

    def test_item_padding_row_does_not_leak(self):
        # Padding slots index row 0 before the mask; pick an example that never buys row 0.
        e = self.model.item_repr(self.data.z)
        example = next(ex for ex in self.data.examples
                       if all(self.data.row_of[i] != 0 for v in ex.history for i in v.items))
        batch = history_batch([example], self.data.row_of, TINY)
        moved = e.clone()
        moved[0] += 100.0
        torch.testing.assert_close(self.model.query(e, batch), self.model.query(moved, batch), atol=1e-5, rtol=0)

    def test_catalog_scores_shape(self):
        scores = catalog_scores(self.model, self.data, self.data.examples[:5])
        self.assertEqual(tuple(scores.shape), (5, N_ITEMS))
        self.assertTrue(torch.isfinite(scores).all())


class Training(unittest.TestCase):
    def test_training_generalises_to_new_customers_and_is_repeatable(self):
        # Without item IDs the model must learn "score what this history looks like" from text
        # vectors alone, which is slow; the check asks for a clear drop, not for the optimum.
        seen = seller_data(customers=200, seed=0)
        new = seller_data(customers=40, seed=1)
        new = SellerData(new.seller_id, new.items, seen.z, new.examples)  # same items, other customers
        runs = []
        for _ in range(2):
            torch.manual_seed(0)
            model = TextOnlyRecommender(TINY)
            before = validation_loss(model, [new], n_negatives=20)
            log = train(model, [seen], TrainConfig(steps=200, batch_size=32, n_negatives=20, lr=3e-3))
            scores = catalog_scores(model, new, new.examples[:20])
            runs.append((before, validation_loss(model, [new], n_negatives=20), scores, log))
        (before, after, scores, log), (_, after2, scores2, _) = runs
        self.assertEqual(log["steps"], 200)
        self.assertLess(after, before - 0.15)
        # Repeats agree on outputs, not bits: the attention key bias has no true gradient, so
        # AdamW moves it by rounding noise without changing any score.
        self.assertAlmostEqual(after, after2, places=5)
        torch.testing.assert_close(scores, scores2, atol=1e-4, rtol=0)

    def test_validation_loss_is_fixed_across_calls(self):
        torch.manual_seed(0)
        model = TextOnlyRecommender(TINY)
        data = seller_data()
        self.assertEqual(validation_loss(model, [data], n_negatives=20), validation_loss(model, [data], n_negatives=20))

    def test_no_examples_means_no_step(self):
        torch.manual_seed(0)
        model = TextOnlyRecommender(TINY)
        before = digest(model)
        empty = SellerData("ic-client-990101", (item_id(1),), torch.zeros(1, TINY.d_text), [])
        log = train(model, [empty], TrainConfig(steps=5))
        self.assertEqual((log["steps"], digest(model)), (0, before))
