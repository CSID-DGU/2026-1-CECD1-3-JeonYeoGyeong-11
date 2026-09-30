import dataclasses
import math
import random
import unicodedata
import unittest

import torch

from commerce.packages.recommender.harex import (
    HAREX_ARCHITECTURES, PAD, UNKNOWN, HarexConfig, HarexRecommender, WordVocabulary, item_sequences, repeat_features,
    word_tokens,
)
from commerce.packages.recommender.tests.test_model import N_ITEMS, seller_data
from commerce.packages.recommender.training import TrainConfig, batch_loss, catalog_scores, train, validation_loss

TINY = HarexConfig("test.T_hx", "hx", False, d_model=16, n_heads=2, d_ffn=32, dropout=0.0, max_items=12,
                   max_tokens=6, d_text=8, d_relation=4, mlp_hidden=16, d_time=4)


def texts():
    # Items share words in groups, so a word token links several items as in GCI.
    return ["[NAME] Fixture product %d family%d [AISLE] aisle%d" % (i, i % 5, i % 3) for i in range(1, N_ITEMS + 1)]


def with_tokens(data):
    vocab = WordVocabulary(texts())
    data.tokens = vocab.encode(texts(), TINY.max_tokens)
    return data, vocab


class Tokens(unittest.TestCase):
    def test_space_tokenization_keeps_case_and_nfc(self):
        self.assertEqual(word_tokens("  Organic   Whole\tMilk "), ["Organic", "Whole", "Milk"])
        self.assertEqual(word_tokens(unicodedata.normalize("NFD", "유기농 우유")), ["유기농", "우유"])
        self.assertEqual(word_tokens(None), [])

    def test_vocabulary_pads_and_marks_unknown_words(self):
        vocab = WordVocabulary(["a b", "b c"])
        self.assertEqual(len(vocab), 5)  # pad, unknown, a, b, c
        ids = vocab.encode(["a c", "zzz"], max_tokens=4)
        self.assertEqual(ids.tolist(), [[vocab.index["a"], vocab.index["c"]], [UNKNOWN, PAD]])

    def test_item_sequences_are_the_latest_items_right_aligned(self):
        data = seller_data()
        example = data.examples[5]
        index, mask = item_sequences([example], data.row_of, 4)
        flat = [data.row_of[i] for v in example.history for i in v.items]
        self.assertEqual(index[0].tolist()[-min(4, len(flat)):], flat[-4:])
        self.assertEqual(int(mask.sum()), min(4, len(flat)))


class Variants(unittest.TestCase):
    def setUp(self):
        self.data, self.vocab = with_tokens(seller_data(relations=True))

    def model(self, text="hx", relation=False, seed=0):
        torch.manual_seed(seed)
        config = dataclasses.replace(TINY, text=text, relation=relation)
        return HarexRecommender(config, vocab_size=len(self.vocab) if text == "hx" else None)

    def test_registry(self):
        base = ["harex.R_hx.v1", "harex.R_lm.v1", "harex.T_hx.v1", "harex.T_lm.v1"]
        self.assertEqual(sorted(HAREX_ARCHITECTURES), sorted(
            base + [v.replace(".v1", suffix) for v in base for suffix in ("_rep.v1", "_rep2.v1")]))
        # D0022's comparison keeps the repeat path off; D0023's service configuration turns it on.
        self.assertFalse(any(HAREX_ARCHITECTURES[v].repeat for v in base))
        self.assertTrue(all(HAREX_ARCHITECTURES[v.replace(".v1", "_rep.v1")].repeat for v in base))
        self.assertTrue(all(HAREX_ARCHITECTURES[v.replace(".v1", "_rep2.v1")].repeat_item for v in base))
        self.assertFalse(any(HAREX_ARCHITECTURES[v.replace(".v1", "_rep.v1")].repeat_item for v in base))
        gci = HAREX_ARCHITECTURES["harex.T_hx.v1"]
        self.assertEqual((gci.n_layers, gci.d_model, gci.n_heads, gci.d_ffn, gci.dropout), (1, 128, 4, 256, 0.2))

    def test_t_and_r_start_from_the_same_shared_weights(self):
        for text in ("hx", "lm"):
            plain, related = self.model(text, False, seed=3), self.model(text, True, seed=3)
            related_state = related.state_dict()
            for name, tensor in plain.state_dict().items():
                self.assertTrue(torch.equal(tensor, related_state[name]), (text, name))

    def test_local_token_table_never_leaves_the_seller(self):
        model = self.model("hx", True)
        shared = model.shared_state()
        self.assertFalse(any(k.startswith("local_tokens") for k in shared))
        self.assertIn("sequence.layers.0.self_attn.in_proj_weight", shared)
        self.assertFalse(any(len(self.vocab) in v.shape for v in shared.values()))

    def test_relations_are_required_exactly_for_r(self):
        with self.assertRaises(ValueError):
            self.model("hx", False).encode_items(self.data)  # the test seller carries relations
        plain = dataclasses.replace(self.data, relations=None)
        plain.row_of = self.data.row_of
        self.model("hx", False).encode_items(plain)
        with self.assertRaises(ValueError):
            self.model("hx", True).encode_items(plain)

    def test_an_item_without_neighbours_gets_l_zero(self):
        model = self.model("hx", True).eval()
        base = model.base_repr(self.data)
        l = model.relation_repr(base, self.data.relations)
        never_sold = N_ITEMS - 1
        self.assertFalse(bool(self.data.relations.has_neighbor[never_sold]))
        self.assertEqual(float(l[never_sold].detach().abs().sum()), 0.0)

    def test_padding_does_not_move_a_query(self):
        model = self.model("lm", False).eval()
        plain = dataclasses.replace(self.data, relations=None)
        plain.row_of = self.data.row_of
        e = model.encode_items(plain)
        alone = model.encode_queries(e, [plain.examples[0]], plain)
        mixed = model.encode_queries(e, [plain.examples[0]] + plain.examples[5:9], plain)
        torch.testing.assert_close(alone[0], mixed[0], atol=1e-5, rtol=0)

    def test_every_group_and_the_token_table_get_gradients(self):
        model = self.model("hx", True).train()
        loss, used = batch_loss(model, self.data, self.data.examples[:16], 10, random.Random(0))
        self.assertGreater(used, 0)
        loss.backward()
        groups = HarexRecommender.GROUPS + ("local_tokens", "time_mlp", "relation_mlp", "relation_pool")
        for group in groups:
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(group + ".")]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), group)

    def test_all_four_variants_train_and_score(self):
        for text in ("hx", "lm"):
            for relation in (False, True):
                data = self.data if relation else dataclasses.replace(self.data, relations=None)
                data.row_of = self.data.row_of
                model = self.model(text, relation)
                before = validation_loss(model, [data], n_negatives=10)
                log = train(model, [data], TrainConfig(steps=30, batch_size=16, n_negatives=10, lr=3e-3))
                self.assertEqual(log["steps"], 30)
                self.assertTrue(math.isfinite(log["loss_mean"]))
                self.assertLess(validation_loss(model, [data], n_negatives=10), before)
                scores = catalog_scores(model, data, data.examples[:3])
                self.assertEqual(tuple(scores.shape), (3, N_ITEMS))


class WholeCatalogLoss(unittest.TestCase):
    def test_every_item_outside_the_targets_is_a_negative(self):
        data, vocab = with_tokens(seller_data())
        torch.manual_seed(0)
        model = HarexRecommender(TINY, vocab_size=len(vocab)).eval()
        examples = data.examples[:6]
        with torch.no_grad():
            loss, used = batch_loss(model, data, examples, 0, random.Random(0))
            e = model.encode_items(data)
            scores = model.score(model.encode_queries(e, examples, data), e)
        rng = random.Random(0)
        expected = []
        for row, ex in enumerate(examples):
            targets = {data.row_of[i] for i in ex.target_items}
            positive = data.row_of[rng.choice(sorted(ex.target_items))]
            others = [scores[row, j] for j in range(len(data.items)) if j not in targets]
            expected.append(float(torch.logsumexp(torch.stack([scores[row, positive]] + others), 0) - scores[row, positive]))
        self.assertEqual(used, len(examples))
        self.assertAlmostEqual(float(loss), sum(expected) / len(expected), places=5)


class RepeatPath(unittest.TestCase):
    def setUp(self):
        self.data, self.vocab = with_tokens(seller_data())
        self.examples = self.data.examples[:8]

    def model(self, repeat, seed=0, item=False):
        torch.manual_seed(seed)
        return HarexRecommender(dataclasses.replace(TINY, repeat=repeat, repeat_item=item), vocab_size=len(self.vocab))

    def test_features_from_the_customers_own_earlier_visits(self):
        index, values = repeat_features(self.examples, self.data.row_of)
        example = self.examples[0]
        last_items = set(example.history[-1].items)
        for col in range(index.shape[1]):
            row = int(index[0, col])
            if row < 0:
                continue
            item = self.data.items[row]
            self.assertIn(item, example.prior_counts)
            self.assertAlmostEqual(float(values[0, col, 0]), math.log1p(example.prior_counts[item]), places=5)
            self.assertEqual(float(values[0, col, 2]), 1.0 if item in last_items else 0.0)  # bought on the last visit
        self.assertEqual(int((index[0] >= 0).sum()), len(example.prior_counts))

    def test_only_earlier_purchases_get_a_repeat_score(self):
        for item in (False, True):
            model = self.model(True, item=item).eval()
            with torch.no_grad():
                e = model.encode_items(self.data)
                q = model.encode_queries(e, self.examples, self.data)
                extra = model.extra_scores(q, e, self.examples, self.data)
            for row, example in enumerate(self.examples):
                bought = {self.data.row_of[i] for i in example.prior_counts}
                for col in range(len(self.data.items)):
                    if col not in bought:
                        self.assertEqual(float(extra[row, col]), 0.0)

    def test_rep2_reads_the_items_own_representation(self):
        model = self.model(True, item=True)
        loss, _ = batch_loss(model, self.data, self.examples, 20, random.Random(0))
        loss.backward()
        self.assertGreater(float(model.repeat_item.weight.grad.abs().sum()), 0.0)

    def test_other_groups_start_as_without_the_path_and_the_path_learns(self):
        plain, repeat = self.model(False, seed=5), self.model(True, seed=5)
        state = repeat.state_dict()
        for name, tensor in plain.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)
        loss, _ = batch_loss(repeat, self.data, self.examples, 20, random.Random(0))
        loss.backward()
        self.assertGreater(float(repeat.repeat_mlp[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(repeat.repeat_gate.weight.grad.abs().sum()), 0.0)
        scores = catalog_scores(repeat, self.data, self.examples)
        self.assertEqual(tuple(scores.shape), (len(self.examples), len(self.data.items)))
