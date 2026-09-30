import unittest

import torch

from commerce.evaluation.gci_original import (
    END, LOCAL, PAD, SEP, SPECIAL, START, Catalog, GCIModel, Vocabulary, beam_search, client_groups, source_tokens,
    target_tokens, ten_items, warmup_schedule,
)


class Architecture(unittest.TestCase):
    def test_shared_layers_have_the_papers_parameter_counts(self):
        model = GCIModel(500)
        self.assertEqual(sum(p.numel() for p in model.encoder.parameters()), 132480)  # GCI Table 3
        self.assertEqual(sum(p.numel() for p in model.decoder.parameters()), 198784)

    def test_embeddings_and_output_stay_local(self):
        shared = GCIModel(50).shared_state()
        self.assertFalse(any(k.startswith(LOCAL) for k in shared))
        self.assertTrue(any(k.startswith("encoder.") for k in shared))
        self.assertTrue(any(k.startswith("decoder.") for k in shared))

    def test_learning_rate_warms_up_then_decays(self):
        rate = warmup_schedule(128, 4000)
        self.assertLess(rate(0), rate(3999))
        self.assertGreater(rate(3999), rate(20000))


class Tokens(unittest.TestCase):
    def test_four_names_joined_by_separators_and_a_bounded_target(self):
        vocab = Vocabulary(["oat milk dairy", "rye bread bakery"])
        source = source_tokens(vocab, ["oat milk dairy", "rye bread bakery"])
        self.assertEqual(source.count(SEP), 1)
        self.assertEqual(len(source), 7)
        target = target_tokens(vocab, "rye bread bakery")
        self.assertEqual((target[0], target[-1], len(target)), (START, END, 5))
        self.assertEqual(vocab.words("unseen milk")[0], 4)  # UNKNOWN


class Matching(unittest.TestCase):
    def test_jaccard_nearest_and_ten_distinct_items(self):
        texts = ["organic oat milk", "oat milk", "rye bread", "organic rye bread"]
        vocab = Vocabulary(texts)
        catalog = Catalog(range(4), texts, vocab, torch.device("cpu"))
        written = torch.tensor([vocab.words("oat milk") + [END], vocab.words("organic bread") + [PAD]])
        nearest = catalog.nearest(written, k=4)
        self.assertEqual(int(nearest[0, 0]), 1)  # exact name
        self.assertEqual(int(nearest[1, 0]), 3)  # 2 of 3 words in common
        # Two written names that collide on item 1 give items 1, then their next-nearest.
        picked = ten_items(torch.tensor([[1, 0, 3, 2], [1, 3, 0, 2]]), k=3)
        self.assertEqual(picked, [1, 0, 3])


class StubModel:
    """Prefers the name [5, 6] then END whatever the source."""

    def eval(self):
        return self

    def encode(self, source):
        return torch.zeros(source.shape[0], 1, 4), torch.zeros(source.shape[0], 1, dtype=torch.bool)

    def decode(self, memory, memory_pad, target_in):
        n, t = target_in.shape
        logits = torch.zeros(n, t, SPECIAL + 3)
        want = [5, 6, END]
        logits[:, -1, want[min(t - 1, 2)]] = 5.0
        logits[:, -1, 7] = 4.0  # the runner-up
        return logits


class Beam(unittest.TestCase):
    def test_the_best_beam_comes_first(self):
        out = beam_search(StubModel(), torch.ones(2, 3, dtype=torch.long), beam=3, max_len=5)
        self.assertEqual(out.shape[:2], (2, 3))
        self.assertEqual(out[0, 0, :3].tolist(), [5, 6, END])


if __name__ == "__main__":
    unittest.main()


class Federation(unittest.TestCase):
    def args(self, **kw):
        import argparse
        base = dict(clients=4, sellers_per_client=5, menu_size=0, menu_clients=0, menu_sellers_per_client=10)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_mixed_federation_like_the_papers_table_4(self):
        groups = client_groups(self.args(menu_size=200, menu_clients=2))
        self.assertEqual(groups, [(10, True), (10, True), (5, False), (5, False)])

    def test_alike_clients_by_default(self):
        self.assertEqual(client_groups(self.args()), [(5, False)] * 4)
        self.assertEqual(client_groups(self.args(menu_size=200)), [(5, True)] * 4)
        with self.assertRaises(SystemExit):
            client_groups(self.args(menu_clients=2))


class SellerSize(unittest.TestCase):
    def test_small_sellers_keep_the_cohort_ids(self):
        import argparse
        from commerce.evaluation.e_g0 import STAND_IN_TARGETS
        from commerce.evaluation.harex_compare import cohort_targets
        self.assertIs(cohort_targets(argparse.Namespace(seller_size=0)), STAND_IN_TARGETS)
        small = cohort_targets(argparse.Namespace(seller_size=200))
        self.assertEqual(sorted(small), sorted(STAND_IN_TARGETS))
        self.assertEqual(set(small.values()), {200})
