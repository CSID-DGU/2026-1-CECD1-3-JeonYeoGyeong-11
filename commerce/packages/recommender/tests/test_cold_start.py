import random
import unittest

import torch

from commerce.evaluation.harex_compare import held_out_item, parts, seller_info, without_items
from commerce.packages.recommender.harex import UNKNOWN
from commerce.packages.recommender.tests import SELLER, item_id, visits_of

N_ITEMS = 12
ITEMS = tuple(item_id(n) for n in range(1, N_ITEMS + 1))
FRAC = 0.3
HELD = {i for i in ITEMS if held_out_item(i, FRAC, 0)}


def counts():
    return {"examples": {r: 0 for r in ("train", "validation", "test")}, "catalog_items": 0, "held_out_items": 0,
            "test_examples_with_held_out_answers": 0, "training_examples_only_held_out": 0,
            "examples_history_all_held_out": 0}


def texts():
    # A held-out item's second word appears in no other name.
    return ["oat %s" % ("only%d" % n if item_id(n) in HELD else "milk") for n in range(1, N_ITEMS + 1)]


def customers(seed=0):
    rng = random.Random(seed)
    out = {"c%d" % c: visits_of("99%04d" % c, [rng.sample(range(1, N_ITEMS + 1), 3) for _ in range(10)])
           for c in range(6)}
    h = min(int(i.rsplit("-", 1)[1]) - 990000 for i in HELD)
    plain = [n for n in range(1, N_ITEMS + 1) if item_id(n) not in HELD][:2]
    # Visit 5 (train) holds only a held-out item; visit 9 (test) mixes one in.
    baskets = [plain] * 10
    baskets[4], baskets[8] = [h], [h] + plain[:1]
    out["held"] = visits_of("990900", baskets)
    return out


class HeldOutItems(unittest.TestCase):
    def test_fixed_by_seed_and_id_near_the_fraction(self):
        ids = ["ic-p-%d" % n for n in range(990000, 995000)]  # outside the raw ID range (data.md §6)
        chosen = [i for i in ids if held_out_item(i, 0.1, 0)]
        self.assertTrue(0.08 < len(chosen) / len(ids) < 0.12)
        self.assertEqual(chosen, [i for i in ids if held_out_item(i, 0.1, 0)])
        self.assertNotEqual(chosen, [i for i in ids if held_out_item(i, 0.1, 1)])
        self.assertFalse(any(held_out_item(i, 0.0, 0) for i in ids))

    def test_a_history_of_only_held_out_items_is_left_out(self):
        h = min(int(i.rsplit("-", 1)[1]) - 990000 for i in HELD)
        c = counts()
        seller_info(ITEMS, texts(), torch.zeros(N_ITEMS, 8), {"h": visits_of("990901", [[h]] * 10)}, FRAC, 0,
                    False, c)
        self.assertEqual(c["examples_history_all_held_out"], 7)  # targets 4..10
        self.assertEqual(sum(c["examples"].values()), 0)

    def test_removal_keeps_every_visit_and_its_time(self):
        visits = customers()["held"]
        stripped = without_items(visits, HELD)
        self.assertEqual(len(stripped), len(visits))
        self.assertEqual([v.gap_days for v in stripped], [v.gap_days for v in visits])
        self.assertEqual(stripped[4].basket.items, ())
        self.assertFalse(any(v.basket.item_ids & HELD for v in stripped))


class CNew(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELD and len(HELD) < N_ITEMS)
        self.counts = counts()
        z = torch.randn(N_ITEMS, 8, generator=torch.Generator().manual_seed(0))
        self.info = seller_info(ITEMS, texts(), z, customers(), FRAC, 0, True, self.counts)

    def examples(self, *roles):
        return [e for (r, _), ex in self.info["grouped"].items() if r in roles for e in ex]

    def test_training_never_sees_a_held_out_item(self):
        for example in self.examples("train", "validation"):
            self.assertTrue(example.target_items)
            self.assertFalse(example.target_items & HELD)
            self.assertFalse(set(example.prior_counts) & HELD)
            self.assertFalse(any(set(v.items) & HELD for v in example.history))
        self.assertGreaterEqual(self.counts["training_examples_only_held_out"], 1)
        self.assertTrue(all(any(v.items for v in e.history) for e in self.examples("train", "validation", "test")))
        self.assertEqual(self.info["train_items"], tuple(i for i in ITEMS if i not in HELD))

    def test_test_answers_keep_held_out_items_but_history_does_not(self):
        test = self.examples("test")
        mixed = [e for e in test if e.target_items & HELD]
        self.assertTrue(mixed)
        self.assertEqual(self.counts["test_examples_with_held_out_answers"], len(mixed))
        for example in test:
            self.assertFalse(set(example.prior_counts) & HELD)
        self.assertEqual(self.info["held_rows"], {ITEMS.index(i) for i in HELD})

    def test_held_out_words_are_unknown_and_items_have_no_relations(self):
        for item in HELD:
            self.assertEqual(int(self.info["tokens"][ITEMS.index(item), 1]), UNKNOWN)
        for relations in self.info["test_snapshots"].values():
            self.assertEqual(len(relations.has_neighbor), N_ITEMS)
            for item in HELD:
                self.assertFalse(bool(relations.has_neighbor[ITEMS.index(item)]))
        for relations in self.info["train_snapshots"].values():
            self.assertEqual(len(relations.has_neighbor), N_ITEMS - len(HELD))

    def test_parts_train_on_the_reduced_catalog_and_test_on_the_whole(self):
        cpu = torch.device("cpu")
        train = parts(self.info, SELLER, "train", "R_hx", "basket", {}, cpu)
        test = parts(self.info, SELLER, "test", "R_hx", "basket", {}, cpu)
        keep = [ITEMS.index(i) for i in self.info["train_items"]]
        for _, data in train:
            self.assertEqual(data.items, self.info["train_items"])
            torch.testing.assert_close(data.z, self.info["z"][keep])
            self.assertEqual(len(data.tokens), len(keep))
            self.assertEqual(len(data.relations.neighbor), len(keep))
        for _, data in test:
            self.assertEqual(data.items, ITEMS)
            self.assertEqual(len(data.relations.neighbor), N_ITEMS)


class NoHoldout(unittest.TestCase):
    def test_zero_fraction_changes_nothing(self):
        z = torch.randn(N_ITEMS, 8, generator=torch.Generator().manual_seed(0))
        info = seller_info(ITEMS, texts(), z, customers(), 0.0, 0, True, counts())
        self.assertEqual(info["train_items"], ITEMS)
        self.assertEqual(info["held_rows"], set())
        self.assertIs(info["train_snapshots"], info["test_snapshots"])
        test = [e for (r, _), ex in info["grouped"].items() if r == "test" for e in ex]
        self.assertTrue(all(e.target_items for e in test))


if __name__ == "__main__":
    unittest.main()
