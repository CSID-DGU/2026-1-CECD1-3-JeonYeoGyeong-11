import math
import random
import unittest
from unittest import mock

import torch

from commerce.packages.recommender import relations as rel
from commerce.packages.recommender.relations import (
    CENSORED_BIN, N_MAX, build_relations, gap_bin, replay_relations,
)
from commerce.packages.recommender.tests import visits_of


def catalog(n):
    items = tuple("ic-p-%d" % i for i in range(1, n + 1))
    return items, {item: row for row, item in enumerate(items)}


A, B, C, D = 0, 1, 2, 3  # rows of ic-p-1 .. ic-p-4


class HandCounted(unittest.TestCase):
    """u1: {A,B} then {B,C} 3 days later; u2: {A,C} then {A} after a capped 30 days. D never sells."""

    def setUp(self):
        self.items, self.row_of = catalog(4)
        visits = {"u1": visits_of("u1", [[1, 2], [2, 3]], gaps={2: 3.0}),
                  "u2": visits_of("u2", [[1, 3], [1]], gaps={2: 30.0})}
        self.r = build_relations(visits, self.row_of, 4)

    def slot(self, c, j):
        return self.r.neighbor[c].tolist().index(j)

    def test_neighbours_and_the_item_without_any(self):
        self.assertEqual(sorted(j for j in self.r.neighbor[A].tolist() if j >= 0), [B, C])
        self.assertEqual(self.r.has_neighbor.tolist(), [True, True, True, False])
        self.assertTrue((self.r.neighbor[D] == -1).all())

    def test_pair_features(self):
        f = self.r.features[A, self.slot(A, C)].tolist()
        # customers: A 2, C 2, both 2. baskets: A 3, C 2, both 1. A->C once (u1), C->A once (u2).
        expected = [2 / math.sqrt(2 * 2), math.log1p(2), 1 / math.sqrt(3 * 2), math.log1p(1),
                    math.log1p(2), math.log1p(2), 1.0, 1.0, math.log1p(1), math.log1p(1)]
        for got, want in zip(f, expected):
            self.assertAlmostEqual(got, want, places=5)

    def test_time_bins_keep_direction_and_censoring(self):
        s = self.slot(A, C)
        self.assertEqual(float(self.r.time_forward[A, s, 3]), 1.0)  # A then C, 3 days
        self.assertEqual(float(self.r.time_backward[A, s, CENSORED_BIN]), 1.0)  # C then A, capped 30
        b = self.slot(B, C)
        self.assertEqual(float(self.r.time_forward[B, b].sum()), 1.0)
        self.assertEqual(float(self.r.time_backward[B, b].sum()), 0.0)  # C never came before B

    def test_same_basket_is_not_a_time_pair(self):
        b = self.slot(A, B)
        # A and B share u1's first basket; the only time pair is A -> B across u1's visits.
        self.assertEqual(float(self.r.time_forward[A, b, 3]), 1.0)
        self.assertAlmostEqual(float(self.r.features[A, b, 8]), math.log1p(1), places=6)  # once, not twice
        self.assertEqual(float(self.r.features[A, b, 7]), 0.0)  # no B -> A

    def test_bins(self):
        self.assertEqual([gap_bin(0.0, False), gap_bin(29.9, False), gap_bin(45.0, False), gap_bin(30.0, True)],
                         [0, 29, 30, CENSORED_BIN])


class ThreeRelationsPickDifferentNeighbours(unittest.TestCase):
    """model.md §3: A's top customer, basket and time neighbours are X, Y and Z, all different."""

    def test_round_robin_keeps_one_of_each(self):
        items, row_of = catalog(12)
        a, x, y, z = 1, 2, 3, 4
        visits = {}
        for i, filler in enumerate((5, 6, 7)):  # three customers buy A, then a filler, then X
            visits["x%d" % i] = visits_of("x%d" % i, [[a], [filler], [x]])
        visits["y"] = visits_of("y", [[a, y], [8], [9], [a, y]])  # A and Y share two baskets
        for i in range(2):  # two customers buy Z right after A
            visits["z%d" % i] = visits_of("z%d" % i, [[a], [z]])
        with mock.patch.object(rel, "NEIGHBOR_K", 1), mock.patch.object(rel, "N_MAX", 3):
            r = build_relations(visits, row_of, 12)
        chosen = r.neighbor[row_of["ic-p-%d" % a]].tolist()
        self.assertEqual(chosen[:3], [row_of["ic-p-%d" % i] for i in (x, y, z)])


class Snapshot(unittest.TestCase):
    def test_visits_after_the_prefix_do_not_count(self):
        items, row_of = catalog(20)
        rng = random.Random(0)
        base = {c: [[rng.randint(1, 20) for _ in range(3)] for _ in range(10)] for c in "abcd"}
        changed = {c: b[:4] + [[rng.randint(1, 20) for _ in range(3)] for _ in range(6)] for c, b in base.items()}
        make = lambda baskets: {c: visits_of(c, b) for c, b in baskets.items()}
        first = replay_relations(make(base), 4, row_of, 20)  # bucket 4 shows 4 of 10 visits
        second = replay_relations(make(changed), 4, row_of, 20)
        for name in ("neighbor", "features", "time_forward", "time_backward", "has_neighbor"):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)
        self.assertFalse(replay_relations(make(base), 0, row_of, 20).has_neighbor.any())

    def test_neighbour_budget(self):
        items, row_of = catalog(60)
        rng = random.Random(1)
        visits = {c: visits_of(c, [rng.sample(range(1, 61), 10) for _ in range(6)]) for c in "abcdefgh"}
        r = build_relations(visits, row_of, 60)
        self.assertEqual(r.neighbor.shape[1], N_MAX)
        self.assertLessEqual(r.neighbors_per_item, N_MAX)
