import csv
from pathlib import Path
import random
import tempfile
import unittest

from commerce.packages.data_adapters.assignment import assign_clients, write_assignment
from commerce.packages.data_adapters.instacart import load_assignment, load_instacart, train_cutoff

AISLES = [990001 + i for i in range(6)]
PER_AISLE = 20
TARGETS = {990101: 60, 990102: 40, 990103: 30, 990104: 20}


def product(aisle, k=0):
    """The k-th of PER_AISLE products in an aisle; IDs stay out of the raw range."""
    return 9_900_000 + (aisle - 990000) * 100 + k


def write_tables(root: Path, orders, lines) -> Path:
    """orders: [(order_id, user_id, eval_set, order_number)], lines: [(order_id, product_id)]."""
    def put(name, header, rows):
        with open(root / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    put("aisles.csv", ("aisle_id", "aisle"), [(a, "aisle %d" % a) for a in AISLES])
    put("departments.csv", ("department_id", "department"), [(990001, "dept")])
    put("products.csv", ("product_id", "product_name", "aisle_id", "department_id"),
        [(product(a, k), "product %d" % product(a, k), a, 990001)
         for a in AISLES for k in range(PER_AISLE)])
    put("orders.csv", ("order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"),
        [(o, u, s, n, "" if n == 1 else "5.0") for o, u, s, n in orders])
    put("order_products__prior.csv", ("order_id", "product_id"), lines)
    return root


def population(users=240, seed=7):
    """Each user has 4..12 prior orders, mostly from one favourite aisle."""
    rng = random.Random(seed)
    orders, lines, favourites = [], [], {}
    for u in range(990001, 990001 + users):
        favourites[u] = rng.choice(AISLES)
        for n in range(1, rng.randint(4, 12) + 1):
            order = u * 100 + n
            orders.append((order, u, "prior", n))
            for _ in range(rng.randint(1, 3)):
                aisle = favourites[u] if rng.random() < 0.8 else rng.choice(AISLES)
                lines.append((order, product(aisle, rng.randrange(PER_AISLE))))
    return orders, lines, favourites


def prior_counts(orders):
    counts = {}
    for _, user, eval_set, _ in orders:
        if eval_set == "prior":
            counts[user] = counts.get(user, 0) + 1
    return counts


class TempTables(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def tables(self, name, orders, lines):
        path = self.root / name
        path.mkdir()
        return write_tables(path, orders, lines)


class TrainCutoff(unittest.TestCase):
    def test_floor_of_seventy_percent(self):
        self.assertEqual([train_cutoff(n) for n in (1, 3, 4, 10, 20, 30, 99)],
                         [0, 2, 2, 7, 14, 21, 69])


class TrainOnly(TempTables):
    def test_label_ignores_validation_test_and_official_orders(self):
        a, b = AISLES[0], AISLES[1]
        orders = [(990001000 + n, 990001, "prior", n) for n in range(1, 11)]  # cutoff 7
        orders.append((990001011, 990001, "train", 11))  # official train order, not prior
        lines = [(990001000 + n, product(a)) for n in range(1, 8)]
        lines += [(990001000 + n, product(b, k)) for n in range(8, 12) for k in range(5)]
        lines.append((990001001, product(a)))  # a duplicate line counts once
        result = assign_clients(self.tables("t", orders, lines), {990101: 1}, alpha=0.25, seed=0)
        self.assertEqual(result.labels, {990001: a})  # b: 15 lines over all prior orders, a: 7 in train
        self.assertEqual(result.loads, {990101: 7})

    def test_later_orders_do_not_move_anyone(self):
        orders, lines, favourites = population()
        n_prior = prior_counts(orders)
        later = {o: u for o, u, _, n in orders if n > train_cutoff(n_prior[u])}
        # Fill every validation/test order with another aisle, enough to flip the label
        # of anyone whose later orders were counted.
        changed = [(o, p) for o, p in lines if o not in later]
        for order, user in later.items():
            other = AISLES[(AISLES.index(favourites[user]) + 1) % len(AISLES)]
            changed += [(order, product(other, k)) for k in range(PER_AISLE)]
        base = assign_clients(self.tables("base", orders, lines), TARGETS, alpha=0.25, seed=3)
        moved = assign_clients(self.tables("moved", orders, changed), TARGETS, alpha=0.25, seed=3)
        self.assertEqual(base.clients, moved.clients)
        self.assertEqual(base.labels, moved.labels)

    def test_customer_without_train_orders_is_not_eligible(self):
        orders = [(990001001, 990001, "prior", 1), (990002001, 990002, "prior", 1),
                  (990002002, 990002, "prior", 2)]
        lines = [(990001001, product(AISLES[0])), (990002001, product(AISLES[0])),
                 (990002002, product(AISLES[0]))]
        result = assign_clients(self.tables("t", orders, lines), {990101: 5}, alpha=0.25, seed=0)
        self.assertEqual(result.clients, {990002: 990101})
        self.assertEqual(result.record["users_without_train_orders"], 1)


class Split(TempTables):
    @classmethod
    def setUpClass(cls):
        cls.orders, cls.lines, _ = population()

    def run_split(self, name, **kwargs):
        params = {"alpha": 0.25, "seed": 11, **kwargs}
        return assign_clients(self.tables(name, self.orders, self.lines), TARGETS, **params)

    def test_pick_stops_near_the_target(self):
        result = self.run_split("a")
        need = sum(TARGETS.values())
        self.assertGreaterEqual(result.record["train_orders_picked"], 1.25 * need)
        self.assertLess(result.record["users_picked"], len(prior_counts(self.orders)))
        self.assertEqual(set(result.clients), set(result.labels))
        self.assertLessEqual(set(result.clients.values()), set(TARGETS))
        self.assertEqual(sum(result.loads.values()), result.record["train_orders_picked"])

    def test_same_seed_same_split(self):
        self.assertEqual(self.run_split("a").clients, self.run_split("b").clients)
        self.assertNotEqual(self.run_split("c").clients, self.run_split("d", seed=12).clients)

    def test_small_alpha_concentrates_labels(self):
        def top_share(result):
            per_client = {}
            for user, client in result.clients.items():
                per_client.setdefault(client, []).append(result.labels[user])
            shares = [max(labels.count(x) for x in labels) / len(labels) for labels in per_client.values()]
            return sum(shares) / len(shares)
        skewed = self.run_split("skewed", alpha=0.05)
        uniform = self.run_split("uniform", alpha=float("inf"))
        self.assertGreater(top_share(skewed), top_share(uniform) + 0.1)
        self.assertEqual(uniform.record["alpha"], "inf")

    def test_split_feeds_the_adapter(self):
        result = self.run_split("a")
        path = self.root / "assignment.csv"
        write_assignment(path, result.clients)
        self.assertEqual(load_assignment(path), result.clients)
        sample = load_instacart(self.root / "a", load_assignment(path))
        sellers = {}
        for event in sample.events:
            sellers.setdefault(event["customer_id_local"], set()).add(event["seller_id"])
        self.assertEqual(len(sellers), len(result.clients))
        self.assertTrue(all(len(s) == 1 for s in sellers.values()))
