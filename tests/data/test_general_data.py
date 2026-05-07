import unittest

import numpy as np

from spectral_fl.general_data import _dirichlet_partition, _ensure_min_samples_per_client


class GeneralDataPartitionTest(unittest.TestCase):
    def test_rebalance_gives_each_client_at_least_one_sample(self):
        rng = np.random.default_rng(123)
        labels = np.repeat(np.arange(10), 20)
        shards = _dirichlet_partition(
            rng=rng,
            labels=labels,
            num_clients=20,
            alpha=0.01,
        )

        balanced = _ensure_min_samples_per_client(
            rng=rng,
            shards=shards,
            min_samples=1,
        )

        self.assertEqual(sum(len(x) for x in balanced), len(labels))
        self.assertTrue(all(len(x) >= 1 for x in balanced))


if __name__ == "__main__":
    unittest.main()
