import math
import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.client_metrics import (
    extract_metric,
    weighted_optional_mean,
)


class GraphFLClientMetricsTest(unittest.TestCase):
    def test_extract_metric_uses_first_available_alias_per_client(self):
        metrics = [
            {"train_accuracy": "0.5"},
            {"accuracy": 0.75},
            {"accuracy": "bad", "train_accuracy": 0.25},
            {},
        ]

        values = extract_metric(metrics, "accuracy", "train_accuracy")

        self.assertEqual(values, [0.5, 0.75, 0.25, None])

    def test_extract_metric_returns_none_when_all_aliases_missing_or_invalid(self):
        metrics = [{"loss": "bad"}, {}]

        values = extract_metric(metrics, "loss")

        self.assertIsNone(values)

    def test_weighted_optional_mean_skips_missing_values(self):
        value = weighted_optional_mean(
            [1.0, None, 3.0],
            np.array([2.0, 100.0, 1.0], dtype=np.float64),
        )

        self.assertAlmostEqual(value, 5.0 / 3.0)

    def test_weighted_optional_mean_returns_nan_without_valid_denominator(self):
        self.assertTrue(
            math.isnan(weighted_optional_mean(None, np.array([1.0])))
        )
        self.assertTrue(
            math.isnan(weighted_optional_mean([None], np.array([1.0])))
        )


if __name__ == "__main__":
    unittest.main()
