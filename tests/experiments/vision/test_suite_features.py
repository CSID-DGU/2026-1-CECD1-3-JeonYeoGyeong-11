import math
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.experiments.suites.vision.features import (
    collect_run_features,
    load_preloaded_fedavg_accs,
    missing_timing_features,
    rank_key,
    truthy,
)
from graphfl_lab.experiments.suites.result_writer import write_json


class VisionSuiteFeatureTest(unittest.TestCase):
    def test_collect_run_features_exports_diagnostic_means_and_aliases(self):
        result = {
            "results": {
                "ours": {
                    "round_trace": [
                        {
                            "graph_mode": "knn",
                            "spectral_filter_strength": 0.5,
                            "h_spec": 0.2,
                            "di_pre": 0.9,
                            "di_post": 0.4,
                        },
                        {
                            "graph_mode": "random",
                            "graph_filter_strength": 0.7,
                            "h_spec": 0.6,
                            "di_pre": 0.7,
                            "di_post": 0.3,
                        },
                    ]
                }
            }
        }

        features = collect_run_features(result, "ours")

        self.assertEqual(features["graph_mode"], "random")
        self.assertEqual(features["graph_filter_strength"], "0.7")
        self.assertEqual(features["spectral_filter_strength"], "0.5")
        self.assertAlmostEqual(features["mean_h_spec"], 0.4)
        self.assertAlmostEqual(features["mean_di_pre"], 0.8)
        self.assertAlmostEqual(features["mean_di_post"], 0.35)

    def test_collect_run_features_ignores_baseline_methods(self):
        self.assertEqual(collect_run_features({"results": {}}, "fedavg"), {})

    def test_load_preloaded_fedavg_accs_reads_latest_supported_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "result_general_fedavg_seed1.json",
                {
                    "meta": {"seed": 1},
                    "results": {
                        "fedavg": {
                            "metrics_distributed": {
                                "accuracy": [[1, 0.1], [2, 0.2]]
                            }
                        }
                    },
                },
            )
            write_json(
                root / "result_vision_fedavg_seed1.json",
                {
                    "meta": {"experiment": {"seed": 1}},
                    "results": {
                        "fedavg": {
                            "metrics_distributed": {
                                "accuracy": [[1, 0.3], [2, 0.4]]
                            }
                        }
                    },
                },
            )

            accs = load_preloaded_fedavg_accs(root)

        self.assertEqual(accs, {1: 0.4})

    def test_missing_timing_features_and_truthy_and_rank_key(self):
        timing = missing_timing_features("preloaded_fedavg")

        self.assertTrue(math.isnan(timing["run_wall_time_sec"]))
        self.assertEqual(timing["timing_source"], "preloaded_fedavg")
        self.assertTrue(truthy("yes"))
        self.assertFalse(truthy("0"))
        self.assertGreater(
            rank_key({"variant": "ours", "mean_delta": 1.0}),
            rank_key({"variant": "fedavg"}),
        )


if __name__ == "__main__":
    unittest.main()
