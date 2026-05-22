from argparse import Namespace
import unittest

from graphfl_lab.flower_runner import args_to_run_config


def _args(**overrides):
    values = {
        "engine": "print-flwr-run",
        "config": "",
        "graph_preset": "none",
        "graph_method": "default_similarity_knn",
        "graph_source": "update",
        "graph_mode": "dense",
        "aggregation_target": "update",
        "graph_scale_sigma": 1.0,
        "knn_k": 2,
        "data_root": "./data/torchvision",
        "out_dir": "./experiments_current/test",
        "projection_dim": 0,
        "_user_arg_dests": frozenset({"graph_method"}),
    }
    values.update(overrides)
    return Namespace(**values)


class FlowerRunnerConfigTest(unittest.TestCase):
    def test_graph_method_is_resolved_before_run_config_serialization(self):
        args = _args()

        cfg = args_to_run_config(args, track="vision-fl")

        self.assertEqual(cfg["graph-method"], "default_similarity_knn")
        self.assertEqual(cfg["graph-source"], "update")
        self.assertEqual(cfg["graph-mode"], "rbf_knn")
        self.assertEqual(cfg["aggregation-target"], "graph_filtered_update")
        self.assertEqual(cfg["graph-scale-sigma"], 0.0)
        self.assertEqual(args.aggregation_target, "update")

    def test_explicit_lower_level_override_survives_graph_method_resolution(self):
        args = _args(_user_arg_dests=frozenset({"graph_method", "aggregation_target"}))

        cfg = args_to_run_config(args, track="vision-fl")

        self.assertEqual(cfg["graph-method"], "default_similarity_knn")
        self.assertEqual(cfg["graph-mode"], "rbf_knn")
        self.assertEqual(cfg["aggregation-target"], "update")


if __name__ == "__main__":
    unittest.main()
