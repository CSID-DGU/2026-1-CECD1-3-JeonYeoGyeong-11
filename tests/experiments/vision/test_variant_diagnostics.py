import unittest

from graphfl_lab.experiments.suites.vision.variant_diagnostics import (
    parse_diagnostic_variant,
)


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class VisionVariantDiagnosticTest(unittest.TestCase):
    def test_parse_diagnostic_variant_handles_real_and_control_graphs(self):
        _, label, real_args = parse_diagnostic_variant("ours_real_graph", 2)
        self.assertEqual(label, "ours_real_graph")
        self.assertEqual(_last_flag_value(real_args, "--correction-family"), "real_graph")
        self.assertEqual(_last_flag_value(real_args, "--knn-k"), "2")

        _, label, control_args = parse_diagnostic_variant("ours_shuffled_control_k3", 2)
        self.assertEqual(label, "ours_shuffled_control_k3")
        self.assertEqual(_last_flag_value(control_args, "--correction-family"), "control_graph")
        self.assertEqual(_last_flag_value(control_args, "--control-graph-mode"), "shuffled")
        self.assertEqual(_last_flag_value(control_args, "--knn-k"), "3")

    def test_parse_diagnostic_variant_handles_cluster_and_graph_free_controls(self):
        _, _, cluster_args = parse_diagnostic_variant("ours_cluster_only", 4)
        self.assertEqual(_last_flag_value(cluster_args, "--correction-family"), "clustering_only")
        self.assertEqual(_last_flag_value(cluster_args, "--cluster-auto-k"), "true")
        self.assertEqual(_last_flag_value(cluster_args, "--knn-k"), "4")

        _, _, graph_free_args = parse_diagnostic_variant("ours_graphfree_cap", 4)
        self.assertEqual(_last_flag_value(graph_free_args, "--correction-family"), "graph_free")
        self.assertEqual(_last_flag_value(graph_free_args, "--graph-free-mode"), "contribution_cap")
        self.assertEqual(_last_flag_value(graph_free_args, "--contribution-cap"), "0.35")

    def test_parse_diagnostic_variant_returns_none_for_other_families(self):
        self.assertIsNone(parse_diagnostic_variant("ours_knn_k2", 2))


if __name__ == "__main__":
    unittest.main()
