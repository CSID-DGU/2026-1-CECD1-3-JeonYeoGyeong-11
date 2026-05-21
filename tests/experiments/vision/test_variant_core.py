import unittest

from graphfl_lab.experiments.suites.vision.variant_core import parse_core_graph_variant


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class VisionVariantCoreTest(unittest.TestCase):
    def test_parse_core_graph_variant_handles_default_and_basic_graphs(self):
        _, label, args = parse_core_graph_variant("ours_default_graph_k3", 2)
        self.assertEqual(label, "ours_default_graph_k3")
        self.assertEqual(_last_flag_value(args, "--graph-method"), "default_similarity_knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "3")

        _, _, args = parse_core_graph_variant("ours_knn", 4)
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "4")

    def test_parse_core_graph_variant_handles_random_and_custom_modes(self):
        _, _, args = parse_core_graph_variant("ours_random_matched_k5", 2)
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "random")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "5")

        _, label, args = parse_core_graph_variant("ours_graph_mode_my_relation", 2)
        self.assertEqual(label, "ours_graph_mode_my_relation")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "my_relation")

    def test_parse_core_graph_variant_returns_none_for_source_specific_family(self):
        self.assertIsNone(parse_core_graph_variant("ours_weight_graph_knn_k2", 2))


if __name__ == "__main__":
    unittest.main()
