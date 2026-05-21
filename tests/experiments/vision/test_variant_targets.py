import unittest

from graphfl_lab.experiments.suites.vision.variant_targets import parse_target_variant


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class VisionVariantTargetTest(unittest.TestCase):
    def test_parse_target_variant_handles_graph_filtered_family(self):
        _, _, args = parse_target_variant("ours_graph_filtered_magnitude_knn_k2")

        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "graph_filtered_update")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "magnitude_knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "2")

    def test_parse_target_variant_preserves_legacy_spectral_family(self):
        _, _, args = parse_target_variant("ours_spectral_filtered_random_matched_k3")

        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "spectral_filtered_update")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "random")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "3")

    def test_parse_target_variant_handles_weight_agg_and_returns_none_for_sources(self):
        _, label, args = parse_target_variant("ours_weight_agg")
        self.assertEqual(label, "ours_weight_agg")
        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "weight")
        self.assertIsNone(parse_target_variant("ours_weight_graph_knn_k2"))


if __name__ == "__main__":
    unittest.main()
