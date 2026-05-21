import unittest

from graphfl_lab.experiments.suites.vision.variant_sources import parse_source_variant


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class VisionVariantSourceTest(unittest.TestCase):
    def test_parse_source_variant_handles_weight_and_head_graphs(self):
        _, _, args = parse_source_variant("ours_weight_graph_filtered_weight_agg_knn_k2")
        self.assertEqual(_last_flag_value(args, "--graph-source"), "weight")
        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "graph_filtered_weight")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "2")

        _, _, args = parse_source_variant("ours_head_weight_graph_spectral_weight_agg_knn_k3")
        self.assertEqual(_last_flag_value(args, "--graph-source"), "classifier_head_weight")
        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "spectral_filtered_weight")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "3")

    def test_parse_source_variant_handles_layerwise_ema_and_tail_sources(self):
        _, _, args = parse_source_variant("ours_layerwise_head_ema_graph_knn_k4")
        self.assertEqual(_last_flag_value(args, "--graph-source"), "layerwise_classifier_head_ema_update")

        _, _, args = parse_source_variant("ours_ema_signal_knn_k2")
        self.assertEqual(_last_flag_value(args, "--graph-source"), "ema_update")
        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "graph_filtered_ema_update")

        _, _, args = parse_source_variant("ours_tail_m2_knn_k1")
        self.assertEqual(_last_flag_value(args, "--graph-source"), "layer_slice_update")
        self.assertEqual(_last_flag_value(args, "--graph-layer-start"), "-2")

    def test_parse_source_variant_returns_none_for_target_only_family(self):
        self.assertIsNone(parse_source_variant("ours_graph_filtered_knn_k2"))


if __name__ == "__main__":
    unittest.main()
