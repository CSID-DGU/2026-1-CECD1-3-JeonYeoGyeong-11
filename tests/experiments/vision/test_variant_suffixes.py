import argparse
import unittest

from graphfl_lab.experiments.suites.vision.variant_suffixes import parse_suffix_variant


def base_parser(variant, args):
    if variant == "base":
        return "ours", "base", ["--graph-mode", "knn"]
    if variant == "fedavg":
        return "fedavg", "fedavg", []
    suffix = parse_suffix_variant(variant, args, base_parser)
    if suffix is not None:
        return suffix
    raise ValueError(variant)


class VisionVariantSuffixTest(unittest.TestCase):
    def _args(self):
        return argparse.Namespace(
            ours_server_learning_rate=1.0,
            server_momentum=0.9,
        )

    def test_parse_suffix_variant_handles_tau_lowpass_and_server_momentum(self):
        _, label, args = parse_suffix_variant("base_lp0p5_serverm_fixed_tau", self._args(), base_parser)

        self.assertEqual(label, "base_lp0p5_serverm_fixed_tau")
        self.assertIn("--graph-filter-strength", args)
        self.assertIn("0.5", args)
        self.assertIn("--ours-server-momentum", args)
        self.assertIn("0.9", args)
        self.assertIn("--disable-adaptive-tau", args)

    def test_parse_suffix_variant_handles_filter_only_and_rejects_baselines(self):
        _, label, args = parse_suffix_variant("base_graph_filter_only", self._args(), base_parser)

        self.assertEqual(label, "base_graph_filter_only")
        self.assertIn("--conflict-mix", args)
        self.assertIn("0.0", args)

        with self.assertRaises(ValueError):
            parse_suffix_variant("fedavg_graph_filter_only", self._args(), base_parser)

    def test_parse_suffix_variant_returns_none_without_suffix(self):
        self.assertIsNone(parse_suffix_variant("base", self._args(), base_parser))


if __name__ == "__main__":
    unittest.main()
