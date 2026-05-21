import unittest

from graphfl_lab.experiments.suites.vision.variant_legacy import (
    parse_legacy_residual_variant,
)


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class VisionVariantLegacyTest(unittest.TestCase):
    def test_parse_legacy_residual_variant_handles_old_and_compat_tokens(self):
        _, label, args = parse_legacy_residual_variant("ours_residual_reweight_knn_k2")
        self.assertEqual(label, "ours_residual_reweight_knn_k2")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "2")
        self.assertEqual(_last_flag_value(args, "--aggregation-target"), "update")

        _, label, args = parse_legacy_residual_variant(
            "ours_legacy_residual_reweight_magnitude_knn_k3"
        )
        self.assertEqual(label, "ours_legacy_residual_reweight_magnitude_knn_k3")
        self.assertEqual(_last_flag_value(args, "--graph-mode"), "magnitude_knn")
        self.assertEqual(_last_flag_value(args, "--knn-k"), "3")

    def test_parse_legacy_residual_variant_returns_none_for_current_tokens(self):
        self.assertIsNone(parse_legacy_residual_variant("ours_knn_k2"))


if __name__ == "__main__":
    unittest.main()
