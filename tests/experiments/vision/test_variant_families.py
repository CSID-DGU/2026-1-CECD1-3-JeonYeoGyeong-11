import unittest

from graphfl_lab.experiments.suites.vision.variant_families import parse_baseline_variant


class VisionVariantFamilyTest(unittest.TestCase):
    def test_parse_baseline_variant_handles_fedopt_suffixes(self):
        self.assertEqual(
            parse_baseline_variant("fedadam_eta0p05_etal0p01", 2),
            (
                "fedadam",
                "fedadam_eta0p05_etal0p01",
                ["--fedopt-eta", "0.05", "--fedopt-eta-l", "0.01"],
            ),
        )

    def test_parse_baseline_variant_handles_fedsim_default_and_explicit_k(self):
        self.assertEqual(
            parse_baseline_variant("fedsim", 3),
            ("fedsim", "fedsim", ["--graph-mode", "knn", "--knn-k", "3"]),
        )
        self.assertEqual(
            parse_baseline_variant("fedsim_rbf_knn_k2", 3),
            ("fedsim", "fedsim_rbf_knn_k2", ["--graph-mode", "rbf_knn", "--knn-k", "2"]),
        )

    def test_parse_baseline_variant_returns_none_for_ours(self):
        self.assertIsNone(parse_baseline_variant("ours_knn_k2", 2))


if __name__ == "__main__":
    unittest.main()
