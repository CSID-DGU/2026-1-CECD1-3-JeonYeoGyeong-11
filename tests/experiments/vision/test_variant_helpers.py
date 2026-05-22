import tempfile
import unittest
from pathlib import Path

from graphfl_lab.experiments.suites.vision.variant_helpers import (
    canonical_result_path_for_variant,
    compatibility_result_path_for_variant,
    diagnostic_graph_args,
    diagnostic_graph_free_args,
    legacy_residual_reweight_args,
    resolve_result_path_for_variant,
    result_path_for_variant,
    token_float,
)


class VisionVariantHelperTest(unittest.TestCase):
    def test_token_float_converts_compact_float_text(self):
        self.assertEqual(token_float("0p01"), "0.01")
        self.assertEqual(token_float("1.5"), "1.5")

    def test_legacy_residual_reweight_args_keeps_old_control_path(self):
        args = legacy_residual_reweight_args("random", "3")

        self.assertIn("--graph-source", args)
        self.assertIn("update", args)
        self.assertIn("--aggregation-target", args)
        self.assertIn("--knn-k", args)
        self.assertEqual(args[-1], "3")

    def test_diagnostic_graph_args_adds_cluster_auto_k_for_cluster_only(self):
        args = diagnostic_graph_args(
            correction_family="clustering_only",
            knn_k="2",
            control_graph_mode="shuffled",
            cluster_method="spectral",
        )

        self.assertEqual(args[args.index("--knn-k") + 1], "2")
        self.assertEqual(args[args.index("--control-graph-mode") + 1], "shuffled")
        self.assertEqual(args[args.index("--cluster-method") + 1], "spectral")
        self.assertEqual(args[args.index("--cluster-auto-k") + 1], "true")

    def test_diagnostic_graph_free_args_sets_mode_specific_knobs(self):
        self.assertIn("--clip-quantile", diagnostic_graph_free_args("norm_clip"))
        self.assertIn("--contribution-cap", diagnostic_graph_free_args("contribution_cap"))
        self.assertIn("--graph-free-gamma", diagnostic_graph_free_args("dominance_reweight"))

    def test_result_path_for_variant_uses_canonical_vision_filename(self):
        path = result_path_for_variant(Path("out"), "ours", 7, "ours_knn_k2_seed7")

        self.assertEqual(
            path,
            Path("out") / "result_vision_ours_seed7_ours_knn_k2_seed7.json",
        )

    def test_resolve_result_path_for_variant_reuses_compatibility_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            compatibility = compatibility_result_path_for_variant(
                out_dir, "ours", 7, "ours_knn_k2_seed7"
            )
            compatibility.write_text("{}", encoding="utf-8")
            resolved = resolve_result_path_for_variant(
                out_dir, "ours", 7, "ours_knn_k2_seed7"
            )

        self.assertEqual(resolved, compatibility)

    def test_resolve_result_path_for_variant_prefers_canonical_when_both_exist(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            canonical = canonical_result_path_for_variant(
                out_dir, "ours", 7, "ours_knn_k2_seed7"
            )
            compatibility = compatibility_result_path_for_variant(
                out_dir, "ours", 7, "ours_knn_k2_seed7"
            )
            canonical.write_text('{"canonical": true}', encoding="utf-8")
            compatibility.write_text('{"legacy": true}', encoding="utf-8")
            resolved = resolve_result_path_for_variant(
                out_dir, "ours", 7, "ours_knn_k2_seed7"
            )

        self.assertEqual(resolved, canonical)


if __name__ == "__main__":
    unittest.main()
