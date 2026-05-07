import csv
import tempfile
import unittest
from pathlib import Path

from spectral_fl.experiments.general.suite import (
    _variant_k_number,
    append_validation_verdict,
    collect_timing_features,
    write_knn_vs_random_matched_csv,
)


class GeneralSuiteReportTest(unittest.TestCase):
    def test_variant_k_number_accepts_tau_suffix(self):
        self.assertEqual(_variant_k_number("ours_knn_k1_fixed_tau"), 1)
        self.assertEqual(_variant_k_number("ours_knn_k3_norm_tau"), 3)
        self.assertEqual(
            _variant_k_number(
                "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only"
            ),
            1,
        )
        self.assertEqual(
            _variant_k_number("ours_legacy_residual_reweight_knn_k2_fixed_tau"),
            2,
        )
        self.assertIsNone(_variant_k_number("ours_random_matched_k1_fixed_tau"))

    def test_knn_random_csv_matches_same_suffix_and_k1(self):
        summary_rows = [
            {
                "variant": "ours_knn_k1_fixed_tau",
                "mean_delta": 0.010,
                "min_delta": 0.004,
                "win_rate": 1.0,
                "mean_graph_density": 0.2,
            },
            {
                "variant": "ours_random_matched_k1_fixed_tau",
                "mean_delta": -0.005,
                "min_delta": -0.010,
                "win_rate": 0.0,
                "mean_graph_density": 0.2,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_knn_vs_random_matched_csv(Path(tmp), summary_rows)
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["k"], "1")
        self.assertEqual(rows[0]["variant_suffix"], "_fixed_tau")
        self.assertEqual(rows[0]["knn_variant"], "ours_knn_k1_fixed_tau")
        self.assertEqual(rows[0]["random_variant"], "ours_random_matched_k1_fixed_tau")
        self.assertAlmostEqual(float(rows[0]["difference_mean_delta"]), 0.015)
        self.assertEqual(rows[0]["interpretation"], "similarity_graph_helpful")

    def test_knn_random_csv_matches_spectral_filtered_suffixes(self):
        summary_rows = [
            {
                "variant": (
                    "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only"
                ),
                "mean_delta": 0.034,
                "min_delta": 0.034,
                "win_rate": 1.0,
                "mean_graph_density": 0.09,
            },
            {
                "variant": (
                    "ours_spectral_filtered_random_matched_k1_lp0p5_serverm_fixed_tau_spectral_only"
                ),
                "mean_delta": 0.053,
                "min_delta": 0.053,
                "win_rate": 1.0,
                "mean_graph_density": 0.09,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_knn_vs_random_matched_csv(Path(tmp), summary_rows)
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["k"], "1")
        self.assertEqual(
            rows[0]["variant_suffix"],
            "_lp0p5_serverm_fixed_tau_spectral_only",
        )
        self.assertEqual(
            rows[0]["knn_variant"],
            "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only",
        )
        self.assertEqual(
            rows[0]["random_variant"],
            "ours_spectral_filtered_random_matched_k1_lp0p5_serverm_fixed_tau_spectral_only",
        )
        self.assertAlmostEqual(float(rows[0]["difference_mean_delta"]), -0.019)
        self.assertEqual(rows[0]["interpretation"], "random_better")

    def test_knn_random_csv_matches_legacy_residual_reweight_suffixes(self):
        summary_rows = [
            {
                "variant": "ours_legacy_residual_reweight_knn_k1_serverm_fixed_tau",
                "mean_delta": 0.021,
                "min_delta": 0.010,
                "win_rate": 1.0,
                "mean_graph_density": 0.09,
            },
            {
                "variant": (
                    "ours_legacy_residual_reweight_random_matched_k1_serverm_fixed_tau"
                ),
                "mean_delta": 0.018,
                "min_delta": 0.006,
                "win_rate": 1.0,
                "mean_graph_density": 0.09,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_knn_vs_random_matched_csv(Path(tmp), summary_rows)
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant_suffix"], "_serverm_fixed_tau")
        self.assertEqual(
            rows[0]["knn_variant"],
            "ours_legacy_residual_reweight_knn_k1_serverm_fixed_tau",
        )
        self.assertEqual(
            rows[0]["random_variant"],
            "ours_legacy_residual_reweight_random_matched_k1_serverm_fixed_tau",
        )
        self.assertAlmostEqual(float(rows[0]["difference_mean_delta"]), 0.003)

    def test_validation_verdict_matches_spectral_filtered_pairs(self):
        summary_rows = [
            {"variant": "fedavg", "mean_delta": 0.0},
            {
                "variant": (
                    "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only"
                ),
                "mean_delta": 0.034,
                "min_delta": 0.034,
                "win_rate": 1.0,
            },
            {
                "variant": (
                    "ours_spectral_filtered_random_matched_k1_lp0p5_serverm_fixed_tau_spectral_only"
                ),
                "mean_delta": 0.053,
                "min_delta": 0.053,
                "win_rate": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interpretation.md"
            path.write_text("# interpretation\n", encoding="utf-8")
            append_validation_verdict(Path(tmp), summary_rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn(
            "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only",
            text,
        )
        self.assertIn(
            "ours_spectral_filtered_random_matched_k1_lp0p5_serverm_fixed_tau_spectral_only",
            text,
        )
        self.assertIn("Mixed outcome", text)

    def test_collect_timing_features_prefers_observed_suite_time(self):
        result = {
            "meta": {
                "experiment": {"rounds": 4},
                "timing": {"total_wall_time_sec": 12.0},
            },
            "results": {"ours": {"timing": {"wall_time_sec": 10.0}}},
        }

        features = collect_timing_features(
            result_obj=result,
            method="ours",
            observed_wall_time_sec=14.0,
            reused_existing_result=False,
        )

        self.assertEqual(features["run_wall_time_sec"], 14.0)
        self.assertEqual(features["result_method_wall_time_sec"], 10.0)
        self.assertEqual(features["result_total_wall_time_sec"], 12.0)
        self.assertEqual(features["seconds_per_round"], 3.5)
        self.assertEqual(features["timing_source"], "suite_observed")

    def test_collect_timing_features_uses_result_time_for_reused_runs(self):
        result = {
            "meta": {"experiment": {"rounds": 5}},
            "results": {"fedavg": {"timing": {"wall_time_sec": 20.0}}},
        }

        features = collect_timing_features(
            result_obj=result,
            method="fedavg",
            observed_wall_time_sec=None,
            reused_existing_result=True,
        )

        self.assertEqual(features["run_wall_time_sec"], 20.0)
        self.assertEqual(features["seconds_per_round"], 4.0)
        self.assertEqual(features["timing_source"], "result_method_timing")
        self.assertTrue(features["reused_existing_result"])


if __name__ == "__main__":
    unittest.main()
