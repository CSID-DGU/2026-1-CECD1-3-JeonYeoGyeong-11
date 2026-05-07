import unittest

from spectral_fl.experiments.general.stress_grid import (
    build_auto_review_rows,
    expand_variant_templates,
    variant_k_from_label,
)


class GeneralStressGridTest(unittest.TestCase):
    def test_expand_variant_templates_deduplicates_fedavg_and_expands_k(self):
        variants = expand_variant_templates(
            [
                "fedavg",
                "ours_knn_k{k}_fixed_tau",
                "ours_random_matched_k{k}_fixed_tau",
                "fedavg",
            ],
            [1, 3],
        )

        self.assertEqual(
            variants,
            [
                "fedavg",
                "ours_knn_k1_fixed_tau",
                "ours_knn_k3_fixed_tau",
                "ours_random_matched_k1_fixed_tau",
                "ours_random_matched_k3_fixed_tau",
            ],
        )

    def test_variant_k_from_label_handles_supported_tokens(self):
        self.assertEqual(variant_k_from_label("ours_knn_k2_fixed_tau"), 2)
        self.assertEqual(variant_k_from_label("ours_tail_m2_knn_k3_fixed_tau"), 3)
        self.assertEqual(variant_k_from_label("ours_random_matched_k1"), 1)
        self.assertIsNone(variant_k_from_label("fedavg"))

    def test_auto_review_promotes_rescue_that_beats_random(self):
        rows = [
            {
                "variant": "fedavg",
                "mean_acc": "0.30",
                "dirichlet_alpha": 0.03,
                "local_epochs": 2,
                "num_clients": 10,
                "train_subset_size": 1000,
                "test_subset_size": 1000,
            },
            {
                "variant": "ours_knn_k1_fixed_tau",
                "mean_fedavg_acc": "0.30",
                "mean_acc": "0.35",
                "mean_delta": "0.05",
                "min_delta": "0.02",
                "win_rate": "1.0",
                "dirichlet_alpha": 0.03,
                "local_epochs": 2,
                "num_clients": 10,
                "train_subset_size": 1000,
                "test_subset_size": 1000,
            },
            {
                "variant": "ours_random_matched_k1_fixed_tau",
                "mean_fedavg_acc": "0.30",
                "mean_acc": "0.31",
                "mean_delta": "0.01",
                "min_delta": "0.00",
                "win_rate": "0.5",
                "dirichlet_alpha": 0.03,
                "local_epochs": 2,
                "num_clients": 10,
                "train_subset_size": 1000,
                "test_subset_size": 1000,
            },
        ]

        review = build_auto_review_rows(
            rows,
            collapse_acc_threshold=0.45,
            meaningful_delta=0.01,
            random_margin=0.005,
        )
        knn = next(r for r in review if r["variant"] == "ours_knn_k1_fixed_tau")

        self.assertEqual(knn["verdict"], "promising_rescue")
        self.assertTrue(knn["fedavg_collapsed"])
        self.assertAlmostEqual(knn["delta_vs_random"], 0.04)

    def test_auto_review_flags_saturated_min_client_weight(self):
        rows = [
            {
                "variant": "fedavg",
                "mean_acc": "0.30",
                "dirichlet_alpha": 0.03,
                "local_epochs": 2,
                "num_clients": 20,
                "min_client_weight": 0.05,
                "train_subset_size": 1000,
                "test_subset_size": 1000,
            },
            {
                "variant": "ours_knn_k2_fixed_tau",
                "mean_fedavg_acc": "0.30",
                "mean_acc": "0.40",
                "mean_delta": "0.10",
                "min_delta": "0.05",
                "win_rate": "1.0",
                "dirichlet_alpha": 0.03,
                "local_epochs": 2,
                "num_clients": 20,
                "min_client_weight": 0.05,
                "train_subset_size": 1000,
                "test_subset_size": 1000,
            },
        ]

        review = build_auto_review_rows(
            rows,
            collapse_acc_threshold=0.45,
            meaningful_delta=0.01,
            random_margin=0.005,
        )
        knn = next(r for r in review if r["variant"] == "ours_knn_k2_fixed_tau")

        self.assertEqual(knn["verdict"], "weight_floor_saturated")
        self.assertTrue(knn["weight_floor_saturated"])
        self.assertAlmostEqual(knn["weight_floor_total"], 1.0)


if __name__ == "__main__":
    unittest.main()
