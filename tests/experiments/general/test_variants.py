import argparse
import unittest

from spectral_fl.experiments.suites.general.variants import parse_variant


def _last_flag_value(args, flag):
    indexes = [i for i, value in enumerate(args) if value == flag]
    if not indexes:
        return None
    return args[indexes[-1] + 1]


class GeneralSuiteVariantTest(unittest.TestCase):
    def _args(self):
        return argparse.Namespace(
            knn_k=2,
            server_momentum=0.9,
            ours_server_learning_rate=1.0,
            ours_server_momentum=0.0,
            fedopt_eta=0.1,
            fedopt_eta_l=0.1,
            fedopt_beta1=0.9,
            fedopt_beta2=0.99,
            fedopt_tau=1e-9,
            client_update_ema_alpha=0.8,
        )

    def test_tau_suffix_extends_base_variant(self):
        method, label, extras = parse_variant("ours_knn_k3_norm_tau", self._args())

        self.assertEqual(method, "ours")
        self.assertEqual(label, "ours_knn_k3_norm_tau")
        self.assertIn("--graph-mode", extras)
        self.assertIn("knn", extras)
        self.assertIn("--tau-source", extras)
        self.assertIn("h_spec_normalized", extras)

    def test_fixed_tau_suffix_extends_uniform_variant(self):
        _, label, extras = parse_variant("ours_uniform_fixed_tau", self._args())

        self.assertEqual(label, "ours_uniform_fixed_tau")
        self.assertIn("--graph-mode", extras)
        self.assertIn("uniform", extras)
        self.assertIn("--disable-adaptive-tau", extras)
        self.assertIn("true", extras)

    def test_new_extension_variants_are_wrapped(self):
        cases = {
            "fedavgm": ("fedavgm", []),
            "fedadagrad": ("fedadagrad", []),
            "fedadagrad_eta0p05": ("fedadagrad", ["--fedopt-eta", "0.05"]),
            "fedadam": ("fedadam", []),
            "fedadam_eta0p05_etal0p01": (
                "fedadam",
                ["--fedopt-eta", "0.05", "--fedopt-eta-l", "0.01"],
            ),
            "fedyogi": ("fedyogi", []),
            "fedyogi_etal0p03": ("fedyogi", ["--fedopt-eta-l", "0.03"]),
            "fednova": ("fednova", []),
            "fednova_slr0p5": ("fednova", ["--server-learning-rate", "0.5"]),
            "fedprox_mu0p01": ("fedprox", ["--fedprox-mu", "0.01"]),
            "fedmedian": ("fedmedian", []),
            "fedtrimmedavg_beta0p1": ("fedtrimmedavg", ["--trimmed-beta", "0.1"]),
            "fedsim_k2": ("fedsim", ["--graph-mode", "knn", "--knn-k", "2"]),
            "ours_layerwise_knn_k2": ("--graph-source", "layerwise_update"),
            "ours_signed_abs": ("--graph-mode", "signed_abs"),
            "ours_negative_knn_k2": ("--graph-mode", "negative_knn"),
            "ours_rbf_knn_k2": ("--graph-mode", "rbf_knn"),
            "ours_learned_smooth_knn_k2": ("--graph-mode", "learned_smooth_knn"),
            "ours_magnitude_knn_k2": ("--graph-mode", "magnitude_knn"),
            "ours_grad_graph_grad_agg_knn_k2": ("--graph-source", "update"),
            "ours_update_graph_update_agg_knn_k2": ("--aggregation-target", "update"),
            "ours_residual_reweight_knn_k2": ("--conflict-mix", "0.2"),
            "ours_residual_reweight_random_matched_k2": (
                "--graph-mode",
                "random",
            ),
            "ours_legacy_residual_reweight_knn_k2": (
                "--aggregation-target",
                "update",
            ),
            "ours_legacy_residual_reweight_random_matched_k2": (
                "--graph-mode",
                "random",
            ),
            "ours_legacy_residual_reweight_magnitude_knn_k2": (
                "--graph-mode",
                "magnitude_knn",
            ),
            "ours_legacy_residual_reweight_uniform": (
                "--graph-mode",
                "uniform",
            ),
            "ours_weight_graph_weight_agg_knn_k2": (
                "--aggregation-target",
                "weight",
            ),
            "ours_weight_graph_spectral_weight_agg_knn_k2": (
                "--aggregation-target",
                "spectral_filtered_weight",
            ),
            "ours_head_graph_knn_k2": (
                "--graph-source",
                "classifier_head_update",
            ),
            "ours_head_ema_graph_knn_k2": (
                "--graph-source",
                "classifier_head_ema_update",
            ),
            "ours_head_weight_graph_knn_k2": (
                "--graph-source",
                "classifier_head_weight",
            ),
            "ours_head_weight_graph_spectral_weight_agg_knn_k2": (
                "--aggregation-target",
                "spectral_filtered_weight",
            ),
            "ours_layerwise_head_graph_knn_k2": (
                "--graph-source",
                "layerwise_classifier_head_update",
            ),
            "ours_layerwise_head_ema_graph_knn_k2": (
                "--graph-source",
                "layerwise_classifier_head_ema_update",
            ),
            "ours_ema_graph_knn_k2": (
                "--graph-source",
                "ema_update",
            ),
            "ours_ema_signal_knn_k2": (
                "--aggregation-target",
                "spectral_filtered_ema_update",
            ),
            "ours_tail_m2_knn_k1": ("--graph-source", "layer_slice_update"),
            "ours_layerwise_tail_m2_knn_k1": (
                "--graph-source",
                "layerwise_slice_update",
            ),
            "ours_spectral_filtered_knn_k2": (
                "--aggregation-target",
                "spectral_filtered_update",
            ),
            "ours_spectral_filtered_random_matched_k2": (
                "--graph-mode",
                "random",
            ),
            "ours_spectral_filtered_uniform": (
                "--graph-mode",
                "uniform",
            ),
            "ours_spectral_filtered_magnitude_knn_k2": (
                "--graph-mode",
                "magnitude_knn",
            ),
            "ours_spectral_filtered_rbf_knn_k2": (
                "--graph-mode",
                "rbf_knn",
            ),
        }
        for variant, expected in cases.items():
            with self.subTest(variant=variant):
                method, label, extras = parse_variant(variant, self._args())
                self.assertEqual(label, variant)
                if variant.startswith("fed"):
                    self.assertEqual(method, expected[0])
                    for item in expected[1]:
                        self.assertIn(item, extras)
                else:
                    self.assertEqual(method, "ours")
                    self.assertIn(expected[0], extras)
                    self.assertIn(expected[1], extras)

    def test_tail_variant_forwards_negative_layer_start(self):
        _, _, extras = parse_variant("ours_tail_m2_knn_k1_fixed_tau", self._args())

        self.assertIn("--graph-layer-start", extras)
        idx = extras.index("--graph-layer-start")
        self.assertEqual(extras[idx + 1], "-2")
        self.assertIn("--disable-adaptive-tau", extras)

    def test_lowpass_strength_suffix_can_stack_with_tau_suffix(self):
        _, label, extras = parse_variant(
            "ours_spectral_filtered_magnitude_knn_k1_lp2p0_fixed_tau",
            self._args(),
        )

        self.assertEqual(
            label,
            "ours_spectral_filtered_magnitude_knn_k1_lp2p0_fixed_tau",
        )
        self.assertIn("--spectral-filter-strength", extras)
        idx = extras.index("--spectral-filter-strength")
        self.assertEqual(extras[idx + 1], "2.0")
        self.assertIn("--disable-adaptive-tau", extras)

    def test_server_momentum_suffix_can_stack_with_tau_suffix(self):
        _, label, extras = parse_variant(
            "ours_spectral_filtered_magnitude_knn_k1_serverm_fixed_tau",
            self._args(),
        )

        self.assertEqual(
            label,
            "ours_spectral_filtered_magnitude_knn_k1_serverm_fixed_tau",
        )
        self.assertIn("--ours-server-momentum", extras)
        idx = extras.index("--ours-server-momentum")
        self.assertEqual(extras[idx + 1], "0.9")
        self.assertIn("--disable-adaptive-tau", extras)

    def test_spectral_only_suffix_disables_conflict_floor_and_momentum(self):
        _, label, extras = parse_variant(
            "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only",
            self._args(),
        )

        self.assertEqual(
            label,
            "ours_spectral_filtered_knn_k1_lp0p5_serverm_fixed_tau_spectral_only",
        )
        self.assertIn("--spectral-filter-strength", extras)
        idx = extras.index("--spectral-filter-strength")
        self.assertEqual(extras[idx + 1], "0.5")
        self.assertIn("--ours-server-momentum", extras)
        self.assertEqual(_last_flag_value(extras, "--ours-server-learning-rate"), "1.0")
        self.assertEqual(_last_flag_value(extras, "--ours-server-momentum"), "0.0")
        self.assertIn("--disable-adaptive-tau", extras)
        self.assertIn("--conflict-mix", extras)
        self.assertEqual(_last_flag_value(extras, "--conflict-mix"), "0.0")
        self.assertIn("--min-client-weight", extras)
        self.assertEqual(_last_flag_value(extras, "--min-client-weight"), "0.0")

    def test_legacy_residual_reweight_pins_old_alpha_path(self):
        _, label, extras = parse_variant(
            "ours_legacy_residual_reweight_magnitude_knn_k1_serverm_fixed_tau",
            self._args(),
        )

        self.assertEqual(
            label,
            "ours_legacy_residual_reweight_magnitude_knn_k1_serverm_fixed_tau",
        )
        self.assertEqual(_last_flag_value(extras, "--graph-source"), "update")
        self.assertEqual(_last_flag_value(extras, "--aggregation-target"), "update")
        self.assertEqual(_last_flag_value(extras, "--graph-mode"), "magnitude_knn")
        self.assertEqual(_last_flag_value(extras, "--knn-k"), "1")
        self.assertEqual(_last_flag_value(extras, "--conflict-mix"), "0.2")
        self.assertEqual(_last_flag_value(extras, "--min-client-weight"), "0.0")
        self.assertEqual(_last_flag_value(extras, "--ours-server-momentum"), "0.9")
        self.assertIn("--disable-adaptive-tau", extras)


if __name__ == "__main__":
    unittest.main()
