import unittest
from argparse import Namespace

from graphfl_lab.experiments.suites.vision.variant_commands import build_base_cmd


def command_args(**overrides):
    values = dict(
        python_bin="python",
        dataset="mnist",
        num_clients=4,
        rounds=2,
        local_epochs=1,
        batch_size=8,
        model="linear",
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        partition="dirichlet",
        dirichlet_alpha=0.3,
        data_root="data",
        projection_dim=16,
        compression_seed=7,
        ema_alpha=0.9,
        tau_gain=1.0,
        tau_max=1.0,
        conflict_mix=0.2,
        warmup_rounds=1,
        graph_seed=3,
        graph_plugin="",
        use_ema_graph=True,
        disable_adaptive_tau=False,
        fixed_tau=0.5,
        tau_source="h_spec",
        graph_filter_strength=1.0,
        client_update_ema_alpha=0.8,
        diagnostic_only=False,
        correction_family="none",
        control_graph_mode="random",
        cluster_method="kmeans",
        cluster_k=2,
        cluster_auto_k=False,
        graph_free_mode="none",
        graph_free_gamma=1.0,
        clip_quantile=0.9,
        contribution_cap=0.35,
        diagnostics_enable=True,
        save_round_graphs=False,
        graph_snapshot_rounds="",
        save_update_arrays=False,
        loo_enabled=False,
        e_std_threshold=0.0,
        min_client_weight=0.0,
        server_learning_rate=1.0,
        server_momentum=0.0,
        ours_server_learning_rate=1.0,
        ours_server_momentum=0.0,
        fedprox_mu=0.0,
        fedopt_eta=0.1,
        fedopt_eta_l=0.1,
        fedopt_beta1=0.9,
        fedopt_beta2=0.99,
        fedopt_tau=1e-9,
        trimmed_beta=0.1,
        out_dir="out",
        graph_method="none",
        graph_source="update",
        aggregation_target="update",
        edge_threshold=0.0,
        graph_scale_sigma=1.0,
        learned_graph_lambda=0.0,
        graph_layer_start=0,
        graph_layer_end=-1,
        train_subset_size=0,
        test_subset_size=0,
        _user_arg_dests=(),
    )
    values.update(overrides)
    return Namespace(**values)


class VisionVariantCommandTest(unittest.TestCase):
    def test_build_base_cmd_preserves_core_command_contract(self):
        cmd = build_base_cmd(command_args(graph_method="default_similarity_knn"))

        self.assertEqual(cmd[:2], ["python", "run_vision_experiment.py"])
        self.assertIn("--dataset", cmd)
        self.assertIn("mnist", cmd)
        self.assertEqual(cmd[cmd.index("--graph-method") + 1], "default_similarity_knn")
        self.assertNotIn("--train-subset-size", cmd)

    def test_build_base_cmd_forwards_only_explicit_user_graph_args(self):
        cmd = build_base_cmd(
            command_args(
                _user_arg_dests=("graph_source", "aggregation_target"),
                graph_source="classifier_head_update",
                aggregation_target="graph_filtered_update",
                edge_threshold=0.5,
                train_subset_size=20,
                test_subset_size=10,
            )
        )

        self.assertEqual(cmd[cmd.index("--graph-source") + 1], "classifier_head_update")
        self.assertEqual(
            cmd[cmd.index("--aggregation-target") + 1],
            "graph_filtered_update",
        )
        self.assertNotIn("--edge-threshold", cmd)
        self.assertEqual(cmd[cmd.index("--train-subset-size") + 1], "20")
        self.assertEqual(cmd[cmd.index("--test-subset-size") + 1], "10")


if __name__ == "__main__":
    unittest.main()
