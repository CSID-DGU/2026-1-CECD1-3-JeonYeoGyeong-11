from argparse import Namespace
from pathlib import Path
import unittest

from graphfl_lab.experiments.cora.graph_ablation import variant_command


def _args():
    return Namespace(
        python_bin="python",
        num_clients=2,
        rounds=1,
        local_epochs=1,
        hidden_dim=64,
        partition="iid",
        dirichlet_alpha=0.2,
        data_root="./data",
        compression_dim=256,
        compression_seed=0,
        warmup_rounds=0,
        tau_max=2.0,
        tau_gain=2.0,
        conflict_mix=0.2,
        ema_alpha=0.8,
        graph_source="update",
        aggregation_target="update",
        graph_seed=0,
        knn_k=1,
        edge_threshold=0.0,
        diagnostic_only=True,
        e_std_threshold=0.02,
        min_client_weight=0.05,
        fixed_tau=1.0,
    )


class CoraGraphAblationCommandTest(unittest.TestCase):
    def test_subprocess_uses_explicit_cora_track(self):
        cmd, method, result_path = variant_command(
            "ours_knn",
            _args(),
            seed=42,
            out_dir=Path("out"),
            run_tag="tag",
        )

        self.assertEqual(method, "ours")
        self.assertEqual(result_path, Path("out") / "result_ours_seed42_tag.json")
        self.assertEqual(cmd[:4], ["python", "run_experiment.py", "--track", "cora"])


if __name__ == "__main__":
    unittest.main()
