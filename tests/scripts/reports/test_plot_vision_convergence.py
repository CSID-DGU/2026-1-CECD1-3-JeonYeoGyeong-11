import importlib.util
import unittest
from pathlib import Path


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


SCRIPT_PATH = repo_root() / "scripts" / "reports" / "plot_vision_convergence.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_vision_convergence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PlotVisionConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot = load_plot_module()

    def test_infer_variant_accepts_bare_suite_run_tag(self):
        obj = {"meta": {"run_tag": "ours_spectral_filtered_knn_k1_serverm_fixed_tau_seed42"}}
        variant = self.plot.infer_variant(
            Path("result_general_ours_seed42_ours_spectral_filtered_knn_k1_serverm_fixed_tau_seed42.json"),
            obj,
            "ours",
            42,
            ["ours_spectral_filtered_knn_k1_serverm_fixed_tau"],
        )

        self.assertEqual(variant, "ours_spectral_filtered_knn_k1_serverm_fixed_tau")

    def test_result_method_and_seed_accepts_non_fedavg_baseline(self):
        obj = {
            "meta": {"experiment": {"seed": 43}},
            "results": {"fedavgm": {"metrics_distributed": {"accuracy": [[1, 0.4]]}}},
        }
        method, seed = self.plot.result_method_and_seed(Path("result_general_fedavgm_seed43.json"), obj)

        self.assertEqual(method, "fedavgm")
        self.assertEqual(seed, 43)

    def test_result_method_and_seed_accepts_vision_alias_filename(self):
        obj = {"results": {"ours": {"metrics_distributed": {"accuracy": [[1, 0.4]]}}}}
        method, seed = self.plot.result_method_and_seed(Path("result_vision_ours_seed44_alias.json"), obj)

        self.assertEqual(method, "ours")
        self.assertEqual(seed, 44)


if __name__ == "__main__":
    unittest.main()
