import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiagnosticReportScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = repo_root()
        cls.plot_mod = load_module(
            root / "scripts" / "reports" / "generate_diagnostic_plots.py",
            "generate_diagnostic_plots",
        )
        cls.dashboard_mod = load_module(
            root / "scripts" / "reports" / "generate_dashboard_mockup.py",
            "generate_dashboard_mockup",
        )

    def test_generate_plots_writes_fallback_assets_when_matplotlib_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            diag_csv = suite_dir / "diagnostic_summary.csv"
            with diag_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant",
                        "seeds",
                        "mean_final_acc",
                        "mean_delta_vs_fedavg",
                        "mean_di_drop",
                        "mean_neff_gain",
                        "mean_alignment_gain",
                        "mean_loo_drop",
                        "win_rate",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "variant": "ours_knn_k2",
                        "seeds": 1,
                        "mean_final_acc": 0.7,
                        "mean_delta_vs_fedavg": 0.03,
                        "mean_di_drop": 0.1,
                        "mean_neff_gain": 0.2,
                        "mean_alignment_gain": 0.05,
                        "mean_loo_drop": 0.01,
                        "win_rate": 1.0,
                    }
                )
            generated = self.plot_mod.generate_plots(suite_dir)
            self.assertTrue(generated)
            for path in generated:
                self.assertTrue(path.exists())

    def test_dashboard_mockup_script_loads_summary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            summary_csv = suite_dir / "general_suite_summary.csv"
            with summary_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["variant", "mean_delta", "win_rate", "mean_graph_density"],
                )
                writer.writeheader()
                writer.writerow(
                    {"variant": "fedavg", "mean_delta": 0, "win_rate": 0, "mean_graph_density": 0}
                )
                writer.writerow(
                    {
                        "variant": "ours_knn_k2",
                        "mean_delta": 0.02,
                        "win_rate": 1.0,
                        "mean_graph_density": 0.2,
                    }
                )
            rows = self.dashboard_mod._load_summary_rows(summary_csv)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["variant"], "ours_knn_k2")


if __name__ == "__main__":
    unittest.main()
