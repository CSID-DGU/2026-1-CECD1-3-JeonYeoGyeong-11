import ast
import unittest
from pathlib import Path


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


ROOT = repo_root()


def python_files(path: Path):
    return [
        p
        for p in path.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def imported_modules(tree: ast.Module):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def starts_with_any(name: str, prefixes) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class RepositoryBoundaryTest(unittest.TestCase):
    def assert_no_banned_imports(self, path: Path, banned_prefixes):
        tree = parse_file(path)
        offenders = [
            name for name in imported_modules(tree)
            if starts_with_any(name, banned_prefixes)
        ]
        self.assertEqual(
            offenders,
            [],
            f"{path.relative_to(ROOT)} imports across a protected boundary",
        )

    def test_cli_files_stay_parser_only(self):
        banned = (
            "flwr",
            "numpy",
            "torch",
            "graphfl_lab.clients",
            "graphfl_lab.data",
            "graphfl_lab.flower_app",
            "graphfl_lab.flower_runner",
            "graphfl_lab.graph",
            "graphfl_lab.models",
            "graphfl_lab.strategies",
        )
        for path in python_files(ROOT / "graphfl_lab" / "cli"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_graph_package_does_not_depend_on_runners_or_strategies(self):
        banned = (
            "graphfl_lab.cli",
            "graphfl_lab.experiments",
            "graphfl_lab.strategies",
        )
        for path in python_files(ROOT / "graphfl_lab" / "graph"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_graphfl_strategy_does_not_depend_on_cli_or_experiments(self):
        banned = (
            "graphfl_lab.cli",
            "graphfl_lab.experiments",
        )
        for path in python_files(ROOT / "graphfl_lab" / "strategies" / "graphfl"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_compatibility_facades_stay_thin(self):
        facades = [
            "run_experiment.py",
            "run_vision_client_count_sweep.py",
            "run_vision_experiment.py",
            "run_vision_stress_grid.py",
            "run_vision_suite.py",
            "run_graph_ablation.py",
            "graphfl_lab/aggregation.py",
            "graphfl_lab/client.py",
            "graphfl_lab/general_client.py",
            "graphfl_lab/general_data.py",
            "graphfl_lab/general_models.py",
            "graphfl_lab/general_suite_variants.py",
            "graphfl_lab/cli/general_client_count_sweep.py",
            "graphfl_lab/cli/general_experiment.py",
            "graphfl_lab/cli/general_stress_grid.py",
            "graphfl_lab/cli/general_suite.py",
            "graphfl_lab/model.py",
            "graphfl_lab/spectral_diagnostics.py",
            "graphfl_lab/strategy.py",
            "graphfl_lab/suite_stats.py",
            "graphfl_lab/update_graph.py",
            "graphfl_lab/strategies/spectral/__init__.py",
            "graphfl_lab/strategies/spectral/aggregation.py",
            "graphfl_lab/strategies/spectral/config.py",
            "graphfl_lab/strategies/spectral/diagnostics.py",
            "graphfl_lab/strategies/spectral/filtering.py",
            "graphfl_lab/strategies/spectral/momentum.py",
            "graphfl_lab/strategies/spectral/strategy.py",
            "graphfl_lab/strategies/spectral/targets.py",
            "graphfl_lab/strategies/spectral/tracing.py",
            "graphfl_lab/experiments/client_count_sweep.py",
            "graphfl_lab/experiments/graph_ablation.py",
            "graphfl_lab/experiments/stress_grid.py",
            "graphfl_lab/experiments/suite.py",
        ]
        for rel in facades:
            path = ROOT / rel
            with self.subTest(path=rel):
                tree = parse_file(path)
                definitions = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                self.assertEqual(definitions, [])

    def test_question_oriented_config_folders_stay_deep(self):
        broad_dirs = [
            ROOT / "configs" / "cora" / "ablations",
            ROOT / "configs" / "vision" / "probes",
            ROOT / "configs" / "vision" / "stress",
            ROOT / "configs" / "vision" / "sweeps",
        ]
        for directory in broad_dirs:
            with self.subTest(path=directory.relative_to(ROOT)):
                json_files = sorted(p.name for p in directory.glob("*.json"))
                self.assertEqual(json_files, [])


if __name__ == "__main__":
    unittest.main()
