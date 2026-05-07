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
            "spectral_fl.clients",
            "spectral_fl.data",
            "spectral_fl.flower_app",
            "spectral_fl.flower_runner",
            "spectral_fl.graph",
            "spectral_fl.models",
            "spectral_fl.strategies",
        )
        for path in python_files(ROOT / "spectral_fl" / "cli"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_graph_package_does_not_depend_on_runners_or_strategies(self):
        banned = (
            "spectral_fl.cli",
            "spectral_fl.experiments",
            "spectral_fl.strategies",
        )
        for path in python_files(ROOT / "spectral_fl" / "graph"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_spectral_strategy_does_not_depend_on_cli_or_experiments(self):
        banned = (
            "spectral_fl.cli",
            "spectral_fl.experiments",
        )
        for path in python_files(ROOT / "spectral_fl" / "strategies" / "spectral"):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assert_no_banned_imports(path, banned)

    def test_compatibility_facades_stay_thin(self):
        facades = [
            "run_experiment.py",
            "run_general_client_count_sweep.py",
            "run_general_experiment.py",
            "run_general_stress_grid.py",
            "run_general_suite.py",
            "run_graph_ablation.py",
            "spectral_fl/aggregation.py",
            "spectral_fl/client.py",
            "spectral_fl/general_client.py",
            "spectral_fl/general_data.py",
            "spectral_fl/general_models.py",
            "spectral_fl/general_suite_variants.py",
            "spectral_fl/model.py",
            "spectral_fl/spectral_diagnostics.py",
            "spectral_fl/strategy.py",
            "spectral_fl/suite_stats.py",
            "spectral_fl/update_graph.py",
            "spectral_fl/experiments/client_count_sweep.py",
            "spectral_fl/experiments/graph_ablation.py",
            "spectral_fl/experiments/stress_grid.py",
            "spectral_fl/experiments/suite.py",
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
            ROOT / "configs" / "general" / "probes",
            ROOT / "configs" / "general" / "stress",
            ROOT / "configs" / "general" / "sweeps",
        ]
        for directory in broad_dirs:
            with self.subTest(path=directory.relative_to(ROOT)):
                json_files = sorted(p.name for p in directory.glob("*.json"))
                self.assertEqual(json_files, [])


if __name__ == "__main__":
    unittest.main()
