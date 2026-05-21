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


class LifecycleBoundaryTest(unittest.TestCase):
    def assert_file_has_no_banned_imports(self, path: Path, banned):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        offenders = [
            name for name in imported_modules(tree)
            if starts_with_any(name, banned)
        ]
        self.assertEqual(
            offenders,
            [],
            f"{path.relative_to(ROOT)} imports across a protected lifecycle boundary",
        )

    def test_lifecycle_contracts_do_not_depend_on_graph_or_runtime_layers(self):
        banned = (
            "flwr",
            "torch",
            "graphfl_lab.cli",
            "graphfl_lab.experiments",
            "graphfl_lab.flower_app",
            "graphfl_lab.flower_runner",
            "graphfl_lab.graph",
            "graphfl_lab.strategies",
        )
        contract_files = [
            ROOT / "spectral_fl" / "lifecycle" / "context.py",
            ROOT / "spectral_fl" / "lifecycle" / "modules.py",
            ROOT / "spectral_fl" / "lifecycle" / "traces.py",
        ]
        for path in contract_files:
            self.assert_file_has_no_banned_imports(path, banned)

    def test_lifecycle_stage_modules_do_not_import_later_stage_modules(self):
        guarded = {
            "client_state.py": ("graphfl_lab.lifecycle.relation", "graphfl_lab.lifecycle.topology"),
            "relation.py": ("graphfl_lab.lifecycle.topology",),
            "topology.py": (),
        }
        for filename, banned in guarded.items():
            path = ROOT / "spectral_fl" / "lifecycle" / filename
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            offenders = [
                name for name in imported_modules(tree)
                if starts_with_any(name, banned)
            ]
            self.assertEqual(
                offenders,
                [],
                f"{path.relative_to(ROOT)} imports across a protected lifecycle boundary",
            )


if __name__ == "__main__":
    unittest.main()
