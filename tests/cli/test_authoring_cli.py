import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from graphfl_lab.cli.authoring import (
    compose_design,
    poster_session_root,
    scaffold_component,
    validate_component,
)
from graphfl_lab.cli.argparse_types import json_object
from graphfl_lab.cli.main import main as graphfl_main
from graphfl_lab.extensions.run_resolution import dry_run_payload
from graphfl_lab.graph import unregister_graph_builder, unregister_graph_source
from graphfl_lab.strategies.graphfl.targets import unregister_aggregation_target


class AuthoringCliTest(unittest.TestCase):
    def setUp(self):
        self._clear_component_registries()
        poster_session_root().mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            dir=poster_session_root(),
            prefix="contract_",
        )
        self.workspace = Path(self.temp.name)

    def tearDown(self):
        self._clear_component_registries()
        self.temp.cleanup()

    @staticmethod
    def _clear_component_registries():
        unregister_graph_source("update_stats")
        unregister_graph_builder("mutual_knn")
        unregister_aggregation_target("residual_graph_mix")

    def _scaffold_all(self):
        outputs = {}
        for kind, name in (
            ("source", "update_stats"),
            ("builder", "mutual_knn"),
            ("aggregation", "residual_graph_mix"),
        ):
            outputs[kind] = scaffold_component(
                kind=kind,
                name=name,
                workspace=self.workspace,
            )
        return outputs

    def test_scaffold_validate_compose_and_dry_run(self):
        outputs = self._scaffold_all()
        plugin = Path(outputs["source"]["plugin_path"])

        for kind, name in (
            ("source", "update_stats"),
            ("builder", "mutual_knn"),
            ("aggregation", "residual_graph_mix"),
        ):
            report = validate_component(
                plugin=plugin,
                component=f"{kind}:{name}",
            )
            self.assertTrue(report["ok"], report)
            self.assertEqual(
                [item["name"] for item in report["checks"]],
                ["Registry", "Shape", "Finite", "Metadata", "Trace/Artifact"],
            )

        composed = compose_design(
            plugin=plugin,
            name="poster_contract",
            source="update_stats",
            builder="mutual_knn",
            aggregation="residual_graph_mix",
            knn_k=2,
            aggregation_params={"beta": 0.5},
        )
        config_path = Path(composed["config_path"])
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(
            payload["args"]["aggregation_params"],
            {"beta": 0.5},
        )

        resolved = dry_run_payload(
            "single",
            [
                "--track",
                "vision",
                "--config",
                str(config_path),
            ],
        )
        self.assertEqual(resolved["track"], "vision")
        self.assertEqual(
            resolved["resolved_components"],
            {
                "source": "update_stats",
                "builder": "mutual_knn",
                "aggregation": "residual_graph_mix",
                "parameters": {
                    "knn_k": 2,
                    "aggregation": {"beta": 0.5},
                },
                "variants": [],
            },
        )

    def test_rejects_invalid_names_paths_and_overwrite(self):
        with self.assertRaises(ValueError):
            scaffold_component(
                kind="source",
                name="../update_stats",
                workspace=self.workspace,
            )
        with self.assertRaises(ValueError):
            scaffold_component(
                kind="source",
                name="update_stats",
                workspace=self.workspace.parent.parent,
            )

        scaffold_component(
            kind="source",
            name="update_stats",
            workspace=self.workspace,
        )
        with self.assertRaises(FileExistsError):
            scaffold_component(
                kind="source",
                name="update_stats",
                workspace=self.workspace,
            )

    def test_each_poster_branch_composes_and_resolves(self):
        branches = (
            (
                "source",
                "update_stats",
                "update_stats",
                "knn",
                "graph_filtered_update",
                {},
            ),
            (
                "builder",
                "mutual_knn",
                "update",
                "mutual_knn",
                "graph_filtered_update",
                {},
            ),
            (
                "aggregation",
                "residual_graph_mix",
                "update",
                "knn",
                "residual_graph_mix",
                {"beta": 0.5},
            ),
        )
        for index, (
            kind,
            component_name,
            source,
            builder,
            aggregation,
            aggregation_params,
        ) in enumerate(branches):
            with self.subTest(kind=kind):
                branch_dir = self.workspace / f"branch_{index}"
                scaffolded = scaffold_component(
                    kind=kind,
                    name=component_name,
                    workspace=branch_dir,
                )
                plugin = scaffolded["plugin_path"]
                report = validate_component(
                    plugin=plugin,
                    component=f"{kind}:{component_name}",
                )
                self.assertTrue(report["ok"], report)
                composed = compose_design(
                    plugin=plugin,
                    name=f"poster_{component_name}",
                    source=source,
                    builder=builder,
                    aggregation=aggregation,
                    knn_k=2,
                    aggregation_params=aggregation_params,
                )
                resolved = dry_run_payload(
                    "single",
                    [
                        "--track",
                        "vision",
                        "--config",
                        composed["config_path"],
                    ],
                )
                self.assertEqual(
                    resolved["resolved_components"]["source"],
                    source,
                )
                self.assertEqual(
                    resolved["resolved_components"]["builder"],
                    builder,
                )
                self.assertEqual(
                    resolved["resolved_components"]["aggregation"],
                    aggregation,
                )

    def test_unified_single_dry_run_preserves_track(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = graphfl_main(
                ["run", "single", "--track", "vision", "--dry-run"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["track"], "vision")

    def test_aggregation_params_accept_powershell_safe_key_value(self):
        self.assertEqual(
            json_object("beta=0.7,enabled=true,label=poster"),
            {"beta": 0.7, "enabled": True, "label": "poster"},
        )

    def test_multi_run_dry_run_resolves_variant_components(self):
        payload = dry_run_payload("suite", [])
        self.assertEqual(
            payload["resolved_components"]["builder"],
            "variant-defined",
        )
        variants = payload["resolved_components"]["variants"]
        self.assertTrue(variants)
        knn = next(
            row for row in variants if row["variant"] == "ours_knn_k2"
        )
        self.assertEqual(knn["source"], "update")
        self.assertEqual(knn["builder"], "knn")
        self.assertEqual(knn["aggregation"], "update")


if __name__ == "__main__":
    unittest.main()
