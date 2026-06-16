import unittest
from pathlib import Path


class DemoContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.html = (
            root / "docs" / "demos" / "graphfl-assembly-scratch.html"
        ).read_text(encoding="utf-8")
        cls.demo_js = (
            root / "docs" / "demos" / "graphfl-assembly-scratch.js"
        ).read_text(encoding="utf-8")
        cls.demo_css = (
            root / "docs" / "demos" / "graphfl-assembly-scratch.css"
        ).read_text(encoding="utf-8")
        cls.mock_js = (
            root / "docs" / "demos" / "graphfl-assembly-mock-system.js"
        ).read_text(encoding="utf-8")
        cls.capabilities_js = (
            root / "docs" / "demos" / "graphfl-authoring-capabilities.js"
        ).read_text(encoding="utf-8")

    def test_capability_manifest_loads_before_demo_runtime(self):
        manifest_index = self.html.index("graphfl-authoring-capabilities.js")
        runtime_index = self.html.index("graphfl-assembly-scratch.js")
        self.assertLess(manifest_index, runtime_index)

    def test_authoring_panel_has_three_contract_surfaces(self):
        for element_id in (
            "authoring-cards",
            "authoring-contract",
            "authoring-commands",
            "authoring-checks",
            "authoring-validate",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("Actual validation", self.html)
        self.assertIn("Mock FL execution", self.html)

    def test_submission_v2_includes_authoring_design_and_signature(self):
        self.assertIn("schema_version: 2", self.demo_js)
        self.assertIn("authoring,", self.demo_js)
        self.assertIn("design,", self.demo_js)
        self.assertIn("parameters: {", self.demo_js)
        self.assertIn("graphfl run single", self.demo_js)
        self.assertIn("graphfl run suite", self.demo_js)
        self.assertIn("graphfl run ablation", self.demo_js)
        self.assertIn("--aggregation-params beta=", self.demo_js)

    def test_mock_db_uses_v2_without_deleting_v1(self):
        self.assertIn('const STORAGE_KEY = "graphfl-demo-runs-v2"', self.mock_js)
        self.assertNotIn('removeItem("graphfl-demo-runs-v1")', self.mock_js)
        self.assertIn("component_validation_report", self.mock_js)
        self.assertIn("module_traces.jsonl", self.mock_js)

    def test_existing_config_result_branch_handles_specific_types(self):
        self.assertIn('previewType.startsWith("existing")', self.demo_js)
        self.assertNotIn('previewType === "existing"', self.demo_js)

    def test_demo_surface_uses_neutral_demo_names(self):
        combined = "\n".join([self.html, self.demo_js, self.capabilities_js])
        self.assertNotIn("poster", combined.lower())
        self.assertIn("tmp/demo_sessions", combined)
        self.assertIn("demo_mutual_knn", combined)

    def test_player_can_recover_from_config_input(self):
        self.assertNotIn(".config-direct-row {\n      display: none;", self.demo_css)
        self.assertIn('if (!authoringState.enabled) authoringState.enabled = true;', self.demo_js)


if __name__ == "__main__":
    unittest.main()
