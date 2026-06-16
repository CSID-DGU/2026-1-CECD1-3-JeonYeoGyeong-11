import unittest
from pathlib import Path

from graphfl_lab.presentation.capabilities import (
    capability_manifest,
    render_capabilities_javascript,
)


class PresentationCapabilitiesTest(unittest.TestCase):
    def test_manifest_covers_three_authoring_components(self):
        manifest = capability_manifest()
        by_kind = {
            component["kind"]: component
            for component in manifest["components"]
        }

        self.assertEqual(set(by_kind), {"source", "builder", "aggregation"})
        self.assertEqual(by_kind["source"]["name"], "update_stats")
        self.assertEqual(by_kind["builder"]["parameter"]["choices"], [1, 2, 3])
        self.assertEqual(
            by_kind["aggregation"]["parameter"]["choices"],
            [0.3, 0.5, 0.7],
        )
        self.assertEqual(
            manifest["validation"]["checks"],
            ["Registry", "Shape", "Finite", "Metadata", "Trace/Artifact"],
        )

    def test_tracked_javascript_has_no_runtime_drift(self):
        root = Path(__file__).resolve().parents[2]
        tracked = (
            root
            / "docs"
            / "demos"
            / "graphfl-authoring-capabilities.js"
        )
        self.assertEqual(
            tracked.read_text(encoding="utf-8"),
            render_capabilities_javascript(),
        )


if __name__ == "__main__":
    unittest.main()
