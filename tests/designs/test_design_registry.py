import unittest

from spectral_fl.designs import (
    ComponentSpec,
    DesignRegistry,
    GraphFLDesign,
    design_names,
    resolve_design,
)


class DesignRegistryTest(unittest.TestCase):
    def test_builtin_design_resolves_and_exports_trace_metadata(self):
        design = resolve_design("default_similarity_knn")

        self.assertEqual(design.support_level, "core-supported")
        self.assertEqual(design.client_state.kind, "ClientStateExtractor")
        self.assertEqual(design.relation.kind, "RelationEstimator")
        self.assertEqual(design.topology.kind, "TopologyOperator")
        self.assertEqual(design.aggregation.kind, "AggregationOperator")
        metadata = design.trace_metadata()
        self.assertEqual(metadata["design_name"], "default_similarity_knn")
        self.assertEqual(metadata["topology.name"], "rbf_knn")
        self.assertEqual(metadata["aggregation.name"], "graph_filtered_update")

    def test_registry_handles_aliases(self):
        registry = DesignRegistry()
        base = resolve_design("head_knn_filtered_update")
        registry.register(base)
        registry.register_alias("unit_alias", base.name)

        self.assertEqual(registry.resolve("unit_alias").name, base.name)
        self.assertIn("unit_alias", registry.names(include_aliases=True))

    def test_duplicate_registration_requires_override(self):
        registry = DesignRegistry()
        design = resolve_design("head_knn_filtered_update")
        registry.register(design)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(design)

    def test_unknown_design_reports_known_names(self):
        with self.assertRaisesRegex(ValueError, "Known designs"):
            resolve_design("does_not_exist")

    def test_component_and_design_validate_support_levels(self):
        with self.assertRaisesRegex(ValueError, "support_level"):
            ComponentSpec(kind="RelationEstimator", name="bad", support_level="exact")
        with self.assertRaisesRegex(ValueError, "support_level"):
            GraphFLDesign(
                name="bad",
                client_state=resolve_design("head_knn_filtered_update").client_state,
                relation=resolve_design("head_knn_filtered_update").relation,
                topology=resolve_design("head_knn_filtered_update").topology,
                aggregation=resolve_design("head_knn_filtered_update").aggregation,
                support_level="diagnostic_proxy",
            )

    def test_builtin_names_include_compat_aliases(self):
        names = design_names(include_aliases=True)

        self.assertIn("default_graph", names)
        self.assertIn("default_similarity_knn", names)
        self.assertIn("fedamp_like", names)
        self.assertIn("fedamp_proxy", names)


if __name__ == "__main__":
    unittest.main()
