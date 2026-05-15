import unittest

from spectral_fl.designs import ComponentSpec, interface_target_designs, resolve_design
from spectral_fl.graph.presets import graph_preset_names, resolve_graph_preset_spec


class DesignPresetTest(unittest.TestCase):
    def test_default_similarity_knn_preset_uses_graph_canonical_knobs(self):
        spec = resolve_graph_preset_spec("default_graph")

        self.assertEqual(spec["graph_design"], "default_similarity_knn")
        self.assertEqual(spec["graph_method"], "default_similarity_knn")
        self.assertEqual(spec["graph_source"], "update")
        self.assertEqual(spec["graph_mode"], "rbf_knn")
        self.assertEqual(spec["knn_k"], 2)
        self.assertEqual(spec["graph_scale_sigma"], 0.0)
        self.assertEqual(spec["aggregation_target"], "graph_filtered_update")

    def test_design_to_legacy_args_matches_current_strategy_knobs(self):
        design = resolve_design("pfedgraph_proxy")

        args = design.to_legacy_args()

        self.assertEqual(args["graph_design"], "pfedgraph_proxy")
        self.assertEqual(args["graph_method"], "pfedgraph")
        self.assertEqual(args["graph_source"], "update")
        self.assertEqual(args["graph_mode"], "pfedgraph_qp")
        self.assertEqual(args["aggregation_target"], "spectral_filtered_update")

    def test_legacy_graph_preset_alias_resolves_through_design(self):
        spec = resolve_graph_preset_spec("pfedgraph_like")

        self.assertEqual(spec["graph_design"], "pfedgraph_proxy")
        self.assertEqual(spec["graph_method"], "pfedgraph")
        self.assertEqual(spec["graph_source"], "update")
        self.assertEqual(spec["graph_mode"], "pfedgraph_qp")
        self.assertEqual(spec["aggregation_target"], "spectral_filtered_update")

    def test_graph_preset_names_include_designs_and_legacy_aliases(self):
        names = graph_preset_names()

        self.assertIn("default_graph", names)
        self.assertIn("default_similarity_knn", names)
        self.assertIn("head_knn_filtered_update", names)
        self.assertIn("pfedgraph_like", names)

    def test_design_mutation_helpers_replace_single_component(self):
        design = resolve_design("head_knn_filtered_update")
        mutated = design.with_topology(name="matched_random", params={"graph_mode": "matched_random"})

        self.assertEqual(design.topology.name, "knn")
        self.assertEqual(mutated.topology.name, "matched_random")
        self.assertEqual(mutated.to_legacy_args()["graph_mode"], "matched_random")

    def test_component_replacement_helper_supports_counterfactual_axes(self):
        design = resolve_design("head_knn_filtered_update")
        replacement = ComponentSpec(
            kind="RelationEstimator",
            name="rbf",
            params={"graph_scale_sigma": 0.5},
            input_kind=("client_state",),
            output_kind="relation_matrix",
        )

        mutated = design.with_relation(replacement)

        self.assertEqual(mutated.relation.name, "rbf")
        self.assertEqual(mutated.to_legacy_args()["graph_scale_sigma"], 0.5)

    def test_interface_target_prior_work_is_explicit(self):
        targets = interface_target_designs()

        self.assertEqual(targets["FED-PUB"]["support_level"], "interface-target")


if __name__ == "__main__":
    unittest.main()
