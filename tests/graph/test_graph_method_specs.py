import unittest
from argparse import Namespace

from graphfl_lab.graph.method_specs import get_graph_fl_method_spec
from graphfl_lab.graph.presets import (
    apply_graph_preset_to_namespace,
    graph_method_names,
    graph_preset_names,
    resolve_graph_method_spec,
    resolve_graph_preset_spec,
)


class GraphMethodSpecTest(unittest.TestCase):
    def test_pfedgraph_like_preset_uses_method_spec(self):
        spec = resolve_graph_preset_spec("pfedgraph_like")

        self.assertEqual(spec["graph_method"], "pfedgraph")
        self.assertEqual(spec["graph_source"], "update")
        self.assertEqual(spec["graph_mode"], "pfedgraph_qp")
        self.assertEqual(spec["aggregation_target"], "spectral_filtered_update")

    def test_interface_only_method_is_not_exposed_as_runnable_preset(self):
        self.assertNotIn("fedpub_like", graph_preset_names())
        method = get_graph_fl_method_spec("fedpub")
        self.assertEqual(method.support_level, "interface-target")
        self.assertEqual(method.config_overrides["aggregation_target"], "personalized_weight")
        self.assertEqual(method.design_name, "fedpub_interface_target")

    def test_method_preset_records_method_name_on_namespace(self):
        args = Namespace(graph_preset="fedaga_like")

        info = apply_graph_preset_to_namespace(args)

        self.assertEqual(args.graph_method, "fedaga")
        self.assertEqual(args.graph_source, "ema_update")
        self.assertIn("graph_method", info["applied"])

    def test_graph_method_resolves_runnable_design(self):
        spec = resolve_graph_method_spec("default_similarity_knn")

        self.assertEqual(spec["graph_design"], "default_similarity_knn")
        self.assertEqual(spec["graph_method"], "default_similarity_knn")
        self.assertEqual(spec["graph_mode"], "rbf_knn")
        self.assertEqual(spec["aggregation_target"], "graph_filtered_update")
        self.assertIn("default_similarity_knn", graph_method_names())

    def test_graph_method_preserves_explicit_lower_level_overrides(self):
        args = Namespace(
            graph_preset="none",
            graph_method="default_similarity_knn",
            knn_k=5,
            _user_arg_dests=frozenset({"graph_method", "knn_k"}),
        )

        info = apply_graph_preset_to_namespace(args)

        self.assertEqual(info["source"], "graph_method")
        self.assertEqual(args.graph_method, "default_similarity_knn")
        self.assertEqual(args.graph_mode, "rbf_knn")
        self.assertEqual(args.aggregation_target, "graph_filtered_update")
        self.assertEqual(args.knn_k, 5)
        self.assertNotIn("knn_k", info["applied"])

    def test_interface_target_method_is_not_runnable_by_name(self):
        with self.assertRaises(ValueError):
            resolve_graph_method_spec("fedpub")


if __name__ == "__main__":
    unittest.main()
