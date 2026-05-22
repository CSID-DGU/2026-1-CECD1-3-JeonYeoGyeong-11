import importlib
import pickle
import sys
import unittest


class PackageAliasTest(unittest.TestCase):
    def test_graphfl_lab_imports_flower_app(self):
        module = importlib.import_module("graphfl_lab.flower_app")

        self.assertTrue(hasattr(module, "server_app"))
        self.assertTrue(hasattr(module, "client_app"))

    def test_spectral_fl_shim_removed(self):
        sys.modules.pop("spectral_fl", None)
        importlib.invalidate_caches()
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("spectral_fl")

    def test_pickle_round_trip_for_canonical_import(self):
        from graphfl_lab.diagnostics.schema import RoundDiagnostics

        row = RoundDiagnostics(
            run_id="run",
            variant="v",
            seed=1,
            round=1,
            accuracy=0.5,
            loss=1.0,
            di_pre=0.4,
            di_post=0.3,
            neff_pre=2.0,
            neff_post=3.0,
            align_mean_pre=0.1,
            align_mean_post=0.2,
            loo_mean_pre=0.5,
            loo_mean_post=0.4,
            graph_density=0.5,
            graph_entropy=0.7,
            alpha_entropy=0.6,
            wall_time_sec=1.2,
            graph_method="m",
            correction_family="real_graph",
            graph_source="update",
            graph_variant="update",
            aggregation_target="graph_filtered_update",
            graph_kind="knn",
        )

        restored = pickle.loads(pickle.dumps(row))

        self.assertEqual(restored.to_dict(), row.to_dict())


if __name__ == "__main__":
    unittest.main()
