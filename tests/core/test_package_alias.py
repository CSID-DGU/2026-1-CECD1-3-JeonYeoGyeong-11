import importlib
import os
import pickle
import sys
import unittest
import warnings


class PackageAliasTest(unittest.TestCase):
    def test_graphfl_lab_imports_flower_app(self):
        module = importlib.import_module("graphfl_lab.flower_app")

        self.assertTrue(hasattr(module, "server_app"))
        self.assertTrue(hasattr(module, "client_app"))

    def test_spectral_fl_warns_by_default(self):
        import spectral_fl

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.reload(spectral_fl)

        self.assertTrue(
            any("spectral_fl" in str(item.message) for item in caught),
            [str(item.message) for item in caught],
        )

    def test_spectral_fl_warning_can_be_silenced(self):
        import spectral_fl

        old = os.environ.get("GRAPHFL_LAB_SILENCE_DEPRECATION")
        os.environ["GRAPHFL_LAB_SILENCE_DEPRECATION"] = "1"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                importlib.reload(spectral_fl)
        finally:
            if old is None:
                os.environ.pop("GRAPHFL_LAB_SILENCE_DEPRECATION", None)
            else:
                os.environ["GRAPHFL_LAB_SILENCE_DEPRECATION"] = old

        self.assertEqual([item for item in caught if item.category is DeprecationWarning], [])

    def test_sys_modules_alias_roots_exist(self):
        import graphfl_lab
        import spectral_fl

        self.assertIs(sys.modules["graphfl_lab"], graphfl_lab)
        self.assertIs(sys.modules["spectral_fl"], spectral_fl)
        self.assertEqual(list(spectral_fl.__path__), list(graphfl_lab.__path__))

    def test_legacy_submodule_import_still_resolves(self):
        module = importlib.import_module("spectral_fl.diagnostics.schema")

        self.assertTrue(hasattr(module, "RoundDiagnostics"))

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

    def test_pickle_round_trip_for_legacy_import(self):
        from spectral_fl.diagnostics.schema import RoundDiagnostics

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
