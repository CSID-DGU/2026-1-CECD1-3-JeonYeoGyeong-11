"""Scaffold boundary checks; none of these proves a1/b1/b2/c1/g2/g3/g4."""
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented, JobBusyError
from commerce.packages.contracts.ports import RecommenderRuntime
from commerce.packages.fl_client.lifecycle import FLClientConfig
from commerce.packages.recommender.runtime import open_runtime
from commerce.services.central_api.main import app as central_app
from commerce.services.fl_coordinator.main import app as coordinator_app
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = MerchantSettings("synthetic-seller-1", self.root / "features.sqlite", self.root / "models")

    def test_context_shares_runtime_and_jobs_with_client_and_isolates_sellers(self):
        first = build_context(self.settings)
        second = build_context(MerchantSettings("synthetic-seller-2", self.root / "other.sqlite", self.root / "other-models"))
        self.addCleanup(first.jobs.close)
        self.addCleanup(second.jobs.close)
        self.assertIsInstance(first.runtime, RecommenderRuntime)
        self.assertIs(first.runtime, first.fl_client.runtime)
        self.assertIs(first.jobs, first.fl_client.jobs)
        self.assertIsNot(first.runtime, second.runtime)
        self.assertIsNot(first.jobs, second.jobs)
        self.assertNotEqual(first.runtime.feature_db_path, second.runtime.feature_db_path)

    def test_stub_never_acknowledges_persistence_or_fabricates_model_results(self):
        runtime = open_runtime(self.settings.seller_id, self.settings.feature_db_path, self.settings.model_dir)
        calls = [
            lambda: runtime.ingest_purchase_event({}), lambda: runtime.upsert_catalog_item({}, 1),
            lambda: runtime.predict_local({}), lambda: runtime.get_local_data_ref(),
            lambda: runtime.get_shared_manifest(), lambda: runtime.export_shared_state(),
            lambda: runtime.train_round("synthetic-ref", {}), lambda: runtime.install_release({}, {}, {}),
            lambda: runtime.personalize_local("synthetic-ref", {}), lambda: runtime.compare_local({}),
        ]
        for call in calls:
            with self.assertRaises(FeatureNotImplemented):
                call()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_enabled_fl_fails_closed_in_both_modes(self):
        for mode in ("protected", "synthetic_plaintext"):
            settings = MerchantSettings(self.settings.seller_id, self.settings.feature_db_path,
                                        self.settings.model_dir, FLClientConfig(enabled=True, mode=mode))
            with self.assertRaises(FeatureNotImplemented):
                with TestClient(create_app(settings)):
                    self.fail("Enabled FL should not start before its implementation")

    def test_health_is_responsive_during_local_job_and_second_job_is_rejected(self):
        started, release = Event(), Event()
        app = create_app(self.settings)
        def operation():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test release missing")
            return "finished"
        with TestClient(app) as client:
            jobs = app.state.merchant.jobs
            future = jobs.submit(operation)
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(client.get("/healthz").json()["ready"], False)
                with self.assertRaises(JobBusyError):
                    jobs.submit(lambda: "must not run")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=2), "finished")
        self.assertFalse(hasattr(app.state, "merchant"))
        with self.assertRaises(RuntimeError):
            jobs.submit(lambda: "closed executor")

    def test_failed_job_releases_slot(self):
        context = build_context(self.settings)
        self.addCleanup(context.jobs.close)
        def operation():
            raise ValueError("synthetic failure")
        with self.assertRaises(ValueError):
            context.jobs.submit(operation).result(timeout=2)
        self.assertEqual(context.jobs.submit(lambda: 7).result(timeout=2), 7)

    def test_public_and_coordinator_apps_have_no_private_or_submission_routes(self):
        for app, service in [(central_app, "central"), (coordinator_app, "coordinator")]:
            with TestClient(app) as client:
                self.assertEqual(client.get("/healthz").json(), {"service": service, "status": "scaffold", "ready": False})
                self.assertFalse(hasattr(app.state, "merchant"))
                self.assertEqual(client.post("/orders", json={"synthetic": True}).status_code, 404)
                self.assertEqual(client.post("/rounds/synthetic/submissions", json={}).status_code, 404)

    def test_error_response_matches_schema_without_echoing_private_fields(self):
        error = ContractError("NOT_FOUND", "/synthetic-customer")
        schema_path = Path(__file__).resolve().parents[2] / "packages/contracts/schemas/contract_error.v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(error.to_payload())
        self.assertNotIn("synthetic-customer", json.dumps(error.to_payload()))
        self.assertEqual(error.field_path, "/synthetic-customer")
        with self.assertRaises(ValueError):
            ContractError("NOT_IMPLEMENTED")


if __name__ == "__main__":
    unittest.main()
