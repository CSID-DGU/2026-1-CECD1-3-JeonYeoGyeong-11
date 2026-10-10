"""c1: coordinator HTTP routes over the round core. Generated tensors only, in-process TestClient."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import FeatureNotImplemented
from commerce.services.fl_coordinator import auth
from commerce.services.fl_coordinator.main import CoordinatorSettings, create_app
from commerce.services.fl_coordinator.round_core import ModelRegistry, check_contract
from commerce.tests.e2e import dummy_round as dummy

SELLERS = ("seller-a", "seller-b", "seller-c")
OFFSETS = {"seller-a": 0.25, "seller-b": 0.5, "seller-c": 1.0, "seller-z": 0.0}


class _HttpCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.manifest = dummy.dummy_manifest()
        self.settings = CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds",
                                            mode="synthetic_plaintext")
        ModelRegistry(self.settings.registry_dir).register("model-0", self.manifest,
                                                           dummy.base_tensors(self.manifest))
        self.tokens = {seller: auth.issue(self.settings.auth_file, seller) for seller in (*SELLERS, "seller-z")}
        self.client = self.start()

    def start(self):
        client = TestClient(create_app(self.settings))
        self.addCleanup(client.close)
        return client

    def h(self, seller):
        return {"Authorization": "Bearer " + self.tokens[seller]}

    def open_round(self, client=None, **kwargs):
        return (client or self.client).app.state.coordinator.open_round(SELLERS, **kwargs)

    def envelope(self, seller, config, *, offset=None):
        result = dummy.DummyTrainer(seller, OFFSETS[seller] if offset is None else offset).train_round(
            dummy.base_tensors(self.manifest), config)
        return dummy.build_submission(seller, self.manifest, config, result)

    def submit(self, seller, config, *, offset=None, client=None):
        client = client or self.client
        submission, payload = self.envelope(seller, config, offset=offset)
        posted = client.post("/rounds/%s/submissions" % config["round_id"], content=json.dumps(submission),
                             headers=self.h(seller))
        if posted.status_code != 201:
            return posted
        return client.put(posted.headers["Location"], content=payload,
                          headers=dict(self.h(seller), **{"Content-Type": "application/octet-stream"}))

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        check_contract("contract_error.v1", body)
        self.assertEqual(body["code"], code)
        self.assertNotIn("seller-", response.text)  # no identifier or input is echoed


class Authentication(_HttpCase):
    def test_every_route_but_health_needs_a_valid_token(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        routes = [("get", "/rounds/current"), ("get", "/models/latest"), ("get", "/models/model-0/manifest"),
                  ("get", "/models/model-0/weights"), ("post", "/rounds/r/submissions"),
                  ("put", "/rounds/r/submissions/delta"), ("get", "/rounds/r/result")]
        for headers in ({}, {"Authorization": "Bearer seller-a.forged"}, {"Authorization": "Basic x"}):
            for method, path in routes:
                response = getattr(self.client, method)(path, headers=headers)
                self.assertEqual(response.status_code, 401, (method, path))
                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                self.assertEqual(response.content, b"")  # body waits for OQ13

    def test_health_reports_readiness_without_secrets(self):
        body = self.client.get("/healthz").json()
        self.assertEqual((body["service"], body["ready"]), ("coordinator", True))
        self.assertEqual(set(body), {"service", "status", "ready"})


class Models(_HttpCase):
    def test_latest_manifest_and_weights_verify(self):
        descriptor = self.client.get("/models/latest", headers=self.h("seller-z")).json()
        check_contract("model_release.v1", descriptor)
        manifest = self.client.get("/models/model-0/manifest", headers=self.h("seller-z")).json()
        self.assertEqual(manifest["manifest_hash"], descriptor["manifest_hash"])
        weights = self.client.get("/models/model-0/weights", headers=self.h("seller-z"))
        self.assertEqual(weights.headers["content-type"], "application/octet-stream")
        self.assertEqual(hashlib.sha256(weights.content).hexdigest(), descriptor["weights_sha256"])
        self.assertEqual(len(weights.content), descriptor["weights_size_bytes"])

    def test_unknown_or_unsafe_versions_are_not_found(self):
        for version in ("model-9", "..", "a%2F..%2Fb"):
            self.assert_error(self.client.get("/models/%s/manifest" % version, headers=self.h("seller-a")),
                              404, "NOT_FOUND")


class Rounds(_HttpCase):
    def test_current_round_is_for_cohort_members_only(self):
        self.assertEqual(self.client.get("/rounds/current", headers=self.h("seller-a")).status_code, 204)
        config = self.open_round()
        response = self.client.get("/rounds/current", headers=self.h("seller-a"))
        self.assertEqual(response.json(), config)
        self.assertEqual(self.client.get("/rounds/current", headers=self.h("seller-z")).status_code, 204)

    def test_full_round_over_http_publishes_a_new_latest(self):
        config = self.open_round()
        for seller in SELLERS:
            response = self.submit(seller, config)
            self.assertEqual(response.status_code, 202, response.text)
            check_contract("round_submit_ack.v1", response.json())
        self.assertEqual(response.json()["disposition"], "aggregated_on_time")
        latest = self.client.get("/models/latest", headers=self.h("seller-z")).json()
        self.assertNotEqual(latest["model_version"], "model-0")
        result = self.client.get("/rounds/%s/result" % config["round_id"], headers=self.h("seller-a"))
        self.assertEqual(result.json()["disposition"], "aggregated_on_time")

    def test_retry_with_the_same_bytes_and_conflict_with_different_bytes(self):
        config = self.open_round()
        first = self.submit("seller-a", config)
        self.assertEqual(self.submit("seller-a", config).json(), first.json())
        self.assert_error(self.submit("seller-a", config, offset=9.0), 409, "DUPLICATE_ROUND_SUBMIT")

    def test_envelope_checks(self):
        config = self.open_round()
        path = "/rounds/%s/submissions" % config["round_id"]
        submission, _ = self.envelope("seller-b", config)
        self.assert_error(self.client.post(path, content=json.dumps(submission), headers=self.h("seller-a")),
                          403, "FORBIDDEN")
        self.assert_error(self.client.post(path, content=b"{not json", headers=self.h("seller-a")),
                          422, "SCHEMA_INVALID")
        self.assert_error(self.client.post(path, content=b'{"x": NaN}', headers=self.h("seller-a")),
                          422, "SCHEMA_INVALID")
        own, _ = self.envelope("seller-a", config)
        self.assert_error(self.client.post(path, content=json.dumps(dict(own, extra=1)), headers=self.h("seller-a")),
                          422, "UNKNOWN_FIELD")
        self.assert_error(self.submit("seller-z", config), 403, "FORBIDDEN")
        self.assert_error(self.client.post("/rounds/round-none/submissions", content=json.dumps(own),
                                           headers=self.h("seller-a")), 404, "NOT_FOUND")

    def test_upload_before_the_envelope_is_a_state_conflict(self):
        config = self.open_round()
        self.assert_error(self.client.put("/rounds/%s/submissions/delta" % config["round_id"], content=b"x",
                                          headers=self.h("seller-a")), 409, "ILLEGAL_STATE_TRANSITION")

    def test_transfer_limit_applies_while_receiving(self):
        config = self.open_round()
        submission, _ = self.envelope("seller-a", config)
        self.client.post("/rounds/%s/submissions" % config["round_id"], content=json.dumps(submission),
                         headers=self.h("seller-a"))

        def chunks():  # no Content-Length: the limit must hold on the running total
            for _ in range(9):
                yield b"\0" * (1024 * 1024)

        response = self.client.put("/rounds/%s/submissions/delta" % config["round_id"], content=chunks(),
                                   headers=self.h("seller-a"))
        self.assert_error(response, 413, "SCHEMA_INVALID")

    def test_restart_discards_the_open_round(self):
        config = self.open_round()
        self.submit("seller-a", config)
        restarted = self.start()
        self.assert_error(restarted.get("/rounds/%s/result" % config["round_id"], headers=self.h("seller-a")),
                          409, "ROUND_DISCARDED")
        self.assertEqual(restarted.get("/rounds/current", headers=self.h("seller-a")).status_code, 204)


class Configuration(unittest.TestCase):
    def test_unconfigured_app_serves_health_only(self):
        with TestClient(create_app(None)) as client:
            self.assertEqual(client.get("/healthz").json()["ready"], False)
            self.assertEqual(client.get("/models/latest").status_code, 404)

    def test_path_kinds_and_protected_mode(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "file").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                CoordinatorSettings(root / "file", root / "auth.json", root / "rounds")
            with self.assertRaises(ValueError):
                CoordinatorSettings(root / "registry", root, root / "rounds")
            with self.assertRaises(ValueError):
                CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds", mode="plaintext")
            with self.assertRaises(FeatureNotImplemented):
                create_app(CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds"))

    def test_model_variant_must_match_the_registry(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = dummy.dummy_manifest()  # architecture_version 1 = text_only (D0024)
            ModelRegistry(root / "registry").register("model-0", manifest, dummy.base_tensors(manifest))
            settings = lambda variant: CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds",
                                                           mode="synthetic_plaintext", model_variant=variant)
            create_app(settings("text_only"))
            create_app(settings(None))  # not set: not checked
            with self.assertRaises(ValueError):
                create_app(settings("text_relation"))
            with self.assertRaises(ValueError):
                settings("text_other")


if __name__ == "__main__":
    unittest.main()
