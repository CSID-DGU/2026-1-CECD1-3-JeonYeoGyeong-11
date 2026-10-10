"""c1: the seller FL client over coordinator HTTP, with generated-tensor runtimes. Not g3: B is faked.

The client functions are driven directly; FLClient.start still refuses both modes
until OQ17 (synthetic input provenance) and g4 (protection) are settled.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import TrainingResult
from commerce.packages.fl_client import rounds
from commerce.packages.fl_client.submission import build_submission, verify_release
from commerce.packages.fl_client.transport import CoordinatorRefused, CoordinatorTransport
from commerce.services.fl_coordinator import auth
from commerce.services.fl_coordinator.main import CoordinatorSettings, create_app
from commerce.services.fl_coordinator.round_core import ModelRegistry
from commerce.services.merchant_api.jobs import SellerJobs
from commerce.tests.e2e import dummy_round as dummy

SELLERS = ("seller-a", "seller-b", "seller-c")
OFFSETS = {"seller-a": 0.25, "seller-b": 0.5, "seller-c": 1.0}
FL_BOUNDARY = {"get_shared_manifest", "get_local_data_ref", "train_round", "install_release"}


class _ClientCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.manifest = dummy.dummy_manifest()
        settings = CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds",
                                       mode="synthetic_plaintext")
        ModelRegistry(settings.registry_dir).register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        self.tokens = {seller: auth.issue(settings.auth_file, seller) for seller in (*SELLERS, "seller-new")}
        self.http = TestClient(create_app(settings))
        self.addCleanup(self.http.close)
        self.coordinator = self.http.app.state.coordinator

    def seller(self, seller_id, *, offset=None, completed=True, model_variant="text_only"):
        runtime = dummy.DummyRuntime(seller_id, self.manifest, OFFSETS.get(seller_id, 0.0) if offset is None else offset,
                                     completed=completed, model_variant=model_variant)
        jobs = SellerJobs()
        self.addCleanup(jobs.close)
        return runtime, jobs, CoordinatorTransport(self.http, self.tokens[seller_id])

    def installed_sellers(self, **overrides):
        out = []
        for seller_id in SELLERS:
            runtime, jobs, transport = self.seller(seller_id, **overrides.get(seller_id, {}))
            rounds.install_latest(runtime, jobs, transport, "text_only")
            out.append((runtime, jobs, transport))
        return out


class SyntheticRoundOverHttp(_ClientCase):
    def test_three_sellers_train_from_the_base_and_a_new_seller_installs_the_result(self):
        sellers = self.installed_sellers()
        self.coordinator.open_round(SELLERS)
        acks = [rounds.participate(runtime, jobs, transport, "text_only") for runtime, jobs, transport in sellers]
        self.assertEqual([ack["disposition"] for ack in acks],
                         ["accepted_on_time", "accepted_on_time", "aggregated_on_time"])
        latest = self.coordinator.registry.latest()
        expected = np.float32(sum(OFFSETS.values()) / 3)
        for array in self.coordinator.registry.tensors(latest["model_version"]).values():
            np.testing.assert_allclose(array, expected, rtol=1e-6)
        # a seller with no round history installs the new release after verifying it
        runtime, jobs, transport = self.seller("seller-new")
        self.assertEqual(rounds.install_latest(runtime, jobs, transport, "text_only"), latest)
        self.assertEqual(runtime.serving, latest["model_version"])
        # members move to the new base the same way; reinstalling is idempotent
        for runtime, jobs, transport in sellers:
            rounds.install_latest(runtime, jobs, transport, "text_only")
            rounds.install_latest(runtime, jobs, transport, "text_only")
            self.assertEqual(runtime.serving, latest["model_version"])

    def test_the_client_stays_inside_the_fl_boundary(self):
        sellers = self.installed_sellers()
        self.coordinator.open_round(SELLERS)
        for runtime, jobs, transport in sellers:
            rounds.participate(runtime, jobs, transport, "text_only")
            self.assertLessEqual(set(runtime.calls), FL_BOUNDARY)  # no serving, export or personalization

    def test_submitted_delta_is_the_training_result_not_the_personal_tail(self):
        runtime, _jobs, _transport = self.seller("seller-a")
        config = dummy.round_config(self.manifest, "model-0")
        result = dummy.DummyTrainer("seller-a", 0.25).train_round(dummy.base_tensors(self.manifest), config)
        _submission, payload = build_submission("seller-a", self.manifest, config, result)
        decoded = verify_like_coordinator(payload, self.manifest)
        for name, array in decoded.items():
            np.testing.assert_array_equal(array, result.shared_delta[name])
            self.assertFalse(np.array_equal(array, runtime.personal_tail[name]))

    def test_an_incomplete_seller_sends_a_zero_delta_and_the_round_is_discarded(self):
        sellers = self.installed_sellers(**{"seller-b": {"completed": False}})
        config = self.coordinator.open_round(SELLERS)
        rounds.participate(*sellers[0], "text_only")
        ack = rounds.participate(*sellers[1], "text_only")
        self.assertEqual(ack["disposition"], "dropped_incomplete")
        self.assertEqual(sellers[0][2].result(config["round_id"])["disposition"], "round_discarded")
        # the discarded round is no longer offered, so the third seller does not train for nothing
        self.assertIsNone(rounds.participate(*sellers[2], "text_only"))
        self.assertNotIn("train_round", sellers[2][0].calls)
        self.assertEqual(self.coordinator.registry.latest()["model_version"], "model-0")

    def test_no_round_or_not_selected_means_no_training(self):
        runtime, jobs, transport = self.seller("seller-a")
        rounds.install_latest(runtime, jobs, transport, "text_only")
        self.assertIsNone(rounds.participate(runtime, jobs, transport, "text_only"))
        self.coordinator.open_round(("seller-b", "seller-c", "seller-new"))
        self.assertIsNone(rounds.participate(runtime, jobs, transport, "text_only"))
        self.assertNotIn("train_round", runtime.calls)


class Refusals(_ClientCase):
    def test_a_round_for_another_variant_or_base_is_not_trained(self):
        runtime, jobs, transport = self.seller("seller-a")
        rounds.install_latest(runtime, jobs, transport, "text_only")
        self.coordinator.open_round(SELLERS)
        runtime.manifest = dict(self.manifest, architecture_version=2)  # B reports another architecture
        runtime.manifest["manifest_hash"] = dummy.ids.manifest_hash(runtime.manifest)
        with self.assertRaises(ContractError) as caught:
            rounds.participate(runtime, jobs, transport, "text_only")
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")
        self.assertNotIn("train_round", runtime.calls)
        with self.assertRaises(ContractError):  # the configured variant is passed to B unchanged
            rounds.participate(runtime, jobs, transport, "text_relation")

    def test_a_tampered_release_is_never_installed(self):
        descriptor = self.coordinator.registry.latest()
        manifest = self.coordinator.registry.manifest("model-0")
        weights = bytearray(self.coordinator.registry.weights("model-0"))
        weights[-1] ^= 0xFF
        with self.assertRaises(ContractError):
            verify_release(descriptor, manifest, bytes(weights))
        bad_manifest = dict(manifest, architecture_version=9)
        with self.assertRaises(ContractError):
            verify_release(descriptor, bad_manifest, self.coordinator.registry.weights("model-0"))

    def test_invalid_training_results_are_not_sent(self):
        config = dummy.round_config(self.manifest, "model-0")
        good = dummy.DummyTrainer("seller-a", 0.25).train_round(dummy.base_tensors(self.manifest), config)
        first = self.manifest["tensors"][0]["name"]
        cases = [
            TrainingResult({first: good.shared_delta[first]}, good.metrics, True),  # missing tensor
            TrainingResult(dict(good.shared_delta, **{first: good.shared_delta[first].astype(np.float64)}),
                           good.metrics, True),
            TrainingResult(dict(good.shared_delta, **{first: np.full_like(good.shared_delta[first], np.nan)}),
                           good.metrics, True),
            TrainingResult(good.shared_delta, {"loss_mean": float("inf"), "grad_norm_mean": None}, True),
            TrainingResult(good.shared_delta, {"loss_mean": -1.0, "grad_norm_mean": None}, True),
        ]
        for result in cases:
            with self.assertRaises(ContractError):
                build_submission("seller-a", self.manifest, config, result)

    def test_wrong_token_is_refused(self):
        transport = CoordinatorTransport(self.http, "seller-a.forged")
        with self.assertRaises(CoordinatorRefused) as caught:
            transport.current_round()
        self.assertEqual(caught.exception.status, 401)


def verify_like_coordinator(payload, manifest):
    from commerce.services.fl_coordinator.npz_payload import decode_npz
    return decode_npz(payload, manifest["tensors"])


if __name__ == "__main__":
    unittest.main()
