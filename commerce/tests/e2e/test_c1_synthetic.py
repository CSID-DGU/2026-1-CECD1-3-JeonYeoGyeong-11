"""c1: the synthetic_plaintext client loop and its attested-input check (D0025, OQ17)."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import FeatureNotImplemented
from commerce.packages.fl_client.attestation import load_attestation, write_attestation
from commerce.packages.fl_client.lifecycle import FLClient, FLClientConfig
from commerce.packages.fl_client.transport import CoordinatorTransport
from commerce.services.fl_coordinator import auth
from commerce.services.fl_coordinator.main import CoordinatorSettings, create_app
from commerce.services.fl_coordinator.round_core import ModelRegistry
from commerce.services.fl_coordinator.service import RoundSettings
from commerce.services.merchant_api.jobs import SellerJobs
from commerce.tests.e2e import dummy_round as dummy

SELLERS = ("seller-a", "seller-b", "seller-c")
OFFSETS = {"seller-a": 0.25, "seller-b": 0.5, "seller-c": 1.0}


class SnapshotDigest(unittest.TestCase):
    def test_order_free_content_hash(self):
        catalog = {"i1": {"t": 1}, "i2": {"t": 2}}
        self.assertEqual(ids.snapshot_digest(["e1", "e2"], catalog), ids.snapshot_digest(["e2", "e1"], dict(reversed(catalog.items()))))
        self.assertNotEqual(ids.snapshot_digest(["e1", "e2"], catalog), ids.snapshot_digest(["e1", "e2", "e3"], catalog))
        self.assertNotEqual(ids.snapshot_digest(["e1"], catalog), ids.snapshot_digest(["e1"], dict(catalog, i2={"t": 9})))
        with self.assertRaises(ValueError):
            ids.snapshot_digest(["e1", "e1"], catalog)


class _LoopCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = dummy.dummy_manifest()
        settings = CoordinatorSettings(self.root / "registry", self.root / "auth.json", self.root / "rounds",
                                       mode="synthetic_plaintext")
        ModelRegistry(settings.registry_dir).register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        self.tokens = {s: auth.issue(settings.auth_file, s) for s in SELLERS}
        self.http = TestClient(create_app(settings))
        self.addCleanup(self.http.close)
        self.coordinator = self.http.app.state.coordinator

    def client(self, seller, *, attest_digest=None):
        runtime = dummy.DummyRuntime(seller, self.manifest, OFFSETS[seller])
        path = self.root / ("%s.attest.json" % seller)
        write_attestation(path, seller, attest_digest or runtime.expected_digest(), "test")
        jobs = SellerJobs()
        self.addCleanup(jobs.close)
        config = FLClientConfig(enabled=True, mode="synthetic_plaintext", model_variant="text_only",
                                coordinator_url="http://testserver", token=self.tokens[seller],
                                synthetic_attestation=path)
        client = FLClient(runtime, jobs, config)
        # what start() sets up, wired to the in-process coordinator
        client._transport = CoordinatorTransport(self.http, self.tokens[seller])
        client._attestation = load_attestation(path)
        return client


class InstallOnly(_LoopCase):
    """A seller outside the cohort (e.g. a new seller) receives releases and never submits."""

    def install_only_client(self, seller):
        jobs = SellerJobs()
        self.addCleanup(jobs.close)
        config = FLClientConfig(enabled=True, mode="synthetic_plaintext", model_variant="text_only",
                                coordinator_url="http://testserver", token=self.tokens[seller], install_only=True)
        client = FLClient(dummy.DummyRuntime(seller, self.manifest, OFFSETS[seller]), jobs, config)
        client._transport = CoordinatorTransport(self.http, self.tokens[seller])
        return client

    def test_installs_and_never_joins_even_when_selected(self):
        client = self.install_only_client("seller-a")
        config = self.coordinator.open_round(SELLERS, RoundSettings(local_steps=1, batch_size=8, n_neg=8))
        client.tick()
        self.assertEqual(client.installed, "model-0")
        self.assertEqual(client.runtime.serving, "model-0")
        self.assertFalse({"train_round", "snapshot_digest", "get_local_data_ref"} & set(client.runtime.calls))
        with self.assertRaises(Exception):  # nothing was submitted for this seller
            client._transport.result(config["round_id"])

    def test_starts_without_an_attestation(self):
        client = self.install_only_client("seller-b")
        asyncio.run(client.start())  # no attestation needed; the loop runs against http://testserver
        asyncio.run(client.stop())


class AttestedLoop(_LoopCase):
    def test_attested_sellers_install_then_join_and_the_round_aggregates(self):
        clients = [self.client(s) for s in SELLERS]
        self.coordinator.open_round(SELLERS, RoundSettings(local_steps=1, batch_size=8, n_neg=8))
        for client in clients:
            client.tick()
        self.assertEqual([c.last_outcome for c in clients],
                         ["round:accepted_on_time", "round:accepted_on_time", "round:aggregated_on_time"])
        latest = self.coordinator.registry.latest()["model_version"]
        for client in clients:
            client.tick()  # next poll installs the new base; the closed round is not joined again
            self.assertEqual((client.installed, client.runtime.serving), (latest, latest))
            self.assertEqual(client.runtime.calls.count("train_round"), 1)

    def test_one_live_event_after_attestation_stops_that_seller(self):
        clients = [self.client(s) for s in SELLERS]
        clients[1].runtime.event_ids.append("live-order-1")  # e.g. an order from the seller screen
        config = self.coordinator.open_round(SELLERS, RoundSettings(local_steps=1, batch_size=8, n_neg=8,
                                                                    deadline_seconds=60))
        for client in clients:
            client.tick()
        self.assertEqual(clients[1].last_outcome, "refused:not_attested_synthetic_input")
        self.assertNotIn("train_round", clients[1].runtime.calls)  # nothing trained, nothing sent
        self.assertEqual(self.coordinator.ledger.get(config["round_id"])["state"], "open")  # waits, then discards
        self.assertEqual(self.coordinator.registry.latest()["model_version"], "model-0")

    def test_an_attestation_of_other_content_or_a_runtime_without_digest_is_refused(self):
        forged = self.client("seller-a", attest_digest="0" * 64)
        self.coordinator.open_round(SELLERS, RoundSettings(local_steps=1, batch_size=8, n_neg=8))
        forged.tick()
        self.assertEqual(forged.last_outcome, "refused:not_attested_synthetic_input")
        plain = self.client("seller-b")
        plain.runtime.snapshot_digest = None  # a runtime that cannot report its snapshot
        plain.tick()
        self.assertEqual(plain.last_outcome, "refused:not_attested_synthetic_input")


class StartRules(_LoopCase):
    def run_start(self, config, runtime=None):
        client = FLClient(runtime or dummy.DummyRuntime("seller-a", self.manifest, 0.0), SellerJobs(), config)
        self.addCleanup(client.jobs.close)
        asyncio.run(client.start())
        asyncio.run(client.stop())

    def test_plaintext_needs_coordinator_token_and_attestation(self):
        for config in (FLClientConfig(enabled=True, mode="synthetic_plaintext"),
                       FLClientConfig(enabled=True, mode="synthetic_plaintext", coordinator_url="http://x", token="t"),
                       FLClientConfig(enabled=True, mode="protected", coordinator_url="http://x", token="t",
                                      synthetic_attestation=Path("a.json"))):
            with self.assertRaises(FeatureNotImplemented):
                self.run_start(config)

    def test_attestation_for_another_seller_is_refused_at_start(self):
        path = self.root / "other.json"
        write_attestation(path, "seller-b", "1" * 64, "test")
        with self.assertRaises(ValueError):
            self.run_start(FLClientConfig(enabled=True, mode="synthetic_plaintext", coordinator_url="http://x",
                                          token="t", synthetic_attestation=path))

    def test_token_is_not_in_the_config_repr(self):
        config = FLClientConfig(enabled=True, mode="synthetic_plaintext", token="seller-a.secret")
        self.assertNotIn("secret", repr(config))


if __name__ == "__main__":
    unittest.main()
