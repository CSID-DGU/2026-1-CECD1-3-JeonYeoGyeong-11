"""g3: synthetic FL with B's real SellerRuntime and C's coordinator (working-agreement §4).

Three synthetic sellers run one round, the base changes, personalization is rebuilt on
the new base, and a fourth seller with its own catalog (not in the cohort) scores with it.
The input is B's g3 scenario (recommender/scenario.py) at CI size: TINY_ARCHITECTURES and
FakeText. The coordinator runs in-process over HTTP (TestClient); the same flow as separate
processes is commerce.deploy.fl_demo (D0025). Generated synthetic input only; no protection (g4).
Ported from B's rehearsal (issue #35), with its fixed settings, under which personalization
is known to pass validation at this size.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.packages.fl_client import rounds
from commerce.packages.fl_client.transport import CoordinatorTransport
from commerce.packages.recommender import scenario as sc
from commerce.services.fl_coordinator import auth
from commerce.services.fl_coordinator.main import CoordinatorSettings, create_app
from commerce.services.fl_coordinator.round_core import ModelRegistry
from commerce.services.fl_coordinator.service import RoundSettings
from commerce.services.merchant_api.jobs import SellerJobs

VARIANT = "text_relation"
PERSONAL = {"steps": 30, "batch_size": 8, "lr": 0.05, "n_neg": 8, "seed": 1,
            "min_train_examples": 5, "min_val_examples": 1}
ROUND = RoundSettings(local_steps=6, max_local_epochs=2, n_neg=8, batch_size=8, learning_rate=0.01, seed=0)


def request(seller, customer, top_n=10):
    return {"schema_version": "recommendation_request.v1", "seller_id": seller, "customer_id_local": customer,
            "as_of": sc.AS_OF, "candidate_item_ids": None, "top_n": top_n}


class G3Flow(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.data = sc.scenario()
        text = sc.FakeText()
        self.runtimes = {}
        for seller_id, seller in self.data.sellers.items():
            runtime = sc.open_test_runtime(seller_id, root, text=text)
            closer = getattr(runtime, "close", None)
            if closer is not None:
                self.addCleanup(closer)
            sc.load_seller(runtime, seller)
            self.runtimes[seller_id] = runtime
        settings = CoordinatorSettings(root / "registry", root / "auth.json", root / "rounds",
                                       mode="synthetic_plaintext")
        manifest, self.base_tensors = sc.random_init_release(self.runtimes[sc.COHORT[0]], VARIANT)
        ModelRegistry(settings.registry_dir).register("base-0", manifest, self.base_tensors,
                                                      provenance={"source": "import", "kind": "random_init"})
        tokens = {s: auth.issue(settings.auth_file, s) for s in self.data.sellers}
        self.http = TestClient(create_app(settings))
        self.addCleanup(self.http.close)
        self.coordinator = self.http.app.state.coordinator
        self.jobs = {s: SellerJobs() for s in self.data.sellers}
        for jobs in self.jobs.values():
            self.addCleanup(jobs.close)
        self.transport = {s: CoordinatorTransport(self.http, tokens[s]) for s in self.data.sellers}

    def predict(self, seller, **kwargs):
        customer = self.data.sellers[seller].customers[0]
        return self.runtimes[seller].predict_local(request(seller, customer), **kwargs)

    def install_everywhere(self):
        for s in self.data.sellers:
            rounds.install_latest(self.runtimes[s], self.jobs[s], self.transport[s], VARIANT)

    def test_round_changes_the_base_personalization_follows_and_a_fourth_seller_scores(self):
        a, b = sc.COHORT[0], sc.COHORT[1]

        # 1 every seller installs base-0 through C and serves model scores
        self.install_everywhere()
        for s in self.data.sellers:
            served = self.predict(s)
            self.assertEqual((served["model_version"], served["fallback_reason"]), ("base-0", None), s)

        # 2 seller-a personalizes on base-0
        result = self.runtimes[a].personalize_local(self.runtimes[a].get_local_data_ref(), PERSONAL,
                                                    model_variant=VARIANT)
        self.assertEqual(result.status, "installed", result.reason)
        self.assertTrue(self.predict(a)["model_version"].startswith("personal."))

        # 3 one round over the fixed cohort; the fourth seller is not in it
        before = {s: self.runtimes[s].export_shared_state(model_variant=VARIANT) for s in sc.COHORT}
        config = self.coordinator.open_round(sc.COHORT, ROUND)
        acks = [rounds.participate(self.runtimes[s], self.jobs[s], self.transport[s], VARIANT)
                for s in (*sc.COHORT, sc.NEW_SELLER)]
        self.assertEqual([ack["disposition"] for ack in acks[:3]],
                         ["accepted_on_time", "accepted_on_time", "aggregated_on_time"])
        self.assertIsNone(acks[3])
        latest = self.coordinator.registry.latest()["model_version"]
        self.assertNotEqual(latest, "base-0")
        self.assertEqual(self.coordinator.ledger.get(config["round_id"])["result_model_version"], latest)
        moved = sum(float(np.abs(t - self.base_tensors[k]).sum())
                    for k, t in self.coordinator.registry.tensors(latest).items())
        self.assertGreater(moved, 0)
        for s in sc.COHORT:  # training used a copy; the serving base did not move
            now = self.runtimes[s].export_shared_state(model_variant=VARIANT)
            self.assertTrue(all(np.array_equal(before[s][k], now[k]) for k in before[s]), s)

        # 4 everyone installs the new base; the old personalization is not attached, then rebuilt
        self.install_everywhere()
        for s in self.data.sellers:
            served = self.predict(s)
            self.assertEqual((served["model_version"], served["fallback_reason"]), (latest, None), s)
        with self.assertRaises(ContractError) as caught:
            self.predict(a, mode="personalized")
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        compare = self.runtimes[a].compare_local(request(a, self.data.sellers[a].customers[0]))
        self.assertEqual(compare.arms[3].unavailable_reason, "personalization_not_ready")
        again = self.runtimes[a].personalize_local(self.runtimes[a].get_local_data_ref(), PERSONAL,
                                                   model_variant=VARIANT)
        self.assertEqual((again.status, again.base_model_version), ("installed", latest), again.reason)

        # 5 the fourth seller scores only its own catalog with the new base
        new = self.data.sellers[sc.NEW_SELLER]
        own = {item["item_id_local"] for item in new.catalog}
        rec = self.runtimes[sc.NEW_SELLER].predict_local(request(sc.NEW_SELLER, new.customers[0], top_n=len(own)))
        self.assertEqual((rec["fallback_reason"], rec["model_version"]), (None, latest))
        self.assertEqual({item["item_id_local"] for item in rec["items"]}, own)
        fresh = self.runtimes[sc.NEW_SELLER].predict_local(request(sc.NEW_SELLER, "never-seen-customer"))
        self.assertEqual(fresh["fallback_reason"], "no_customer_history")

        # 6 a controlled purchase gives the never-bought item a relation
        customer = self.data.sellers[b].customers[1]

        def score():
            items = self.runtimes[b].predict_local(request(b, customer, top_n=100))["items"]
            return {item["item_id_local"]: item["score"] for item in items}[sc.NEW_ITEM]

        epoch, s0 = self.runtimes[b].store.feature_epoch, score()
        self.runtimes[b].ingest_purchase_event(sc.new_item_purchase(
            b, self.data.sellers[b].customers[2], self.data.sellers[b].catalog[0]["item_id_local"]))
        self.assertEqual(self.runtimes[b].store.feature_epoch, epoch + 1)
        self.assertGreater(abs(score() - s0), 1e-6)


if __name__ == "__main__":
    unittest.main()
