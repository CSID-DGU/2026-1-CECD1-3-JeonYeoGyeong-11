"""The seller runtime end to end on a tiny synthetic live seller (no model download).

A fake frozen encoder hashes each text into a fixed vector, and the two service
architectures are shrunk to a few units, so every path runs in seconds on a CPU.
"""
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from commerce.packages.contracts.errors import ContractError, JobBusyError
from commerce.packages.contracts.ids import manifest_hash, purchase_event_id
from commerce.packages.contracts.ports import RecommenderRuntime
from commerce.packages.recommender import seller_runtime
from commerce.packages.recommender.feature_store import FeatureStore
from commerce.packages.recommender.harex import HarexConfig
from commerce.packages.recommender.seller_runtime import (
    FALLBACK_MODEL_VERSION, SellerRuntime, is_validation_customer, planned_steps, round_train_seed,
)
from commerce.packages.recommender.serving import (
    SERVICE_ENCODER, build_manifest, build_model, canonical_npz, read_npz, shared_tensors,
)

SELLER = "seller-test"
D_TEXT = 8
TINY = {
    1: HarexConfig("test.T_lm", "lm", False, d_model=16, n_heads=2, d_ffn=32, dropout=0.0, max_items=12,
                   d_text=D_TEXT, d_relation=4, mlp_hidden=16, d_time=4),
    2: HarexConfig("test.R_lm", "lm", True, d_model=16, n_heads=2, d_ffn=32, dropout=0.0, max_items=12,
                   d_text=D_TEXT, d_relation=4, mlp_hidden=16, d_time=4),
}
N_ITEMS = 12
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


class FakeText:
    text_artifact_hash = "a" * 64
    dim = D_TEXT

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        rows = [np.frombuffer(hashlib.sha256(t.encode("utf-8")).digest()[:D_TEXT], dtype=np.uint8) for t in texts]
        return (np.stack(rows).astype(np.float32) / 255.0) if rows else np.zeros((0, D_TEXT), np.float32)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def item(i: int, *, status="active", title=None) -> dict:
    return {"schema_version": "catalog_item.v1", "seller_id": SELLER, "item_id_local": "item-%02d" % i,
            "source": "live", "title_text": title or "테스트 상품 %d" % i, "description_text": None,
            "category_path": ["식품", "group%d" % (i % 3)], "listing_status": status,
            "first_listed_at": "2026-08-01T00:00:00Z"}


def event(customer: str, basket: str, when: datetime, items: list[int], seller=SELLER) -> dict:
    return {"schema_version": "purchase_event.v1", "seller_id": seller, "source": "live",
            "seller_partition": "platform_seller", "customer_id_local": customer, "basket_id_local": basket,
            "purchase_event_id": purchase_event_id(seller, "live", basket),
            "time": {"kind": "absolute", "value": iso(when)}, "order_rank": None,
            "items": [{"item_id_local": "item-%02d" % i, "quantity_observed": 1} for i in sorted(set(items))]}


def request(customer: str, as_of: datetime, candidates=None, top_n=None) -> dict:
    return {"schema_version": "recommendation_request.v1", "seller_id": SELLER, "customer_id_local": customer,
            "as_of": iso(as_of), "candidate_item_ids": candidates, "top_n": top_n}


def history(n_customers=10, visits=6, seed=0, popular=()) -> list[dict]:
    """Each customer keeps two favourites and adds two random items; popular items join every visit."""
    rng = random.Random(seed)
    out = []
    for c in range(n_customers):
        favourite = [c % N_ITEMS + 1, (c + 3) % N_ITEMS + 1] + list(popular)
        for v in range(visits):
            items = favourite + rng.sample(range(1, N_ITEMS + 1), 2)
            out.append(event("cust-%02d" % c, "b-%02d-%02d" % (c, v), START + timedelta(days=7 * v + c), items))
    return out


def release_for(runtime: SellerRuntime, variant: str, version: str, *, seed=0, tensors=None) -> tuple:
    manifest = runtime.get_shared_manifest(model_variant=variant)
    config = TINY[manifest["architecture_version"]]
    tensors = tensors if tensors is not None else shared_tensors(build_model(config, seed=seed))
    data = canonical_npz(tensors, [t["name"] for t in manifest["tensors"]])
    release = {"schema_version": "model_release.v1", "model_version": version,
               "manifest_hash": manifest["manifest_hash"], "weights_sha256": hashlib.sha256(data).hexdigest(),
               "weights_size_bytes": len(data)}
    return release, manifest, tensors


def round_config(runtime: SellerRuntime, variant: str, **overrides) -> dict:
    handle_manifest = runtime.get_shared_manifest(model_variant=variant)
    config = {"schema_version": "round_config.v1", "round_id": "round-0001",
              "model_version": runtime._handles[variant].model_version,
              "manifest_hash": handle_manifest["manifest_hash"],
              "architecture_version": handle_manifest["architecture_version"], "min_clients": 1,
              "deadline": "2026-10-10T00:00:00Z", "local_steps": 6, "max_local_epochs": 2, "n_neg": 8,
              "batch_size": 4, "learning_rate": 0.01, "seed": 7}
    config.update(overrides)
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.text = FakeText()
        self.runtime = self.open()

    def tearDown(self):
        self.tmp.cleanup()

    def open(self) -> SellerRuntime:
        return SellerRuntime(SELLER, self.root / "features.sqlite", self.root / "models", text=self.text,
                             architectures=TINY)

    def fill(self, runtime=None, catalog=True, events=True):
        runtime = runtime or self.runtime
        if catalog:
            for i in range(1, N_ITEMS + 1):
                runtime.upsert_catalog_item(item(i), i)
        if events:
            for e in history():
                runtime.ingest_purchase_event(e)

    def install(self, variant="text_relation", version="base-1", **kwargs):
        release, manifest, tensors = release_for(self.runtime, variant, version, **kwargs)
        self.runtime.install_release(release, manifest, tensors, model_variant=variant)
        return release, manifest, tensors


class FeatureStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeatureStore(Path(self.tmp.name) / "f.sqlite", SELLER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_event_is_stored_once_and_a_changed_body_is_rejected(self):
        e = event("c1", "b1", START, [1, 2])
        self.assertTrue(self.store.ingest_event(e))
        self.assertFalse(self.store.ingest_event(dict(e)))
        self.assertEqual(self.store.feature_epoch, 1)
        changed = event("c1", "b1", START, [1, 3])
        with self.assertRaises(ContractError) as caught:
            self.store.ingest_event(changed)
        self.assertEqual(caught.exception.code, "DUPLICATE_EVENT")
        self.assertEqual(self.store.feature_epoch, 1)

    def test_another_sellers_event_is_forbidden(self):
        with self.assertRaises(ContractError) as caught:
            self.store.ingest_event(event("c1", "b1", START, [1], seller="seller-other"))
        self.assertEqual(caught.exception.code, "FORBIDDEN")

    def test_catalog_keeps_the_newest_source_seq_and_old_epochs_stay_readable(self):
        self.assertTrue(self.store.upsert_item(item(1, title="old name"), 5))
        epoch_old = self.store.feature_epoch
        self.assertTrue(self.store.upsert_item(item(1, title="new name"), 7))
        self.assertFalse(self.store.upsert_item(item(1, title="late older"), 6))  # late, older: ignored
        self.assertFalse(self.store.upsert_item(item(1, title="new name"), 7))  # same seq, same body
        with self.assertRaises(ContractError) as caught:
            self.store.upsert_item(item(1, title="other"), 7)
        self.assertEqual(caught.exception.code, "DUPLICATE_EVENT")
        self.assertEqual(self.store.snapshot().catalog["item-01"]["title_text"], "new name")
        self.assertEqual(self.store.snapshot(epoch_old).catalog["item-01"]["title_text"], "old name")

    def test_a_store_belongs_to_one_seller(self):
        with self.assertRaises(ValueError):
            FeatureStore(self.store.path, "seller-other")


class ServingTest(Base):
    def test_it_is_a_recommender_runtime(self):
        self.assertIsInstance(self.runtime, RecommenderRuntime)

    def test_without_a_model_it_falls_back_to_local_popularity(self):
        self.fill()
        out = self.runtime.predict_local(request("cust-00", START + timedelta(days=60)))
        self.assertTrue(out["is_cold_start"])
        self.assertEqual(out["fallback_reason"], "no_shared_model")
        self.assertEqual(out["model_version"], FALLBACK_MODEL_VERSION)
        scores = [i["score"] for i in out["items"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(out["items"]), 10)

    def test_a_seller_without_history_lists_by_id(self):
        self.fill(events=False)
        self.install()
        out = self.runtime.predict_local(request("cust-00", START), model_variant="text_relation")
        self.assertEqual(out["fallback_reason"], "no_seller_history")
        self.assertEqual([i["item_id_local"] for i in out["items"]], ["item-%02d" % i for i in range(1, 11)])

    def test_the_model_ranks_known_customers_and_falls_back_for_new_ones(self):
        self.fill()
        self.install(version="base-1")
        out = self.runtime.predict_local(request("cust-03", START + timedelta(days=60), top_n=5))
        self.assertFalse(out["is_cold_start"])
        self.assertIsNone(out["fallback_reason"])
        self.assertEqual(out["model_version"], "base-1")
        self.assertEqual(len(out["items"]), 5)
        new = self.runtime.predict_local(request("cust-new", START + timedelta(days=60)))
        self.assertEqual(new["fallback_reason"], "no_customer_history")
        self.assertEqual(new["model_version"], "base-1")

    def test_events_at_or_after_as_of_are_not_history(self):
        self.fill()
        self.install()
        out = self.runtime.predict_local(request("cust-00", START))
        self.assertEqual(out["fallback_reason"], "no_seller_history")

    def test_explicit_candidates(self):
        self.fill()
        self.runtime.upsert_catalog_item(item(3, status="inactive"), 100)
        self.install()
        late = START + timedelta(days=60)
        out = self.runtime.predict_local(request("cust-01", late, candidates=["item-02", "item-03", "item-05"]))
        self.assertEqual(sorted(i["item_id_local"] for i in out["items"]), ["item-02", "item-05"])
        self.assertEqual(self.runtime.predict_local(request("cust-01", late, candidates=[]))["items"], [])
        with self.assertRaises(ContractError) as caught:
            self.runtime.predict_local(request("cust-01", late, candidates=["item-99"]))
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        auto = self.runtime.predict_local(request("cust-01", late))
        self.assertNotIn("item-03", [i["item_id_local"] for i in auto["items"]])

    def test_explicit_personalized_mode_without_a_personalization_is_not_found(self):
        self.fill()
        self.install()
        with self.assertRaises(ContractError) as caught:
            self.runtime.predict_local(request("cust-01", START + timedelta(days=60)), mode="personalized")
        self.assertEqual(caught.exception.code, "NOT_FOUND")

    def test_another_sellers_request_is_forbidden(self):
        r = request("cust-01", START)
        r["seller_id"] = "seller-other"
        with self.assertRaises(ContractError) as caught:
            self.runtime.predict_local(r)
        self.assertEqual(caught.exception.code, "FORBIDDEN")

    def test_text_vectors_are_cached_in_the_feature_store(self):
        self.fill()
        self.install()
        late = START + timedelta(days=60)
        self.runtime.predict_local(request("cust-01", late))
        calls = self.text.calls
        reopened = self.open()
        reopened.predict_local(request("cust-01", late))
        self.assertEqual(self.text.calls, calls)  # the z rows came from features.sqlite


class ReleaseTest(Base):
    def test_manifests_name_only_shared_float32_tensors(self):
        only = self.runtime.get_shared_manifest(model_variant="text_only")
        rel = self.runtime.get_shared_manifest(model_variant="text_relation")
        self.assertEqual(only["manifest_hash"], manifest_hash(only))
        self.assertTrue(all(t["name"].startswith("shared.") and t["dtype"] == "float32" for t in rel["tensors"]))
        groups = lambda m: {t["name"].split("_")[0] for t in m["tensors"]}
        self.assertEqual(groups(rel) - groups(only), {"shared.time", "shared.relation"})
        self.assertNotEqual(only["architecture_version"], rel["architecture_version"])
        self.assertEqual(rel["text_artifact_hash"], FakeText.text_artifact_hash)

    def test_install_serves_exports_and_survives_a_restart(self):
        self.fill()
        _, _, tensors = self.install(version="base-1")
        exported = self.runtime.export_shared_state(model_variant="text_relation")
        self.assertEqual(set(exported), set(tensors))
        self.assertTrue(all(np.array_equal(exported[k], tensors[k]) for k in tensors))
        late = START + timedelta(days=60)
        before = self.runtime.predict_local(request("cust-02", late))
        after = self.open().predict_local(request("cust-02", late))
        self.assertEqual(before, after)

    def test_a_repeated_release_is_idempotent_and_a_changed_one_is_rejected(self):
        self.fill()
        release, manifest, tensors = self.install(version="base-1")
        self.runtime.install_release(release, manifest, tensors, model_variant="text_relation")
        other, other_manifest, other_tensors = release_for(self.runtime, "text_relation", "base-1", seed=5)
        with self.assertRaises(ContractError) as caught:
            self.runtime.install_release(other, other_manifest, other_tensors, model_variant="text_relation")
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")
        self.assertTrue(np.array_equal(self.runtime.export_shared_state()[next(iter(tensors))],
                                       tensors[next(iter(tensors))]))

    def test_a_release_for_another_text_encoder_or_preprocessing_is_rejected(self):
        # C fixes one manifest per variant; only B knows the encoder installed at this seller.
        self.fill()
        release, own, tensors = release_for(self.runtime, "text_relation", "base-1")
        arch = self.runtime._arch("text_relation")
        for field in ("text_artifact_hash", "preprocessing_version"):
            fields = {"text_artifact_hash": own["text_artifact_hash"],
                      "preprocessing_version": own["preprocessing_version"], field: "b" * 64}
            foreign = build_manifest(arch, **fields)
            with self.assertRaises(ContractError) as caught:
                self.runtime.install_release(dict(release, manifest_hash=foreign["manifest_hash"]), foreign,
                                             tensors, model_variant="text_relation")
            self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")
        served = self.runtime.predict_local(request("cust-02", START + timedelta(days=60)))
        self.assertEqual(served["fallback_reason"], "no_shared_model")

    def test_a_failed_release_keeps_the_old_base_serving(self):
        self.fill()
        self.install(version="base-1")
        late = START + timedelta(days=60)
        served = self.runtime.predict_local(request("cust-04", late))
        release, manifest, tensors = release_for(self.runtime, "text_relation", "base-2", seed=3)
        bad = dict(tensors)
        name = next(iter(bad))
        bad[name] = bad[name].copy()
        bad[name].flat[0] = np.nan
        with self.assertRaises(ContractError):
            self.runtime.install_release(release, manifest, bad, model_variant="text_relation")
        wrong_manifest = dict(release, manifest_hash="0" * 64)
        with self.assertRaises(ContractError) as caught:
            self.runtime.install_release(wrong_manifest, manifest, tensors, model_variant="text_relation")
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")
        missing = {k: v for k, v in tensors.items() if k != name}
        with self.assertRaises(ContractError) as caught:
            self.runtime.install_release(release, manifest, missing, model_variant="text_relation")
        self.assertEqual(caught.exception.code, "TENSOR_SET_MISMATCH")
        self.assertEqual(self.runtime.predict_local(request("cust-04", late)), served)
        self.assertEqual(self.open().predict_local(request("cust-04", late)), served)

    def test_a_release_c_encoded_with_np_savez_installs(self):
        """C's registry writes np.savez bytes and its client checks them; B gets the decoded tensors."""
        import io
        self.fill()
        _, manifest, tensors = release_for(self.runtime, "text_relation", "base-c")
        buffer = io.BytesIO()
        np.savez(buffer, **tensors)
        data = buffer.getvalue()
        release = {"schema_version": "model_release.v1", "model_version": "base-c",
                   "manifest_hash": manifest["manifest_hash"], "weights_sha256": hashlib.sha256(data).hexdigest(),
                   "weights_size_bytes": len(data)}
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            decoded = {k: archive[k] for k in archive.files}
        self.runtime.install_release(release, manifest, decoded, model_variant="text_relation")
        self.runtime.install_release(release, manifest, decoded, model_variant="text_relation")  # repeated
        late = START + timedelta(days=60)
        self.assertEqual(self.runtime.predict_local(request("cust-02", late))["model_version"], "base-c")
        self.assertEqual(self.open().predict_local(request("cust-02", late))["model_version"], "base-c")

    def test_a_release_for_the_other_variant_is_rejected(self):
        release, manifest, tensors = release_for(self.runtime, "text_only", "base-1")
        with self.assertRaises(ContractError) as caught:
            self.runtime.install_release(release, manifest, tensors, model_variant="text_relation")
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")

    def test_a_new_base_replaces_the_old_one(self):
        self.fill()
        self.install(version="base-1")
        self.install(version="base-2", seed=9)
        late = START + timedelta(days=60)
        self.assertEqual(self.runtime.predict_local(request("cust-05", late))["model_version"], "base-2")
        self.assertEqual(self.open().predict_local(request("cust-05", late))["model_version"], "base-2")

    def test_serving_scores_equal_the_lab_path(self):
        """model-lab.md §6.8: the same input scores the same through training code and the service."""
        import torch
        from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
        from commerce.packages.data_adapters.text import catalog_item_text
        from commerce.packages.recommender.examples import query_example
        from commerce.packages.recommender.relations import build_relations
        from commerce.packages.recommender.serving import load_shared
        from commerce.packages.recommender.training import SellerData, catalog_scores

        self.fill()
        for variant, version in (("text_only", "base-t"), ("text_relation", "base-r")):
            _, _, tensors = self.install(variant=variant, version=version, seed=11)
            config = TINY[self.runtime.get_shared_manifest(model_variant=variant)["architecture_version"]]
            model = build_model(config)
            load_shared(model, tensors)
            items = tuple("item-%02d" % i for i in range(1, N_ITEMS + 1))
            z = torch.from_numpy(self.text.encode([catalog_item_text(item(i)) for i in range(1, N_ITEMS + 1)]))
            baskets = [basket_from_event(e) for e in history()]
            visits = {c: customer_visits([b for b in baskets if b.customer_id_local == c])
                      for c in {b.customer_id_local for b in baskets}}
            row_of = {i: r for r, i in enumerate(items)}
            relations = build_relations(visits, row_of, len(items)) if config.relation else None
            example = query_example(visits["cust-06"])
            lab = catalog_scores(model, SellerData(SELLER, items, z, [example], relations), [example])[0]
            served = self.runtime.predict_local(request("cust-06", START + timedelta(days=60), top_n=N_ITEMS),
                                                model_variant=variant)
            got = {i["item_id_local"]: i["score"] for i in served["items"]}
            for row, item_id in enumerate(items):
                self.assertAlmostEqual(got[item_id], float(lab[row]), places=5)

    def test_canonical_npz_is_stable_and_round_trips(self):
        manifest = self.runtime.get_shared_manifest(model_variant="text_only")
        tensors = shared_tensors(build_model(TINY[1]))
        order = [t["name"] for t in manifest["tensors"]]
        data = canonical_npz(tensors, order)
        self.assertEqual(data, canonical_npz({k: tensors[k].copy() for k in reversed(order)}, order))
        back = read_npz(data, manifest)
        self.assertTrue(all(np.array_equal(back[k], tensors[k]) for k in tensors))

    def test_the_service_encoder_is_the_probe_choice(self):
        from commerce.evaluation.encoder_probe import CANDIDATES
        self.assertEqual(SERVICE_ENCODER, CANDIDATES["minilm-l12"])


class TrainRoundTest(Base):
    def setUp(self):
        super().setUp()
        self.fill()
        self.install(variant="text_relation", version="base-1")
        self.install(variant="text_only", version="base-t1")

    def test_a_round_returns_a_delta_and_leaves_the_serving_base_alone(self):
        for variant in ("text_only", "text_relation"):
            before = self.runtime.export_shared_state(model_variant=variant)
            ref = self.runtime.get_local_data_ref()
            result = self.runtime.train_round(ref, round_config(self.runtime, variant), model_variant=variant)
            self.assertTrue(result.completed)
            self.assertEqual(set(result.shared_delta), set(before))
            self.assertTrue(all(result.shared_delta[k].shape == before[k].shape for k in before))
            self.assertTrue(any(np.abs(d).sum() > 0 for d in result.shared_delta.values()))
            self.assertTrue(result.metrics["grad_norm_mean"] is not None)
            after = self.runtime.export_shared_state(model_variant=variant)
            self.assertTrue(all(np.array_equal(before[k], after[k]) for k in before))

    def test_one_run_seed_samples_anew_each_round_but_keeps_its_validation_customers(self):
        ref = self.runtime.get_local_data_ref()
        seen = []
        real = self.runtime.training_parts

        def spy(epoch, variant, run_seed):
            seen.append(run_seed)
            return real(epoch, variant, run_seed)

        self.runtime.training_parts = spy

        def delta(round_id):
            config = round_config(self.runtime, "text_only", round_id=round_id)
            return self.runtime.train_round(ref, config, model_variant="text_only").shared_delta

        first, again, second = delta("round-0001"), delta("round-0001"), delta("round-0002")
        self.assertTrue(all(np.array_equal(first[k], again[k]) for k in first))
        self.assertFalse(all(np.array_equal(first[k], second[k]) for k in first))
        self.assertEqual(seen, [7, 7, 7])
        self.assertNotEqual(round_train_seed(SELLER, 7, "round-0001"), round_train_seed(SELLER, 7, "round-0002"))
        self.assertNotEqual(round_train_seed(SELLER, 7, "round-0001"), round_train_seed("other", 7, "round-0001"))

    def test_a_round_must_name_the_installed_base(self):
        ref = self.runtime.get_local_data_ref()
        with self.assertRaises(ContractError) as caught:
            self.runtime.train_round(ref, round_config(self.runtime, "text_relation", model_version="base-0"))
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        with self.assertRaises(ContractError) as caught:
            self.runtime.train_round(ref, round_config(self.runtime, "text_relation", manifest_hash="f" * 64))
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")

    def test_zero_local_steps_is_an_incomplete_zero_delta(self):
        result = self.runtime.train_round(self.runtime.get_local_data_ref(),
                                          round_config(self.runtime, "text_only", local_steps=0),
                                          model_variant="text_only")
        self.assertFalse(result.completed)
        self.assertTrue(all(not d.any() for d in result.shared_delta.values()))

    def test_a_second_job_is_refused_while_one_runs(self):
        self.runtime._job.acquire()
        try:
            with self.assertRaises(JobBusyError):
                self.runtime.train_round(self.runtime.get_local_data_ref(), round_config(self.runtime, "text_only"),
                                         model_variant="text_only")
        finally:
            self.runtime._job.release()

    def test_local_data_ref_pins_the_snapshot_and_belongs_to_the_seller(self):
        ref = self.runtime.get_local_data_ref()
        epoch = int(ref.split(".")[1])
        train_parts, _ = self.runtime.training_parts(epoch, "text_only", 7)
        n = sum(len(p.examples) for p in train_parts)
        for v in range(6, 9):  # later visits add examples to the ledger, not to the pinned snapshot
            self.runtime.ingest_purchase_event(event("cust-00", "late-%d" % v, START + timedelta(days=7 * v), [1]))
        again, _ = self.runtime.training_parts(epoch, "text_only", 7)
        self.assertEqual(sum(len(p.examples) for p in again), n)
        with self.assertRaises(ContractError) as caught:
            self.runtime.train_round(ref[:-1] + ("0" if ref[-1] != "0" else "1"),
                                     round_config(self.runtime, "text_only"), model_variant="text_only")
        self.assertEqual(caught.exception.code, "FORBIDDEN")

    def test_the_validation_customers_depend_on_seller_and_run_seed_only(self):
        _, val = self.runtime.training_parts(self.runtime.store.feature_epoch, "text_relation", 7)
        held = {ex.customer_id_local for part in val for ex in part.examples}
        expected = {"cust-%02d" % c for c in range(10) if is_validation_customer(SELLER, 7, "cust-%02d" % c)}
        self.assertEqual(held, expected)
        train_parts, _ = self.runtime.training_parts(self.runtime.store.feature_epoch, "text_relation", 7)
        trained = {ex.customer_id_local for part in train_parts for ex in part.examples}
        self.assertFalse(held & trained)

    def test_relation_snapshots_never_see_the_target_visit(self):
        train_parts, _ = self.runtime.training_parts(self.runtime.store.feature_epoch, "text_relation", 7)
        self.assertGreater(len(train_parts), 1)
        for part in train_parts:
            self.assertIsNotNone(part.relations)

    def test_planned_steps(self):
        config = {"local_steps": 40, "max_local_epochs": 6, "batch_size": 64}
        self.assertEqual(planned_steps(100, config), 12)
        self.assertEqual(planned_steps(10000, config), 40)
        self.assertEqual(planned_steps(100, dict(config, max_local_epochs=0)), 40)
        self.assertEqual(planned_steps(100, dict(config, local_steps=0)), 0)
        self.assertEqual(planned_steps(0, config), 0)


class AgreementTest(Base):
    """What A and C can rely on for OQ07 (a purchase before its item), OQ08 (jobs, restarts)
    and OQ09 (fallback model_version and scores)."""

    def ranked(self, out) -> list[tuple[str, float]]:
        return [(i["item_id_local"], i["score"]) for i in out["items"]]

    def test_a_purchase_before_its_item_is_kept_and_counts_once_the_item_arrives(self):
        for i in (1, 2, 3):
            self.runtime.upsert_catalog_item(item(i), i)
        self.runtime.ingest_purchase_event(event("cust-a", "b-1", START, [1, 4]))
        self.runtime.ingest_purchase_event(event("cust-a", "b-2", START + timedelta(days=1), [4]))
        late = START + timedelta(days=30)
        before = self.runtime.predict_local(request("cust-a", late))
        self.assertEqual(self.ranked(before), [("item-01", 1.0), ("item-02", 0.0), ("item-03", 0.0)])
        epoch = self.runtime.store.feature_epoch
        self.runtime.upsert_catalog_item(item(4), 4)
        self.assertGreater(self.runtime.store.feature_epoch, epoch)
        after = self.runtime.predict_local(request("cust-a", late))
        self.assertEqual(self.ranked(after)[:2], [("item-04", 2.0), ("item-01", 1.0)])

    def test_a_redelivery_after_a_restart_is_applied_once(self):
        self.fill()
        epoch = self.runtime.store.feature_epoch
        late = START + timedelta(days=60)
        served = self.runtime.predict_local(request("cust-00", late))
        restarted = self.open()
        for e in history():
            restarted.ingest_purchase_event(e)  # A's pending rows after a crash before "delivered"
        self.assertEqual(restarted.store.feature_epoch, epoch)
        self.assertEqual(restarted.predict_local(request("cust-00", late)), served)

    def test_fallback_scores_are_basket_counts_and_model_version_names_the_serving_base(self):
        self.fill()
        late = START + timedelta(days=60)
        counts = {}
        for e in history():
            for i in e["items"]:
                counts[i["item_id_local"]] = counts.get(i["item_id_local"], 0) + 1
        out = self.runtime.predict_local(request("cust-new", late, top_n=N_ITEMS))
        self.assertEqual(out["model_version"], FALLBACK_MODEL_VERSION)
        expected = {"item-%02d" % i: float(counts.get("item-%02d" % i, 0)) for i in range(1, N_ITEMS + 1)}
        self.assertEqual(dict(self.ranked(out)), expected)
        self.install(version="base-1")
        new = self.runtime.predict_local(request("cust-new", late, top_n=N_ITEMS))
        self.assertEqual((new["model_version"], new["fallback_reason"], new["is_cold_start"]),
                         ("base-1", "no_customer_history", True))
        self.assertEqual(new["items"], out["items"])

    def test_serving_goes_on_while_a_round_trains_and_a_second_job_is_refused(self):
        self.fill()
        self.install(version="base-1")
        started, release = threading.Event(), threading.Event()
        real_train = seller_runtime.train

        def slow_train(*args, **kwargs):
            started.set()
            release.wait(30)
            return real_train(*args, **kwargs)

        results = []
        with mock.patch.object(seller_runtime, "train", slow_train):
            worker = threading.Thread(target=lambda: results.append(self.runtime.train_round(
                self.runtime.get_local_data_ref(), round_config(self.runtime, "text_relation"))))
            worker.start()
            try:
                self.assertTrue(started.wait(30))
                out = self.runtime.predict_local(request("cust-03", START + timedelta(days=60)))
                self.assertEqual((out["model_version"], out["fallback_reason"]), ("base-1", None))
                self.runtime.ingest_purchase_event(event("cust-03", "during", START + timedelta(days=61), [1]))
                with self.assertRaises(JobBusyError):
                    self.runtime.personalize_local(self.runtime.get_local_data_ref(), {})
            finally:
                release.set()
                worker.join(60)
        self.assertTrue(results and results[0].completed)

    def test_a_failed_round_keeps_the_base_serving_and_frees_the_job(self):
        self.fill()
        self.install(version="base-1")
        before = self.runtime.export_shared_state()
        with mock.patch.object(seller_runtime, "train", side_effect=RuntimeError("killed mid-round")):
            with self.assertRaises(RuntimeError):
                self.runtime.train_round(self.runtime.get_local_data_ref(), round_config(self.runtime, "text_relation"))
        after = self.runtime.export_shared_state()
        self.assertTrue(all(np.array_equal(before[k], after[k]) for k in before))
        self.assertEqual(self.runtime.predict_local(request("cust-03", START + timedelta(days=60)))["model_version"],
                         "base-1")
        result = self.runtime.train_round(self.runtime.get_local_data_ref(), round_config(self.runtime, "text_relation"))
        self.assertTrue(result.completed)

    def test_a_seller_with_nothing_to_learn_returns_an_incomplete_zero_delta(self):
        self.fill(events=False)
        self.install(version="base-1")
        result = self.runtime.train_round(self.runtime.get_local_data_ref(), round_config(self.runtime, "text_relation"))
        self.assertFalse(result.completed)
        self.assertEqual(result.metrics, {"loss_mean": None, "grad_norm_mean": None})
        self.assertTrue(all(not d.any() for d in result.shared_delta.values()))


class WarmUpTest(Base):
    """open_runtime's warm=True: the first request finds z, relations and e already computed."""

    def tearDown(self):
        for runtime in getattr(self, "warm_runtimes", []):
            runtime.close()  # stop the thread before the folder goes
        super().tearDown()

    def open_warm(self) -> SellerRuntime:
        runtime = SellerRuntime(SELLER, self.root / "features.sqlite", self.root / "models", text=self.text,
                                architectures=TINY, warm=True, clock=lambda: START + timedelta(days=90))
        self.warm_runtimes = getattr(self, "warm_runtimes", []) + [runtime]
        return runtime

    def test_the_first_request_after_warm_up_computes_nothing(self):
        self.fill()
        self.install(version="base-1")
        warm = self.open_warm()
        self.assertTrue(warm.wait_warm(30))
        key = ("text_relation", "base-1", warm.store.feature_epoch)
        self.assertIn(key, warm._e_cache)
        cached, calls = warm._e_cache[key], self.text.calls
        served = warm.predict_local(request("cust-03", START + timedelta(days=60)))
        self.assertIsNone(served["fallback_reason"])
        self.assertEqual(self.text.calls, calls)  # no text encoded on the request
        self.assertIs(warm._e_cache[key], cached)  # e not recomputed either
        self.assertEqual(served, self.open().predict_local(request("cust-03", START + timedelta(days=60))))

    def test_a_new_event_or_release_is_warmed_again(self):
        self.fill()
        warm = self.open_warm()
        self.assertTrue(warm.wait_warm(30))
        release, manifest, tensors = release_for(warm, "text_only", "base-t")
        warm.install_release(release, manifest, tensors, model_variant="text_only")
        self.assertTrue(warm.wait_warm(30))
        self.assertIn(("text_only", "base-t", warm.store.feature_epoch), warm._e_cache)
        warm.ingest_purchase_event(event("cust-00", "late-1", START + timedelta(days=50), [1, 2]))
        self.assertTrue(warm.wait_warm(30))
        self.assertIn(("text_only", "base-t", warm.store.feature_epoch), warm._e_cache)

    def test_without_warm_there_is_no_thread_and_close_is_safe(self):
        self.assertIsNone(self.runtime._warmer)
        self.assertTrue(self.runtime.wait_warm(0))
        self.runtime.close()
        warm = self.open_warm()
        warm.close()
        self.assertIsNone(warm._warmer)
        self.fill(runtime=warm)  # still answers, computing on demand
        self.assertEqual(warm.predict_local(request("cust-01", START + timedelta(days=60)))["fallback_reason"],
                         "no_shared_model")

    def test_events_after_now_leave_the_rest_to_the_request(self):
        self.fill()
        self.install(version="base-1")
        early = SellerRuntime(SELLER, self.root / "features.sqlite", self.root / "models", text=self.text,
                              architectures=TINY, warm=True, clock=lambda: START + timedelta(days=10))
        self.warm_runtimes = [early]
        self.assertTrue(early.wait_warm(30))
        self.assertEqual(early._e_cache, {})  # z only: a live request cuts the ledger at its own as_of
        self.assertIsNotNone(early._catalog_cache)

    def test_open_runtime_warms_and_copes_without_an_encoder(self):
        from commerce.packages.recommender.runtime import open_runtime
        runtime = open_runtime(SELLER, self.root / "f2.sqlite", self.root / "m2")
        self.warm_runtimes = [runtime]
        self.assertIsNotNone(runtime._warmer)
        self.fill(runtime=runtime)
        self.assertTrue(runtime.wait_warm(30))
        self.assertNotIn("warm", runtime.load_errors)
        self.assertEqual(runtime.predict_local(request("cust-01", START + timedelta(days=60)))["fallback_reason"],
                         "no_shared_model")


class BundleTest(Base):
    def test_a_lab_run_becomes_an_installable_release(self):
        import json
        import torch
        from commerce.evaluation.release_bundle import bundle
        from commerce.packages.recommender.serving import load_bundle
        from commerce.packages.recommender.z_cache import preprocessing_version

        run = self.root / "run"
        run.mkdir()
        state = build_model(TINY[2], seed=4).shared_state()
        torch.save({"architecture": TINY[2].architecture_version, "variant": "R_lm", "best_round": 3,
                    "final_round_shared": state, "best_round_shared": state}, run / "shared_weights.pt")
        record = {"encoder": {"text_artifact_hash": FakeText.text_artifact_hash,
                              "preprocessing_version": preprocessing_version()},
                  "settings": {"sellers": 3, "target": "basket", "rounds": 5, "seed": 0}, "code": None,
                  "labels": ["비보호 FL 시뮬레이션"]}
        (run / "record.json").write_text(json.dumps(record), encoding="utf-8")
        out = self.root / "release"
        bundle(run, out, "lab-R-1", FakeText.text_artifact_hash, architectures=TINY)
        release, manifest, tensors = load_bundle(out)
        self.fill()
        self.runtime.install_release(release, manifest, tensors, model_variant="text_relation")
        served = self.runtime.predict_local(request("cust-01", START + timedelta(days=60)))
        self.assertEqual(served["model_version"], "lab-R-1")
        self.assertFalse(served["is_cold_start"])
        self.assertTrue((out / "provenance.json").is_file())
        wrong = dict(record, encoder=dict(record["encoder"], preprocessing_version="0" * 64))
        (run / "record.json").write_text(json.dumps(wrong), encoding="utf-8")
        with self.assertRaises(SystemExit):
            bundle(run, self.root / "release-2", "lab-R-2", FakeText.text_artifact_hash, architectures=TINY)


LEARN = {"steps": 30, "batch_size": 8, "lr": 0.05, "n_neg": 8, "seed": 3, "min_train_examples": 5,
         "min_val_examples": 1}


class PersonalizationTest(Base):
    def setUp(self):
        super().setUp()
        for i in range(1, N_ITEMS + 1):
            self.runtime.upsert_catalog_item(item(i), i)
        for e in history(n_customers=30, visits=7, popular=(1, 2)):  # something every customer shares
            self.runtime.ingest_purchase_event(e)
        self.install(variant="text_only", version="base-t1", seed=2)
        self.install(variant="text_relation", version="base-r1", seed=2)
        self.late = START + timedelta(days=90)

    def personalize(self, variant="text_relation", **config):
        return self.runtime.personalize_local(self.runtime.get_local_data_ref(), dict(LEARN, **config),
                                              model_variant=variant)

    def test_only_query_proj_and_scorer_change_and_the_base_stays(self):
        before = self.runtime.export_shared_state(model_variant="text_relation")
        result = self.personalize()
        self.assertEqual(result.status, "installed", result)
        self.assertEqual(result.base_model_version, "base-r1")
        after = self.runtime.export_shared_state(model_variant="text_relation")
        self.assertTrue(all(np.array_equal(before[k], after[k]) for k in before))  # the base hash is unchanged
        heads = self.runtime._personal["text_relation"].tensors
        self.assertTrue(heads and all(k.startswith(("personal.query_proj", "personal.scorer")) for k in heads))
        self.assertFalse(any(k.startswith("personal") for k in self.runtime.export_shared_state()))

    def test_auto_serves_the_personalization_and_global_the_base(self):
        result = self.personalize()
        auto = self.runtime.predict_local(request("cust-04", self.late))
        self.assertEqual(auto["model_version"], "personal.%s" % result.personalization_revision)
        self.assertEqual(self.runtime.predict_local(request("cust-04", self.late), mode="global")["model_version"],
                         "base-r1")
        self.assertEqual(self.runtime.predict_local(request("cust-04", self.late), mode="personalized"), auto)
        self.assertEqual(self.open().predict_local(request("cust-04", self.late)), auto)  # survives a restart

    def test_a_new_base_drops_the_old_personalization(self):
        self.personalize()
        self.install(variant="text_relation", version="base-r2", seed=8)
        served = self.runtime.predict_local(request("cust-04", self.late))
        self.assertEqual(served["model_version"], "base-r2")
        with self.assertRaises(ContractError):
            self.runtime.predict_local(request("cust-04", self.late), mode="personalized")
        self.assertEqual(self.open().predict_local(request("cust-04", self.late))["model_version"], "base-r2")

    def test_a_personalization_made_for_another_base_is_not_loaded(self):
        result = self.personalize()
        folder = self.root / "models" / "personal" / "text_relation" / "base-r1" / result.personalization_revision
        meta = folder / "meta.json"
        import json
        body = json.loads(meta.read_text(encoding="utf-8"))
        body["base_weights_sha256"] = "0" * 64
        meta.write_text(json.dumps(body), encoding="utf-8")
        reopened = self.open()
        self.assertEqual(reopened.predict_local(request("cust-04", self.late))["model_version"], "base-r1")
        self.assertIn("text_relation.personal", reopened.load_errors)

    def test_no_improvement_is_rejected_and_too_little_data_is_skipped(self):
        rejected = self.personalize(variant="text_only", lr=0.0)
        self.assertEqual((rejected.status, rejected.reason), ("rejected", "validation_rejected"))
        skipped = self.personalize(variant="text_relation", min_train_examples=10 ** 6)
        self.assertEqual((skipped.status, skipped.reason), ("skipped", "insufficient_data"))
        self.assertIsNone(skipped.personalization_revision)
        self.assertEqual(self.runtime.predict_local(request("cust-04", self.late))["model_version"], "base-r1")
        with self.assertRaises(ValueError):
            self.personalize(unknown_key=1)

    def test_the_config_seed_never_picks_the_validation_customers(self):
        from commerce.packages.recommender.seller_runtime import PERSONAL_SPLIT_SEED
        seen = []
        original = self.runtime.training_parts

        def spy(epoch, variant, run_seed):
            seen.append(run_seed)
            return original(epoch, variant, run_seed)
        self.runtime.training_parts = spy
        self.personalize(seed=11)
        self.personalize(variant="text_only", seed=12)
        self.assertEqual(seen, [PERSONAL_SPLIT_SEED, PERSONAL_SPLIT_SEED])

    def test_compare_pins_one_snapshot_and_reports_each_arm(self):
        self.personalize(variant="text_relation")
        self.personalize(variant="text_only", lr=0.0)  # rejected: T-P says why
        result = self.runtime.compare_local(request("cust-05", self.late, top_n=5))
        self.assertEqual([a.arm_id for a in result.arms], ["T-G", "R-G", "T-P", "R-P"])
        tg, rg, tp, rp = result.arms
        self.assertTrue(tg.available and rg.available and rp.available)
        self.assertEqual((tp.available, tp.unavailable_reason, tp.recommendation), (False, "validation_rejected", None))
        self.assertEqual(rp.base_model_version, rg.base_model_version)
        self.assertEqual(rp.recommendation["model_version"], "personal.%s" % rp.personalization_revision)
        self.assertEqual(rg.recommendation, self.runtime.predict_local(request("cust-05", self.late, top_n=5),
                                                                       mode="global"))
        self.assertEqual(result.feature_snapshot_id, "fe-%d" % self.runtime.store.feature_epoch)
        expected = hashlib.sha256(json_bytes(sorted("item-%02d" % i for i in range(1, N_ITEMS + 1)))).hexdigest()
        self.assertEqual(result.candidate_set_hash, expected)

    def test_a_pinned_pair_is_compared_while_the_serving_base_moves_on(self):
        result = self.personalize()  # on base-r1
        self.runtime.pin_comparison(text_only="base-t1", text_relation="base-r1")
        req = request("cust-05", self.late, top_n=5)
        before = self.runtime.compare_local(req)
        self.install(variant="text_relation", version="base-r2", seed=4)
        self.assertEqual(self.runtime.predict_local(req, mode="global")["model_version"], "base-r2")
        tg, rg, tp, rp = self.runtime.compare_local(req).arms
        self.assertEqual((tg.base_model_version, rg.base_model_version), ("base-t1", "base-r1"))
        self.assertEqual(rg.recommendation["model_version"], "base-r1")
        self.assertEqual(rg.recommendation, before.arms[1].recommendation)
        self.assertEqual((rp.available, rp.personalization_revision), (True, result.personalization_revision))
        self.assertEqual(rp.recommendation, before.arms[3].recommendation)
        # A restart keeps the pin; removing it makes compare follow the serving bases again.
        self.assertEqual(self.open().compare_local(req).arms[1].base_model_version, "base-r1")
        self.runtime.pin_comparison()
        unpinned = self.runtime.compare_local(req).arms
        self.assertEqual(unpinned[1].base_model_version, "base-r2")
        self.assertEqual(unpinned[3].unavailable_reason, "personalization_not_ready")

    def test_only_installed_readable_bases_are_compared(self):
        with self.assertRaises(ContractError) as caught:
            self.runtime.pin_comparison(text_relation="base-r9")
        self.assertEqual(caught.exception.code, "NOT_FOUND")
        self.assertEqual(self.runtime.comparison_pin(), {})
        self.runtime.pin_comparison(text_relation="base-r1")
        self.install(variant="text_relation", version="base-r2", seed=4)
        (self.root / "models" / "base" / "text_relation" / "base-r1" / "weights.npz").write_bytes(b"not weights")
        reopened = self.open()  # nothing of base-r1 left in memory
        rg = reopened.compare_local(request("cust-05", self.late)).arms[1]
        self.assertEqual((rg.available, rg.unavailable_reason), (False, "model_not_ready"))
        self.assertIn("text_relation.comparison", reopened.load_errors)

    def test_compare_without_models_says_not_ready(self):
        fresh = SellerRuntime(SELLER, self.root / "other.sqlite", self.root / "other-models", text=self.text,
                              architectures=TINY)
        result = fresh.compare_local(request("cust-05", self.late))
        self.assertTrue(all(not a.available and a.unavailable_reason == "model_not_ready" for a in result.arms))


def json_bytes(value) -> bytes:
    from commerce.packages.contracts.ids import canonical_json
    return canonical_json(value)


if __name__ == "__main__":
    unittest.main()
