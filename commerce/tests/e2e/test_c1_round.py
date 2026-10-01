"""c1: synthetic round core with generated tensors. Not a g3 check: no real B runtime, no HTTP."""
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import ContractError
from commerce.services.fl_coordinator import import_release
from commerce.services.fl_coordinator.npz_payload import PayloadTooLarge, decode_npz, encode_npz
from commerce.services.fl_coordinator.round_core import (
    ModelRegistry, RegistryCorrupt, SyntheticRound, aggregate_uniform, check_contract,
)
from commerce.tests.e2e import dummy_round as dummy

SELLERS = ("seller-a", "seller-b", "seller-c")
OFFSETS = {"seller-a": 0.25, "seller-b": 0.5, "seller-c": 1.0}


class _Clock:
    def __init__(self):
        self.now = datetime.now(timezone.utc)
        self.t = 0.0

    def __call__(self):
        return self.now


class _RoundCase(unittest.TestCase):
    def setUp(self):
        self.manifest = dummy.dummy_manifest()
        self.registry = ModelRegistry()
        self.base = self.registry.register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        self.clock = _Clock()
        self.config = dummy.round_config(self.manifest, "model-0", deadline=self.clock.now + timedelta(minutes=5))

    def new_round(self, cohort=SELLERS, config=None):
        return SyntheticRound(self.registry, config or self.config, cohort, now=self.clock)

    def submission(self, seller, *, offset=None, completed=True, config=None):
        config = config or self.config
        trainer = dummy.DummyTrainer(seller, OFFSETS.get(seller, 0.0) if offset is None else offset, completed=completed)
        result = trainer.train_round(dummy.base_tensors(self.manifest), config)
        return dummy.build_submission(seller, self.manifest, config, result)

    def code(self, call, *args):
        with self.assertRaises(ContractError) as caught:
            call(*args)
        return caught.exception.code


class UniformMeanRelease(_RoundCase):
    def test_all_complete_gives_plain_mean_and_a_new_immutable_release(self):
        rnd = self.new_round()
        for seller in SELLERS[:-1]:
            ack = rnd.submit(seller, *self.submission(seller))
            self.assertEqual(ack["disposition"], "accepted_on_time")
            self.assertIsNone(ack["aggregated_in_round_id"])
        ack = rnd.submit(SELLERS[-1], *self.submission(SELLERS[-1]))
        check_contract("round_submit_ack.v1", ack)
        self.assertEqual((ack["disposition"], ack["staleness_rounds"], ack["aggregated_in_round_id"]),
                         ("aggregated_on_time", 0, self.config["round_id"]))
        release = rnd.release
        check_contract("model_release.v1", release)
        self.assertNotEqual(release["model_version"], "model-0")
        self.assertEqual(self.registry.latest(), release)
        expected = np.float32(sum(OFFSETS.values()) / 3)
        for array in self.registry.tensors(release["model_version"]).values():
            np.testing.assert_allclose(array, expected, rtol=1e-6)
        # the base release is untouched
        for array in self.registry.tensors("model-0").values():
            self.assertFalse(array.any())
        for seller in SELLERS:  # every member sees the same final result
            self.assertEqual(rnd.result(seller)["disposition"], "aggregated_on_time")

    def test_new_seller_installs_latest_after_verifying_hashes(self):
        rnd = self.new_round()
        for seller in SELLERS:
            rnd.submit(seller, *self.submission(seller))
        descriptor = self.registry.latest()
        weights = self.registry.weights(descriptor["model_version"])
        self.assertEqual(hashlib.sha256(weights).hexdigest(), descriptor["weights_sha256"])
        self.assertEqual(len(weights), descriptor["weights_size_bytes"])
        manifest = self.registry.manifest(descriptor["model_version"])
        self.assertEqual(manifest["manifest_hash"], descriptor["manifest_hash"])
        self.assertEqual(manifest["manifest_hash"], ids.manifest_hash(manifest))
        decode_npz(weights, manifest["tensors"])

    def test_deltas_are_dropped_after_the_round_closes(self):
        rnd = self.new_round()
        for seller in SELLERS:
            rnd.submit(seller, *self.submission(seller))
        self.assertTrue(all(slot.delta is None for slot in rnd._slots.values()))


class DiscardWholeRound(_RoundCase):
    def test_incomplete_member_discards_everyone_and_keeps_the_last_model(self):
        rnd = self.new_round()
        rnd.submit("seller-a", *self.submission("seller-a"))
        ack = rnd.submit("seller-b", *self.submission("seller-b", completed=False))
        self.assertEqual(ack["disposition"], "dropped_incomplete")
        self.assertEqual(rnd.result("seller-a")["disposition"], "round_discarded")
        self.assertEqual(self.code(rnd.submit, "seller-c", *self.submission("seller-c")), "ROUND_DISCARDED")
        self.assertEqual(self.registry.latest()["model_version"], "model-0")
        self.assertIsNone(rnd.release)

    def test_missing_member_at_the_deadline_discards_with_no_carry_over(self):
        rnd = self.new_round()
        for seller in SELLERS[:2]:
            rnd.submit(seller, *self.submission(seller))
        self.clock.now += timedelta(minutes=6)
        self.assertEqual(self.code(rnd.submit, "seller-c", *self.submission("seller-c")), "ROUND_DISCARDED")
        self.assertEqual(rnd.result("seller-a")["disposition"], "round_discarded")
        self.assertEqual(self.registry.latest()["model_version"], "model-0")

    def test_no_result_for_a_seller_that_never_submitted(self):
        rnd = self.new_round()
        self.assertEqual(self.code(rnd.result, "seller-a"), "NOT_FOUND")

    def test_new_process_means_a_new_empty_round(self):
        rnd = self.new_round()
        rnd.submit("seller-a", *self.submission("seller-a"))
        restarted = self.new_round()  # a restart keeps no round state
        self.assertEqual(self.code(restarted.result, "seller-a"), "NOT_FOUND")


class RetriesAndSlots(_RoundCase):
    def test_same_bytes_again_returns_the_same_result(self):
        rnd = self.new_round()
        first = rnd.submit("seller-a", *self.submission("seller-a"))
        self.assertEqual(rnd.submit("seller-a", *self.submission("seller-a")), first)

    def test_different_bytes_in_a_filled_slot_are_refused(self):
        rnd = self.new_round()
        rnd.submit("seller-a", *self.submission("seller-a"))
        self.assertEqual(self.code(rnd.submit, "seller-a", *self.submission("seller-a", offset=9.0)),
                         "DUPLICATE_ROUND_SUBMIT")

    def test_retry_after_aggregation_returns_the_final_ack(self):
        rnd = self.new_round()
        for seller in SELLERS:
            rnd.submit(seller, *self.submission(seller))
        self.assertEqual(rnd.submit("seller-a", *self.submission("seller-a"))["disposition"], "aggregated_on_time")

    def test_seller_outside_the_cohort_is_refused(self):
        rnd = self.new_round()
        self.assertEqual(self.code(rnd.submit, "seller-z", *self.submission("seller-z")), "FORBIDDEN")

    def test_delta_manifest_seller_must_match_the_caller(self):
        rnd = self.new_round()
        submission, payload = self.submission("seller-b")
        self.assertEqual(self.code(rnd.submit, "seller-a", submission, payload), "FORBIDDEN")


class ManifestAndTensorChecks(_RoundCase):
    def test_wrong_model_manifest_or_architecture_is_refused(self):
        rnd = self.new_round()
        for key, bad in (("model_version", "model-9"), ("manifest_hash", "0" * 64), ("architecture_version", 2)):
            submission, payload = self.submission("seller-a")
            submission["delta_manifest"][key] = bad
            self.assertEqual(self.code(rnd.submit, "seller-a", submission, payload), "MANIFEST_MISMATCH", key)

    def test_wrong_round_id_is_not_found(self):
        rnd = self.new_round()
        submission, payload = self.submission("seller-a")
        submission["delta_manifest"]["round_id"] = "round-other"
        self.assertEqual(self.code(rnd.submit, "seller-a", submission, payload), "NOT_FOUND")

    def test_tensor_list_must_equal_the_manifest(self):
        rnd = self.new_round()
        submission, payload = self.submission("seller-a")
        submission["delta_manifest"]["tensors"][0]["shape"] = [3, 9]
        self.assertEqual(self.code(rnd.submit, "seller-a", submission, payload), "TENSOR_SET_MISMATCH")
        submission, payload = self.submission("seller-a")
        submission["delta_manifest"]["tensors"].pop()
        self.assertEqual(self.code(rnd.submit, "seller-a", submission, payload), "TENSOR_SET_MISMATCH")

    def test_payload_length_and_hash_must_match_the_declaration(self):
        rnd = self.new_round()
        submission, payload = self.submission("seller-a")
        self.assertEqual(self.code(rnd.submit, "seller-a", dict(submission, payload_nbytes=len(payload) + 1), payload),
                         "SCHEMA_INVALID")
        self.assertEqual(self.code(rnd.submit, "seller-a", dict(submission, payload_sha256="0" * 64), payload),
                         "SCHEMA_INVALID")

    def test_variant_mixing_is_refused_even_with_equal_shapes(self):
        other = copy.deepcopy(self.manifest)
        other["architecture_version"] = 2  # a different variant has a different architecture config
        other["manifest_hash"] = ids.manifest_hash(other)
        self.assertEqual(self.code(self.registry.register, "model-x", other, dummy.base_tensors(other)),
                         "MANIFEST_MISMATCH")
        config = dict(self.config, manifest_hash=other["manifest_hash"], architecture_version=2)
        self.assertEqual(self.code(SyntheticRound, self.registry, config, SELLERS, ), "MANIFEST_MISMATCH")

    def test_cohort_below_three_or_with_duplicates_is_refused(self):
        self.assertEqual(self.code(SyntheticRound, self.registry, dict(self.config, min_clients=1), SELLERS[:2]),
                         "SCHEMA_INVALID")
        self.assertEqual(self.code(SyntheticRound, self.registry, self.config, ("seller-a",) * 3), "SCHEMA_INVALID")

    def test_round_must_start_from_the_latest_release(self):
        rnd = self.new_round()
        for seller in SELLERS:
            rnd.submit(seller, *self.submission(seller))
        self.assertEqual(self.code(SyntheticRound, self.registry, self.config, SELLERS), "MANIFEST_MISMATCH")

    def test_registry_never_rewrites_a_version(self):
        self.assertEqual(self.code(self.registry.register, "model-0", self.manifest, dummy.base_tensors(self.manifest)),
                         "ILLEGAL_STATE_TRANSITION")
        self.assertEqual(self.code(self.registry.release, "missing"), "NOT_FOUND")


class RegistryOnDisk(unittest.TestCase):
    """REGISTRY_DIR: atomic publish, verified reload, variant pin, no partial release."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "text_only" / "registry"
        self.manifest = dummy.dummy_manifest()

    def aggregated_registry(self):
        registry = ModelRegistry(self.root)
        registry.register("model-0", self.manifest, dummy.base_tensors(self.manifest),
                          provenance={"source": "import", "kind": "random_init"})
        config = dummy.round_config(self.manifest, "model-0")
        rnd = SyntheticRound(registry, config, SELLERS)
        for seller in SELLERS:
            result = dummy.DummyTrainer(seller, OFFSETS[seller]).train_round(dummy.base_tensors(self.manifest), config)
            rnd.submit(seller, *dummy.build_submission(seller, self.manifest, config, result))
        return registry, rnd

    def test_reload_restores_releases_latest_and_provenance(self):
        registry, rnd = self.aggregated_registry()
        reloaded = ModelRegistry(self.root)
        self.assertEqual(reloaded.latest(), registry.latest())
        version = rnd.release["model_version"]
        self.assertEqual(reloaded.weights(version), registry.weights(version))
        self.assertEqual(reloaded.provenance("model-0")["kind"], "random_init")
        record = reloaded.provenance(version)
        self.assertEqual((record["source"], record["round_id"], record["base_model_version"], record["cohort_size"]),
                         ("fl_round", "round-0001", "model-0", 3))

    def test_only_releases_reach_the_disk(self):
        self.aggregated_registry()
        names = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        allowed = {"latest"} | {"%s/%s" % (v, f) for v in {n.split("/")[0] for n in names if "/" in n}
                                for f in ("release.json", "manifest.json", "weights.npz", "provenance.json")}
        self.assertTrue(set(names) <= allowed, names)

    def test_a_version_is_never_rewritten_after_a_restart(self):
        self.aggregated_registry()
        reloaded = ModelRegistry(self.root)
        with self.assertRaises(ContractError) as caught:
            reloaded.register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        self.assertEqual(caught.exception.code, "ILLEGAL_STATE_TRANSITION")

    def test_variant_pin_survives_a_restart(self):
        self.aggregated_registry()
        other = copy.deepcopy(self.manifest)
        other["architecture_version"] = 2
        other["manifest_hash"] = ids.manifest_hash(other)
        with self.assertRaises(ContractError) as caught:
            ModelRegistry(self.root).register("model-x", other, dummy.base_tensors(other))
        self.assertEqual(caught.exception.code, "MANIFEST_MISMATCH")

    def test_tampered_weights_or_a_dangling_latest_refuse_to_load(self):
        registry, _ = self.aggregated_registry()
        weights = self.root / "model-0" / "weights.npz"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 0xFF
        weights.write_bytes(bytes(data))
        with self.assertRaises(RegistryCorrupt):
            ModelRegistry(self.root)
        weights.write_bytes(registry.weights("model-0"))
        ModelRegistry(self.root)  # restored
        (self.root / "latest").write_text("model-missing", encoding="ascii")
        with self.assertRaises(RegistryCorrupt):
            ModelRegistry(self.root)

    def test_an_interrupted_publish_is_invisible(self):
        registry = ModelRegistry(self.root)
        registry.register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        staging = self.root / ".staging-crash"
        staging.mkdir()
        (staging / "weights.npz").write_bytes(b"partial")
        reloaded = ModelRegistry(self.root)
        self.assertEqual(reloaded.latest()["model_version"], "model-0")
        self.assertFalse(staging.exists())

    def test_unsafe_version_names_are_refused(self):
        registry = ModelRegistry(self.root)
        for bad in ("../escape", "a/b", ".hidden", ""):
            with self.assertRaises(ContractError):
                registry.register(bad, self.manifest, dummy.base_tensors(self.manifest))


class ImportRelease(unittest.TestCase):
    """Local import of a lab release: no upload API, provenance recorded."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.registry = self.dir / "registry"
        self.manifest = dummy.dummy_manifest()
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.dir / "weights.npz").write_bytes(encode_npz(dummy.base_tensors(self.manifest)))

    def run_import(self, *extra, version="model-init-1", kind="trained"):
        argv = ["--registry", str(self.registry), "--manifest", str(self.dir / "manifest.json"),
                "--weights", str(self.dir / "weights.npz"), "--model-version", version, "--kind", kind, *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = import_release.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_import_publishes_latest_and_records_the_kind(self):
        code, out, err = self.run_import("--note", "single-seller pretraining", kind="random_init")
        self.assertEqual(code, 0, err)
        descriptor = json.loads(out)
        check_contract("model_release.v1", descriptor)
        registry = ModelRegistry(self.registry)
        self.assertEqual(registry.latest(), descriptor)
        record = registry.provenance("model-init-1")
        self.assertEqual((record["source"], record["kind"], record["note"]),
                         ("import", "random_init", "single-seller pretraining"))
        self.assertIn("not checked", err)  # B's config registry is not verified yet

    def test_bad_manifest_hash_or_tensor_set_is_refused(self):
        self.manifest["manifest_hash"] = "0" * 64
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.assertEqual(self.run_import()[0], 1)
        manifest = dummy.dummy_manifest()
        (self.dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        first = manifest["tensors"][0]["name"]
        (self.dir / "weights.npz").write_bytes(encode_npz({first: np.zeros(manifest["tensors"][0]["shape"], np.float32)}))
        self.assertEqual(self.run_import()[0], 1)
        self.assertIsNone(ModelRegistry(self.registry).latest())

    def test_the_same_version_cannot_be_imported_twice(self):
        self.assertEqual(self.run_import()[0], 0)
        code, _, err = self.run_import()
        self.assertEqual(code, 1)
        self.assertIn("ILLEGAL_STATE_TRANSITION", err)


class LabAggregation(unittest.TestCase):
    """aggregate_uniform is the entry point for B's lab runner (unprotected simulation, D0020)."""

    def test_uniform_mean_when_everyone_completed(self):
        mean = aggregate_uniform([{"w": np.array([1.0, 3.0])}, {"w": np.array([3.0, 5.0])}])
        self.assertEqual(mean["w"].dtype, np.float32)
        np.testing.assert_array_equal(mean["w"], [2.0, 4.0])

    def test_one_missing_member_or_an_empty_cohort_discards(self):
        self.assertIsNone(aggregate_uniform([{"w": np.ones(2)}, None]))
        self.assertIsNone(aggregate_uniform([]))

    def test_no_sample_weighting(self):
        # three members: the mean of the three, whatever their data sizes were
        mean = aggregate_uniform([{"w": np.array([0.0])}, {"w": np.array([0.0])}, {"w": np.array([3.0])}])
        np.testing.assert_array_equal(mean["w"], [1.0])

    def test_tensor_set_or_shape_disagreement_raises(self):
        with self.assertRaises(ValueError):
            aggregate_uniform([{"w": np.ones(1)}, {"v": np.ones(1)}])
        with self.assertRaises(ValueError):
            aggregate_uniform([{"w": np.ones(1)}, {"w": np.ones(2)}])

    def test_round_core_uses_the_same_rule(self):
        manifest = dummy.dummy_manifest()
        registry = ModelRegistry()
        registry.register("model-0", manifest, dummy.base_tensors(manifest))  # zero base
        config = dummy.round_config(manifest, "model-0")
        rnd = SyntheticRound(registry, config, SELLERS)
        deltas = []
        for seller in SELLERS:
            result = dummy.DummyTrainer(seller, OFFSETS[seller]).train_round(dummy.base_tensors(manifest), config)
            deltas.append(result.shared_delta)
            rnd.submit(seller, *dummy.build_submission(seller, manifest, config, result))
        expected = aggregate_uniform(deltas)
        for name, array in registry.tensors(rnd.release["model_version"]).items():
            np.testing.assert_array_equal(array, expected[name])


class NpzRules(unittest.TestCase):
    def setUp(self):
        self.specs = dummy.dummy_manifest()["tensors"]
        self.good = dummy.base_tensors(dummy.dummy_manifest())

    def code(self, data):
        with self.assertRaises(ContractError) as caught:
            decode_npz(data, self.specs)
        return caught.exception.code

    def test_exact_tensor_set_round_trips(self):
        out = decode_npz(encode_npz(self.good), self.specs)
        self.assertEqual(sorted(out), sorted(self.good))

    def test_transfer_length_is_not_the_tensor_length(self):
        from commerce.services.fl_coordinator.npz_payload import tensor_nbytes
        self.assertGreater(len(encode_npz(self.good)), tensor_nbytes(self.specs))

    def test_missing_extra_or_wrong_shape_entries(self):
        first, second = (spec["name"] for spec in self.specs)
        self.assertEqual(self.code(encode_npz({first: self.good[first]})), "TENSOR_SET_MISMATCH")
        self.assertEqual(self.code(encode_npz(dict(self.good, **{"shared.extra": np.zeros(1, np.float32)}))),
                         "TENSOR_SET_MISMATCH")
        self.assertEqual(self.code(encode_npz(dict(self.good, **{second: np.zeros((2, 2), np.float32)}))),
                         "TENSOR_SET_MISMATCH")

    def test_wrong_dtype_object_arrays_and_non_finite_values(self):
        first = self.specs[0]["name"]
        self.assertEqual(self.code(encode_npz(dict(self.good, **{first: self.good[first].astype(np.float64)}))),
                         "TENSOR_SET_MISMATCH")
        pickled = io.BytesIO()
        np.savez(pickled, **dict(self.good, **{first: np.array([{"x": 1}], dtype=object)}))
        self.assertIn(self.code(pickled.getvalue()), ("TENSOR_SET_MISMATCH", "SCHEMA_INVALID"))
        for bad in (np.nan, np.inf):
            tensors = {name: array.copy() for name, array in self.good.items()}
            tensors[first][0, 0] = bad
            self.assertEqual(self.code(encode_npz(tensors)), "SCHEMA_INVALID")

    def test_garbage_compressed_and_oversize_payloads(self):
        self.assertEqual(self.code(b"not a zip"), "SCHEMA_INVALID")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **self.good)
        self.assertEqual(self.code(buffer.getvalue()), "SCHEMA_INVALID")
        with self.assertRaises(PayloadTooLarge):
            decode_npz(b"\0" * (8 * 1024 * 1024 + 1), self.specs)

    def test_entry_larger_than_its_declared_tensor_is_refused_before_reading(self):
        first = self.specs[0]["name"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for name in (spec["name"] for spec in self.specs):
                array = np.zeros((1000, 1000), np.float32) if name == first else self.good[name]
                inner = io.BytesIO()
                np.lib.format.write_array(inner, array)
                archive.writestr(name + ".npy", inner.getvalue())
        self.assertEqual(self.code(buffer.getvalue()), "TENSOR_SET_MISMATCH")


if __name__ == "__main__":
    unittest.main()
