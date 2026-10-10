"""c1: coordinator state over the round core (cohort, ledger, restart). Generated tensors only."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented
from commerce.services.fl_coordinator.round_core import ModelRegistry, check_contract
from commerce.services.fl_coordinator.service import Coordinator, RoundLedger, RoundSettings
from commerce.tests.e2e import dummy_round as dummy

SELLERS = ("seller-a", "seller-b", "seller-c")
OFFSETS = {"seller-a": 0.25, "seller-b": 0.5, "seller-c": 1.0}


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class _CoordinatorCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = dummy.dummy_manifest()
        ModelRegistry(self.root / "registry").register("model-0", self.manifest, dummy.base_tensors(self.manifest))
        self.clock = _Clock()

    def coordinator(self):
        """A fresh process: registry and ledger are reloaded from disk, round state is not."""
        return Coordinator(ModelRegistry(self.root / "registry"), RoundLedger(self.root / "rounds"),
                           mode="synthetic_plaintext", now=self.clock)

    def submit(self, coordinator, seller, config, *, completed=True):
        result = dummy.DummyTrainer(seller, OFFSETS.get(seller, 0.0), completed=completed).train_round(
            dummy.base_tensors(self.manifest), config)
        submission, payload = dummy.build_submission(seller, self.manifest, config, result)
        coordinator.declare(seller, config["round_id"], submission)
        return coordinator.upload(seller, config["round_id"], payload)

    def code(self, call, *args, **kwargs):
        with self.assertRaises(ContractError) as caught:
            call(*args, **kwargs)
        return caught.exception.code


class OpeningRounds(_CoordinatorCase):
    def test_round_config_is_valid_and_fixes_the_cohort(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS, RoundSettings(deadline_seconds=120))
        check_contract("round_config.v1", config)
        self.assertEqual((config["model_version"], config["min_clients"], config["deadline"]),
                         ("model-0", 3, "2026-10-02T09:02:00Z"))
        for seller in SELLERS:
            self.assertEqual(coordinator.current_config(seller), config)
        self.assertIsNone(coordinator.current_config("seller-z"))  # not selected: no active round

    def test_one_open_round_at_a_time_and_no_round_without_a_model(self):
        coordinator = self.coordinator()
        coordinator.open_round(SELLERS)
        self.assertEqual(self.code(coordinator.open_round, SELLERS), "ILLEGAL_STATE_TRANSITION")
        empty = Coordinator(ModelRegistry(self.root / "empty"), RoundLedger(self.root / "rounds2"),
                            mode="synthetic_plaintext")
        self.assertEqual(self.code(empty.open_round, SELLERS), "NOT_FOUND")

    def test_a_cohort_below_three_is_refused(self):
        self.assertEqual(self.code(self.coordinator().open_round, SELLERS[:2]), "SCHEMA_INVALID")

    def test_protected_mode_fails_closed(self):
        with self.assertRaises(FeatureNotImplemented):
            Coordinator(ModelRegistry(self.root / "registry"), RoundLedger(self.root / "rounds"), mode="protected")


class RoundOutcome(_CoordinatorCase):
    def test_complete_cohort_publishes_and_the_ledger_records_it(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS)
        for seller in SELLERS:
            ack = self.submit(coordinator, seller, config)
        self.assertEqual(ack["disposition"], "aggregated_on_time")
        latest = coordinator.registry.latest()["model_version"]
        self.assertNotEqual(latest, "model-0")
        entry = coordinator.ledger.get(config["round_id"])
        self.assertEqual((entry["state"], entry["result_model_version"]), ("aggregated", latest))
        self.assertIsNone(coordinator.current_config("seller-a"))
        # the next round starts from the new release
        self.assertEqual(coordinator.open_round(SELLERS)["model_version"], latest)

    def test_deadline_discards_and_the_ledger_records_it(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS, RoundSettings(deadline_seconds=60))
        self.submit(coordinator, "seller-a", config)
        self.clock.now += timedelta(seconds=61)
        self.assertIsNone(coordinator.current_config("seller-b"))
        self.assertEqual(coordinator.ledger.get(config["round_id"])["state"], "discarded")
        self.assertEqual(coordinator.result("seller-a", config["round_id"])["disposition"], "round_discarded")

    def test_retry_after_aggregation_returns_the_final_ack(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS)
        for seller in SELLERS:
            self.submit(coordinator, seller, config)
        self.assertEqual(self.submit(coordinator, "seller-a", config)["disposition"], "aggregated_on_time")

    def test_upload_needs_a_declared_envelope_from_a_cohort_member(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS)
        self.assertEqual(self.code(coordinator.upload, "seller-a", config["round_id"], b""), "ILLEGAL_STATE_TRANSITION")
        result = dummy.DummyTrainer("seller-z", 0.0).train_round(dummy.base_tensors(self.manifest), config)
        submission, _ = dummy.build_submission("seller-z", self.manifest, config, result)
        self.assertEqual(self.code(coordinator.declare, "seller-z", config["round_id"], submission), "FORBIDDEN")

    def test_envelope_must_match_caller_round_and_mode(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS)
        result = dummy.DummyTrainer("seller-b", 0.0).train_round(dummy.base_tensors(self.manifest), config)
        submission, _ = dummy.build_submission("seller-b", self.manifest, config, result)
        self.assertEqual(self.code(coordinator.declare, "seller-a", config["round_id"], submission), "FORBIDDEN")
        self.assertEqual(self.code(coordinator.declare, "seller-b", "round-other", submission), "NOT_FOUND")
        protected = dict(submission, transport_mode="protected")
        self.assertIn(self.code(coordinator.declare, "seller-b", config["round_id"], protected),
                      ("FORBIDDEN", "SCHEMA_INVALID", "INVALID_ENUM_VALUE"))


class Restart(_CoordinatorCase):
    def test_an_open_round_is_discarded_by_a_restart_and_its_id_is_never_reused(self):
        before = self.coordinator()
        config = before.open_round(SELLERS, round_id="round-restart")
        self.submit(before, "seller-a", config)
        after = self.coordinator()  # restart
        self.assertEqual(after.ledger.get("round-restart")["state"], "discarded")
        self.assertEqual(self.code(after.result, "seller-a", "round-restart"), "ROUND_DISCARDED")
        self.assertEqual(self.code(after.open_round, SELLERS, round_id="round-restart"), "ILLEGAL_STATE_TRANSITION")
        self.assertEqual(after.open_round(SELLERS)["model_version"], "model-0")  # last model kept

    def test_ledger_holds_round_metadata_only(self):
        coordinator = self.coordinator()
        config = coordinator.open_round(SELLERS)
        for seller in SELLERS:
            self.submit(coordinator, seller, config)
        files = list((self.root / "rounds").iterdir())
        self.assertEqual([path.suffix for path in files], [".json"])
        entry = json.loads(files[0].read_bytes())
        self.assertLessEqual(set(entry), {"round_id", "state", "model_version", "deadline", "cohort_size", "seed",
                                          "result_model_version", "reason"})
        self.assertNotIn("seller-a", files[0].read_text(encoding="utf-8"))

    def test_unknown_round_is_not_found(self):
        self.assertEqual(self.code(self.coordinator().result, "seller-a", "round-none"), "NOT_FOUND")


class FixedSeed(_CoordinatorCase):
    """B picks each seller's validation customers from round_config.seed (model.md §6), so one
    run keeps one seed; a new seed would move customers between validation and training."""

    def test_later_rounds_must_keep_the_first_seed(self):
        coordinator = self.coordinator()
        first = coordinator.open_round(SELLERS, RoundSettings(seed=7, deadline_seconds=60))
        self.clock.now += timedelta(seconds=61)  # discarded at the deadline
        self.assertEqual(self.code(coordinator.open_round, SELLERS, RoundSettings(seed=8)), "ILLEGAL_STATE_TRANSITION")
        self.assertEqual(coordinator.open_round(SELLERS, RoundSettings(seed=7))["seed"], first["seed"])

    def test_the_seed_survives_a_restart(self):
        self.coordinator().open_round(SELLERS, RoundSettings(seed=7))
        after = self.coordinator()  # restart: the open round is discarded, the seed stays
        self.assertEqual(self.code(after.open_round, SELLERS, RoundSettings(seed=0)), "ILLEGAL_STATE_TRANSITION")
        self.assertEqual(after.open_round(SELLERS, RoundSettings(seed=7))["seed"], 7)

    def test_a_new_round_state_dir_is_a_new_run(self):
        self.coordinator().open_round(SELLERS, RoundSettings(seed=7))
        other = Coordinator(ModelRegistry(self.root / "registry"), RoundLedger(self.root / "rounds-next"),
                            mode="synthetic_plaintext", now=self.clock)
        self.assertEqual(other.open_round(SELLERS, RoundSettings(seed=8))["seed"], 8)


if __name__ == "__main__":
    unittest.main()
