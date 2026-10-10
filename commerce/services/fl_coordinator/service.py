"""Coordinator state over the round core: one variant, one registry, at most one open round.

HTTP-free, so tests and a future trigger drive it directly. Opening a round is an
operator action; who triggers rounds and when is still open (OQ08). The cohort is
fixed when the round opens and never changes during it (D0017).

Restart rule (interfaces.md §6): a round that was open when the coordinator stopped
is recorded as discarded on the next start, and its round_id is never reused. The
ledger in ROUND_STATE_DIR holds round metadata only: no delta, no metric, no secret.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented
from commerce.packages.contracts.types import Payload
from commerce.services.fl_coordinator.round_core import (
    SAFE_ID, ModelRegistry, SyntheticRound, atomic_write, check_contract,
)

MODES = ("protected", "synthetic_plaintext")


@dataclass(frozen=True)
class RoundSettings:
    """Training fields of round_config.v1 plus the deadline. Defaults are model.md §6 starting values."""
    local_steps: int = 40
    max_local_epochs: int = 6
    n_neg: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 0
    deadline_seconds: int = 600  # cover the whole cohort when one PC trains sellers in turn


class RoundLedger:
    """`<root>/<round_id>.json` per round: state, base model, deadline, cohort size, seed, result model.

    One ledger directory is one run, and `seed` is the round seed that run uses."""

    STATES = ("open", "aggregated", "discarded")

    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self.seed: int | None = None
        for path in self._root.glob("*.json"):
            entry = json.loads(path.read_bytes())
            if entry["state"] == "open":  # the coordinator stopped while this round was running
                self._write(dict(entry, state="discarded", reason="coordinator_restart"))
            if entry.get("seed") is not None:
                self.seed = entry["seed"]

    def get(self, round_id: str) -> Payload | None:
        path = self._root / (round_id + ".json")
        return json.loads(path.read_bytes()) if SAFE_ID.fullmatch(round_id) and path.exists() else None

    def record(self, round_id: str, **fields) -> None:
        entry = dict(self.get(round_id) or {"round_id": round_id}, **fields)
        if entry["state"] not in self.STATES:
            raise ValueError("unknown round state")
        self._write(entry)
        if entry.get("seed") is not None:
            self.seed = entry["seed"]

    def _write(self, entry: Payload) -> None:
        atomic_write(self._root / (entry["round_id"] + ".json"), ids.canonical_json(entry))


class Coordinator:
    def __init__(self, registry: ModelRegistry, ledger: RoundLedger, *, mode: str = "protected",
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        if mode not in MODES:
            raise ValueError("Unknown FL mode")
        if mode == "protected":
            # g4 replaces this. Never fall back to plaintext (architecture.md §3).
            raise FeatureNotImplemented("C: protected aggregation is not implemented")
        self.registry, self.ledger, self.mode, self._now = registry, ledger, mode, now
        self._round: SyntheticRound | None = None
        self._declared: dict[str, Payload] = {}

    # --- operator side ---------------------------------------------------------

    def open_round(self, cohort: Sequence[str], settings: RoundSettings = RoundSettings(), *,
                   round_id: str | None = None) -> Payload:
        self._sync()
        if self._round is not None and self._round.state == "open":
            raise ContractError("ILLEGAL_STATE_TRANSITION")
        latest = self.registry.latest()
        if latest is None:
            raise ContractError("NOT_FOUND", "/model_version")
        round_id = round_id or "round-%s-%s" % (self._now().strftime("%Y%m%d%H%M%S"), secrets.token_hex(3))
        if not SAFE_ID.fullmatch(round_id):
            raise ContractError("SCHEMA_INVALID", "/round_id")
        if self.ledger.get(round_id) is not None:
            raise ContractError("ILLEGAL_STATE_TRANSITION", "/round_id")  # never reuse a round_id
        if self.ledger.seed is not None and settings.seed != self.ledger.seed:
            # B picks validation customers from the seed (model.md §6): a new seed needs a new run.
            raise ContractError("ILLEGAL_STATE_TRANSITION", "/seed")
        manifest = self.registry.manifest(latest["model_version"])
        deadline = self._now() + timedelta(seconds=settings.deadline_seconds)
        config = {
            "schema_version": "round_config.v1", "round_id": round_id,
            "model_version": latest["model_version"], "manifest_hash": manifest["manifest_hash"],
            "architecture_version": manifest["architecture_version"], "min_clients": len(cohort),
            "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local_steps": settings.local_steps, "max_local_epochs": settings.max_local_epochs,
            "n_neg": settings.n_neg, "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate, "seed": settings.seed,
        }
        self._round = SyntheticRound(self.registry, config, cohort, now=self._now)
        self._declared = {}
        self.ledger.record(round_id, state="open", model_version=config["model_version"],
                           deadline=config["deadline"], cohort_size=len(cohort), seed=settings.seed)
        return dict(config)

    # --- seller side -----------------------------------------------------------

    def current_config(self, seller_id: str) -> Payload | None:
        """The open round's config for a cohort member, else None (no active round for this seller)."""
        self._sync()
        if self._round is None or self._round.state != "open" or seller_id not in self._round.cohort:
            return None
        return dict(self._round.config)

    def declare(self, seller_id: str, round_id: str, submission: Payload) -> None:
        """First step of a submission: the round_submission envelope, checked before the npz arrives."""
        rnd = self._find(round_id)
        if seller_id not in rnd.cohort:
            raise ContractError("FORBIDDEN")
        check_contract("round_submission.v1", submission)
        if submission["delta_manifest"]["seller_id"] != seller_id:
            raise ContractError("FORBIDDEN", "/delta_manifest/seller_id")
        if submission["delta_manifest"]["round_id"] != round_id:
            raise ContractError("NOT_FOUND", "/delta_manifest/round_id")
        if submission["transport_mode"] != self.mode:
            raise ContractError("FORBIDDEN", "/transport_mode")
        self._declared[seller_id] = dict(submission)

    def upload(self, seller_id: str, round_id: str, payload: bytes) -> Payload:
        """Second step: the npz bytes for the declared envelope. Returns round_submit_ack.v1."""
        rnd = self._find(round_id)
        submission = self._declared.get(seller_id)
        if submission is None:
            raise ContractError("ILLEGAL_STATE_TRANSITION")
        try:
            return rnd.submit(seller_id, submission, payload)
        finally:
            self._sync()

    def result(self, seller_id: str, round_id: str) -> Payload:
        rnd = self._find(round_id)
        try:
            return rnd.result(seller_id)
        finally:
            self._sync()

    # --- internals -------------------------------------------------------------

    def _find(self, round_id: str) -> SyntheticRound:
        self._sync()
        if self._round is not None and self._round.round_id == round_id:
            return self._round
        entry = self.ledger.get(round_id)
        if entry is not None and entry["state"] == "discarded":
            raise ContractError("ROUND_DISCARDED")
        raise ContractError("NOT_FOUND")

    def _sync(self) -> None:
        """Mirror a closed round into the ledger and drop the declared envelopes."""
        rnd = self._round
        if rnd is None:
            return
        rnd.expire()
        if rnd.state == "open":
            return
        entry = self.ledger.get(rnd.round_id)
        if entry is not None and entry["state"] == "open":
            fields = {"state": rnd.state}
            if rnd.release is not None:
                fields["result_model_version"] = rnd.release["model_version"]
            self.ledger.record(rnd.round_id, **fields)
            self._declared = {}
