"""Stand-in for B's runtime in synthetic rounds: generated tensors only, no real data.

The tensor names and shapes come from the dummy_tensors_for_round_bringup.json
fixture. Its manifest_hash only illustrates the form, so it is recomputed here.
The dummy trainer is handed to the round driver directly; A's build_context is
not involved (docs/development.md, replacing the other module).
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from commerce.packages.contracts import ids
from commerce.packages.contracts.types import Payload, TensorMap, TrainingResult
from commerce.services.fl_coordinator.npz_payload import encode_npz

FIXTURE = (Path(__file__).resolve().parents[2] / "packages" / "contracts" / "fixtures"
           / "shared_model_manifest.v1" / "valid" / "dummy_tensors_for_round_bringup.json")


def dummy_manifest() -> Payload:
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest["manifest_hash"] = ids.manifest_hash(manifest)
    return manifest


def base_tensors(manifest: Payload) -> TensorMap:
    return {spec["name"]: np.zeros(spec["shape"], dtype=np.float32) for spec in manifest["tensors"]}


def round_config(manifest: Payload, model_version: str, *, round_id: str = "round-0001",
                 min_clients: int = 3, deadline: datetime | None = None) -> Payload:
    deadline = deadline or datetime.now(timezone.utc) + timedelta(minutes=5)
    return {
        "schema_version": "round_config.v1", "round_id": round_id, "model_version": model_version,
        "manifest_hash": manifest["manifest_hash"], "architecture_version": manifest["architecture_version"],
        "min_clients": min_clients, "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_steps": 1, "max_local_epochs": 6, "n_neg": 200, "batch_size": 8,
        "learning_rate": 0.001, "seed": 0,
    }


class DummyTrainer:
    """Fake train_round: a deterministic delta per seller, so the expected mean is known."""

    def __init__(self, seller_id: str, offset: float, *, completed: bool = True):
        self.seller_id, self.offset, self.completed = seller_id, offset, completed

    def train_round(self, base: TensorMap, round_config: Payload) -> TrainingResult:
        delta = {name: np.full(array.shape, self.offset, dtype=np.float32) for name, array in base.items()}
        return TrainingResult(shared_delta=delta, metrics={"loss_mean": 1.0, "grad_norm_mean": 0.5},
                              completed=self.completed)


def build_submission(seller_id: str, manifest: Payload, config: Payload, result: TrainingResult) -> tuple[Payload, bytes]:
    """Wrap a TrainingResult as (round_submission.v1, npz bytes), as the FL client will."""
    payload = encode_npz(result.shared_delta)
    submission = {
        "schema_version": "round_submission.v1", "transport_mode": "synthetic_plaintext",
        "delta_manifest": {
            "schema_version": "delta_manifest.v1", "seller_id": seller_id, "round_id": config["round_id"],
            "model_version": config["model_version"], "manifest_hash": config["manifest_hash"],
            "architecture_version": config["architecture_version"], "completed": result.completed,
            "tensors": [dict(spec) for spec in manifest["tensors"]],
        },
        "aggregate_metrics": {"loss_mean": result.metrics["loss_mean"], "grad_norm_mean": result.metrics["grad_norm_mean"]},
        "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_nbytes": len(payload),
    }
    return submission, payload
