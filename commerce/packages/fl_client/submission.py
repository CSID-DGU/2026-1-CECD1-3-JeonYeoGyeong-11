"""Seller side of a synthetic round: build a submission from B's TrainingResult, verify a release.

Pure functions, no network. The submission is made only from TrainingResult.shared_delta,
which B computes from the shared base of the round (model.md §4-5); the client never
reads the serving model or a personalized tail. Rules: interfaces.md §5.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import Payload, TensorMap, TrainingResult
from commerce.packages.contracts.validate import load_schemas, validate_payload
from commerce.services.fl_coordinator.npz_payload import MAX_PAYLOAD_BYTES, PayloadTooLarge, decode_npz, encode_npz

_SCHEMAS = None


def _check(contract: str, payload) -> None:
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = load_schemas()
    verdict = validate_payload(_SCHEMAS[contract], contract, payload)
    if not verdict.ok:
        raise ContractError(verdict.code, verdict.field_path)


def _metric(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ContractError("SCHEMA_INVALID", "/aggregate_metrics")
    return float(value)


def check_round_matches(config: Payload, manifest: Payload) -> None:
    """Refuse to train when the round's model does not match B's shared manifest (interfaces.md §5)."""
    if (config["manifest_hash"] != manifest["manifest_hash"]
            or config["architecture_version"] != manifest["architecture_version"]
            or manifest["manifest_hash"] != ids.manifest_hash(manifest)):
        raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")


def build_submission(seller_id: str, manifest: Payload, config: Payload,
                     result: TrainingResult) -> tuple[Payload, bytes]:
    """(round_submission.v1, npz bytes) for one seller. Synthetic plaintext only."""
    _check("round_config.v1", config)
    check_round_matches(config, manifest)
    expected = {spec["name"]: tuple(spec["shape"]) for spec in manifest["tensors"]}
    delta: Mapping = result.shared_delta
    if set(delta) != set(expected):
        raise ContractError("TENSOR_SET_MISMATCH", "/delta_manifest/tensors")
    arrays: TensorMap = {}
    for name, shape in expected.items():
        array = np.asarray(delta[name])
        if array.dtype != np.float32 or array.shape != shape:
            raise ContractError("TENSOR_SET_MISMATCH", "/delta_manifest/tensors")
        if not np.isfinite(array).all():
            raise ContractError("SCHEMA_INVALID", "/delta_manifest/tensors")
        arrays[name] = array
    payload = encode_npz(arrays)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise PayloadTooLarge("delta exceeds the transfer limit")
    submission = {
        "schema_version": "round_submission.v1", "transport_mode": "synthetic_plaintext",
        "delta_manifest": {
            "schema_version": "delta_manifest.v1", "seller_id": seller_id, "round_id": config["round_id"],
            "model_version": config["model_version"], "manifest_hash": config["manifest_hash"],
            "architecture_version": config["architecture_version"], "completed": bool(result.completed),
            "tensors": [dict(spec) for spec in manifest["tensors"]],
        },
        "aggregate_metrics": {"loss_mean": _metric(result.metrics.get("loss_mean")),
                              "grad_norm_mean": _metric(result.metrics.get("grad_norm_mean"))},
        "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_nbytes": len(payload),
    }
    _check("round_submission.v1", submission)
    return submission, payload


def verify_release(descriptor: Payload, manifest: Payload, weights: bytes) -> TensorMap:
    """Tensors of a downloaded release, after every check that install_release relies on."""
    _check("model_release.v1", descriptor)
    _check("shared_model_manifest.v1", manifest)
    if (manifest["manifest_hash"] != ids.manifest_hash(manifest)
            or descriptor["manifest_hash"] != manifest["manifest_hash"]):
        raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
    if len(weights) != descriptor["weights_size_bytes"] or hashlib.sha256(weights).hexdigest() != descriptor["weights_sha256"]:
        raise ContractError("SCHEMA_INVALID", "/weights_sha256")
    return decode_npz(weights, manifest["tensors"])
