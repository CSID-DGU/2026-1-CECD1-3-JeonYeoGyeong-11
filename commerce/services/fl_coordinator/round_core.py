"""Synthetic FL round core: model registry, one round, uniform mean. No HTTP.

Invariants are in docs/design/interfaces.md §5-6 (D0017): the cohort is fixed before
the round; either every member submits a completed, valid delta before the deadline
and the plain mean is added to the base model, or the whole round is discarded and
the last model stays. No sample weighting, no late carry-over. Round state is in
memory only, so a restart discards a running round and no individual delta is ever
written. Only released models reach the disk (ModelRegistry with a root).

One registry and one round object serve one model variant. A registry pins the
manifest_hash of its first release, so two variants cannot share a latest pointer.
"""
import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import Payload, TensorMap
from commerce.packages.contracts.validate import load_schemas, validate_payload
from commerce.services.fl_coordinator.npz_payload import (
    MAX_PAYLOAD_BYTES, PayloadTooLarge, decode_npz, encode_npz,
)

MIN_SYNTHETIC_CLIENTS = 3
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@lru_cache(maxsize=1)
def _validators():
    return load_schemas()


def check_contract(contract: str, payload) -> None:
    verdict = validate_payload(_validators()[contract], contract, payload)
    if not verdict.ok:
        raise ContractError(verdict.code, verdict.field_path)


def _specs_by_name(specs: Sequence[Mapping]) -> dict:
    return {spec["name"]: (tuple(spec["shape"]), spec["dtype"]) for spec in specs}


def aggregate_uniform(deltas: Sequence[Mapping[str, object] | None]) -> TensorMap | None:
    """The D0017 rule on its own: every member complete, else discard; then the plain mean.

    One entry per pre-fixed cohort member. None marks a member without a completed delta,
    so the whole round is discarded and None is returned, as for an empty cohort. Arrays
    may be numpy arrays or detached CPU tensors. The mean is taken in float64 and returned
    as float32; there is no sample weighting. Key or shape disagreement is a caller bug and
    raises ValueError. B's lab runner calls this directly (D0018, D0020: not protected).
    """
    if not deltas or any(delta is None for delta in deltas):
        return None
    names = set(deltas[0])
    if any(set(delta) != names for delta in deltas):
        raise ValueError("deltas do not share one tensor set")
    mean: TensorMap = {}
    for name in sorted(names):
        arrays = [np.asarray(delta[name], dtype=np.float64) for delta in deltas]
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("tensor shapes differ across deltas")
        mean[name] = np.mean(arrays, axis=0).astype(np.float32)
    return mean


class RegistryCorrupt(RuntimeError):
    """A REGISTRY_DIR that does not verify. The coordinator refuses to start rather than serve it."""


def _write_json(path: Path, obj) -> None:
    path.write_bytes(ids.canonical_json(obj))


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class ModelRegistry:
    """Immutable releases. The descriptor and latest pointer are published after the weights.

    With `root` (REGISTRY_DIR, one directory per variant) each release is written to
    `<root>/<model_version>/` through a staging directory that is renamed into place,
    and `<root>/latest` is replaced last, so a crash never exposes a partial release.
    `provenance.json` is a local operations record (trained, random init, FL round);
    it is never served and is not part of model_release.v1.
    """

    def __init__(self, root: Path | str | None = None):
        self._releases: dict[str, tuple[Payload, Payload, bytes]] = {}
        self._latest: str | None = None
        self._manifest_hash: str | None = None
        self._root = None if root is None else Path(root)
        if self._root is not None:
            self._load()

    def register(self, model_version: str, manifest: Payload, tensors: TensorMap, *,
                 provenance: Mapping | None = None) -> Payload:
        check_contract("shared_model_manifest.v1", manifest)
        if not SAFE_ID.fullmatch(model_version):
            raise ContractError("SCHEMA_INVALID", "/model_version")
        if manifest["manifest_hash"] != ids.manifest_hash(manifest):
            raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
        if self._manifest_hash not in (None, manifest["manifest_hash"]):
            raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
        if model_version in self._releases:
            raise ContractError("ILLEGAL_STATE_TRANSITION")
        weights = encode_npz({name: np.asarray(array, dtype=np.float32) for name, array in tensors.items()})
        if len(weights) > MAX_PAYLOAD_BYTES:
            raise PayloadTooLarge("weights exceed the transfer limit")
        decode_npz(weights, manifest["tensors"])  # same name/shape/dtype/finite checks as a submission
        descriptor = {
            "schema_version": "model_release.v1", "model_version": model_version,
            "manifest_hash": manifest["manifest_hash"],
            "weights_sha256": hashlib.sha256(weights).hexdigest(), "weights_size_bytes": len(weights),
        }
        check_contract("model_release.v1", descriptor)
        if self._root is not None:
            self._persist(model_version, descriptor, manifest, weights, provenance)
        self._releases[model_version] = (descriptor, dict(manifest), weights)
        self._manifest_hash = manifest["manifest_hash"]
        self._latest = model_version  # published last
        return dict(descriptor)

    def _persist(self, model_version: str, descriptor: Payload, manifest: Payload, weights: bytes,
                 provenance: Mapping | None) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        final = self._root / model_version
        if final.exists():
            raise ContractError("ILLEGAL_STATE_TRANSITION")
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self._root))
        try:
            (staging / "weights.npz").write_bytes(weights)
            _write_json(staging / "manifest.json", manifest)
            _write_json(staging / "provenance.json", dict(provenance or {"source": "unspecified"}))
            _write_json(staging / "release.json", descriptor)
            os.replace(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        atomic_write(self._root / "latest", model_version.encode("ascii"))

    def _load(self) -> None:
        if not self._root.is_dir():
            return
        for entry in sorted(self._root.iterdir()):
            if entry.name.startswith("."):
                # leftovers of an interrupted publish: never visible, safe to drop
                if entry.name.startswith(".staging-"):
                    shutil.rmtree(entry, ignore_errors=True)
                elif entry.name.startswith(".tmp-"):
                    entry.unlink(missing_ok=True)
                continue
            if entry.name == "latest":
                continue
            try:
                descriptor = json.loads((entry / "release.json").read_bytes())
                manifest = json.loads((entry / "manifest.json").read_bytes())
                weights = (entry / "weights.npz").read_bytes()
                check_contract("model_release.v1", descriptor)
                check_contract("shared_model_manifest.v1", manifest)
                decode_npz(weights, manifest["tensors"])
            except (OSError, ValueError, ContractError) as exc:
                raise RegistryCorrupt("release %s does not verify" % entry.name) from exc
            if (descriptor["model_version"] != entry.name
                    or descriptor["weights_sha256"] != hashlib.sha256(weights).hexdigest()
                    or descriptor["weights_size_bytes"] != len(weights)
                    or descriptor["manifest_hash"] != manifest["manifest_hash"]
                    or manifest["manifest_hash"] != ids.manifest_hash(manifest)):
                raise RegistryCorrupt("release %s does not verify" % entry.name)
            if self._manifest_hash not in (None, manifest["manifest_hash"]):
                raise RegistryCorrupt("two manifests in one variant registry")
            self._manifest_hash = manifest["manifest_hash"]
            self._releases[entry.name] = (descriptor, manifest, weights)
        latest = self._root / "latest"
        if latest.exists():
            name = latest.read_text(encoding="ascii").strip()
            if name not in self._releases:
                raise RegistryCorrupt("latest points to a missing release")
            self._latest = name

    def provenance(self, model_version: str) -> Payload:
        """Local operations record of a persisted release; empty for an in-memory registry."""
        self._entry(model_version)
        if self._root is None:
            return {}
        return json.loads((self._root / model_version / "provenance.json").read_bytes())

    def latest(self) -> Payload | None:
        return None if self._latest is None else dict(self._releases[self._latest][0])

    def release(self, model_version: str) -> Payload:
        return dict(self._entry(model_version)[0])

    def manifest(self, model_version: str) -> Payload:
        return dict(self._entry(model_version)[1])

    def weights(self, model_version: str) -> bytes:
        return self._entry(model_version)[2]

    def tensors(self, model_version: str) -> TensorMap:
        _descriptor, manifest, weights = self._entry(model_version)
        return decode_npz(weights, manifest["tensors"])

    def _entry(self, model_version: str):
        try:
            return self._releases[model_version]
        except KeyError:
            raise ContractError("NOT_FOUND") from None


def _parse_deadline(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class _Slot:
    def __init__(self, digest: str, arrival_t_s: float, completed: bool, delta: TensorMap | None):
        self.digest, self.arrival_t_s, self.completed, self.delta = digest, arrival_t_s, completed, delta


class SyntheticRound:
    """One round over a fixed cohort. States: open -> aggregated | discarded."""

    def __init__(self, registry: ModelRegistry, config: Payload, cohort: Sequence[str], *,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 monotonic: Callable[[], float] | None = None):
        check_contract("round_config.v1", config)
        latest = registry.latest()
        if latest is None or config["model_version"] != latest["model_version"]:
            raise ContractError("MANIFEST_MISMATCH", "/model_version")
        manifest = registry.manifest(config["model_version"])
        if (config["manifest_hash"] != manifest["manifest_hash"]
                or config["architecture_version"] != manifest["architecture_version"]):
            raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
        if not SAFE_ID.fullmatch(config["round_id"]):
            raise ContractError("SCHEMA_INVALID", "/round_id")
        cohort = tuple(cohort)
        if (len(set(cohort)) != len(cohort) or not all(SAFE_ID.fullmatch(seller) for seller in cohort)
                or len(cohort) < max(config["min_clients"], MIN_SYNTHETIC_CLIENTS)):
            raise ContractError("SCHEMA_INVALID", "/min_clients")
        self._registry, self.config, self.cohort = registry, dict(config), cohort
        self._manifest = manifest
        self._deadline = _parse_deadline(config["deadline"])
        self._now = now
        self._monotonic = monotonic or time.monotonic
        self._started = self._monotonic()
        self._slots: dict[str, _Slot] = {}
        self.state = "open"
        self.release: Payload | None = None

    @property
    def round_id(self) -> str:
        return self.config["round_id"]

    def submit(self, seller_id: str, submission: Payload, payload: bytes) -> Payload:
        """First valid submission per seller is accepted; the same bytes again return the same result."""
        self.expire()
        if seller_id not in self.cohort:
            raise ContractError("FORBIDDEN")
        digest = hashlib.sha256(ids.canonical_json(submission) + b"\0" + payload).hexdigest()
        slot = self._slots.get(seller_id)
        if slot is not None:
            if slot.digest == digest:
                return self.result(seller_id)
            raise ContractError("DUPLICATE_ROUND_SUBMIT")
        if self.state != "open":
            raise ContractError("ROUND_DISCARDED")
        delta = self._validate(seller_id, submission, payload)
        completed = submission["delta_manifest"]["completed"]
        self._slots[seller_id] = _Slot(digest, self._monotonic() - self._started, completed, delta)
        if not completed:
            self._discard()
        elif len(self._slots) == len(self.cohort):
            self._aggregate()
        return self.result(seller_id)

    def result(self, seller_id: str) -> Payload:
        self.expire()
        slot = self._slots.get(seller_id)
        if slot is None:
            raise ContractError("NOT_FOUND")
        if self.state == "aggregated":
            disposition, staleness, aggregated_in = "aggregated_on_time", 0, self.round_id
        elif self.state == "discarded":
            disposition = "round_discarded" if slot.completed else "dropped_incomplete"
            staleness, aggregated_in = None, None
        else:
            disposition, staleness, aggregated_in = "accepted_on_time", None, None
        return {
            "schema_version": "round_submit_ack.v1", "round_id": self.round_id, "seller_id": seller_id,
            "disposition": disposition, "arrival_t_s": slot.arrival_t_s,
            "staleness_rounds": staleness, "aggregated_in_round_id": aggregated_in,
        }

    def expire(self) -> None:
        """Past the deadline an open round can no longer be complete; there is no late carry-over."""
        if self.state == "open" and self._now() > self._deadline:
            self._discard()

    def _validate(self, seller_id: str, submission: Payload, payload: bytes) -> TensorMap:
        check_contract("round_submission.v1", submission)
        delta_manifest = submission["delta_manifest"]
        if delta_manifest["seller_id"] != seller_id:
            raise ContractError("FORBIDDEN", "/delta_manifest/seller_id")
        if delta_manifest["round_id"] != self.round_id:
            raise ContractError("NOT_FOUND", "/delta_manifest/round_id")
        for key in ("model_version", "manifest_hash", "architecture_version"):
            if delta_manifest[key] != self.config[key]:
                raise ContractError("MANIFEST_MISMATCH", "/delta_manifest/" + key)
        if _specs_by_name(delta_manifest["tensors"]) != _specs_by_name(self._manifest["tensors"]):
            raise ContractError("TENSOR_SET_MISMATCH", "/delta_manifest/tensors")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise PayloadTooLarge("payload exceeds the transfer limit")
        if len(payload) != submission["payload_nbytes"]:
            raise ContractError("SCHEMA_INVALID", "/payload_nbytes")
        if hashlib.sha256(payload).hexdigest() != submission["payload_sha256"]:
            raise ContractError("SCHEMA_INVALID", "/payload_sha256")
        return decode_npz(payload, self._manifest["tensors"])

    def _discard(self) -> None:
        self.state = "discarded"
        for slot in self._slots.values():
            slot.delta = None

    def _aggregate(self) -> None:
        base = self._registry.tensors(self.config["model_version"])
        mean = aggregate_uniform([slot.delta for slot in self._slots.values()])
        new = {name: (base[name].astype(np.float64) + mean[name].astype(np.float64)).astype(np.float32)
               for name in base}
        weights_sha = hashlib.sha256(encode_npz(new)).hexdigest()
        version = "model-" + hashlib.sha256(
            ids.canonical_json([self.config["model_version"], self.round_id, weights_sha])).hexdigest()[:16]
        self.release = self._registry.register(version, self._manifest, new, provenance={
            "source": "fl_round", "mode": "synthetic_plaintext", "round_id": self.round_id,
            "base_model_version": self.config["model_version"], "cohort_size": len(self.cohort)})
        self.state = "aggregated"
        for slot in self._slots.values():
            slot.delta = None

