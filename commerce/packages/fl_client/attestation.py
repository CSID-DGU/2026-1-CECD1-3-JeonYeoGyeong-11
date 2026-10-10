"""Synthetic input attestation for FL_MODE=synthetic_plaintext (D0025, OQ17).

The synthetic demo runner writes one file per seller after it fills a fresh feature
store from a deterministic generator. The file records the content digest of exactly
what was generated (ids.snapshot_digest over event IDs and catalog bodies). Before
every round the FL client asks B's runtime for the digest of the snapshot it is
about to train on and compares the two. Any other input, one live order included,
changes the digest and the client submits nothing. A `source` field is never read.

This guards against accidentally wiring real data into the plaintext path. It is not
a defence against someone who can rewrite the attestation file on the seller host.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from commerce.packages.contracts import ids

SCHEMA = "c.synthetic_attestation.v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SyntheticInputRefused(RuntimeError):
    """The snapshot is not the attested synthetic input. Nothing is submitted."""


@dataclass(frozen=True)
class Attestation:
    seller_id: str
    snapshot_digest: str
    generator: str


def write_attestation(path: Path | str, seller_id: str, digest: str, generator: str) -> None:
    if not _DIGEST.fullmatch(digest):
        raise ValueError("digest must be 64 lowercase hex characters")
    Path(path).write_bytes(ids.canonical_json({
        "schema": SCHEMA, "seller_id": seller_id, "snapshot_digest": digest, "generator": generator}))


def load_attestation(path: Path | str) -> Attestation:
    data = json.loads(Path(path).read_bytes())
    if data.get("schema") != SCHEMA or set(data) != {"schema", "seller_id", "snapshot_digest", "generator"}:
        raise ValueError("not a synthetic attestation")
    if not isinstance(data["seller_id"], str) or not _DIGEST.fullmatch(str(data["snapshot_digest"])):
        raise ValueError("malformed synthetic attestation")
    return Attestation(data["seller_id"], data["snapshot_digest"], str(data["generator"]))


def check_snapshot(runtime, local_data_ref: str, attestation: Attestation) -> None:
    """Raise SyntheticInputRefused unless the runtime's snapshot is the attested one."""
    if attestation.seller_id != runtime.seller_id:
        raise SyntheticInputRefused("attestation is for another seller")
    digest = getattr(runtime, "snapshot_digest", None)
    if digest is None:
        raise SyntheticInputRefused("the runtime cannot report a snapshot digest")
    if digest(local_data_ref) != attestation.snapshot_digest:
        raise SyntheticInputRefused("the snapshot is not the attested synthetic input")
