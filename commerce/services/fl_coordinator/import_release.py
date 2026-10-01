"""Local import of a lab-made release into one variant registry (docs/design/model-lab.md §6).

    python -m commerce.services.fl_coordinator.import_release \
        --registry commerce/deploy/var/fl/text_only/registry \
        --manifest manifest.json --weights weights.npz --model-version model-init-1 --kind trained

There is deliberately no upload API: an operator runs this on the coordinator host.
The manifest (schema, recomputed manifest_hash) and the weights (exact float32 tensor
set, finite values, transfer limit) are checked, the variant pin of the registry is
enforced, and the release is published atomically as the new latest. Whether the
weights were trained or are a random init is recorded in the local provenance file.

Not checked yet: that architecture_version matches B's immutable config registry,
which does not exist yet. The import prints this so it is not read as verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import Payload
from commerce.services.fl_coordinator.npz_payload import PayloadTooLarge, decode_npz
from commerce.services.fl_coordinator.round_core import ModelRegistry, check_contract

KINDS = ("trained", "random_init")


def import_release(registry_dir: Path | str, manifest_path: Path | str, weights_path: Path | str,
                   model_version: str, kind: str, note: str | None = None) -> Payload:
    if kind not in KINDS:
        raise ValueError("kind must be one of %s" % ", ".join(KINDS))
    manifest = json.loads(Path(manifest_path).read_bytes())
    check_contract("shared_model_manifest.v1", manifest)
    weights = Path(weights_path).read_bytes()
    tensors = decode_npz(weights, manifest["tensors"])
    provenance = {"source": "import", "kind": kind, "input_weights_sha256": hashlib.sha256(weights).hexdigest()}
    if note:
        provenance["note"] = note
    return ModelRegistry(registry_dir).register(model_version, manifest, tensors, provenance=provenance)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, help="REGISTRY_DIR of one variant")
    parser.add_argument("--manifest", required=True, help="shared_model_manifest.v1 JSON from B")
    parser.add_argument("--weights", required=True, help="npz of the float32 shared tensors")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--note", help="local provenance note, e.g. single-seller pretraining")
    args = parser.parse_args(argv)
    try:
        descriptor = import_release(args.registry, args.manifest, args.weights, args.model_version,
                                    args.kind, args.note)
    except ContractError as exc:
        print("import refused: %s" % exc.code, file=sys.stderr)
        return 1
    except (PayloadTooLarge, OSError, ValueError) as exc:
        print("import refused: %s" % type(exc).__name__, file=sys.stderr)
        return 1
    print(json.dumps(descriptor, sort_keys=True))
    print("not checked: architecture_version against B's config registry (not available yet)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
