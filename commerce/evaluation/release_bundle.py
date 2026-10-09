"""The first service release from a lab FL run (model-lab.md §6).

    python -m commerce.evaluation.release_bundle --run commerce/evaluation/runs/fl_100/<run folder> \\
        --encoder-dir commerce/evaluation/cache/encoders/<model>/<revision> \\
        --model-version ic100-R_lm-r500 --out commerce/evaluation/outputs/releases/ic100-R_lm-r500

Reads the run's shared_weights.pt and record.json, loads the chosen round (the
final round by default, evaluation.md §5) into the registered service
architecture of the run's variant, and writes release.json, manifest.json,
weights.npz and provenance.json. The run's text artifact and preprocessing must
equal this checkout's, so the release scores the text the seller encodes.

C imports the folder (release.json, manifest.json, weights.npz) into its registry
and hands it to each seller's install_release; provenance.json stays with the lab
records. The weights come from an unprotected FL simulation over Instacart
virtual sellers (D0020); the held-out A-0 sellers took no part.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from commerce.packages.recommender.serving import (
    SERVICE_ARCHITECTURES, SERVICE_ENCODER, VARIANT_OF_ARCHITECTURE, architecture_for, build_manifest, build_model,
    canonical_npz, shared_tensors, write_bundle,
)
from commerce.packages.recommender.text_encoder import artifact_hash
from commerce.packages.recommender.z_cache import preprocessing_version

VARIANT_OF_LAB = {"T_lm": "text_only", "R_lm": "text_relation"}


def bundle(run: Path, out: Path, model_version: str, text_artifact_hash: str, *, which: str = "final",
           architectures=SERVICE_ARCHITECTURES, variants=VARIANT_OF_ARCHITECTURE) -> dict:
    record = json.loads((run / "record.json").read_text(encoding="utf-8"))
    saved = torch.load(run / "shared_weights.pt", map_location="cpu", weights_only=True)
    lab_variant = saved["variant"]
    if lab_variant not in VARIANT_OF_LAB:
        raise SystemExit("only the lm variants are service models, not %s" % lab_variant)
    arch = architecture_for(VARIANT_OF_LAB[lab_variant], architectures, variants)
    if saved["architecture"] != arch.config.architecture_version:
        raise SystemExit("the run used %s, the service registers %s" % (saved["architecture"],
                                                                         arch.config.architecture_version))
    preprocessing = preprocessing_version()
    if record["encoder"]["text_artifact_hash"] != text_artifact_hash:
        raise SystemExit("the run encoded text with another artifact than --encoder-dir")
    if record["encoder"]["preprocessing_version"] != preprocessing:
        raise SystemExit("the run's text builder or relation definition differs from this checkout")
    model = build_model(arch.config)
    model.load_state_dict(saved["%s_round_shared" % which], strict=True)
    tensors = shared_tensors(model)
    manifest = build_manifest(arch, text_artifact_hash, preprocessing)
    data = canonical_npz(tensors, [t["name"] for t in manifest["tensors"]])
    release = {"schema_version": "model_release.v1", "model_version": model_version,
               "manifest_hash": manifest["manifest_hash"], "weights_sha256": hashlib.sha256(data).hexdigest(),
               "weights_size_bytes": len(data)}
    write_bundle(out, release, manifest, data)
    settings = record["settings"]
    provenance = {
        "kind": "unprotected FL simulation (D0020), not a protected FL release",
        "data": "Instacart virtual sellers, %d sellers, target %s" % (settings["sellers"], settings["target"]),
        "run": run.name, "variant": lab_variant, "service_variant": arch.variant,
        "architecture_version": arch.version, "round": which,
        "round_number": settings["rounds"] if which == "final" else saved["best_round"],
        "rounds": settings["rounds"], "seed": settings["seed"], "code": record.get("code"),
        "labels": record.get("labels"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"release": release, "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", choices=("final", "best"), default="final")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("%s exists; a release is never rewritten" % args.out)
    result = bundle(args.run, args.out, args.model_version, artifact_hash(args.encoder_dir, SERVICE_ENCODER),
                    which=args.round)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
