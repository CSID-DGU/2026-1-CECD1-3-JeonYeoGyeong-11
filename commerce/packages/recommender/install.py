"""Install B's seller-side files into a seller's MODEL_DIR (model-lab.md §6.5-6.6).

    python -m commerce.packages.recommender.install --model-dir <MODEL_DIR> \\
        --encoder-dir commerce/evaluation/cache/encoders/<org>__<name>/<revision>
    python -m commerce.packages.recommender.install --model-dir <MODEL_DIR> \\
        --release-dir <release folder> --variant text_relation

The frozen encoder goes to {MODEL_DIR}/frozen_text/ only if its files hash to the
service's text_artifact_hash, the value every release manifest names; the files are
hard-linked when the source is on the same drive, copied otherwise, and an encoder
already there with another hash is left alone and reported. A release folder
(release.json, manifest.json, weights.npz, as release_bundle.py writes it) goes
through SellerRuntime.install_release, the same checks as a release from C's
coordinator, for a seller that shows the trained model without running FL. Neither
step touches the seller's feature store.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from commerce.packages.contracts.types import ModelVariant
from commerce.packages.recommender.serving import SERVICE_ARCHITECTURES, SERVICE_ENCODER, load_bundle
from commerce.packages.recommender.text_encoder import EncoderSpec, artifact_files, artifact_hash

# artifact_hash of SERVICE_ENCODER's files (MiniLM-L12 e8f8c21), as in the first releases' manifests.
SERVICE_TEXT_ARTIFACT_HASH = "e84ab6c5207aa16046297e5e5d02c9d47319176c039a3ed04354370ba683660c"


class InstallError(RuntimeError):
    """Nothing was changed; the message says why."""


def install_encoder(source: str | Path, model_dir: str | Path, *, spec: EncoderSpec = SERVICE_ENCODER,
                    expected_hash: str = SERVICE_TEXT_ARTIFACT_HASH) -> str:
    """Put the frozen encoder at {model_dir}/frozen_text. Returns what was done."""
    source, target = Path(source), Path(model_dir) / "frozen_text"
    if not (source / "config.json").is_file():
        raise InstallError("no encoder files at %s" % source)
    found = artifact_hash(source, spec)
    if found != expected_hash:
        raise InstallError("%s is not the service encoder (text_artifact_hash %s, expected %s)"
                           % (source, found, expected_hash))
    if target.exists():
        present = artifact_hash(target, spec) if (target / "config.json").is_file() else None
        if present == expected_hash:
            return "already installed"
        raise InstallError("%s holds other files (text_artifact_hash %s); move it away first" % (target, present))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (".frozen_text-%s" % uuid.uuid4().hex)
    linked = 0
    try:
        for path in artifact_files(source):
            dest = staging / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, dest)
                linked += 1
            except OSError:
                shutil.copy2(path, dest)
        if artifact_hash(staging, spec) != expected_hash:
            raise InstallError("the copied files do not hash to the service encoder")
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return "installed (%d of %d files hard-linked)" % (linked, len(artifact_files(target)))


def install_release_dir(model_dir: str | Path, release_dir: str | Path, variant: ModelVariant, *,
                        text=None, architectures=SERVICE_ARCHITECTURES) -> str:
    """install_release of a release folder, with a throwaway feature store. Returns the model_version."""
    from commerce.packages.contracts.errors import ContractError
    from commerce.packages.recommender.seller_runtime import SellerRuntime
    release, manifest, tensors = load_bundle(release_dir)
    with tempfile.TemporaryDirectory() as tmp:
        runtime = SellerRuntime("install", Path(tmp) / "features.sqlite", model_dir, text=text,
                                architectures=architectures)
        try:
            runtime.install_release(release, manifest, tensors, model_variant=variant)
        except FileNotFoundError:
            raise InstallError("install the frozen encoder first (--encoder-dir): the release names its hash")
        except ContractError as error:
            raise InstallError("%s at %s: this release is not for this B package or variant"
                               % (error.code, error.field_path))
        finally:
            runtime.close()
    return release["model_version"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install B's frozen encoder and/or a release into MODEL_DIR.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, help="downloaded encoder files of the pinned revision")
    parser.add_argument("--release-dir", type=Path, help="a release folder to install without FL")
    parser.add_argument("--variant", default="text_relation", choices=("text_only", "text_relation"))
    args = parser.parse_args(argv)
    if args.encoder_dir is None and args.release_dir is None:
        parser.error("give --encoder-dir, --release-dir or both")
    report = {}
    try:
        if args.encoder_dir is not None:
            report["frozen_text"] = install_encoder(args.encoder_dir, args.model_dir)
        if args.release_dir is not None:
            report[args.variant] = install_release_dir(args.model_dir, args.release_dir, args.variant)
    except InstallError as error:
        raise SystemExit("not installed: %s" % error)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
