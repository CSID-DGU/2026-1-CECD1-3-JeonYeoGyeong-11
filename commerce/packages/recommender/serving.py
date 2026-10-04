"""Service models and the B ↔ C tensor boundary (model-lab.md §6, interfaces.md §5).

The service runs the two item representations that beat the HAREX-style
baseline in D0022 on the same backbone: text_only is harex.T_lm.v1 (frozen
MiniLM text + a shared projection) and text_relation is harex.R_lm.v1 (the same
plus the seller's own purchase relations). The repeat path (D0023) is not part
of it.

architecture_version is the manifest's integer name for one registered config.
A number never changes meaning once used; a changed config gets a new number.

Tensors cross the boundary under flat names, shared.<state_dict key with dots as
underscores>, in the model's own parameter order. B writes its weight files as a
canonical npz: stored (uncompressed) zip members in manifest order with fixed
timestamps, so the same tensors always give the same bytes. A file C serves may
be encoded otherwise; C's client checks those bytes before install_release.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.ids import manifest_hash
from commerce.packages.contracts.types import ModelVariant, Payload, TensorMap
from commerce.packages.recommender.harex import HAREX_ARCHITECTURES, HarexConfig, HarexRecommender
from commerce.packages.recommender.text_encoder import EncoderSpec, FrozenTextEncoder, artifact_hash

SERVICE_ENCODER = EncoderSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                              "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", max_length=64)
SERVICE_ARCHITECTURES: dict[int, HarexConfig] = {
    1: HAREX_ARCHITECTURES["harex.T_lm.v1"],
    2: HAREX_ARCHITECTURES["harex.R_lm.v1"],
}
VARIANT_OF_ARCHITECTURE = {1: "text_only", 2: "text_relation"}
TASK_KIND = "next_purchase"
PERSONAL_GROUPS = ("query_proj", "scorer")  # D0019: the only groups personalization trains
MAX_WEIGHTS_BYTES = 8 * 1024 * 1024  # model_release.v1 weights_size_bytes maximum
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ServiceArchitecture:
    version: int
    variant: ModelVariant
    config: HarexConfig


def architecture_for(variant: ModelVariant, architectures: Mapping[int, HarexConfig] = SERVICE_ARCHITECTURES,
                     variants: Mapping[int, str] = VARIANT_OF_ARCHITECTURE) -> ServiceArchitecture:
    matches = [v for v, name in variants.items() if name == variant and v in architectures]
    if len(matches) != 1:
        raise ValueError("model_variant must be text_only or text_relation")
    config = architectures[matches[0]]
    if config.text != "lm" or config.relation != (variant == "text_relation"):
        raise ValueError("a service architecture must be lm text with relations only for text_relation")
    return ServiceArchitecture(matches[0], variant, config)


def build_model(config: HarexConfig, *, seed: int = 0) -> HarexRecommender:
    torch.manual_seed(seed)
    model = HarexRecommender(config)
    model.eval()
    return model


def tensor_names(model: HarexRecommender) -> dict[str, str]:
    """state_dict key -> manifest name, in parameter order."""
    names = {key: "shared." + key.replace(".", "_") for key in model.shared_state()}
    if len(set(names.values())) != len(names):
        raise ValueError("two shared tensors map to the same manifest name")
    return names


def shared_tensors(model: HarexRecommender) -> TensorMap:
    state = model.shared_state()
    return {name: state[key].detach().cpu().numpy().astype(np.float32, copy=True)
            for key, name in tensor_names(model).items()}


def load_shared(model: HarexRecommender, tensors: Mapping[str, np.ndarray]) -> None:
    names = tensor_names(model)
    if set(tensors) != set(names.values()):
        raise ContractError("TENSOR_SET_MISMATCH", "/tensors")
    model.load_state_dict({key: torch.from_numpy(np.array(tensors[name], dtype=np.float32))
                           for key, name in names.items()}, strict=True)


def build_manifest(arch: ServiceArchitecture, text_artifact_hash: str, preprocessing_version: str) -> Payload:
    model = build_model(arch.config)
    manifest = {
        "schema_version": "shared_model_manifest.v1",
        "architecture_version": arch.version,
        "task_kind": TASK_KIND,
        "preprocessing_version": preprocessing_version,
        "text_artifact_hash": text_artifact_hash,
        "tensors": [{"name": name, "shape": list(array.shape), "dtype": "float32"}
                    for name, array in shared_tensors(model).items()],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def check_tensors(manifest: Payload, tensors: Mapping[str, np.ndarray]) -> None:
    """Key set, dtype, shape and finiteness against the manifest (interfaces.md §5)."""
    expected = {t["name"]: tuple(t["shape"]) for t in manifest["tensors"]}
    if set(tensors) != set(expected):
        raise ContractError("TENSOR_SET_MISMATCH", "/tensors")
    for name, shape in expected.items():
        array = tensors[name]
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or tuple(array.shape) != shape:
            raise ContractError("MANIFEST_MISMATCH", "/tensors")
        if not np.isfinite(array).all():
            raise ContractError("MANIFEST_MISMATCH", "/tensors")


def canonical_npz(tensors: Mapping[str, np.ndarray], order: Sequence[str]) -> bytes:
    """The one byte form of a tensor map: stored members in manifest order, fixed metadata."""
    if set(order) != set(tensors) or len(order) != len(tensors):
        raise ValueError("order must list every tensor once")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in order:
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(tensors[name], dtype="<f4"),
                                      version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3  # the same bytes on every OS
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    return out.getvalue()


def read_npz(data: bytes, manifest: Payload) -> TensorMap:
    """Tensors from npz bytes; no pickle, members checked before any is read."""
    tensors = read_tensors(data, [t["name"] for t in manifest["tensors"]])
    check_tensors(manifest, tensors)
    return tensors


def read_tensors(data: bytes, names: Sequence[str]) -> TensorMap:
    """The arrays of an npz whose members must be exactly names; float32 only, no pickle."""
    if len(data) > MAX_WEIGHTS_BYTES:
        raise ContractError("MANIFEST_MISMATCH", "/weights_size_bytes")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if sorted(m.filename for m in members) != sorted(n + ".npy" for n in names):
            raise ContractError("TENSOR_SET_MISMATCH", "/tensors")
        if sum(m.file_size for m in members) > MAX_WEIGHTS_BYTES:
            raise ContractError("MANIFEST_MISMATCH", "/weights_size_bytes")
        tensors = {}
        for name in names:
            with archive.open(name + ".npy") as member:
                array = np.lib.format.read_array(io.BytesIO(member.read()), allow_pickle=False)
            if array.dtype != np.float32:
                raise ContractError("MANIFEST_MISMATCH", "/tensors")
            tensors[name] = array
    return tensors


def write_bundle(folder: str | Path, release: Payload, manifest: Payload, data: bytes) -> None:
    """A release folder as B hands it over: release.json, manifest.json, weights.npz."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "weights.npz").write_bytes(data)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / "release.json").write_text(json.dumps(release, indent=1, ensure_ascii=False), encoding="utf-8")


def load_bundle(folder: str | Path) -> tuple[Payload, Payload, TensorMap]:
    """(release, manifest, tensors) of a release folder, with the byte hash and manifest hash checked.
    install_release still checks everything against the installing seller's own package."""
    folder = Path(folder)
    release = json.loads((folder / "release.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data = (folder / "weights.npz").read_bytes()
    if len(data) != release["weights_size_bytes"] or hashlib.sha256(data).hexdigest() != release["weights_sha256"]:
        raise ContractError("MANIFEST_MISMATCH", "/weights_sha256")
    if manifest_hash(manifest) != manifest["manifest_hash"] or release["manifest_hash"] != manifest["manifest_hash"]:
        raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
    return release, manifest, read_npz(data, manifest)


class FrozenText:
    """The installed frozen encoder of SERVICE_ENCODER: hashed when opened, loaded on first use."""

    def __init__(self, encoder_dir: str | Path, spec: EncoderSpec = SERVICE_ENCODER):
        self.dir = Path(encoder_dir)
        config = self.dir / "config.json"
        if not config.is_file():
            raise FileNotFoundError("no frozen text encoder installed at %s" % self.dir)
        self.spec = spec
        self.text_artifact_hash = artifact_hash(self.dir, spec)
        self.dim = int(json.loads(config.read_text(encoding="utf-8"))["hidden_size"])
        self._encoder = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            self._encoder = FrozenTextEncoder(self.dir, self.spec)
            if self._encoder.dim != self.dim:
                raise ValueError("the encoder's width differs from its config")
        return self._encoder.encode(texts)
