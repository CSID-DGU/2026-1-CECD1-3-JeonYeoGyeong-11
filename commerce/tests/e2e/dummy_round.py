"""Stand-in for B's runtime in synthetic rounds: generated tensors only, no real data.

The tensor names and shapes come from the dummy_tensors_for_round_bringup.json
fixture. Its manifest_hash only illustrates the form, so it is recomputed here.
The dummy runtime is handed to the C client functions directly; A's build_context
is not involved (docs/development.md, replacing the other module). It follows the
RecommenderRuntime rules C relies on (model.md §4-5): training starts from the
round's base, install verifies and is idempotent, and nothing personalized is exposed.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from commerce.packages.contracts import ids
from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented
from commerce.packages.contracts.types import ModelVariant, Payload, TensorMap, TrainingResult
from commerce.packages.fl_client import submission as client_submission
from commerce.packages.recommender.runtime import UnimplementedRuntime

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
        if not self.completed:  # no training example: a zero delta of the same shape (model.md §4)
            return TrainingResult(shared_delta={name: np.zeros_like(array) for name, array in base.items()},
                                  metrics={"loss_mean": None, "grad_norm_mean": None}, completed=False)
        delta = {name: np.full(array.shape, self.offset, dtype=np.float32) for name, array in base.items()}
        return TrainingResult(shared_delta=delta, metrics={"loss_mean": 1.0, "grad_norm_mean": 0.5},
                              completed=True)


def build_submission(seller_id: str, manifest: Payload, config: Payload, result: TrainingResult) -> tuple[Payload, bytes]:
    """The FL client's own builder, so tests and the client share one code path."""
    return client_submission.build_submission(seller_id, manifest, config, result)


class DummyRuntime(UnimplementedRuntime):
    """B runtime double for one seller and one variant. Calls outside the FL boundary are recorded."""

    def __init__(self, seller_id: str, manifest: Payload, offset: float, *, completed: bool = True,
                 model_variant: ModelVariant = "text_only"):
        super().__init__(seller_id, Path("unused-features.sqlite"), Path("unused-models"))
        self.manifest, self.model_variant = manifest, model_variant
        self.trainer = DummyTrainer(seller_id, offset, completed=completed)
        self.installed: dict[str, tuple[str, TensorMap]] = {}  # model_version -> (weights hash, tensors)
        self.serving: str | None = None
        self.personal_tail = {name: np.full(spec["shape"], 99.0, np.float32) for name, spec in
                              ((spec["name"], spec) for spec in manifest["tensors"])}
        self.calls: list[str] = []

    def _variant(self, model_variant: ModelVariant) -> None:
        if model_variant != self.model_variant:
            raise ContractError("MANIFEST_MISMATCH")

    def get_local_data_ref(self) -> str:
        self.calls.append("get_local_data_ref")
        return "synthetic:" + self.seller_id

    def get_shared_manifest(self, *, model_variant: ModelVariant = "text_relation") -> Payload:
        self.calls.append("get_shared_manifest")
        self._variant(model_variant)
        return dict(self.manifest)

    def train_round(self, local_data_ref: str, round_config: Payload, *,
                    model_variant: ModelVariant = "text_relation") -> TrainingResult:
        self.calls.append("train_round")
        self._variant(model_variant)
        base = self.installed.get(round_config["model_version"])
        if base is None:  # the round's base must be installed; never substitute another version
            raise ContractError("NOT_FOUND", "/model_version")
        return self.trainer.train_round(base[1], round_config)

    def install_release(self, release: Payload, manifest: Payload, tensors: TensorMap, *,
                        model_variant: ModelVariant = "text_relation") -> None:
        self.calls.append("install_release")
        self._variant(model_variant)
        version, digest = release["model_version"], release["weights_sha256"]
        if version in self.installed:
            if self.installed[version][0] != digest:
                raise ContractError("MANIFEST_MISMATCH", "/weights_sha256")
            return  # same version and hash again: idempotent
        self.installed[version] = (digest, {name: np.array(array) for name, array in tensors.items()})
        self.serving = version

    def personalize_local(self, *args, **kwargs):
        self.calls.append("personalize_local")
        raise FeatureNotImplemented("not part of the FL boundary")

    def export_shared_state(self, *args, **kwargs):
        self.calls.append("export_shared_state")
        raise FeatureNotImplemented("the client submits TrainingResult.shared_delta only")

    def predict_local(self, *args, **kwargs):
        self.calls.append("predict_local")
        raise FeatureNotImplemented("not part of the FL boundary")

    def compare_local(self, *args, **kwargs):
        self.calls.append("compare_local")
        raise FeatureNotImplemented("not part of the FL boundary")
