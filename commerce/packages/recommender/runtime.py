"""B's callable skeleton. No method pretends to persist or train successfully.

UnimplementedRuntime is the reference stub that gate scaffold checks. Build the
real runtime as a separate class and have open_runtime return it; keep this one.
"""
from pathlib import Path

from commerce.packages.contracts.errors import FeatureNotImplemented
from commerce.packages.contracts.ports import RecommenderRuntime
from commerce.packages.contracts.types import (
    ComparisonResult, ModelVariant, Payload, PersonalizationResult,
    ServingMode, TensorMap, TrainingResult,
)


class UnimplementedRuntime:
    def __init__(self, seller_id: str, feature_db_path: Path, model_dir: Path):
        self._seller_id = seller_id
        self.feature_db_path = feature_db_path
        self.model_dir = model_dir

    @property
    def seller_id(self) -> str:
        return self._seller_id

    def ingest_purchase_event(self, event: Payload) -> None:
        raise FeatureNotImplemented("B: durable purchase ingestion")

    def upsert_catalog_item(self, item: Payload, source_seq: int) -> None:
        raise FeatureNotImplemented("B: durable catalog ingestion")

    def predict_local(self, request: Payload, *, model_variant: ModelVariant = "text_relation", mode: ServingMode = "auto") -> Payload:
        raise FeatureNotImplemented("B: recommendation")

    def get_local_data_ref(self) -> str:
        raise FeatureNotImplemented("B: local data snapshot")

    def get_shared_manifest(self, *, model_variant: ModelVariant = "text_relation") -> Payload:
        raise FeatureNotImplemented("B: shared manifest")

    def export_shared_state(self, *, model_variant: ModelVariant = "text_relation") -> TensorMap:
        raise FeatureNotImplemented("B: shared base export")

    def train_round(self, local_data_ref: str, round_config: Payload, *, model_variant: ModelVariant = "text_relation") -> TrainingResult:
        raise FeatureNotImplemented("B: round training")

    def install_release(self, release: Payload, manifest: Payload, tensors: TensorMap, *, model_variant: ModelVariant = "text_relation") -> None:
        raise FeatureNotImplemented("B: verified model installation")

    def personalize_local(self, local_data_ref: str, personal_config: Payload, *, model_variant: ModelVariant = "text_relation") -> PersonalizationResult:
        raise FeatureNotImplemented("B: local personalization")

    def compare_local(self, request: Payload) -> ComparisonResult:
        raise FeatureNotImplemented("B: comparison snapshot")


def open_runtime(seller_id: str, feature_db_path: str | Path, model_dir: str | Path) -> RecommenderRuntime:
    if not seller_id:
        raise ValueError("seller_id is required")
    return UnimplementedRuntime(seller_id, Path(feature_db_path), Path(model_dir))
