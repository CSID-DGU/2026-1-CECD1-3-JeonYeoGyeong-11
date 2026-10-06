"""A's own success-path test double for B's runtime (development.md "대체하는 방법").

Not a shared fake: only commerce/services/merchant_api/tests/ uses this. It
exists so A can exercise the durable-delivery paths in orders_service before B
implements the real runtime. It must never be imported by non-test A code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented
from commerce.packages.contracts.types import (
    ComparisonResult, ModelVariant, Payload, PersonalizationResult, ServingMode, TensorMap, TrainingResult,
)


class FakeRecommenderRuntime:
    """In-memory runtime double honoring interfaces.md §2-3 dedup rules only."""

    def __init__(self, seller_id: str, feature_db_path: Path | None = None, model_dir: Path | None = None):
        self._seller_id = seller_id
        self.ingested_events: dict[str, Payload] = {}
        self.catalog: dict[str, tuple[int, Payload]] = {}
        self.ingest_calls = 0
        self.catalog_calls = 0

    @property
    def seller_id(self) -> str:
        return self._seller_id

    def ingest_purchase_event(self, event: Payload) -> None:
        self.ingest_calls += 1
        event_id = event["purchase_event_id"]
        prior = self.ingested_events.get(event_id)
        if prior is not None:
            if prior == event:
                return  # identical redelivery succeeds (interfaces.md §2 step 5)
            raise ContractError("DUPLICATE_EVENT", "/purchase_event_id")
        self.ingested_events[event_id] = event

    def upsert_catalog_item(self, item: Payload, source_seq: int) -> None:
        self.catalog_calls += 1
        item_id = item["item_id_local"]
        prior = self.catalog.get(item_id)
        if prior is not None:
            prior_seq, prior_item = prior
            if source_seq <= prior_seq:
                return  # stale/out-of-order delivery ignored (interfaces.md §3)
        self.catalog[item_id] = (source_seq, item)

    def get_local_data_ref(self) -> str:
        raise FeatureNotImplemented("fake: local data snapshot")

    def predict_local(self, request: Payload, *, model_variant: ModelVariant = "text_relation", mode: ServingMode = "auto") -> Payload:
        raise FeatureNotImplemented("fake: recommendation")

    def get_shared_manifest(self, *, model_variant: ModelVariant = "text_relation") -> Payload:
        raise FeatureNotImplemented("fake: shared manifest")

    def export_shared_state(self, *, model_variant: ModelVariant = "text_relation") -> TensorMap:
        raise FeatureNotImplemented("fake: shared base export")

    def train_round(self, local_data_ref: str, round_config: Payload, *, model_variant: ModelVariant = "text_relation") -> TrainingResult:
        raise FeatureNotImplemented("fake: round training")

    def install_release(self, release: Payload, manifest: Payload, tensors: TensorMap, *, model_variant: ModelVariant = "text_relation") -> None:
        raise FeatureNotImplemented("fake: model installation")

    def personalize_local(self, local_data_ref: str, personal_config: Payload, *, model_variant: ModelVariant = "text_relation") -> PersonalizationResult:
        raise FeatureNotImplemented("fake: local personalization")

    def compare_local(self, request: Payload) -> ComparisonResult:
        raise FeatureNotImplemented("fake: comparison snapshot")


class ScriptedRuntime(FakeRecommenderRuntime):
    """Answers predict_local/compare_local with whatever a test sets, or raises it.

    `recommendation`/`comparison` may be a value to return or an exception to
    raise; left as None they keep the parent's FeatureNotImplemented.
    """

    def __init__(self, seller_id: str, *args, **kwargs):
        super().__init__(seller_id, *args, **kwargs)
        self.recommendation: Any = None
        self.comparison: Any = None
        self.requests: list[Payload] = []

    def predict_local(self, request: Payload, *, model_variant: ModelVariant = "text_relation", mode: ServingMode = "auto") -> Payload:
        self.requests.append(request)
        if self.recommendation is None:
            return super().predict_local(request)
        if isinstance(self.recommendation, BaseException):
            raise self.recommendation
        return self.recommendation

    def compare_local(self, request: Payload) -> ComparisonResult:
        self.requests.append(request)
        if self.comparison is None:
            return super().compare_local(request)
        if isinstance(self.comparison, BaseException):
            raise self.comparison
        return self.comparison


class AlwaysFailingRuntime(FakeRecommenderRuntime):
    """Simulates B being unreachable: every call raises, nothing is ever delivered."""

    def ingest_purchase_event(self, event: Payload) -> None:
        raise RuntimeError("synthetic: B unreachable")

    def upsert_catalog_item(self, item: Payload, source_seq: int) -> None:
        raise RuntimeError("synthetic: B unreachable")
