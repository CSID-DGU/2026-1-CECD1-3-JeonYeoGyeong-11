"""Local Python return types. These are not coordinator JSON messages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

ModelVariant: TypeAlias = Literal["text_only", "text_relation"]
ServingMode: TypeAlias = Literal["global", "personalized", "auto"]
TensorMap: TypeAlias = dict[str, NDArray[np.float32]]
Payload: TypeAlias = dict[str, Any]
UnavailableReason: TypeAlias = Literal[
    "model_not_ready", "personalization_not_ready", "insufficient_data",
    "validation_rejected", "base_mismatch",
]


@dataclass(frozen=True)
class TrainingResult:
    shared_delta: TensorMap
    metrics: dict[str, float | None]
    completed: bool


@dataclass(frozen=True)
class PersonalizationResult:
    status: Literal["installed", "skipped", "rejected"]
    reason: str | None
    base_model_version: str
    personalization_revision: str | None


@dataclass(frozen=True)
class ComparisonArm:
    arm_id: Literal["T-G", "R-G", "T-P", "R-P"]
    model_variant: ModelVariant
    mode: Literal["global", "personalized"]
    available: bool
    unavailable_reason: UnavailableReason | None
    base_model_version: str | None
    personalization_revision: str | None
    recommendation: Payload | None


@dataclass(frozen=True)
class ComparisonResult:
    comparison_id: str
    as_of: str
    feature_snapshot_id: str
    candidate_set_hash: str
    arms: tuple[ComparisonArm, ...]
