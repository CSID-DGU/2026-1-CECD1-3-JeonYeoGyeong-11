"""Client-state extraction adapters for lifecycle graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .context import StateExtractionContext
from .modules import ModuleResult
from .traces import TraceRecord


@dataclass(frozen=True)
class ClientStatePayload:
    vectors: tuple[np.ndarray, ...] | None = None
    tensors: tuple[Mapping[str, np.ndarray], ...] | None = None
    scalar_features: np.ndarray | None = None
    pairwise_ready: Any | None = None


@dataclass(frozen=True)
class ClientStateOutput:
    state_kind: str
    payload: ClientStatePayload
    per_client_meta: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    source_used: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_kind", str(self.state_kind))
        object.__setattr__(self, "source_used", str(self.source_used))
        object.__setattr__(self, "per_client_meta", tuple(dict(meta) for meta in self.per_client_meta))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def vector_matrix(self) -> np.ndarray:
        if self.payload.vectors is None:
            raise ValueError(f"state_kind={self.state_kind!r} does not expose flat vectors")
        if not self.payload.vectors:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack([np.asarray(vector, dtype=np.float64).reshape(1, -1) for vector in self.payload.vectors])


def _state_kind_for_source(source: str) -> str:
    key = str(source).lower()
    if "classifier_head" in key or "head" in key:
        return "classifier_head_update" if "update" in key or "delta" in key else "classifier_head"
    if "ema" in key or "momentum" in key:
        return "ema_update"
    if "weight" in key:
        return "weights"
    if "gradient" in key:
        return "pseudo_gradients"
    if "update" in key or "delta" in key:
        return "updates"
    return "hybrid"


def _vector_stats(vectors: Sequence[np.ndarray]) -> dict[str, Any]:
    if not vectors:
        return {
            "state_norm_mean": 0.0,
            "state_norm_min": 0.0,
            "state_norm_max": 0.0,
            "state_cosine_mean": 0.0,
            "state_cosine_std": 0.0,
        }
    matrix = np.vstack([np.asarray(vector, dtype=np.float64).reshape(1, -1) for vector in vectors])
    norms = np.linalg.norm(matrix, axis=1)
    safe = matrix / np.maximum(norms[:, None], 1e-12)
    cosine = safe @ safe.T
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return {
        "state_norm_mean": float(np.mean(norms)),
        "state_norm_min": float(np.min(norms)),
        "state_norm_max": float(np.max(norms)),
        "state_cosine_mean": float(np.mean(upper)) if upper.size else 0.0,
        "state_cosine_std": float(np.std(upper)) if upper.size else 0.0,
    }


def _client_meta(context: StateExtractionContext) -> tuple[Mapping[str, Any], ...]:
    metas: list[dict[str, Any]] = []
    for index, cid in enumerate(context.round_context.cids):
        meta = {
            "cid": cid,
            "num_examples": int(context.num_examples[index]) if index < len(context.num_examples) else 0,
        }
        if index < len(context.client_metrics):
            meta.update(dict(context.client_metrics[index]))
        metas.append(meta)
    return tuple(metas)


class GraphSourceClientStateExtractor:
    """Adapter around the existing spectral graph-source vector extractor."""

    def __init__(self, source: str = "update", *, layer_start: int = 0, layer_end: int = 0) -> None:
        self.source = str(source)
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)

    def run(self, context: StateExtractionContext) -> ModuleResult:
        from spectral_fl.graph.sources import GraphSourceConfig, graph_vectors_for_spectral

        try:
            ema_updates = context.round_context.config.get("ema_updates")
            vectors, source_used = graph_vectors_for_spectral(
                local_weights=list(context.local_weights),
                local_updates=list(context.local_updates),
                ema_updates=ema_updates,
                config=GraphSourceConfig(
                    source=self.source,
                    layer_start=self.layer_start,
                    layer_end=self.layer_end,
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised through error result contract
            return ModuleResult.error(exc, support_level="core-supported")

        tuple_vectors = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
        state_kind = _state_kind_for_source(source_used)
        output = ClientStateOutput(
            state_kind=state_kind,
            payload=ClientStatePayload(vectors=tuple_vectors),
            per_client_meta=_client_meta(context),
            source_used=source_used,
            metadata={
                "requested_source": self.source,
                "layer_start": self.layer_start,
                "layer_end": self.layer_end,
            },
        )
        trace_values = {
            "status": "ok",
            "support_level": "core-supported",
            "component_kind": "ClientStateExtractor",
            "component_name": self.source,
            "output_kind": state_kind,
            "source_used": source_used,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
        }
        trace_values.update(_vector_stats(tuple_vectors))
        trace = TraceRecord(
            phase="client_state",
            module="graph_source",
            name=self.source,
            round=context.round_context.server_round,
            values=trace_values,
        )
        return ModuleResult.ok(output=output, trace_records=trace)


class UnsupportedClientStateExtractor:
    def __init__(self, name: str, *, support_level: str = "interface-target", reason: str = "") -> None:
        self.name = str(name)
        self.support_level = str(support_level)
        self.reason = reason or f"client state {self.name!r} is not executable yet"

    def run(self, context: StateExtractionContext) -> ModuleResult:
        trace = TraceRecord(
            phase="client_state",
            module="unsupported",
            name=self.name,
            round=context.round_context.server_round,
            values={
                "status": "unsupported",
                "support_level": self.support_level,
                "component_kind": "ClientStateExtractor",
                "component_name": self.name,
                "reason": self.reason,
            },
        )
        return ModuleResult.unsupported(
            support_level=self.support_level,
            message=self.reason,
            trace_records=trace,
        )


__all__ = [
    "ClientStateOutput",
    "ClientStatePayload",
    "GraphSourceClientStateExtractor",
    "UnsupportedClientStateExtractor",
]
