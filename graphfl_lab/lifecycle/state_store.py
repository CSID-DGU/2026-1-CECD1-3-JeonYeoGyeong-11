"""State storage for lifecycle modules across rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .traces import TraceRecord


@dataclass
class StateStore:
    ema_updates: Any | None = None
    ema_graph: np.ndarray | None = None
    previous_relation: np.ndarray | None = None
    personalized_models: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_ema_updates(self, updates: Sequence[Any], *, alpha: float = 0.9) -> Any:
        current = [np.asarray(update, dtype=np.float64).copy() for update in updates]
        if self.ema_updates is None:
            self.ema_updates = current
        else:
            beta = float(alpha)
            self.ema_updates = [
                beta * np.asarray(prev, dtype=np.float64) + (1.0 - beta) * np.asarray(now, dtype=np.float64)
                for prev, now in zip(self.ema_updates, current)
            ]
        return self.ema_updates

    def set_ema_graph(self, graph: np.ndarray) -> None:
        self.ema_graph = np.asarray(graph, dtype=np.float64).copy()

    def set_previous_relation(self, relation: np.ndarray) -> None:
        self.previous_relation = np.asarray(relation, dtype=np.float64).copy()

    def set_personalized_model(self, cid: str, model: Any) -> None:
        self.personalized_models[str(cid)] = model

    def get_personalized_model(self, cid: str) -> Any | None:
        return self.personalized_models.get(str(cid))

    def trace_record(self, *, round_number: int | None = None) -> TraceRecord:
        ema_norm = 0.0
        if self.ema_updates is not None:
            ema_norm = float(np.mean([np.linalg.norm(np.asarray(update, dtype=np.float64)) for update in self.ema_updates]))
        return TraceRecord(
            phase="state_store",
            module="state_store",
            name="round_state",
            round=round_number,
            values={
                "status": "ok",
                "support_level": "core-supported",
                "ema_updates_available": self.ema_updates is not None,
                "ema_update_norm": ema_norm,
                "ema_graph_available": self.ema_graph is not None,
                "previous_relation_available": self.previous_relation is not None,
                "personalized_model_count": len(self.personalized_models),
            },
        )


def state_store_from_mapping(mapping: Mapping[str, Any] | None) -> StateStore:
    if isinstance(mapping, StateStore):
        return mapping
    data = {} if mapping is None else dict(mapping)
    store = StateStore()
    store.ema_updates = data.get("ema_updates")
    store.ema_graph = data.get("ema_graph")
    store.previous_relation = data.get("previous_relation")
    store.personalized_models = dict(data.get("personalized_models", {}))
    store.metadata = dict(data.get("metadata", {}))
    return store


__all__ = ["StateStore", "state_store_from_mapping"]
