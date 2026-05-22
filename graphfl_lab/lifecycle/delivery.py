"""Delivery policies for lifecycle graph-FL execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .modules import ModuleResult
from .state_store import StateStore
from .traces import TraceRecord


class MissingPersonalizedStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeliveryContext:
    server_round: int
    cids: Sequence[str]
    global_model: Any
    state_store: StateStore


class GlobalDeliveryPolicy:
    def run(self, context: DeliveryContext) -> ModuleResult:
        delivered = {str(cid): context.global_model for cid in context.cids}
        trace = TraceRecord(
            phase="delivery",
            module="delivery_policy",
            name="global",
            round=context.server_round,
            values={
                "status": "ok",
                "support_level": "core-supported",
                "delivery_policy": "global",
                "personalized_state_available": False,
            },
        )
        return ModuleResult.ok(output=delivered, trace_records=trace)


class PreviousPersonalizedDeliveryPolicy:
    def __init__(self, *, strict: bool = False) -> None:
        self.strict = bool(strict)

    def run(self, context: DeliveryContext) -> ModuleResult:
        delivered = {}
        cold_started = []
        for cid in context.cids:
            model = context.state_store.get_personalized_model(str(cid))
            if model is None:
                if self.strict:
                    raise MissingPersonalizedStateError(f"missing personalized model for cid={cid}")
                model = context.global_model
                cold_started.append(str(cid))
            delivered[str(cid)] = model
        trace = TraceRecord(
            phase="delivery",
            module="delivery_policy",
            name="previous_personalized",
            round=context.server_round,
            values={
                "status": "ok",
                "support_level": "proxy-supported",
                "delivery_policy": "previous_personalized",
                "delivery_cold_start": "global_with_trace" if cold_started else "none",
                "personalized_state_available": len(cold_started) == 0,
                "cold_start_cids": cold_started,
            },
        )
        return ModuleResult.ok(output=delivered, support_level="proxy-supported", trace_records=trace)


class InterfaceTargetDeliveryPolicy:
    def __init__(self, name: str) -> None:
        self.name = str(name)

    def run(self, context: DeliveryContext) -> ModuleResult:
        reason = f"delivery policy {self.name!r} is an interface target"
        trace = TraceRecord(
            phase="delivery",
            module="delivery_policy",
            name=self.name,
            round=context.server_round,
            values={
                "status": "unsupported",
                "support_level": "interface-target",
                "delivery_policy": self.name,
                "reason": reason,
            },
        )
        return ModuleResult.unsupported(
            support_level="interface-target",
            message=reason,
            trace_records=trace,
        )


__all__ = [
    "DeliveryContext",
    "GlobalDeliveryPolicy",
    "InterfaceTargetDeliveryPolicy",
    "MissingPersonalizedStateError",
    "PreviousPersonalizedDeliveryPolicy",
]
