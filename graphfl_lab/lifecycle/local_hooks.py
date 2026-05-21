"""Local objective hook contracts and lightweight configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .modules import ModuleResult
from .traces import TraceRecord


@dataclass(frozen=True)
class LocalHookContext:
    server_round: int
    delivered_model: Any | None = None


class NoneLocalObjectiveHook:
    def run(self, context: LocalHookContext) -> ModuleResult:
        trace = TraceRecord(
            phase="local_objective",
            module="local_hook",
            name="none",
            round=context.server_round,
            values={
                "status": "ok",
                "support_level": "core-supported",
                "local_objective_hook": "none",
            },
        )
        return ModuleResult.ok(output={"fit_config": {"local_objective_hook": "none"}}, trace_records=trace)


class ProximalToDeliveredModelHook:
    def __init__(self, *, mu: float = 0.01) -> None:
        self.mu = float(mu)

    def run(self, context: LocalHookContext) -> ModuleResult:
        trace = TraceRecord(
            phase="local_objective",
            module="local_hook",
            name="proximal_to_delivered_model",
            round=context.server_round,
            values={
                "status": "ok",
                "support_level": "proxy-supported",
                "local_objective_hook": "proximal_to_delivered_model",
                "proximal_mu": self.mu,
                "requires_client_support": True,
            },
        )
        return ModuleResult.ok(
            output={
                "fit_config": {
                    "local_objective_hook": "proximal_to_delivered_model",
                    "proximal_mu": self.mu,
                }
            },
            support_level="proxy-supported",
            trace_records=trace,
        )


class InterfaceTargetLocalObjectiveHook:
    def __init__(self, name: str) -> None:
        self.name = str(name)

    def run(self, context: LocalHookContext) -> ModuleResult:
        reason = f"local objective hook {self.name!r} is an interface target"
        trace = TraceRecord(
            phase="local_objective",
            module="local_hook",
            name=self.name,
            round=context.server_round,
            values={
                "status": "unsupported",
                "support_level": "interface-target",
                "local_objective_hook": self.name,
                "reason": reason,
            },
        )
        return ModuleResult.unsupported(
            support_level="interface-target",
            message=reason,
            trace_records=trace,
        )


__all__ = [
    "InterfaceTargetLocalObjectiveHook",
    "LocalHookContext",
    "NoneLocalObjectiveHook",
    "ProximalToDeliveredModelHook",
]
