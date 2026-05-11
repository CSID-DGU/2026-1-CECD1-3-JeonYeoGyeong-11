"""Typed diagnostic records for round/client artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RoundDiagnostics:
    run_id: str
    round: int
    di_pre: float
    di_post: float
    neff_pre: float
    neff_post: float
    align_mean_pre: float
    align_mean_post: float
    loo_mean_pre: float
    loo_mean_post: float
    alpha_entropy: float
    correction_family: str
    graph_variant: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClientRoundDiagnostics:
    run_id: str
    round: int
    cid: str
    num_examples: int
    update_norm_raw: float
    update_norm_corrected: float
    q_raw: float
    q_corrected: float
    alignment_raw: float
    alignment_corrected: float
    loo_raw: float
    loo_corrected: float
    cluster_id: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


__all__ = ["ClientRoundDiagnostics", "RoundDiagnostics"]
