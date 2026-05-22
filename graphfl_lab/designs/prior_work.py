"""Prior-work design support metadata."""

from __future__ import annotations


PRIOR_WORK_DESIGN_ALIASES = {
    "FedAMP": "fedamp_proxy",
    "pFedGraph": "pfedgraph_proxy",
    "SFL": "sfl_proxy",
    "FedAGA": "ema_magnitude_knn_filtered",
}

INTERFACE_TARGET_DESIGNS = {
    "FED-PUB": {
        "support_level": "interface-target",
        "reason": "Requires functional embeddings, mask regularization, and personalized model delivery.",
    },
    "pFedGAT": {
        "support_level": "interface-target",
        "reason": "Requires learned attention graph modules outside the current executable core.",
    },
}


def prior_work_design_aliases() -> dict[str, str]:
    return dict(PRIOR_WORK_DESIGN_ALIASES)


def interface_target_designs() -> dict[str, dict[str, str]]:
    return {name: dict(meta) for name, meta in INTERFACE_TARGET_DESIGNS.items()}


__all__ = [
    "INTERFACE_TARGET_DESIGNS",
    "PRIOR_WORK_DESIGN_ALIASES",
    "interface_target_designs",
    "prior_work_design_aliases",
]
