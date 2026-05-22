"""Prior-work graph-FL design profiles.

These specs describe how representative graph-based FL papers actually use
client representations, relation estimators, graph topology, and aggregation.
They are not all exact implementations in this repository; the support level
states whether a profile is runnable, a diagnostic proxy, or an interface
target for future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from graphfl_lab.graph.sources.config import normalize_key


@dataclass(frozen=True)
class GraphFLMethodSpec:
    name: str
    paper_label: str
    client_state: str
    relation_estimator: str
    topology_operator: str
    aggregation_operator: str
    personalization_site: str
    local_objective_hook: str
    support_level: str
    config_overrides: Mapping[str, object]
    evidence: Tuple[str, ...]
    design_name: str = ""
    approximation_note: str = ""


_SPECS: Dict[str, GraphFLMethodSpec] = {
    "fedamp": GraphFLMethodSpec(
        name="fedamp",
        paper_label="FedAMP / attentive message passing",
        client_state="personalized model parameters",
        relation_estimator="distance-based attentive kernel over client model parameters",
        topology_operator="dense pairwise attention, client-specific cloud model",
        aggregation_operator="server-side attentive message passing followed by client proximal local update",
        personalization_site="server sends a personalized cloud model to each client",
        local_objective_hook="proximal training toward the personalized cloud model",
        support_level="proxy-supported",
        config_overrides={
            "graph_source": "weight",
            "graph_mode": "rbf",
            "aggregation_target": "graph_filtered_weight",
            "correction_family": "real_graph",
        },
        evidence=(
            "AAAI/CiNii summary: FedAMP facilitates pairwise collaborations between similar clients.",
            "FedAMP method summaries describe server-side attentive aggregate over model parameters and local proximal updates.",
        ),
        design_name="fedamp_proxy",
        approximation_note=(
            "Current strategy has one global model, so this is a relation/weight-smoothing proxy, "
            "not an exact personalized cloud-model reproduction."
        ),
    ),
    "sfl": GraphFLMethodSpec(
        name="sfl",
        paper_label="SFL / Personalized Federated Learning With a Graph",
        client_state="local model parameters plus client-wise relation graph",
        relation_estimator="predefined or learned client relation graph",
        topology_operator="GCN-style graph structural aggregation",
        aggregation_operator="server GCN generates client-specific personalized models",
        personalization_site="server-side personalized model generation",
        local_objective_hook="joint global/personalized optimization",
        support_level="proxy-supported",
        config_overrides={
            "graph_source": "weight",
            "graph_mode": "learned_smooth",
            "aggregation_target": "graph_filtered_weight",
            "correction_family": "real_graph",
        },
        evidence=(
            "IJCAI abstract: SFL learns global and personalized models using client-wise relation graphs.",
            "Paper summary describes a GCN module on the server that aggregates local model parameters by graph A.",
        ),
        design_name="sfl_proxy",
        approximation_note=(
            "Exact SFL requires a server GCN aggregation operator; current config is only a graph-filter proxy."
        ),
    ),
    "pfedgraph": GraphFLMethodSpec(
        name="pfedgraph",
        paper_label="pFedGraph / inferred collaboration graph",
        client_state="model update relative to initial/global parameters and local dataset size",
        relation_estimator="pairwise model similarity plus data-size prior optimized on a simplex",
        topology_operator="row-stochastic collaboration graph, often dense directed",
        aggregation_operator="client-specific weighted aggregation of neighbor model parameters",
        personalization_site="server-side personalized neighbor mixture plus client-side local optimization",
        local_objective_hook="local loss assisted/regularized by aggregated cluster model",
        support_level="proxy-supported",
        config_overrides={
            "graph_source": "update",
            "graph_mode": "pfedgraph_qp",
            "aggregation_target": "graph_filtered_update",
            "correction_family": "real_graph",
        },
        evidence=(
            "PMLR abstract: graph is inferred from pairwise model similarity and dataset size at the server.",
            "Official PyTorch code computes model_i - global, solves row-wise simplex QP, then aggregates neighbor models by graph row.",
        ),
        design_name="pfedgraph_proxy",
        approximation_note=(
            "The registered graph mode can reproduce the collaboration weights as a symmetric diagnostic graph, "
            "but exact row-wise personalized model delivery is a future aggregation operator."
        ),
    ),
    "fedaga": GraphFLMethodSpec(
        name="fedaga",
        paper_label="FedAGA / adaptive graph-based aggregation",
        client_state="accumulated gradients over local training",
        relation_estimator="similarity from accumulated gradients with convergence/divergence criteria",
        topology_operator="dynamic graph topology in non-Euclidean relation space",
        aggregation_operator="adaptive graph-based personalized aggregation",
        personalization_site="server-side relation-aware aggregation",
        local_objective_hook="standard local training with accumulated-gradient upload",
        support_level="proxy-supported",
        config_overrides={
            "graph_source": "ema_update",
            "graph_mode": "magnitude_knn",
            "aggregation_target": "graph_filtered_ema_update",
            "correction_family": "real_graph",
            "client_update_ema_alpha": 0.9,
        },
        evidence=(
            "ScienceDirect/Macquarie summaries: FedAGA constructs dynamic graph topology from accumulated-gradient similarities.",
        ),
        design_name="ema_magnitude_knn_filtered",
        approximation_note=(
            "EMA update is a proxy for accumulated gradients; exact convergence/divergence criteria are not implemented."
        ),
    ),
    "fedpub": GraphFLMethodSpec(
        name="fedpub",
        paper_label="FED-PUB / personalized subgraph FL",
        client_state="functional embedding from local GNN response to proxy random graphs",
        relation_estimator="cosine similarity over functional embeddings, optional exponential normalization",
        topology_operator="row-normalized similarity matrix",
        aggregation_operator="global FedAvg plus client-specific weighted local model aggregation",
        personalization_site="server stores personalized model for each client; client learns sparse masks",
        local_objective_hook="local proximal/mask regularization against previous personalized model",
        support_level="interface-target",
        config_overrides={
            "graph_source": "functional_embedding",
            "graph_mode": "cosine_row_stochastic",
            "aggregation_target": "personalized_weight",
            "correction_family": "real_graph",
        },
        evidence=(
            "FED-PUB paper page: random graphs are used as proxy inputs to compute functional embeddings and similarity-weighted averaging.",
            "Official server code computes cosine similarity of functional embeddings and uses each row for personalized local model aggregation.",
        ),
        design_name="fedpub_interface_target",
        approximation_note=(
            "Requires a graph-source plugin that runs model forward on proxy graphs and a personalized aggregation target."
        ),
    ),
}


def graph_fl_method_names() -> list[str]:
    return sorted(_SPECS)


def get_graph_fl_method_spec(name: str) -> GraphFLMethodSpec:
    key = normalize_key(name)
    if key not in _SPECS:
        known = ", ".join(graph_fl_method_names())
        raise ValueError(f"Unknown graph FL method spec {name!r}. Known: {known}")
    return _SPECS[key]


def graph_fl_method_specs() -> Dict[str, GraphFLMethodSpec]:
    return dict(_SPECS)


__all__ = [
    "GraphFLMethodSpec",
    "get_graph_fl_method_spec",
    "graph_fl_method_names",
    "graph_fl_method_specs",
]
