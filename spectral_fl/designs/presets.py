"""Built-in graph-FL design presets."""

from __future__ import annotations

from typing import Any, Mapping

from .design import ComponentSpec, GraphFLDesign


def _state(
    name: str,
    *,
    graph_source: str,
    output_kind: str = "vectors",
    support_level: str = "core-supported",
    params: Mapping[str, Any] | None = None,
) -> ComponentSpec:
    merged = {"graph_source": graph_source}
    if params:
        merged.update(params)
    return ComponentSpec(
        kind="ClientStateExtractor",
        name=name,
        params=merged,
        support_level=support_level,
        output_kind=output_kind,
        trace_keys=("state_norm_mean", "state_norm_min", "state_norm_max", "source_used"),
    )


def _relation(
    name: str,
    *,
    support_level: str = "core-supported",
    params: Mapping[str, Any] | None = None,
    is_learned: bool = False,
) -> ComponentSpec:
    return ComponentSpec(
        kind="RelationEstimator",
        name=name,
        params={} if params is None else params,
        support_level=support_level,
        is_learned=is_learned,
        input_kind=("client_state",),
        output_kind="relation_matrix",
        trace_keys=("relation_score_mean", "relation_score_std", "relation_entropy"),
    )


def _topology(
    name: str,
    *,
    graph_mode: str,
    support_level: str = "core-supported",
    params: Mapping[str, Any] | None = None,
    is_learned: bool = False,
) -> ComponentSpec:
    merged = {"graph_mode": graph_mode}
    if params:
        merged.update(params)
    return ComponentSpec(
        kind="TopologyOperator",
        name=name,
        params=merged,
        support_level=support_level,
        is_learned=is_learned,
        input_kind=("relation_matrix",),
        output_kind="adjacency",
        trace_keys=("graph_density", "graph_entropy", "degree_mean", "row_entropy_mean"),
    )


def _aggregation(
    name: str,
    *,
    aggregation_target: str,
    support_level: str = "core-supported",
    params: Mapping[str, Any] | None = None,
) -> ComponentSpec:
    merged = {
        "aggregation_target": aggregation_target,
        "correction_family": "real_graph",
    }
    if params:
        merged.update(params)
    return ComponentSpec(
        kind="AggregationOperator",
        name=name,
        params=merged,
        support_level=support_level,
        input_kind=("topology", "local_updates"),
        output_kind="global_update",
        trace_keys=("alpha_entropy", "alpha_matrix_entropy", "di_pre", "di_post"),
    )


def _design(
    name: str,
    *,
    client_state: ComponentSpec,
    relation: ComponentSpec,
    topology: ComponentSpec,
    aggregation: ComponentSpec,
    support_level: str = "core-supported",
    tags: tuple[str, ...] = (),
    description: str = "",
    references: tuple[str, ...] = (),
) -> GraphFLDesign:
    return GraphFLDesign(
        name=name,
        client_state=client_state,
        relation=relation,
        topology=topology,
        aggregation=aggregation,
        support_level=support_level,
        tags=tags,
        description=description,
        references=references,
    )


def builtin_designs() -> dict[str, GraphFLDesign]:
    designs = [
        _design(
            "default_similarity_knn",
            client_state=_state(
                "model_update",
                graph_source="update",
                params={"graph_method": "default_similarity_knn"},
            ),
            relation=_relation(
                "rbf",
                params={"sigma": "median_pairwise_update_distance"},
            ),
            topology=_topology(
                "rbf_knn",
                graph_mode="rbf_knn",
                params={"knn_k": 2, "graph_scale_sigma": 0.0},
            ),
            aggregation=_aggregation(
                "graph_filtered_update",
                aggregation_target="graph_filtered_update",
            ),
            tags=("core", "default", "similarity-graph"),
            references=("pFedGraph", "FedSim"),
            description=(
                "Representative default graph-FL preset: RBF similarity over "
                "client updates, kNN topology, and "
                "graph-filtered update aggregation."
            ),
        ),
        _design(
            "head_knn_filtered_update",
            client_state=_state("classifier_head_update", graph_source="classifier_head_update"),
            relation=_relation("cosine"),
            topology=_topology("knn", graph_mode="knn", params={"knn_k": 2}),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            tags=("core", "spectral"),
            description="Classifier-head update similarity with kNN graph-filtered update aggregation.",
        ),
        _design(
            "raw_update_positive_dense",
            client_state=_state("update", graph_source="update"),
            relation=_relation("positive_cosine"),
            topology=_topology("dense", graph_mode="dense", params={"knn_k": 2}),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            tags=("compat", "baseline"),
            description="Compatibility preset for dense positive update graph construction.",
        ),
        _design(
            "raw_update_positive_knn",
            client_state=_state("update", graph_source="update"),
            relation=_relation("positive_cosine"),
            topology=_topology("knn", graph_mode="knn", params={"knn_k": 2}),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            tags=("compat", "baseline"),
            description="Compatibility preset for kNN positive update graph construction.",
        ),
        _design(
            "signed_conflict_knn",
            client_state=_state("update", graph_source="update"),
            relation=_relation("signed_conflict"),
            topology=_topology("signed_abs_knn", graph_mode="signed_abs_knn", params={"knn_k": 2}),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            tags=("compat", "baseline"),
            description="Compatibility preset for signed update-conflict kNN topology.",
        ),
        _design(
            "pfedsim_like",
            client_state=_state("classifier_head_weight", graph_source="classifier_head_weight"),
            relation=_relation("signed_head_similarity"),
            topology=_topology("signed_abs_knn", graph_mode="signed_abs_knn", params={"knn_k": 2}),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            tags=("compat", "prior-work-proxy"),
            description="pFedSim-style head-weight relation proxy.",
        ),
        _design(
            "gfedfilt_like",
            client_state=_state("weight", graph_source="weight"),
            relation=_relation("rbf"),
            topology=_topology(
                "rbf_knn",
                graph_mode="rbf_knn",
                params={
                    "knn_k": 2,
                    "graph_laplacian_type": "normalized",
                    "graph_smoothing_operator": "laplacian",
                },
            ),
            aggregation=_aggregation("spectral_filtered_weight", aggregation_target="spectral_filtered_weight"),
            tags=("compat", "prior-work-proxy"),
            description="Graph-filtered weight smoothing proxy.",
        ),
        _design(
            "ema_magnitude_knn_filtered",
            client_state=_state(
                "ema_update",
                graph_source="ema_update",
                params={"client_update_ema_alpha": 0.9, "graph_method": "fedaga"},
            ),
            relation=_relation("norm_aware_cosine", support_level="proxy-supported"),
            topology=_topology("magnitude_knn", graph_mode="magnitude_knn", params={"knn_k": 2}),
            aggregation=_aggregation(
                "spectral_filtered_ema_update",
                aggregation_target="spectral_filtered_ema_update",
            ),
            support_level="proxy-supported",
            tags=("prior-work-proxy", "fedaga"),
            description="FedAGA-inspired EMA update and magnitude-aware kNN proxy.",
        ),
        _design(
            "pfedgraph_proxy",
            client_state=_state("update", graph_source="update", params={"graph_method": "pfedgraph"}),
            relation=_relation("qp_collaboration", support_level="proxy-supported", params={"uses_sample_prior": True}),
            topology=_topology("row_collaboration_proxy", graph_mode="pfedgraph_qp", support_level="proxy-supported"),
            aggregation=_aggregation("spectral_filtered_update", aggregation_target="spectral_filtered_update"),
            support_level="proxy-supported",
            tags=("prior-work-proxy", "pfedgraph"),
            references=("pFedGraph",),
            description="pFedGraph-inspired collaboration graph proxy, not exact personalized delivery.",
        ),
        _design(
            "fedamp_proxy",
            client_state=_state("weight", graph_source="weight", params={"graph_method": "fedamp"}),
            relation=_relation("rbf", support_level="proxy-supported"),
            topology=_topology("dense_attention_proxy", graph_mode="rbf", support_level="proxy-supported"),
            aggregation=_aggregation("spectral_filtered_weight", aggregation_target="spectral_filtered_weight"),
            support_level="proxy-supported",
            tags=("prior-work-proxy", "fedamp"),
            references=("FedAMP",),
            description="FedAMP-style model-distance weighting proxy, not exact cloud-model delivery.",
        ),
        _design(
            "sfl_proxy",
            client_state=_state("weight", graph_source="weight", params={"graph_method": "sfl"}),
            relation=_relation("learned_smooth_proxy", support_level="proxy-supported", is_learned=True),
            topology=_topology(
                "learned_smooth_proxy",
                graph_mode="learned_smooth",
                support_level="proxy-supported",
                is_learned=True,
            ),
            aggregation=_aggregation("spectral_filtered_weight", aggregation_target="spectral_filtered_weight"),
            support_level="proxy-supported",
            tags=("prior-work-proxy", "sfl"),
            references=("SFL",),
            description="SFL-style learned graph proxy without server GCN personalized generation.",
        ),
        _design(
            "graphfree_dominance_reweight",
            client_state=_state("update", graph_source="update"),
            relation=_relation("none", support_level="proxy-supported"),
            topology=_topology("graph_free", graph_mode="uniform", support_level="proxy-supported"),
            aggregation=_aggregation(
                "dominance_reweight",
                aggregation_target="spectral_filtered_update",
                support_level="proxy-supported",
                params={"correction_family": "graph_free", "graph_free_mode": "dominance_reweight"},
            ),
            support_level="proxy-supported",
            tags=("control", "graph-free"),
            description="Graph-free dominance reweighting control design.",
        ),
    ]
    return {design.name: design for design in designs}


def builtin_aliases() -> dict[str, str]:
    return {
        "default_graph": "default_similarity_knn",
        "representative_graph": "default_similarity_knn",
        "similarity_knn": "default_similarity_knn",
        "fedaga_like": "ema_magnitude_knn_filtered",
        "fedamp_like": "fedamp_proxy",
        "pfedgraph_like": "pfedgraph_proxy",
        "sfl_like": "sfl_proxy",
    }


__all__ = ["builtin_aliases", "builtin_designs"]
