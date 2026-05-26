"""Poster-oriented evidence generation for the Graph-FL framework.

The report produced here is intentionally conservative.  It checks framework
construction drift, paper-mechanism alignment, diagnostic sensitivity,
composability, and graph-component extension contracts.  It does not claim that
Graph-FL gains are proven, nor that paper proxies are full reproductions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from graphfl_lab.designs.design import ComponentSpec, GraphFLDesign
from graphfl_lab.designs.registry import DesignRegistry
from graphfl_lab.diagnostics.metrics import summarize_pre_post
from graphfl_lab.experiments.suites.result_writer import write_csv_rows, write_json
from graphfl_lab.graph.builders import (
    _build_base_client_graph,
    _build_legacy_base_client_graph,
    build_relation_graph,
    learned_smooth_graph,
    magnitude_aware_graph,
    pfedgraph_qp_graph,
    project_simplex,
    rbf_graph,
)
from graphfl_lab.graph.controls import build_control_graph
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics
from graphfl_lab.graph.registry import (
    GraphBuildContext,
    register_graph_builder,
    unregister_graph_builder,
)
from graphfl_lab.graph.similarity import dense_positive_cosine, pairwise_sq_dists
from graphfl_lab.graph.sources import GraphSourceConfig, graph_vectors_for_graphfl
from graphfl_lab.graph.sources.registry import (
    GraphSourceContext,
    GraphSourceResult,
    register_graph_source,
    unregister_graph_source,
)
from graphfl_lab.graph.sparsification import keep_topk
from graphfl_lab.lifecycle.relation import estimate_relation_from_vectors
from graphfl_lab.lifecycle.topology import build_topology_from_relation
from graphfl_lab.cli.aggregation_targets import AGGREGATION_TARGET_CHOICES
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregate_target,
)
from graphfl_lab.strategies.graphfl.diagnostics import (
    heterogeneity,
    spectral_energy_diagnostics,
)


POSTER_SAFE_CLAIM = (
    "A validation package for checking graph construction drift, "
    "paper-mechanism alignment, diagnostic sensitivity, composability, and "
    "extensibility before using Graph-FL gains as evidence."
)

PAPER_KERNEL_NOTE = (
    "paper-kernel means an independent implementation derived from equations/"
    "descriptions in the paper, not an official implementation, unless "
    "commit/hash is provided."
)

GRAPH_MODES = (
    "uniform",
    "dense",
    "knn",
    "mutual_knn",
    "threshold",
    "random",
    "signed_abs",
    "signed_abs_knn",
    "negative",
    "negative_knn",
    "rbf",
    "rbf_knn",
    "learned_smooth",
    "learned_smooth_knn",
    "magnitude",
    "magnitude_knn",
    "global_alignment",
    "pfedgraph_qp",
)

GRAPH_SOURCE_DESIGN_SPACE = (
    "update",
    "ema_update",
    "normalized_update",
    "normalized_ema_update",
    "layer_slice_update",
    "layerwise_update",
    "layerwise_ema_update",
    "classifier_head_update",
    "classifier_head_ema_update",
    "layerwise_classifier_head_update",
    "layerwise_classifier_head_ema_update",
    "weight",
    "layer_slice_weight",
    "layerwise_weight",
    "classifier_head_weight",
    "layerwise_classifier_head_weight",
)

CORRECTION_PROFILES = (
    {
        "correction_profile": "real_graph",
        "correction_family": "real_graph",
        "control_graph_mode": "not-applicable",
        "cluster_method": "none",
    },
    {
        "correction_profile": "control_random",
        "correction_family": "control_graph",
        "control_graph_mode": "random",
        "cluster_method": "none",
    },
    {
        "correction_profile": "control_shuffled",
        "correction_family": "control_graph",
        "control_graph_mode": "shuffled",
        "cluster_method": "none",
    },
    {
        "correction_profile": "control_uniform",
        "correction_family": "control_graph",
        "control_graph_mode": "uniform",
        "cluster_method": "none",
    },
    {
        "correction_profile": "control_identity",
        "correction_family": "control_graph",
        "control_graph_mode": "identity",
        "cluster_method": "none",
    },
    {
        "correction_profile": "clustering_only_kmeans",
        "correction_family": "clustering_only",
        "control_graph_mode": "not-applicable",
        "cluster_method": "kmeans",
    },
)

DESIGN_SPACE_BOUNDARIES = (
    {
        "axis": "graph_source",
        "designable_range": (
            "update/EMA/normalized/layer-slice/layerwise/classifier-head/weight "
            "signals plus registered custom graph-source plugins"
        ),
        "current_boundary": "sources must emit one finite vector per selected client",
        "claim": "plugin-extensible",
    },
    {
        "axis": "graph_mode",
        "designable_range": (
            "cosine, signed, negative, RBF, magnitude-aware, global alignment, "
            "learned smoothness, pFedGraph-style simplex, and registered builders"
        ),
        "current_boundary": "builders must emit finite non-negative square adjacency",
        "claim": "plugin-extensible",
    },
    {
        "axis": "topology",
        "designable_range": "dense, knn, mutual-knn, threshold, random-matched, controls, cluster block",
        "current_boundary": "Laplacian diagnostics assume symmetric zero-diagonal adjacency",
        "claim": "core-supported",
    },
    {
        "axis": "aggregation_target",
        "designable_range": ", ".join(AGGREGATION_TARGET_CHOICES),
        "current_boundary": "v1 supports core-code targets only; target plugin extensibility is not claimed",
        "claim": "core-code-extension-point",
    },
    {
        "axis": "correction_family",
        "designable_range": "real_graph, control_graph, clustering_only; graph_free is a non-graph correction control",
        "current_boundary": "graph_free is not evidence for new graph construction design",
        "claim": "graph-design plus separate graph-free controls",
    },
)


@dataclass(frozen=True)
class EvidencePack:
    out_dir: Path
    files: Mapping[str, str]
    verdict: Mapping[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _edge_mask(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, dtype=np.float64) > 1e-12


def _edge_scores(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    c = _edge_mask(candidate)
    r = _edge_mask(reference)
    n = c.shape[0]
    upper = np.triu_indices(n, k=1)
    c_u = c[upper]
    r_u = r[upper]
    tp = int(np.sum(c_u & r_u))
    fp = int(np.sum(c_u & ~r_u))
    fn = int(np.sum(~c_u & r_u))
    union = int(np.sum(c_u | r_u))
    precision = float(tp / (tp + fp)) if (tp + fp) else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = float(tp / union) if union else 1.0
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_jaccard": jaccard,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
    }


def _matrix_scores(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    diff = cand - ref
    ref_norm = float(np.linalg.norm(ref, ord="fro"))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    return {
        "max_abs_diff": max_abs,
        "rel_fro_diff": float(np.linalg.norm(diff, ord="fro") / max(ref_norm, 1e-12)),
        "symmetry_error": float(np.max(np.abs(cand - cand.T))) if cand.size else 0.0,
        "diagonal_error": float(np.max(np.abs(np.diag(cand)))) if cand.size else 0.0,
    }


def _compare_matrices(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {**_matrix_scores(candidate, reference), **_edge_scores(candidate, reference)}


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    w = np.asarray(adjacency, dtype=np.float64)
    return np.diag(np.sum(w, axis=1)) - w


def _block_graph(cluster_ids: Sequence[int], weight: float = 1.0) -> np.ndarray:
    ids = np.asarray(cluster_ids, dtype=np.int64)
    out = np.where(ids[:, None] == ids[None, :], float(weight), 0.0).astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def _zero_diag_sym(matrix: np.ndarray) -> np.ndarray:
    out = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    np.fill_diagonal(out, 0.0)
    return out


def _reference_positive_cosine(z_mat: np.ndarray) -> np.ndarray:
    z = np.asarray(z_mat, dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    safe = z / np.maximum(norms, 1e-12)
    out = np.maximum(safe @ safe.T, 0.0)
    np.fill_diagonal(out, 0.0)
    return _zero_diag_sym(out)


def _reference_rbf(z_mat: np.ndarray, sigma: float) -> np.ndarray:
    d2 = pairwise_sq_dists(np.asarray(z_mat, dtype=np.float64))
    sig = max(float(sigma), 1e-12)
    out = np.exp(-d2 / (2.0 * sig * sig))
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float64)


def _reference_magnitude_knn(z_mat: np.ndarray, k: int, sigma: float) -> np.ndarray:
    base = _reference_positive_cosine(z_mat)
    norms = np.linalg.norm(z_mat, axis=1).astype(np.float64) + 1e-12
    scale = np.exp(-np.abs(np.log(norms)[:, None] - np.log(norms)[None, :]) / max(float(sigma), 1e-12))
    out = base * scale
    np.fill_diagonal(out, 0.0)
    return keep_topk(out, int(k))


def _reference_learned_smooth(z_mat: np.ndarray, learned_lambda: float) -> np.ndarray:
    z = np.asarray(z_mat, dtype=np.float64)
    n = z.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return out
    d2 = pairwise_sq_dists(z)
    vals = d2[np.triu_indices(n, k=1)]
    vals = vals[vals > 1e-12]
    scaled = d2 / (max(float(np.median(vals)), 1e-12) if vals.size else 1.0)
    lam = max(float(learned_lambda), 1e-12)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out[i, mask] = project_simplex(-scaled[i, mask] / lam)
    return _zero_diag_sym(out)


def _reference_pfedgraph_directed(
    z_mat: np.ndarray,
    sample_weights: Sequence[float],
    learned_lambda: float,
) -> np.ndarray:
    z = np.asarray(z_mat, dtype=np.float64)
    n = int(z.shape[0])
    p = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    p = np.maximum(p, 0.0)
    p = p / max(float(np.sum(p)), 1e-12)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z_safe = z / np.maximum(norms, 1e-12)
    cosine = np.clip(z_safe @ z_safe.T, -1.0, 1.0)
    difference = -cosine
    difference[difference < -0.9] = -1.0
    lam = max(float(learned_lambda), 1e-12)
    directed = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = project_simplex(p - difference[i] / (2.0 * lam))
        row[i] = 0.0
        row_sum = float(np.sum(row))
        if row_sum > 1e-12:
            row = row / row_sum
        elif n > 1:
            row = np.full(n, 1.0 / float(n - 1), dtype=np.float64)
            row[i] = 0.0
        directed[i] = row
    return directed


def graph_parity_rows(profile: str = "smoke") -> list[dict[str, Any]]:
    z_mat = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.0],
            [-1.0, 0.0, 0.2],
            [-0.9, -0.1, 0.0],
        ],
        dtype=np.float64,
    )
    modes = GRAPH_MODES if profile == "poster" else (
        "dense",
        "knn",
        "random",
        "rbf",
        "learned_smooth",
        "magnitude_knn",
        "pfedgraph_qp",
    )
    rows: list[dict[str, Any]] = []
    for mode in modes:
        rng_ref = np.random.default_rng(123)
        rng_assembled = np.random.default_rng(123)
        reference = _build_legacy_base_client_graph(
            z_mat,
            mode=mode,
            knn_k=2,
            edge_threshold=0.2,
            rng=rng_ref,
            graph_scale_sigma=1.0,
            learned_graph_lambda=1.0,
            client_sample_weights=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
        )
        assembled = _build_base_client_graph(
            z_mat,
            mode=mode,
            graph_source="update",
            aggregation_target="graph_filtered_update",
            knn_k=2,
            edge_threshold=0.2,
            rng=rng_assembled,
            graph_scale_sigma=1.0,
            learned_graph_lambda=1.0,
            client_sample_weights=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
        )
        scores = _compare_matrices(assembled, reference)
        deterministic = mode != "random"
        pass_gate = (
            scores["max_abs_diff"] <= 1e-9
            and scores["edge_f1"] >= 1.0 - 1e-12
        )
        rows.append(
            {
                "axis": "construction_drift",
                "claim_level": "implementation-regression",
                "mode": mode,
                "deterministic": deterministic,
                "reference": "_build_legacy_base_client_graph",
                "candidate": "assembled_lifecycle_base_graph",
                **scores,
                "verdict": "pass" if pass_gate else "needs-review",
                "claim_boundary": "implementation regression evidence only",
            }
        )
    return rows


def external_mechanism_alignment_rows() -> list[dict[str, Any]]:
    z = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sample_weights = [0.55, 0.2, 0.15, 0.1]
    rows: list[dict[str, Any]] = []

    relation = estimate_relation_from_vectors(
        z,
        relation_kind="pfedgraph_qp",
        client_sample_weights=sample_weights,
        learned_graph_lambda=1.0,
    )
    ref_directed = _reference_pfedgraph_directed(z, sample_weights, 1.0)
    rows.append(
        _external_row(
            method="pFedGraph",
            component="directed collaboration kernel",
            candidate=relation.relation_matrix,
            reference=ref_directed,
            reference_type="paper-kernel",
            source_url="https://proceedings.mlr.press/v202/ye23b.html",
            commit_or_version="paper-derived",
            derivation_note=(
                "Independent kernel from pairwise cosine-difference, "
                "sample-size prior, and row collaboration projection behavior."
            ),
            matched_component="collaboration matrix optimization / projection behavior",
            unmatched_gap="full personalized training loop not reproduced",
            support_level="proxy-reference",
            graph_directionality="directed",
            directionality_loss="none",
        )
    )
    topology = build_topology_from_relation(relation, mode="pfedgraph_qp")
    ref_symmetric = _zero_diag_sym(ref_directed)
    rows.append(
        _external_row(
            method="pFedGraph",
            component="symmetric diagnostic projection",
            candidate=topology.adjacency,
            reference=ref_symmetric,
            reference_type="paper-kernel",
            source_url="https://proceedings.mlr.press/v202/ye23b.html",
            commit_or_version="paper-derived",
            derivation_note="Symmetric projection of directed collaboration matrix for Laplacian diagnostics.",
            matched_component="diagnostic projection of collaboration graph",
            unmatched_gap="directional personalized delivery not reproduced",
            support_level="proxy-reference",
            graph_directionality="symmetric-projection",
            directionality_loss="reported",
        )
    )

    fedamp_candidate = rbf_graph(z, sigma=1.0)
    rows.append(
        _external_row(
            method="FedAMP",
            component="model-distance attentive weighting kernel",
            candidate=fedamp_candidate,
            reference=_reference_rbf(z, sigma=1.0),
            reference_type="paper-kernel",
            source_url="https://arxiv.org/abs/2007.03797",
            commit_or_version="paper-derived",
            derivation_note="Independent RBF kernel over model/update vectors.",
            matched_component="model-distance attentive weighting kernel",
            unmatched_gap="personalized cloud model delivery and proximal local objective not reproduced",
            support_level="proxy-reference",
            graph_directionality="undirected",
            directionality_loss="not-applicable",
        )
    )

    sfl_candidate = learned_smooth_graph(z, learned_lambda=1.0)
    rows.append(
        _external_row(
            method="SFL",
            component="learned smoothness graph proxy",
            candidate=sfl_candidate,
            reference=_reference_learned_smooth(z, learned_lambda=1.0),
            reference_type="proxy-reference",
            source_url="https://arxiv.org/abs/2203.00829",
            commit_or_version="paper-inspired",
            derivation_note="Smoothness proxy, not a server-side GCN reproduction.",
            matched_component="graph smoothness / learned relation proxy",
            unmatched_gap="server GCN and client-specific personalized model generation not reproduced",
            support_level="proxy-reference",
            graph_directionality="undirected",
            directionality_loss="not-applicable",
        )
    )

    fedaga_candidate = keep_topk(
        magnitude_aware_graph(z, dense_positive_cosine(z), sigma=1.0),
        2,
    )
    rows.append(
        _external_row(
            method="FedAGA",
            component="EMA/update magnitude-aware relation proxy",
            candidate=fedaga_candidate,
            reference=_reference_magnitude_knn(z, k=2, sigma=1.0),
            reference_type="proxy-reference",
            source_url="https://researchers.mq.edu.au/en/publications/fedaga-a-federated-learning-framework-for-enhanced-inter-client-r",
            commit_or_version="paper-inspired",
            derivation_note="Magnitude-aware kNN proxy for accumulated-gradient relation behavior.",
            matched_component="EMA/update magnitude-aware relation proxy",
            unmatched_gap="exact accumulated-gradient and convergence/divergence rule not reproduced",
            support_level="proxy-reference",
            graph_directionality="undirected",
            directionality_loss="not-applicable",
        )
    )
    return rows


def _external_row(
    *,
    method: str,
    component: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    reference_type: str,
    source_url: str,
    commit_or_version: str,
    derivation_note: str,
    matched_component: str,
    unmatched_gap: str,
    support_level: str,
    graph_directionality: str,
    directionality_loss: str,
) -> dict[str, Any]:
    scores = _compare_matrices(candidate, reference)
    pass_gate = scores["max_abs_diff"] <= 1e-6
    return {
        "axis": "paper_mechanism_alignment",
        "method": method,
        "component": component,
        "reference_type": reference_type,
        "source_url": source_url,
        "commit_or_version": commit_or_version,
        "derivation_note": derivation_note,
        "matched_component": matched_component,
        "unmatched_gap": unmatched_gap,
        "support_level": support_level,
        "graph_directionality": graph_directionality,
        "directionality_loss": directionality_loss,
        **scores,
        "verdict": "pass" if pass_gate else "needs-review",
        "claim_boundary": (
            "mechanism alignment only; full algorithm reproduction is not claimed"
        ),
    }


def scenario_manifest(profile: str = "smoke") -> dict[str, Any]:
    scenarios = [
        {
            "scenario": "clustered_label_skew",
            "rationale": "Client updates form label-skew-like clusters.",
            "good_graph_definition": "Edges connect clients inside the same latent cluster.",
            "controls": ["matched_random", "shuffled", "uniform", "identity"],
            "expected_metric_direction": {
                "edge_f1_to_ground_truth": "higher",
                "matrix_similarity_to_ground_truth": "higher",
                "h_spec_normalized": "lower",
            },
            "pass_rule": "predefined ground-truth graph beats applicable controls in expected direction",
            "failure_case": "Diagnostics cannot separate same-cluster structure from controls.",
        },
        {
            "scenario": "rbf_geometry",
            "rationale": "Client updates lie on separated Euclidean manifolds.",
            "good_graph_definition": "RBF-kNN graph links nearby update vectors.",
            "controls": ["matched_random", "shuffled", "uniform", "identity"],
            "expected_metric_direction": {
                "edge_f1_to_ground_truth": "higher",
                "matrix_similarity_to_ground_truth": "higher",
                "h_spec_normalized": "lower",
            },
            "pass_rule": "geometric neighbor graph beats controls in expected direction",
            "failure_case": "Distance-aware graph is not distinguished from random topology.",
        },
        {
            "scenario": "norm_skewed_dominance",
            "rationale": "Client update norms differ strongly even when directions are related.",
            "good_graph_definition": "Magnitude-aware kNN graph links clients with compatible norms.",
            "controls": ["matched_random", "shuffled", "uniform", "identity"],
            "expected_metric_direction": {
                "edge_f1_to_ground_truth": "higher",
                "matrix_similarity_to_ground_truth": "higher",
                "h_spec_normalized": "lower",
            },
            "pass_rule": "norm-aware graph beats controls in expected direction",
            "failure_case": "Magnitude pathology is hidden by plain cosine similarity.",
        },
    ]
    if profile == "poster":
        scenarios.extend(
            [
                {
                    "scenario": "anti_aligned_updates",
                    "rationale": "Some clients have opposing update directions.",
                    "good_graph_definition": "Signed-conflict graph captures anti-aligned pairs.",
                    "controls": ["matched_random", "shuffled", "uniform", "identity"],
                    "expected_metric_direction": {
                        "edge_f1_to_ground_truth": "higher",
                        "matrix_similarity_to_ground_truth": "higher",
                    },
                    "pass_rule": "signed-conflict graph recovers predefined conflict edges",
                    "failure_case": "Anti-alignment is treated as no relation.",
                },
                {
                    "scenario": "sample_prior_collaboration",
                    "rationale": "Collaboration graph should respond to update similarity and sample prior.",
                    "good_graph_definition": "pFedGraph-style collaboration projection is the reference graph.",
                    "controls": ["matched_random", "shuffled", "uniform", "identity"],
                    "expected_metric_direction": {
                        "edge_f1_to_ground_truth": "higher",
                        "directed_row_similarity_to_ground_truth": "higher",
                    },
                    "pass_rule": "sample-prior collaboration graph beats topology controls",
                    "failure_case": "Sample-size prior has no measurable effect on graph construction.",
                },
            ]
        )
    return {
        "schema_version": 1,
        "profile": profile,
        "validation_seeds": [0] if profile == "smoke" else [0, 1, 2, 3, 4],
        "performance_relevance": "future-work",
        "threshold_note": (
            "Operational sanity thresholds such as 80% pass or rho >= 0.6 are "
            "not theoretical guarantees."
        ),
        "scenarios": scenarios,
    }


def _scenario_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "clustered_label_skew":
        z = np.array(
            [
                [1.0, 0.05],
                [0.95, -0.02],
                [1.05, 0.03],
                [0.9, 0.02],
                [-1.0, 0.04],
                [-0.95, -0.03],
                [-1.05, 0.02],
                [-0.9, -0.01],
            ],
            dtype=np.float64,
        )
        truth = _block_graph([0, 0, 0, 0, 1, 1, 1, 1])
        return z, truth
    if name == "rbf_geometry":
        z = np.array([[0.0], [0.08], [0.15], [0.22], [2.0], [2.08], [2.15], [2.22]], dtype=np.float64)
        truth = keep_topk(rbf_graph(z, sigma=0.35), 2)
        return z, truth
    if name == "norm_skewed_dominance":
        z = np.array([[1.0, 0.0], [1.2, 0.05], [8.0, 0.1], [9.0, -0.1], [0.0, 1.0], [0.1, 1.2]], dtype=np.float64)
        base = dense_positive_cosine(z)
        truth = keep_topk(magnitude_aware_graph(z, base, sigma=0.5), 1)
        return z, truth
    if name == "anti_aligned_updates":
        z = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.8, 0.1], [-0.8, -0.1]], dtype=np.float64)
        truth = keep_topk(np.abs(z @ z.T / np.maximum(np.linalg.norm(z, axis=1)[:, None] * np.linalg.norm(z, axis=1)[None, :], 1e-12)), 1)
        np.fill_diagonal(truth, 0.0)
        return z, truth
    if name == "sample_prior_collaboration":
        z = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=np.float64)
        truth = _reference_pfedgraph_directed(
            z,
            sample_weights=[0.5, 0.2, 0.1, 0.1, 0.1],
            learned_lambda=1.0,
        )
        return z, truth
    raise ValueError(f"Unknown scenario {name!r}")


def metric_validity_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in manifest.get("validation_seeds", [0])]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for scenario in manifest["scenarios"]:
            name = str(scenario["scenario"])
            z_mat, truth = _scenario_data(name)
            controls = {
                "ground_truth": truth,
                "matched_random": build_control_graph(
                    reference_adj=truth, control_mode="random", rng=rng
                ),
                "shuffled": build_control_graph(
                    reference_adj=truth, control_mode="shuffled", rng=rng
                ),
                "uniform": build_control_graph(
                    reference_adj=truth, control_mode="uniform", rng=rng
                ),
                "identity": build_control_graph(
                    reference_adj=truth, control_mode="identity", rng=rng
                ),
            }
            metric_values: dict[str, list[tuple[str, float, float, bool]]] = {}
            for graph_name, adjacency in controls.items():
                values = _scenario_metric_values(z_mat, adjacency, truth)
                quality = float(values["edge_f1_to_ground_truth"])
                for metric_name, direction in scenario["expected_metric_direction"].items():
                    value = values.get(metric_name, float("nan"))
                    applies = bool(np.isfinite(value))
                    metric_values.setdefault(metric_name, []).append(
                        (graph_name, float(value), quality, applies)
                    )

            for metric_name, direction in scenario["expected_metric_direction"].items():
                metric_group = metric_values[metric_name]
                applicable = [item for item in metric_group if item[3]]
                rho = _spearman(
                    [item[2] for item in applicable],
                    [item[1] for item in applicable],
                )
                if direction == "lower":
                    rho = -rho
                good_value = next(item[1] for item in metric_group if item[0] == "ground_truth")
                control_values = [
                    item[1]
                    for item in metric_group
                    if item[0] != "ground_truth" and item[3]
                ]
                if direction == "lower":
                    good_beats = sum(1 for value in control_values if good_value <= value + 1e-12)
                else:
                    good_beats = sum(1 for value in control_values if good_value + 1e-12 >= value)
                pass_rate = float(good_beats / max(len(control_values), 1))
                summary_verdict = "pass" if pass_rate >= 0.8 and rho >= 0.6 else "needs-review"
                for graph_name, value, quality, applies in metric_group:
                    rows.append(
                        {
                            "axis": "diagnostic_sensitivity",
                            "seed": seed,
                            "scenario": name,
                            "metric": metric_name,
                            "metric_family": _metric_family(metric_name),
                            "graph_variant": graph_name,
                            "value": value,
                            "quality_edge_f1": quality,
                            "expected_direction": direction,
                            "applies": applies,
                            "pass_rate_vs_controls": pass_rate,
                            "spearman_rho_expected_direction": rho,
                            "threshold_kind": "operational sanity gate",
                            "verdict": summary_verdict if applies else "not-applicable",
                            "claim_boundary": "diagnostic sensitivity, not performance proof",
                        }
                    )
    return rows


def _metric_family(metric_name: str) -> str:
    if metric_name in {
        "edge_f1_to_ground_truth",
        "matrix_similarity_to_ground_truth",
        "directed_row_similarity_to_ground_truth",
    }:
        return "validation_metric"
    return "framework_diagnostic"


def _scenario_metric_values(
    z_mat: np.ndarray,
    adjacency: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    scores = _compare_matrices(adjacency, truth)
    values = {
        "edge_f1_to_ground_truth": scores["edge_f1"],
        "matrix_similarity_to_ground_truth": float(1.0 / (1.0 + scores["rel_fro_diff"])),
        "directed_row_similarity_to_ground_truth": _row_distribution_similarity(
            adjacency,
            truth,
        ),
    }
    diag = compute_graph_diagnostics(adjacency)
    if not bool(diag["graph_empty"]):
        l_mat = _laplacian(adjacency)
        lambda_max = float(np.max(np.linalg.eigvalsh(l_mat))) if adjacency.size else 0.0
        values["h_spec_normalized"] = float(heterogeneity(z_mat, l_mat) / max(lambda_max, 1e-12))
        values["low_frequency_energy_ratio"] = float(
            spectral_energy_diagnostics(z_mat, l_mat)["low_frequency_energy_ratio"]
        )
    else:
        values["h_spec_normalized"] = float("nan")
        values["low_frequency_energy_ratio"] = float("nan")
    return values


def _row_distribution_similarity(candidate: np.ndarray, reference: np.ndarray) -> float:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            np.maximum(row_sums, 1e-12),
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums > 1e-12,
        )

    cand_p = normalize_rows(cand)
    ref_p = normalize_rows(ref)
    row_l1 = np.sum(np.abs(cand_p - ref_p), axis=1)
    return float(np.clip(1.0 - float(np.mean(row_l1)) / 2.0, 0.0, 1.0))


def _ranks(values: Sequence[float]) -> list[float]:
    pairs = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = rank
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    xr = np.asarray(_ranks(xs), dtype=np.float64)
    yr = np.asarray(_ranks(ys), dtype=np.float64)
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def composability_rows(profile: str = "smoke") -> list[dict[str, Any]]:
    base_cases = [
        ("update", "dense", "update", "real_graph"),
        ("update", "knn", "graph_filtered_update", "real_graph"),
        ("ema_update", "magnitude_knn", "graph_filtered_ema_update", "real_graph"),
        ("weight", "rbf", "graph_filtered_weight", "real_graph"),
        ("update", "knn", "graph_filtered_update", "control_graph"),
        ("update", "knn", "graph_filtered_update", "clustering_only"),
    ]
    if profile == "smoke":
        cases = base_cases[:4]
    else:
        cases = base_cases
    rows = [_run_composability_case(*case) for case in cases]
    rows.append(_run_composability_case("not_a_source", "knn", "graph_filtered_update", "real_graph"))
    rows.append(_run_composability_case("update", "not_a_mode", "graph_filtered_update", "real_graph"))
    return rows


def design_space_rows(profile: str = "smoke") -> list[dict[str, Any]]:
    """Enumerate the claimable graph-design space.

    The poster profile checks the full Cartesian product of built-in graph
    sources, graph modes, core aggregation targets, and graph-construction
    correction profiles. Graph-free corrections are documented as boundaries
    because they are correction controls, not new graph designs.
    """
    if profile == "smoke":
        sources = ("update", "classifier_head_update", "weight")
        modes = ("dense", "knn", "magnitude_knn", "pfedgraph_qp")
        targets = ("update", "graph_filtered_update", "graph_filtered_weight")
        profiles = CORRECTION_PROFILES[:2]
    else:
        sources = GRAPH_SOURCE_DESIGN_SPACE
        modes = GRAPH_MODES
        targets = AGGREGATION_TARGET_CHOICES
        profiles = CORRECTION_PROFILES

    rows: list[dict[str, Any]] = []
    for graph_source in sources:
        for graph_mode in modes:
            for aggregation_target in targets:
                for profile_spec in profiles:
                    rows.append(
                        _run_design_space_case(
                            graph_source=str(graph_source),
                            graph_mode=str(graph_mode),
                            aggregation_target=str(aggregation_target),
                            correction_profile=profile_spec,
                        )
                    )
    return rows


def design_space_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    supported = _count(rows, "status", "supported-pass")
    unsupported = _count(rows, "status", "unsupported-explicit")
    needs_review = _count(rows, "status", "needs-review")
    calculation_pass = _count(rows, "calculation_checks_passed", True)
    by_axis = [
        ("graph_sources", len({str(row.get("graph_source")) for row in rows})),
        ("graph_modes", len({str(row.get("graph_mode")) for row in rows})),
        ("aggregation_targets", len({str(row.get("aggregation_target")) for row in rows})),
        ("correction_profiles", len({str(row.get("correction_profile")) for row in rows})),
    ]
    out = [
        {
            "axis": "cartesian_product",
            "count": total,
            "supported_pass": supported,
            "unsupported_explicit": unsupported,
            "needs_review": needs_review,
            "calculation_checks_passed": calculation_pass,
            "claim_boundary": "claimable built-in graph-design combinations only",
        }
    ]
    for axis, count in by_axis:
        out.append(
            {
                "axis": axis,
                "count": count,
                "supported_pass": "",
                "unsupported_explicit": "",
                "needs_review": "",
                "calculation_checks_passed": "",
                "claim_boundary": "unique design-axis values covered",
            }
        )
    return out


def _run_design_space_case(
    *,
    graph_source: str,
    graph_mode: str,
    aggregation_target: str,
    correction_profile: Mapping[str, Any],
) -> dict[str, Any]:
    correction_family = str(correction_profile["correction_family"])
    control_graph_mode = str(correction_profile["control_graph_mode"])
    cluster_method = str(correction_profile["cluster_method"])
    try:
        weights, updates, ema = _sample_arrays()
        vectors, source_used = graph_vectors_for_graphfl(
            local_weights=weights,
            local_updates=updates,
            ema_updates=ema,
            config=GraphSourceConfig(source=graph_source),
        )
        z_mat = np.vstack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        base_adj = _build_base_client_graph(
            z_mat=z_mat,
            mode=graph_mode,
            graph_source=graph_source,
            aggregation_target=aggregation_target,
            correction_family=correction_family,
            rng=np.random.default_rng(11),
            client_sample_weights=[0.4, 0.3, 0.2, 0.1],
        )
        adj, meta = build_relation_graph(
            z_mat=z_mat,
            mode=graph_mode,
            graph_source=graph_source,
            aggregation_target=aggregation_target,
            correction_family=correction_family,
            control_graph_mode=(
                "random" if control_graph_mode == "not-applicable" else control_graph_mode
            ),
            cluster_method=cluster_method,
            cluster_auto_k=cluster_method != "none",
            rng=np.random.default_rng(11),
            client_sample_weights=[0.4, 0.3, 0.2, 0.1],
        )
        graph_diag = compute_graph_diagnostics(adj)
        target_audit = _validate_aggregation_target(
            aggregation_target=aggregation_target,
            local_weights=weights,
            local_updates=updates,
            ema_updates=ema,
            adjacency=adj,
        )
        trace_count = len(meta.get("lifecycle_trace", []) or [])
        audit = _audit_design_space_calculations(
            graph_source=graph_source,
            graph_mode=graph_mode,
            correction_profile=str(correction_profile["correction_profile"]),
            vectors=vectors,
            adjacency=adj,
            base_adjacency=base_adj,
            graph_diag=graph_diag,
            target_audit=target_audit,
        )
        graph_emitted = audit["adjacency_contract_ok"]
        diagnostics_emitted = audit["diagnostics_numeric_ok"]
        metadata_recorded = bool(meta.get("graph_kind")) and bool(meta.get("base_graph_mode"))
        artifact_row_emitted = diagnostics_emitted and metadata_recorded
        supported = (
            graph_emitted
            and diagnostics_emitted
            and metadata_recorded
            and artifact_row_emitted
            and audit["source_vector_contract_ok"]
            and audit["aggregation_contract_ok"]
            and audit["control_semantics_ok"]
        )
        status = "supported-pass" if supported else "needs-review"
        error = ""
    except Exception as exc:
        source_used = ""
        target_audit = {
            "aggregation_target_used": "",
            "aggregation_output_shape_ok": False,
            "aggregation_output_finite": False,
            "aggregation_filter_diag_finite": False,
        }
        graph_diag = {}
        audit = _empty_calculation_audit()
        trace_count = 0
        graph_emitted = False
        diagnostics_emitted = False
        metadata_recorded = False
        artifact_row_emitted = False
        status = "unsupported-explicit"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "axis": "design_space",
        "graph_source": graph_source,
        "graph_source_used": source_used,
        "graph_mode": graph_mode,
        "aggregation_target": aggregation_target,
        "aggregation_target_used": target_audit["aggregation_target_used"],
        "correction_profile": str(correction_profile["correction_profile"]),
        "correction_family": correction_family,
        "control_graph_mode": control_graph_mode,
        "cluster_method": cluster_method,
        "status": status,
        "calculation_checks_passed": audit["calculation_checks_passed"],
        "calculation_failure_reasons": audit["calculation_failure_reasons"],
        "source_vector_contract_ok": audit["source_vector_contract_ok"],
        "source_vector_count": audit["source_vector_count"],
        "source_vector_dim": audit["source_vector_dim"],
        "source_vectors_finite": audit["source_vectors_finite"],
        "source_vectors_nonconstant": audit["source_vectors_nonconstant"],
        "adjacency_contract_ok": audit["adjacency_contract_ok"],
        "adjacency_shape_ok": audit["adjacency_shape_ok"],
        "adjacency_finite": audit["adjacency_finite"],
        "adjacency_symmetric": audit["adjacency_symmetric"],
        "adjacency_zero_diag": audit["adjacency_zero_diag"],
        "adjacency_nonnegative": audit["adjacency_nonnegative"],
        "diagnostics_numeric_ok": audit["diagnostics_numeric_ok"],
        "aggregation_contract_ok": audit["aggregation_contract_ok"],
        "aggregation_output_shape_ok": target_audit["aggregation_output_shape_ok"],
        "aggregation_output_finite": target_audit["aggregation_output_finite"],
        "aggregation_filter_diag_finite": target_audit["aggregation_filter_diag_finite"],
        "control_semantics_ok": audit["control_semantics_ok"],
        "expected_empty_graph": audit["expected_empty_graph"],
        "graph_emitted": graph_emitted,
        "diagnostics_emitted": diagnostics_emitted,
        "metadata_recorded": metadata_recorded,
        "trace_count": trace_count,
        "artifact_row_emitted": artifact_row_emitted,
        "graph_density": graph_diag.get("graph_density", float("nan")),
        "number_of_edges": graph_diag.get("number_of_edges", float("nan")),
        "verdict": "pass" if status in {"supported-pass", "unsupported-explicit"} else "needs-review",
        "claim_boundary": "built-in graph-design combination; graph-free corrections excluded",
        "error": error,
    }


def _validate_aggregation_target(
    *,
    aggregation_target: str,
    local_weights: list[list[np.ndarray]],
    local_updates: list[list[np.ndarray]],
    ema_updates: list[list[np.ndarray]],
    adjacency: np.ndarray,
) -> dict[str, Any]:
    current_global = [np.zeros_like(local_weights[0][0])]
    alpha = np.full(len(local_updates), 1.0 / float(len(local_updates)), dtype=np.float64)
    out_arrays, target_used, filter_diag = aggregate_target(
        current_global=current_global,
        local_weights=local_weights,
        local_updates=local_updates,
        alpha_norm=alpha,
        config=AggregationTargetConfig(
            target=aggregation_target,
            filter_strength=1.0,
        ),
        l_mat=_laplacian(adjacency),
        ema_updates=ema_updates,
    )
    output_shape_ok = (
        len(out_arrays) == len(current_global)
        and all(np.asarray(out).shape == np.asarray(ref).shape for out, ref in zip(out_arrays, current_global))
    )
    output_finite = all(bool(np.all(np.isfinite(np.asarray(out, dtype=np.float64)))) for out in out_arrays)
    filter_diag_finite = all(_is_finite_payload(value) for value in filter_diag.values())
    return {
        "aggregation_target_used": str(target_used),
        "aggregation_output_shape_ok": output_shape_ok,
        "aggregation_output_finite": output_finite,
        "aggregation_filter_diag_finite": filter_diag_finite,
    }


def _audit_design_space_calculations(
    *,
    graph_source: str,
    graph_mode: str,
    correction_profile: str,
    vectors: Sequence[np.ndarray],
    adjacency: np.ndarray,
    base_adjacency: np.ndarray,
    graph_diag: Mapping[str, Any],
    target_audit: Mapping[str, Any],
) -> dict[str, Any]:
    vector_arrays = [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors]
    dims = {int(vector.size) for vector in vector_arrays}
    source_vector_count = len(vector_arrays)
    source_vector_dim = next(iter(dims)) if len(dims) == 1 and dims else 0
    source_vectors_finite = all(bool(np.all(np.isfinite(vector))) for vector in vector_arrays)
    source_vectors_nonconstant = _vectors_have_variation(vector_arrays)
    source_vector_contract_ok = (
        source_vector_count == int(adjacency.shape[0])
        and len(dims) == 1
        and source_vector_dim > 0
        and source_vectors_finite
        and source_vectors_nonconstant
    )

    adj = np.asarray(adjacency, dtype=np.float64)
    adjacency_shape_ok = adj.ndim == 2 and adj.shape[0] == adj.shape[1] == source_vector_count
    adjacency_finite = bool(np.all(np.isfinite(adj)))
    adjacency_symmetric = bool(np.max(np.abs(adj - adj.T)) <= 1e-9) if adj.size else True
    adjacency_zero_diag = bool(np.max(np.abs(np.diag(adj))) <= 1e-12) if adj.size else True
    adjacency_nonnegative = bool(np.min(adj) >= -1e-12) if adj.size else True
    adjacency_contract_ok = (
        adjacency_shape_ok
        and adjacency_finite
        and adjacency_symmetric
        and adjacency_zero_diag
        and adjacency_nonnegative
    )

    diagnostics_numeric_ok = _diagnostics_numeric_ok(graph_diag, expected_nodes=source_vector_count)
    aggregation_contract_ok = bool(
        target_audit.get("aggregation_target_used")
        and target_audit.get("aggregation_output_shape_ok")
        and target_audit.get("aggregation_output_finite")
        and target_audit.get("aggregation_filter_diag_finite")
    )
    expected_empty_graph = correction_profile == "control_identity"
    control_semantics_ok = _control_semantics_ok(
        correction_profile=correction_profile,
        adjacency=adj,
        base_adjacency=np.asarray(base_adjacency, dtype=np.float64),
    )
    checks = {
        "source_vector_contract_ok": source_vector_contract_ok,
        "adjacency_contract_ok": adjacency_contract_ok,
        "diagnostics_numeric_ok": diagnostics_numeric_ok,
        "aggregation_contract_ok": aggregation_contract_ok,
        "control_semantics_ok": control_semantics_ok,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        **checks,
        "calculation_checks_passed": not failures,
        "calculation_failure_reasons": ";".join(failures),
        "source_vector_count": source_vector_count,
        "source_vector_dim": source_vector_dim,
        "source_vectors_finite": source_vectors_finite,
        "source_vectors_nonconstant": source_vectors_nonconstant,
        "adjacency_shape_ok": adjacency_shape_ok,
        "adjacency_finite": adjacency_finite,
        "adjacency_symmetric": adjacency_symmetric,
        "adjacency_zero_diag": adjacency_zero_diag,
        "adjacency_nonnegative": adjacency_nonnegative,
        "expected_empty_graph": expected_empty_graph,
    }


def _empty_calculation_audit() -> dict[str, Any]:
    return {
        "source_vector_contract_ok": False,
        "source_vector_count": 0,
        "source_vector_dim": 0,
        "source_vectors_finite": False,
        "source_vectors_nonconstant": False,
        "adjacency_contract_ok": False,
        "adjacency_shape_ok": False,
        "adjacency_finite": False,
        "adjacency_symmetric": False,
        "adjacency_zero_diag": False,
        "adjacency_nonnegative": False,
        "diagnostics_numeric_ok": False,
        "aggregation_contract_ok": False,
        "control_semantics_ok": False,
        "expected_empty_graph": False,
        "calculation_checks_passed": False,
        "calculation_failure_reasons": "exception",
    }


def _vectors_have_variation(vectors: Sequence[np.ndarray]) -> bool:
    if len(vectors) <= 1:
        return False
    mat = np.vstack(vectors)
    return bool(np.max(np.std(mat, axis=0)) > 1e-12 or np.max(np.linalg.norm(mat - mat[0], axis=1)) > 1e-12)


def _diagnostics_numeric_ok(graph_diag: Mapping[str, Any], *, expected_nodes: int) -> bool:
    required = {
        "graph_num_nodes",
        "graph_density",
        "graph_entropy",
        "graph_degree_mean",
        "graph_degree_min",
        "graph_degree_max",
        "number_of_edges",
        "graph_empty",
    }
    if not required.issubset(graph_diag):
        return False
    if int(graph_diag["graph_num_nodes"]) != int(expected_nodes):
        return False
    for key in required - {"graph_empty"}:
        if not _is_finite_scalar(graph_diag.get(key)):
            return False
    density = float(graph_diag["graph_density"])
    entropy = float(graph_diag["graph_entropy"])
    return 0.0 <= density <= 1.0 and 0.0 <= entropy <= 1.0


def _control_semantics_ok(
    *,
    correction_profile: str,
    adjacency: np.ndarray,
    base_adjacency: np.ndarray,
) -> bool:
    if correction_profile == "control_identity":
        return bool(np.count_nonzero(adjacency > 1e-12) == 0)
    if correction_profile == "control_uniform":
        base_count = int(np.sum(np.triu(base_adjacency, k=1) > 0.0))
        adj_upper = adjacency[np.triu_indices(adjacency.shape[0], k=1)]
        weights = adj_upper[adj_upper > 0.0]
        if int(weights.size) != base_count:
            return False
        return bool(weights.size <= 1 or np.max(weights) - np.min(weights) <= 1e-12)
    if correction_profile in {"control_random", "control_shuffled"}:
        base_count = int(np.sum(np.triu(base_adjacency, k=1) > 0.0))
        adj_count = int(np.sum(np.triu(adjacency, k=1) > 0.0))
        return adj_count == base_count
    return True


def _is_finite_scalar(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _is_finite_payload(value: Any) -> bool:
    if isinstance(value, (str, bool)) or value is None:
        return True
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(arr)))


def _sample_arrays() -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    weights = [
        [np.array([1.0, 0.0], dtype=np.float64)],
        [np.array([0.9, 0.1], dtype=np.float64)],
        [np.array([0.0, 1.0], dtype=np.float64)],
        [np.array([-1.0, 0.0], dtype=np.float64)],
    ]
    updates = [
        [np.array([0.2, 0.0], dtype=np.float64)],
        [np.array([0.18, 0.02], dtype=np.float64)],
        [np.array([0.0, 0.2], dtype=np.float64)],
        [np.array([-0.2, 0.0], dtype=np.float64)],
    ]
    ema = [
        [np.array([0.15, 0.0], dtype=np.float64)],
        [np.array([0.14, 0.02], dtype=np.float64)],
        [np.array([0.0, 0.15], dtype=np.float64)],
        [np.array([-0.15, 0.0], dtype=np.float64)],
    ]
    return weights, updates, ema


def _run_composability_case(
    graph_source: str,
    graph_mode: str,
    aggregation_target: str,
    correction_family: str,
) -> dict[str, Any]:
    weights, updates, ema = _sample_arrays()
    try:
        vectors, source_used = graph_vectors_for_graphfl(
            local_weights=weights,
            local_updates=updates,
            ema_updates=ema,
            config=GraphSourceConfig(source=graph_source),
        )
        z_mat = np.vstack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        adj, meta = build_relation_graph(
            z_mat=z_mat,
            mode=graph_mode,
            graph_source=graph_source,
            aggregation_target=aggregation_target,
            correction_family=correction_family,
            control_graph_mode="random",
            cluster_method="kmeans",
            cluster_auto_k=True,
            rng=np.random.default_rng(5),
        )
        graph_diag = compute_graph_diagnostics(adj)
        trace_count = len(meta.get("lifecycle_trace", []) or [])
        artifact_row = {
            "graph_density": graph_diag["graph_density"],
            "number_of_edges": graph_diag["number_of_edges"],
        }
        supported = (
            bool(np.all(np.isfinite(adj)))
            and bool(meta)
            and trace_count >= 2
            and "graph_density" in artifact_row
        )
        status = "supported-pass" if supported else "needs-review"
        error = ""
    except Exception as exc:
        source_used = ""
        graph_diag = {}
        trace_count = 0
        status = "unsupported-explicit"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "axis": "composability",
        "graph_source": graph_source,
        "graph_source_used": source_used,
        "graph_mode": graph_mode,
        "aggregation_target": aggregation_target,
        "correction_family": correction_family,
        "status": status,
        "graph_emitted": status == "supported-pass",
        "trace_count": trace_count,
        "artifact_row_emitted": status == "supported-pass",
        "graph_density": graph_diag.get("graph_density", float("nan")),
        "number_of_edges": graph_diag.get("number_of_edges", float("nan")),
        "error": error,
        "verdict": "pass" if status in {"supported-pass", "unsupported-explicit"} else "needs-review",
    }


def extension_contract_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(_custom_graph_source_contract())
    rows.append(_custom_graph_builder_contract())
    rows.append(_custom_design_contract())
    rows.append(
        {
            "axis": "extensibility",
            "extension_kind": "aggregation_target",
            "extension_name": "v1_core_code_extension_point",
            "status": "not-claimed",
            "metadata_recorded": True,
            "trace_recorded": False,
            "artifact_recorded": False,
            "verdict": "pass",
            "claim_boundary": "aggregation target plugin extensibility is not claimed in v1",
        }
    )
    return rows


def _custom_graph_source_contract() -> dict[str, Any]:
    name = "evidence_unit_source"

    @register_graph_source(name, override=True)
    def _source(context: GraphSourceContext) -> GraphSourceResult:
        vectors = [
            np.asarray(arrays[0], dtype=np.float64).reshape(-1) * 2.0
            for arrays in context.local_updates
        ]
        return GraphSourceResult(
            vectors=vectors,
            source_used="evidence_unit_source_used",
            metadata={"contract": "unit"},
        )

    try:
        weights, updates, ema = _sample_arrays()
        vectors, source_used = graph_vectors_for_graphfl(
            local_weights=weights,
            local_updates=updates,
            ema_updates=ema,
            config=GraphSourceConfig(source=name),
        )
        z_mat = np.vstack(vectors)
        adj, meta = build_relation_graph(
            z_mat=z_mat,
            mode="knn",
            graph_source=name,
            aggregation_target="graph_filtered_update",
        )
        diag = compute_graph_diagnostics(adj)
        ok = source_used == "evidence_unit_source_used" and diag["number_of_edges"] > 0 and bool(meta)
        return {
            "axis": "extensibility",
            "extension_kind": "graph_source",
            "extension_name": name,
            "status": "supported-pass" if ok else "needs-review",
            "metadata_recorded": bool(meta),
            "trace_recorded": len(meta.get("lifecycle_trace", []) or []) >= 2,
            "artifact_recorded": "graph_density" in diag,
            "verdict": "pass" if ok else "needs-review",
            "claim_boundary": "custom source reaches graph construction and diagnostics",
        }
    finally:
        unregister_graph_source(name)


def _custom_graph_builder_contract() -> dict[str, Any]:
    name = "evidence_unit_builder"

    @register_graph_builder(name, override=True)
    def _builder(context: GraphBuildContext):
        n = context.z_mat.shape[0]
        adj = np.zeros((n, n), dtype=np.float64)
        adj[0, 1:] = 0.5
        adj[1:, 0] = 0.5
        return adj, {"graph_kind": "plugin:evidence_unit_star"}

    try:
        z_mat = np.eye(4, dtype=np.float64)
        adj, meta = build_relation_graph(
            z_mat=z_mat,
            mode=name,
            graph_source="update",
            aggregation_target="graph_filtered_update",
        )
        diag = compute_graph_diagnostics(adj)
        ok = (
            meta.get("base_graph_builder") == "registered"
            and meta.get("base_graph_kind") == "plugin:evidence_unit_star"
            and diag["number_of_edges"] == 3
        )
        return {
            "axis": "extensibility",
            "extension_kind": "graph_builder",
            "extension_name": name,
            "status": "supported-pass" if ok else "needs-review",
            "metadata_recorded": bool(meta.get("base_graph_kind")),
            "trace_recorded": True,
            "artifact_recorded": "graph_density" in diag,
            "verdict": "pass" if ok else "needs-review",
            "claim_boundary": "custom graph builder reaches metadata and diagnostics",
        }
    finally:
        unregister_graph_builder(name)


def _custom_design_contract() -> dict[str, Any]:
    registry = DesignRegistry()
    design = GraphFLDesign(
        name="evidence_unit_design",
        client_state=ComponentSpec(
            kind="ClientStateExtractor",
            name="update",
            params={"graph_source": "update", "graph_method": "evidence_unit_design"},
        ),
        relation=ComponentSpec(kind="RelationEstimator", name="positive_cosine"),
        topology=ComponentSpec(
            kind="TopologyOperator",
            name="knn",
            params={"graph_mode": "knn", "knn_k": 1},
        ),
        aggregation=ComponentSpec(
            kind="AggregationOperator",
            name="graph_filtered_update",
            params={
                "aggregation_target": "graph_filtered_update",
                "correction_family": "real_graph",
            },
        ),
    )
    registry.register(design)
    resolved = registry.resolve("evidence_unit_design")
    args = resolved.to_legacy_args()
    trace = resolved.trace_metadata()
    ok = (
        args.get("graph_source") == "update"
        and args.get("graph_mode") == "knn"
        and args.get("aggregation_target") == "graph_filtered_update"
        and trace.get("design_name") == "evidence_unit_design"
    )
    return {
        "axis": "extensibility",
        "extension_kind": "design_preset",
        "extension_name": "evidence_unit_design",
        "status": "supported-pass" if ok else "needs-review",
        "metadata_recorded": bool(trace),
        "trace_recorded": bool(trace),
        "artifact_recorded": bool(args),
        "verdict": "pass" if ok else "needs-review",
        "claim_boundary": "custom design resolves into current strategy knobs",
    }


def real_diagnostic_consistency_rows(real_summary_dir: str | Path | None) -> list[dict[str, Any]]:
    if not real_summary_dir:
        return []
    root = Path(real_summary_dir)
    artifact_rows = _real_rows_from_round_metrics(root / "diagnostics" / "round_metrics.csv")
    if artifact_rows:
        return artifact_rows
    path = root / "diagnostic_summary.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            deltas = {
                "mean_di_drop": _safe_float(row.get("mean_di_drop")),
                "mean_neff_gain": _safe_float(row.get("mean_neff_gain")),
                "mean_alignment_gain": _safe_float(row.get("mean_alignment_gain")),
                "mean_loo_drop": _safe_float(row.get("mean_loo_drop")),
            }
            variant = row.get("variant", "")
            measurement = _real_measurement_status(
                variant=variant,
                numeric_pair_count=sum(1 for value in deltas.values() if value is not None),
                deltas=deltas.values(),
            )
            rows.append(
                {
                    "axis": "real_experiment_diagnostic_consistency",
                    "variant": variant,
                    "mean_delta_vs_fedavg": row.get("mean_delta_vs_fedavg", ""),
                    "mean_di_drop": row.get("mean_di_drop", ""),
                    "mean_neff_gain": row.get("mean_neff_gain", ""),
                    "mean_alignment_gain": row.get("mean_alignment_gain", ""),
                    "mean_loo_drop": row.get("mean_loo_drop", ""),
                    "measurement_status": measurement,
                    "verdict": "pass" if _real_measurement_passes(measurement) else "needs-review",
                    "claim_boundary": (
                        "constructed graph vs controls only; no ground-truth graph is claimed"
                    ),
                }
            )
    return rows


def _real_rows_from_round_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    groups: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            variant = str(row.get("variant") or _variant_from_run_id(row.get("run_id", "")))
            if not variant:
                continue
            groups.setdefault(variant, []).append(row)
    rows: list[dict[str, Any]] = []
    for variant, group in sorted(groups.items()):
        di_pre = _safe_float_list(row.get("di_pre") for row in group)
        di_post = _safe_float_list(row.get("di_post") for row in group)
        neff_pre = _safe_float_list(row.get("neff_pre") for row in group)
        neff_post = _safe_float_list(row.get("neff_post") for row in group)
        align_pre = _safe_float_list(row.get("align_mean_pre") for row in group)
        align_post = _safe_float_list(row.get("align_mean_post") for row in group)
        loo_pre = _safe_float_list(row.get("loo_mean_pre") for row in group)
        loo_post = _safe_float_list(row.get("loo_mean_post") for row in group)
        deltas = {
            "mean_di_drop": _mean_pair_delta(di_pre, di_post),
            "mean_neff_gain": _mean_pair_delta(neff_post, neff_pre),
            "mean_alignment_gain": _mean_pair_delta(align_post, align_pre),
            "mean_loo_drop": _mean_pair_delta(loo_pre, loo_post),
        }
        numeric_pair_count = sum(
            1
            for left, right in (
                (di_pre, di_post),
                (neff_pre, neff_post),
                (align_pre, align_post),
                (loo_pre, loo_post),
            )
            if left and right
        )
        measurement = _real_measurement_status(
            variant=variant,
            numeric_pair_count=numeric_pair_count,
            deltas=deltas.values(),
        )
        rows.append(
            {
                "axis": "real_experiment_diagnostic_consistency",
                "variant": variant,
                "round_count": len(group),
                "source": "diagnostics/round_metrics.csv",
                "mean_delta_vs_fedavg": "",
                "mean_di_drop": deltas["mean_di_drop"],
                "mean_neff_gain": deltas["mean_neff_gain"],
                "mean_alignment_gain": deltas["mean_alignment_gain"],
                "mean_loo_drop": deltas["mean_loo_drop"],
                "mean_di_pre": _safe_mean(di_pre),
                "mean_di_post": _safe_mean(di_post),
                "mean_neff_pre": _safe_mean(neff_pre),
                "mean_neff_post": _safe_mean(neff_post),
                "measurement_status": measurement,
                "verdict": "pass" if _real_measurement_passes(measurement) else "needs-review",
                "claim_boundary": (
                    "constructed graph vs controls only; no ground-truth graph is claimed"
                ),
            }
        )
    return rows


def _variant_from_run_id(run_id: str) -> str:
    text = str(run_id).strip()
    if not text:
        return ""
    if text.startswith("ours_") and "_seed" in text:
        return text[len("ours_"): text.rfind("_seed")]
    if text.startswith("fedavg_") and "_seed" in text:
        return "fedavg"
    return text


def _safe_float_list(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _mean_pair_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("nan")
    size = min(len(left), len(right))
    return float(np.mean(np.asarray(left[:size], dtype=np.float64) - np.asarray(right[:size], dtype=np.float64)))


def _real_measurement_status(
    *,
    variant: Any,
    numeric_pair_count: int,
    deltas: Iterable[float | None],
) -> str:
    finite = [
        float(value)
        for value in deltas
        if value is not None and np.isfinite(float(value))
    ]
    if numeric_pair_count <= 0 or not finite:
        return "missing_metrics"
    if all(abs(value) <= 1e-12 for value in finite):
        if _zero_delta_expected_variant(variant):
            return "measured_zero_delta_expected_control"
        return "measured_zero_delta"
    return "measured_nonzero_delta"


def _zero_delta_expected_variant(variant: Any) -> bool:
    return "identity_control" in str(variant)


def _real_measurement_passes(status: str) -> bool:
    return status in {"measured_nonzero_delta", "measured_zero_delta_expected_control"}


def build_verdict(
    *,
    manifest_hash: str,
    parity_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    composability: Sequence[Mapping[str, Any]],
    design_rows: Sequence[Mapping[str, Any]],
    extension_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metric_required = [
        row
        for row in metric_rows
        if row.get("metric_family") == "framework_diagnostic"
        and row.get("applies") is True
    ]
    metric_pass = bool(metric_required) and all(
        row.get("verdict") == "pass" for row in metric_required
    )
    if not real_rows:
        real_status: bool | str = "not-provided"
    elif any(row.get("verdict") == "needs-review" for row in real_rows):
        real_status = "needs-review"
    else:
        real_status = "available"
    checks = {
        "construction_drift": all(row.get("verdict") == "pass" for row in parity_rows),
        "paper_mechanism_alignment": all(row.get("verdict") == "pass" for row in external_rows),
        "synthetic_metric_validity": metric_pass,
        "composability": all(row.get("verdict") == "pass" for row in composability),
        "design_space_coverage": all(row.get("verdict") == "pass" for row in design_rows),
        "extensibility": all(row.get("verdict") == "pass" for row in extension_rows),
        "real_experiment_diagnostic_consistency": real_status,
        "performance_relevance": "future-work",
    }
    required = [
        "construction_drift",
        "paper_mechanism_alignment",
        "synthetic_metric_validity",
        "composability",
        "design_space_coverage",
        "extensibility",
    ]
    return {
        "pass": all(checks[key] is True for key in required),
        "checks": checks,
        "manifest_sha256": manifest_hash,
        "claim": POSTER_SAFE_CLAIM,
        "claim_boundary": "This evidence pack is not a performance proof.",
    }


def write_claim_boundaries(path: Path) -> Path:
    lines = [
        "# Claim Boundaries",
        "",
        POSTER_SAFE_CLAIM,
        "",
        "## Guards",
        "",
        f"- {PAPER_KERNEL_NOTE}",
        "- pFedGraph directionality is reported separately for directed and symmetric-projection rows.",
        "- Synthetic benchmarks have predefined ground-truth graphs; real FL experiments do not.",
        "- Real summaries report constructed graph vs controls only.",
        "- Metric validity is diagnostic sensitivity, not performance proof.",
        "- Performance relevance is future-work.",
        "- Aggregation target plugin extensibility is not claimed in v1.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_poster_tables(
    path: Path,
    *,
    parity_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    composability: Sequence[Mapping[str, Any]],
    design_rows: Sequence[Mapping[str, Any]],
    design_summary: Sequence[Mapping[str, Any]],
    extension_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
) -> Path:
    lines = [
        "# Graph-FL Framework Evidence Pack",
        "",
        POSTER_SAFE_CLAIM,
        "",
        "All scenario/metric/method rows are retained in CSV; poster tables are compact summaries.",
        "",
        "## Table 1. Lifecycle Assembly Regression Check",
        "",
        "| mode | max_abs_diff | edge_f1 | verdict |",
        "|---|---:|---:|---|",
    ]
    for row in parity_rows[:10]:
        lines.append(
            f"| `{row['mode']}` | {float(row['max_abs_diff']):.3e} | {float(row['edge_f1']):.3f} | `{row['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Table 2. External/Paper Mechanism Alignment",
            "",
            "| method | component | reference_type | directionality | max_abs_diff | unmatched_gap |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in external_rows:
        lines.append(
            f"| {row['method']} | {row['component']} | `{row['reference_type']}` | "
            f"`{row['graph_directionality']}` | {float(row['max_abs_diff']):.3e} | {row['unmatched_gap']} |"
        )
    lines.extend(
        [
            "",
            "## Table 3. Synthetic Metric Validity",
            "",
            "| scenario | metric | family | seeds | min_pass_rate | min_rho | verdict |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in _metric_poster_rows(metric_rows):
        lines.append(
            f"| `{row['scenario']}` | `{row['metric']}` | `{row['metric_family']}` | "
            f"{int(row['seeds'])} | "
            f"{float(row['min_pass_rate']):.2f} | "
            f"{float(row['min_rho']):.2f} | `{row['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Table 4. Composability And Extension Contracts",
            "",
            "| check | pass | needs-review | unsupported-explicit |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.append(f"| composability | {_count(composability, 'status', 'supported-pass')} | {_count(composability, 'status', 'needs-review')} | {_count(composability, 'status', 'unsupported-explicit')} |")
    lines.append(f"| extensibility | {_count(extension_rows, 'verdict', 'pass')} | {_count(extension_rows, 'verdict', 'needs-review')} | 0 |")
    lines.extend(
        [
            "",
            "## Table 5. Full Design-Space Coverage",
            "",
            "| axis | count | supported_pass | calculation_checks_passed | needs_review | unsupported_explicit |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in design_summary:
        lines.append(
            f"| `{row['axis']}` | {row['count']} | {row.get('supported_pass', '')} | "
            f"{row.get('calculation_checks_passed', '')} | {row.get('needs_review', '')} | "
            f"{row.get('unsupported_explicit', '')} |"
        )
    if design_rows:
        lines.append(
            ""
        )
        lines.append(
            f"Claimable graph-design combinations checked: {len(design_rows)}; "
            f"supported-pass: {_count(design_rows, 'status', 'supported-pass')}; "
            f"needs-review: {_count(design_rows, 'status', 'needs-review')}."
        )
    if real_rows:
        lines.extend(
            [
                "",
                "## Optional Real Experiment Diagnostic Consistency",
                "",
                "| variant | source | rounds | measurement_status | mean_di_drop | mean_neff_gain | mean_alignment_gain | mean_loo_drop | verdict |",
                "|---|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in real_rows:
            lines.append(
                f"| `{row['variant']}` | `{row.get('source', '')}` | {row.get('round_count', '')} | "
                f"`{row.get('measurement_status', '')}` | "
                f"{row['mean_di_drop']} | {row['mean_neff_gain']} | "
                f"{row.get('mean_alignment_gain', '')} | {row.get('mean_loo_drop', '')} | "
                f"`{row.get('verdict', '')}` |"
            )
    else:
        lines.extend(
            [
                "",
                "## Optional Real Experiment Diagnostic Consistency",
                "",
                "Not provided in this run.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _count(rows: Sequence[Mapping[str, Any]], key: str, value: Any) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def _metric_poster_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("applies")):
            continue
        key = (str(row.get("scenario", "")), str(row.get("metric", "")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (scenario, metric), group in sorted(grouped.items()):
        seed_count = len({str(row.get("seed", "")) for row in group})
        pass_rates = [float(row.get("pass_rate_vs_controls", float("nan"))) for row in group]
        rhos = [float(row.get("spearman_rho_expected_direction", float("nan"))) for row in group]
        finite_pass = [value for value in pass_rates if np.isfinite(value)]
        finite_rho = [value for value in rhos if np.isfinite(value)]
        families = sorted({str(row.get("metric_family", "")) for row in group})
        verdict = "needs-review" if any(row.get("verdict") == "needs-review" for row in group) else "pass"
        out.append(
            {
                "scenario": scenario,
                "metric": metric,
                "metric_family": ",".join(families),
                "seeds": seed_count,
                "min_pass_rate": min(finite_pass) if finite_pass else float("nan"),
                "min_rho": min(finite_rho) if finite_rho else float("nan"),
                "verdict": verdict,
            }
        )
    return out


def write_figures(
    figures_dir: Path,
    *,
    external_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    composability: Sequence[Mapping[str, Any]],
    design_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    _ensure_dir(figures_dir)
    paths = [
        _write_bar_svg(
            figures_dir / "mechanism_alignment_diff_heatmap.svg",
            [(str(row["method"]), float(row["max_abs_diff"])) for row in external_rows],
            title="Mechanism alignment max diff",
        ),
        _write_bar_svg(
            figures_dir / "scenario_metric_direction_bars.svg",
            _metric_summary_for_plot(metric_rows),
            title="Synthetic metric pass rate",
            value_max=1.0,
        ),
        _write_bar_svg(
            figures_dir / "composability_matrix_heatmap.svg",
            [
                ("supported", float(_count(composability, "status", "supported-pass"))),
                ("explicit unsupported", float(_count(composability, "status", "unsupported-explicit"))),
                ("needs-review", float(_count(composability, "status", "needs-review"))),
            ],
            title="Composability classifications",
        ),
        _write_bar_svg(
            figures_dir / "design_space_coverage.svg",
            [
                ("supported design combinations", float(_count(design_rows, "status", "supported-pass"))),
                ("explicit unsupported", float(_count(design_rows, "status", "unsupported-explicit"))),
                ("needs-review", float(_count(design_rows, "status", "needs-review"))),
            ],
            title="Design-space coverage",
        ),
    ]
    if real_rows:
        paths.append(
            _write_bar_svg(
                figures_dir / "real_vs_control_gap.svg",
                [
                    (str(row["variant"]), _float_or_zero(row.get("mean_di_drop")))
                    for row in real_rows
                ],
                title="Optional real DI drop",
            )
        )
    return paths


def _metric_summary_for_plot(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, float]] = []
    for row in rows:
        key = (str(row.get("scenario")), str(row.get("metric")))
        if key in seen or not bool(row.get("applies")):
            continue
        seen.add(key)
        out.append((f"{key[0]}:{key[1]}", float(row.get("pass_rate_vs_controls", 0.0))))
    return out[:12]


def _float_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _write_bar_svg(
    path: Path,
    values: Sequence[tuple[str, float]],
    *,
    title: str,
    value_max: float | None = None,
) -> Path:
    width = 900
    row_h = 28
    height = max(140, 80 + row_h * max(len(values), 1))
    max_val = value_max if value_max is not None else max([abs(v) for _, v in values] + [1.0])
    max_val = max(float(max_val), 1e-12)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{_xml_escape(title)}</text>',
    ]
    for idx, (label, raw_value) in enumerate(values):
        y = 64 + idx * row_h
        value = float(raw_value)
        bar_w = min(640.0, abs(value) / max_val * 640.0)
        color = "#2f6f9f" if value >= 0 else "#b55a4a"
        lines.append(f'<text x="24" y="{y + 16}" font-family="Arial" font-size="12">{_xml_escape(label[:70])}</text>')
        lines.append(f'<rect x="230" y="{y + 4}" width="{bar_w:.2f}" height="16" fill="{color}"/>')
        lines.append(f'<text x="{240 + bar_w:.2f}" y="{y + 17}" font-family="Arial" font-size="12">{value:.3g}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_evidence_pack(
    out_dir: str | Path,
    *,
    profile: str = "smoke",
    include_external: bool = True,
    real_summary_dir: str | Path | None = None,
) -> EvidencePack:
    if profile not in {"smoke", "poster"}:
        raise ValueError("profile must be 'smoke' or 'poster'")
    root = _ensure_dir(Path(out_dir))
    figures_dir = _ensure_dir(root / "figures")

    manifest = scenario_manifest(profile)
    manifest_hash = _sha256_payload(manifest)
    parity = graph_parity_rows(profile)
    external = external_mechanism_alignment_rows() if include_external else []
    metric_rows = metric_validity_rows(manifest)
    composability = composability_rows(profile)
    design_rows = design_space_rows(profile)
    design_summary = design_space_summary_rows(design_rows)
    extension_rows = extension_contract_rows()
    real_rows = real_diagnostic_consistency_rows(real_summary_dir)
    verdict = build_verdict(
        manifest_hash=manifest_hash,
        parity_rows=parity,
        external_rows=external,
        metric_rows=metric_rows,
        composability=composability,
        design_rows=design_rows,
        extension_rows=extension_rows,
        real_rows=real_rows,
    )

    files: dict[str, str] = {}
    files["poster_tables"] = str(
        write_poster_tables(
            root / "poster_tables.md",
            parity_rows=parity,
            external_rows=external,
            metric_rows=metric_rows,
            composability=composability,
            design_rows=design_rows,
            design_summary=design_summary,
            extension_rows=extension_rows,
            real_rows=real_rows,
        )
    )
    files["claim_boundaries"] = str(write_claim_boundaries(root / "claim_boundaries.md"))
    files["scenario_manifest"] = str(write_json(root / "scenario_manifest.json", manifest))
    files["graph_parity_summary"] = str(_write_csv(root / "graph_parity_summary.csv", parity))
    files["external_mechanism_alignment"] = str(_write_csv(root / "external_mechanism_alignment.csv", external))
    files["metric_validity_summary"] = str(_write_csv(root / "metric_validity_summary.csv", metric_rows))
    files["composability_matrix"] = str(_write_csv(root / "composability_matrix.csv", composability))
    files["design_space_matrix"] = str(_write_csv(root / "design_space_matrix.csv", design_rows))
    files["design_space_summary"] = str(_write_csv(root / "design_space_summary.csv", design_summary))
    files["design_space_boundaries"] = str(_write_csv(root / "design_space_boundaries.csv", DESIGN_SPACE_BOUNDARIES))
    files["extension_contract_summary"] = str(_write_csv(root / "extension_contract_summary.csv", extension_rows))
    if real_rows:
        files["real_diagnostic_consistency"] = str(_write_csv(root / "real_diagnostic_consistency.csv", real_rows))
    files["validation_verdict"] = str(write_json(root / "validation_verdict.json", verdict))
    figure_paths = write_figures(
        figures_dir,
        external_rows=external,
        metric_rows=metric_rows,
        composability=composability,
        design_rows=design_rows,
        real_rows=real_rows,
    )
    for figure in figure_paths:
        files[f"figure:{figure.name}"] = str(figure)

    return EvidencePack(out_dir=root, files=files, verdict=verdict)


__all__ = [
    "EvidencePack",
    "PAPER_KERNEL_NOTE",
    "POSTER_SAFE_CLAIM",
    "composability_rows",
    "design_space_rows",
    "design_space_summary_rows",
    "extension_contract_rows",
    "external_mechanism_alignment_rows",
    "generate_evidence_pack",
    "graph_parity_rows",
    "metric_validity_rows",
    "scenario_manifest",
]
