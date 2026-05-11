"""Phase-2 graph-informativeness strategy with simple graph smoothing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from spectral_fl.graph.builders import build_client_graph
from spectral_fl.graph.sources import GraphSourceConfig, graph_vectors_for_spectral
from spectral_fl.projection import flatten_weights, make_gaussian_projection
from spectral_fl.strategies.baselines.ordering import sort_fit_results_by_cid
from spectral_fl.strategies.baselines.tracing import _EvalTracer, _InteractionDiagnostics
from spectral_fl.strategies.spectral.momentum import apply_server_optimizer


def _safe_cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    xn = float(np.linalg.norm(x))
    yn = float(np.linalg.norm(y))
    if xn <= eps or yn <= eps:
        return 0.0
    return float(np.dot(x, y) / (xn * yn + eps))


def _count_components(a_mat: np.ndarray) -> int:
    n = int(a_mat.shape[0])
    if n <= 0:
        return 0
    adj = np.abs(a_mat) > 0.0
    seen = np.zeros(n, dtype=bool)
    comp = 0
    for start in range(n):
        if seen[start]:
            continue
        comp += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            nbrs = np.nonzero(adj[node])[0]
            for nb in nbrs:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
    return int(comp)


def _edge_stats(a_mat: np.ndarray) -> Dict[str, float]:
    iu = np.triu_indices(a_mat.shape[0], k=1)
    vals = a_mat[iu]
    pos = vals[vals > 0.0]
    neg = vals[vals < 0.0]
    nonzero = vals[np.abs(vals) > 0.0]
    if nonzero.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "density": 0.0,
            "edge_count": 0.0,
            "positive_edge_count": 0.0,
            "negative_edge_count": 0.0,
            "signed_edge_ratio": 0.0,
        }
    possible = max(float(vals.size), 1.0)
    return {
        "mean": float(np.mean(nonzero)),
        "std": float(np.std(nonzero)),
        "min": float(np.min(nonzero)),
        "max": float(np.max(nonzero)),
        "density": float(nonzero.size / possible),
        "edge_count": float(nonzero.size),
        "positive_edge_count": float(pos.size),
        "negative_edge_count": float(neg.size),
        "signed_edge_ratio": float(neg.size / max(float(nonzero.size), 1.0)),
    }


def _build_graph_from_variant(
    *,
    base: np.ndarray,
    variant: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(base.shape[0])
    if variant == "update":
        return base
    if variant == "identity":
        return np.zeros((n, n), dtype=np.float64)
    if variant == "uniform":
        a = np.zeros_like(base)
        mask = base > 0.0
        if bool(np.any(mask)):
            fill = float(np.mean(base[mask]))
            a[mask] = fill
            np.fill_diagonal(a, 0.0)
        return a
    iu = np.triu_indices(n, k=1)
    upper = base[iu]
    pos = upper[upper > 0.0]
    if pos.size == 0:
        return np.zeros_like(base)
    if variant == "shuffled":
        all_pairs = upper.copy()
        shuf_vals = pos.copy()
        rng.shuffle(shuf_vals)
        idx = np.nonzero(all_pairs > 0.0)[0]
        all_pairs[:] = 0.0
        all_pairs[idx] = shuf_vals
        out = np.zeros_like(base)
        out[iu] = all_pairs
        out = out + out.T
        np.fill_diagonal(out, 0.0)
        return out
    if variant == "random":
        out = np.zeros_like(base)
        m = int(pos.size)
        choices = np.arange(upper.size, dtype=int)
        pick = rng.choice(choices, size=m, replace=False)
        max_w = float(np.max(pos))
        rand_w = rng.uniform(low=0.0, high=max(max_w, 1e-6), size=m)
        arr = np.zeros_like(upper)
        arr[pick] = rand_w
        out[iu] = arr
        out = out + out.T
        np.fill_diagonal(out, 0.0)
        return out
    raise ValueError(f"Unknown graph variant: {variant!r}")


class TracingGraphSmoothFedAvgM(
    _InteractionDiagnostics, _EvalTracer, fl.server.strategy.FedAvg
):
    """Simple graph smoothing on full updates followed by FedAvgM server step."""

    def __init__(
        self,
        *,
        graph_preset: str = "none",
        graph_variant: str = "update",
        graph_mode: str = "dense",
        graph_source: str = "classifier_head_update",
        knn_k: int = 2,
        edge_threshold: float = 0.0,
        graph_scale_sigma: float = 1.0,
        learned_graph_lambda: float = 1.0,
        graph_layer_start: int = 0,
        graph_layer_end: int = 0,
        graph_smoothing_operator: str = "laplacian",
        graph_dominance_gamma: float = 1.0,
        dominance_mode: str = "sample",
        dominance_cap_kappa: float = 2.0,
        dominance_soft_tau: float = 5.0,
        client_update_ema_alpha: float = 0.8,
        compression_dim: int = 256,
        compression_seed: int = 0,
        graph_seed: int = 0,
        graph_smoothing_lambda: float = 0.05,
        graph_laplacian_type: str = "unnormalized",
        graph_zero_diagonal: bool = True,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        **kwargs,
    ) -> None:
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        _InteractionDiagnostics.__init__(self)
        self.graph_variant = str(graph_variant).strip().lower()
        self.graph_preset = str(graph_preset).strip().lower()
        self.graph_mode = str(graph_mode).strip()
        self.graph_source = str(graph_source).strip()
        self.knn_k = int(knn_k)
        self.edge_threshold = float(edge_threshold)
        self.graph_scale_sigma = float(graph_scale_sigma)
        self.learned_graph_lambda = float(learned_graph_lambda)
        self.graph_layer_start = int(graph_layer_start)
        self.graph_layer_end = int(graph_layer_end)
        self.graph_smoothing_operator = str(graph_smoothing_operator).strip().lower()
        self.graph_dominance_gamma = float(graph_dominance_gamma)
        self.dominance_mode = str(dominance_mode).strip().lower()
        self.dominance_cap_kappa = float(dominance_cap_kappa)
        self.dominance_soft_tau = float(dominance_soft_tau)
        self.client_update_ema_alpha = float(client_update_ema_alpha)
        self.compression_dim = int(compression_dim)
        self.compression_seed = int(compression_seed)
        self.graph_seed = int(graph_seed)
        self.graph_smoothing_lambda = float(graph_smoothing_lambda)
        self.graph_laplacian_type = str(graph_laplacian_type).strip().lower()
        self.graph_zero_diagonal = bool(graph_zero_diagonal)
        self.server_learning_rate = float(server_learning_rate)
        self.server_momentum = float(server_momentum)
        self.server_momentum_vector: Optional[NDArrays] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self._prev_a_mat: Optional[np.ndarray] = None
        self._prev_dominant_cid: Optional[str] = None
        self._client_update_ema: Optional[List[NDArrays]] = None
        self._client_update_ema_cids: Optional[List[str]] = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._cache_current_global(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)

    def _project(self, vec: np.ndarray) -> np.ndarray:
        if vec.size <= self.compression_dim:
            return vec.astype(np.float32, copy=False)
        if self._proj_matrix is None:
            self._proj_matrix = make_gaussian_projection(
                n_features=int(vec.size),
                n_dim=int(self.compression_dim),
                seed=int(self.compression_seed),
            )
        return vec.astype(np.float32, copy=False) @ self._proj_matrix

    def _graph_vectors(
        self,
        cids: List[str],
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
    ) -> Tuple[List[np.ndarray], str]:
        ema_updates = self._update_client_update_ema(local_updates=local_updates, cids=cids)
        vectors, source_used = graph_vectors_for_spectral(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            config=GraphSourceConfig(
                source=self.graph_source,
                layer_start=self.graph_layer_start,
                layer_end=self.graph_layer_end,
            ),
        )
        return (
            [vec.astype(np.float64, copy=False) for vec in vectors],
            source_used,
        )

    def _update_client_update_ema(
        self,
        *,
        local_updates: List[NDArrays],
        cids: List[str],
    ) -> List[NDArrays]:
        alpha = min(max(float(self.client_update_ema_alpha), 0.0), 1.0)
        if self._client_update_ema is None or self._client_update_ema_cids != list(cids):
            ema_updates = [
                [np.array(arr, copy=True) for arr in update]
                for update in local_updates
            ]
        else:
            ema_updates = []
            for old_update, current_update in zip(self._client_update_ema, local_updates):
                ema_updates.append(
                    [
                        alpha * old + (1.0 - alpha) * current
                        for old, current in zip(old_update, current_update)
                    ]
                )
        self._client_update_ema = [
            [np.array(arr, copy=True) for arr in update]
            for update in ema_updates
        ]
        self._client_update_ema_cids = list(cids)
        return ema_updates

    def _laplacian(self, a_mat: np.ndarray) -> np.ndarray:
        deg = np.sum(a_mat, axis=1)
        if self.graph_laplacian_type in {"normalized", "norm"}:
            inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
            d_half = np.diag(inv_sqrt)
            eye = np.eye(a_mat.shape[0], dtype=np.float64)
            return eye - d_half @ a_mat @ d_half
        if self.graph_laplacian_type in {"random_walk", "rw"}:
            inv_deg = 1.0 / np.maximum(deg, 1e-12)
            d_inv = np.diag(inv_deg)
            eye = np.eye(a_mat.shape[0], dtype=np.float64)
            return eye - d_inv @ a_mat
        return np.diag(deg) - a_mat

    @staticmethod
    def _unflatten_like(vec: np.ndarray, ref: NDArrays) -> NDArrays:
        out: NDArrays = []
        cursor = 0
        for arr in ref:
            size = int(arr.size)
            piece = vec[cursor : cursor + size].reshape(arr.shape)
            out.append(piece.astype(arr.dtype, copy=False))
            cursor += size
        return out

    @staticmethod
    def _neighbor_average(
        updates: np.ndarray,
        a_mat: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        denom = np.sum(a_mat, axis=1, keepdims=True)
        safe = np.maximum(denom, eps)
        return (a_mat @ updates) / safe

    @staticmethod
    def _neighbor_average_signed(
        updates: np.ndarray,
        a_mat: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        denom = np.sum(np.abs(a_mat), axis=1, keepdims=True)
        safe = np.maximum(denom, eps)
        return (a_mat @ updates) / safe

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}
        if self._current_global is None:
            return super().aggregate_fit(server_round, results, failures)

        ordered = sort_fit_results_by_cid(list(results))
        local_weights: List[NDArrays] = []
        n_examples: List[int] = []
        cids: List[str] = []
        for proxy, fit_res in ordered:
            metrics = dict(fit_res.metrics or {})
            cids.append(str(metrics.get("cid", getattr(proxy, "cid", "?"))))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(int(fit_res.num_examples))

        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        flat_updates = np.stack(
            [flatten_weights(delta).astype(np.float64, copy=False) for delta in local_updates],
            axis=0,
        )

        n_examples_arr = np.asarray(n_examples, dtype=np.float64)
        p = n_examples_arr / max(float(np.sum(n_examples_arr)), 1e-12)
        p_agg = p.copy()

        graph_vecs, graph_source_used = self._graph_vectors(
            cids=cids,
            local_weights=local_weights,
            local_updates=local_updates,
        )
        z_mat = np.stack([self._project(v) for v in graph_vecs], axis=0).astype(np.float64)
        rng = np.random.default_rng(
            int(self.graph_seed) * 1009 + int(server_round) * 17
        )
        base_graph = build_client_graph(
            z_mat=z_mat,
            mode=self.graph_mode,
            knn_k=self.knn_k,
            edge_threshold=self.edge_threshold,
            rng=rng,
            graph_scale_sigma=self.graph_scale_sigma,
            learned_graph_lambda=self.learned_graph_lambda,
        )
        a_mat = _build_graph_from_variant(
            base=base_graph,
            variant=self.graph_variant,
            rng=rng,
        )
        if self.graph_zero_diagonal:
            np.fill_diagonal(a_mat, 0.0)
        op_key = str(self.graph_smoothing_operator).strip().lower()
        lap_type_used = str(self.graph_laplacian_type)
        if op_key == "unnormalized_laplacian":
            op_key = "laplacian"
            lap_type_used = "unnormalized"
        elif op_key == "normalized_laplacian":
            op_key = "laplacian"
            lap_type_used = "normalized"
        elif op_key == "random_walk_laplacian":
            op_key = "laplacian"
            lap_type_used = "random_walk"
        elif op_key == "residual_neighbor_mixing":
            op_key = "residual"
        l_mat = self._laplacian(a_mat)
        q_raw = p * np.asarray(
            [float(np.linalg.norm(v)) for v in flat_updates], dtype=np.float64
        )
        q_bar = q_raw / (float(np.sum(q_raw)) + 1e-12)
        if self.dominance_mode == "uniform":
            p_agg = np.ones_like(p) / max(float(p.size), 1.0)
            corrected_flat = flat_updates.copy()
            operator_used = "dominance_uniform_weight"
        elif self.dominance_mode == "cap":
            cap = float(self.dominance_cap_kappa) * float(np.median(q_raw))
            scales = np.ones_like(q_raw, dtype=np.float64)
            mask = q_raw > (cap + 1e-12)
            scales[mask] = cap / (q_raw[mask] + 1e-12)
            corrected_flat = flat_updates * scales[:, None]
            operator_used = "dominance_contribution_cap"
        elif self.dominance_mode == "soft_reweight":
            logits = p * np.exp(-float(self.dominance_soft_tau) * q_bar)
            p_agg = logits / (float(np.sum(logits)) + 1e-12)
            corrected_flat = flat_updates.copy()
            operator_used = "dominance_soft_reweight"
        elif self.graph_variant == "identity":
            corrected_flat = flat_updates.copy()
            operator_used = "identity_passthrough"
        elif op_key == "laplacian":
            n_clients = int(a_mat.shape[0])
            original_lap_type = self.graph_laplacian_type
            self.graph_laplacian_type = lap_type_used
            l_mat = self._laplacian(a_mat)
            self.graph_laplacian_type = original_lap_type
            m_mat = np.eye(n_clients, dtype=np.float64) - float(self.graph_smoothing_lambda) * l_mat
            corrected_flat = m_mat @ flat_updates
            operator_used = "laplacian"
        elif op_key == "residual":
            g_bar = self._neighbor_average(flat_updates, a_mat)
            lam = float(self.graph_smoothing_lambda)
            corrected_flat = (1.0 - lam) * flat_updates + lam * g_bar
            operator_used = "residual"
        elif op_key in {"dominance_residual", "dominance_aware_attenuation"}:
            source_scale = np.power(
                np.clip(1.0 - q_bar, 0.0, 1.0),
                float(self.graph_dominance_gamma),
            )
            a_dom = a_mat * source_scale[None, :]
            g_bar = self._neighbor_average(flat_updates, a_dom)
            lam = float(self.graph_smoothing_lambda)
            corrected_flat = (1.0 - lam) * flat_updates + lam * g_bar
            operator_used = "dominance_aware_attenuation"
        elif op_key == "signed_conflict_attenuation":
            g_bar = self._neighbor_average_signed(flat_updates, a_mat)
            lam = float(self.graph_smoothing_lambda)
            corrected_flat = (1.0 - lam) * flat_updates + lam * g_bar
            operator_used = "signed_conflict_attenuation"
        else:
            raise ValueError(
                "Unknown correction mode: "
                f"dominance_mode={self.dominance_mode!r}, "
                f"graph_smoothing_operator={self.graph_smoothing_operator!r}"
            )

        fedavg_delta = np.sum(p[:, None] * flat_updates, axis=0)
        corrected_delta = np.sum(p_agg[:, None] * corrected_flat, axis=0)
        candidate_global = [
            gp + delta for gp, delta in zip(self._current_global, self._unflatten_like(corrected_delta, local_updates[0]))
        ]
        new_global, self.server_momentum_vector, server_diag = apply_server_optimizer(
            current_global=self._current_global,
            candidate_global=candidate_global,
            server_learning_rate=self.server_learning_rate,
            server_momentum=self.server_momentum,
            server_momentum_vector=self.server_momentum_vector,
        )

        self._record_interaction_diagnostics(
            server_round=server_round,
            results=list(results),
            method="graph_smooth",
        )
        base_log = self.round_logs[-1]

        edge = _edge_stats(a_mat)
        degree_strength = np.sum(a_mat, axis=1)
        a_delta_fro = (
            float(np.linalg.norm(a_mat - self._prev_a_mat, ord="fro"))
            if self._prev_a_mat is not None
            else None
        )
        self._prev_a_mat = a_mat.copy()

        corrected_norms = np.asarray(
            [float(np.linalg.norm(v)) for v in corrected_flat], dtype=np.float64
        )
        corrected_weighted_norm_sum = float(np.sum(p * corrected_norms))
        corrected_alignment_ratio = float(
            np.linalg.norm(corrected_delta) / (corrected_weighted_norm_sum + 1e-12)
        )
        graph_smoothness = float(np.sum(flat_updates * (l_mat @ flat_updates)))
        q_corr = p * corrected_norms
        q_corr_bar = q_corr / (float(np.sum(q_corr)) + 1e-12)
        di_raw = float(np.max(q_bar))
        di_corr = float(np.max(q_corr_bar))
        n_eff_raw = float(1.0 / (np.sum(np.square(q_bar)) + 1e-12))
        n_eff_corr = float(1.0 / (np.sum(np.square(q_corr_bar)) + 1e-12))
        dominant_idx = int(np.argmax(q_bar))
        dominant_cid = str(base_log.get("cids", [str(dominant_idx)])[dominant_idx])
        dominant_vec = flat_updates[dominant_idx]
        top3_mass = float(np.sum(np.sort(q_bar)[-3:])) if q_bar.size > 0 else float("nan")
        dominant_changed = (
            None
            if self._prev_dominant_cid is None
            else bool(str(dominant_cid) != self._prev_dominant_cid)
        )
        self._prev_dominant_cid = str(dominant_cid)
        momentum_cos = None
        if self.server_momentum_vector is not None:
            mom_flat = flatten_weights(self.server_momentum_vector).astype(np.float64, copy=False)
            momentum_cos = _safe_cosine(corrected_delta, -mom_flat)

        base_log.update(
            {
                "method": "graph_smooth",
                "graph_variant": self.graph_variant,
                "graph_preset": self.graph_preset,
                "graph_mode": self.graph_mode,
                "graph_source_used": graph_source_used,
                "graph_smoothing_operator": operator_used,
                "operator_type": operator_used,
                "graph_dominance_gamma": float(self.graph_dominance_gamma),
                "dominance_mode": self.dominance_mode,
                "dominance_cap_kappa": float(self.dominance_cap_kappa),
                "dominance_soft_tau": float(self.dominance_soft_tau),
                "graph_laplacian_type": lap_type_used,
                "graph_smoothing_lambda": float(self.graph_smoothing_lambda),
                "lambda": float(self.graph_smoothing_lambda),
                "correction_operator": "G_corrected = (I - lambda L) G",
                "graph_source": graph_source_used,
                "graph_density": float(edge["density"]),
                "graph_edge_count": int(edge["edge_count"]),
                "positive_edge_count": int(edge["positive_edge_count"]),
                "negative_edge_count": int(edge["negative_edge_count"]),
                "signed_edge_ratio": float(edge["signed_edge_ratio"]),
                "graph_edge_weight_mean": float(edge["mean"]),
                "graph_edge_weight_std": float(edge["std"]),
                "graph_edge_weight_min": float(edge["min"]),
                "graph_edge_weight_max": float(edge["max"]),
                "graph_degree_mean": float(np.mean(degree_strength)),
                "mean_degree": float(np.mean(degree_strength)),
                "graph_degree_std": float(np.std(degree_strength)),
                "max_degree": float(np.max(degree_strength)),
                "min_degree": float(np.min(degree_strength)),
                "graph_connected_components": int(_count_components(a_mat)),
                "connected_components": int(_count_components(a_mat)),
                "graph_adj_delta_fro": a_delta_fro,
                "graph_smoothness": graph_smoothness,
                "fedavg_delta_norm": float(np.linalg.norm(fedavg_delta)),
                "base_delta_norm": float(np.linalg.norm(fedavg_delta)),
                "corrected_delta_norm": float(np.linalg.norm(corrected_delta)),
                "corrected_over_raw_norm_ratio": float(
                    np.linalg.norm(corrected_delta) / (np.linalg.norm(fedavg_delta) + 1e-12)
                ),
                "corrected_delta_norm_over_weighted_client_norm": corrected_alignment_ratio,
                "corrected_vs_fedavg_delta_cosine": _safe_cosine(corrected_delta, fedavg_delta),
                "cos_delta_corrected_vs_base": _safe_cosine(corrected_delta, fedavg_delta),
                "direction_change_one_minus_cosine": float(
                    1.0 - _safe_cosine(corrected_delta, fedavg_delta)
                ),
                "relative_delta_change": float(
                    np.linalg.norm(corrected_delta - fedavg_delta)
                    / (np.linalg.norm(fedavg_delta) + 1e-12)
                ),
                "rel_delta_change": float(
                    np.linalg.norm(corrected_delta - fedavg_delta)
                    / (np.linalg.norm(fedavg_delta) + 1e-12)
                ),
                "dominance_ratio_raw": di_raw,
                "dominance_ratio_corrected": di_corr,
                "effective_num_clients_raw": n_eff_raw,
                "effective_num_clients_corrected": n_eff_corr,
                "dominant_client_index": int(dominant_idx),
                "dominant_client_id": str(dominant_cid),
                "dominant_client_changed": dominant_changed,
                "client_contribution_weights": [float(x) for x in p_agg.tolist()],
                "max_contribution": float(np.max(q_corr_bar)),
                "top1_client_id_by_contribution": str(dominant_cid),
                "top3_contribution_mass": top3_mass,
                "contribution_entropy": float(
                    -np.sum(q_corr_bar * np.log(np.maximum(q_corr_bar, 1e-12)))
                ),
                "max_qbar_i_raw": float(np.max(q_bar)),
                "max_qbar_i_corrected": float(np.max(q_corr_bar)),
                "qbar_list_raw": [float(x) for x in q_bar.tolist()],
                "qbar_list_corrected": [float(x) for x in q_corr_bar.tolist()],
                "aggregation_weights_used": [float(x) for x in p_agg.tolist()],
                "cosine_dominant_vs_raw_delta": _safe_cosine(dominant_vec, fedavg_delta),
                "cosine_dominant_vs_corrected_delta": _safe_cosine(dominant_vec, corrected_delta),
                "corrected_vs_server_momentum_cosine": momentum_cos,
                "corrected_client_update_norms": [float(x) for x in corrected_norms.tolist()],
                "update_definition": "g_i = w_i_local - w_t",
                "server_optimizer": str(server_diag.get("server_optimizer", "")),
                "server_candidate_delta_norm": float(
                    server_diag.get("server_candidate_delta_norm", np.nan)
                ),
                "server_applied_delta_norm": float(
                    server_diag.get("server_applied_delta_norm", np.nan)
                ),
            }
        )

        metrics = {
            "graph_density": float(edge["density"]),
            "graph_smoothness": graph_smoothness,
            "corrected_delta_norm": float(np.linalg.norm(corrected_delta)),
            "corrected_vs_fedavg_delta_cosine": _safe_cosine(corrected_delta, fedavg_delta),
            "direction_change": float(1.0 - _safe_cosine(corrected_delta, fedavg_delta)),
            "relative_delta_change": float(
                np.linalg.norm(corrected_delta - fedavg_delta)
                / (np.linalg.norm(fedavg_delta) + 1e-12)
            ),
            "pairwise_cosine_mean": float(base_log.get("pairwise_cosine_mean", np.nan)),
            "conflict_ratio": float(base_log.get("conflict_ratio", np.nan)),
            "cancellation_ratio": float(base_log.get("cancellation_ratio", np.nan)),
            "dominance_ratio": float(base_log.get("dominance_ratio", np.nan)),
            "effective_num_clients": float(base_log.get("effective_num_clients", np.nan)),
        }
        return ndarrays_to_parameters(new_global), metrics


__all__ = ["TracingGraphSmoothFedAvgM"]
