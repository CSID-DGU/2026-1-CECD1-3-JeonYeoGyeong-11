"""FedSim-style similarity-guided cluster aggregation baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

from graphfl_lab.strategies.graphfl.aggregation import weighted_average_by_alpha
from graphfl_lab.graph.sources import graph_vectors_for_fedsim, normalize_key
from graphfl_lab.graph.builders import build_client_graph
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics
from graphfl_lab.projection import make_gaussian_projection
from graphfl_lab.strategies.baselines.ordering import sort_fit_results_by_cid
from graphfl_lab.strategies.baselines.tracing import _EvalTracer


def _connected_components(w_mat: np.ndarray) -> List[List[int]]:
    n = int(w_mat.shape[0])
    seen = np.zeros(n, dtype=bool)
    components: List[List[int]] = []
    for start in range(n):
        if bool(seen[start]):
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            i = stack.pop()
            comp.append(int(i))
            neighbors = np.flatnonzero(w_mat[i] > 0.0)
            for j in neighbors:
                jj = int(j)
                if not bool(seen[jj]):
                    seen[jj] = True
                    stack.append(jj)
        components.append(comp)
    return components


class TracingFedSim(_EvalTracer, fl.server.strategy.FedAvg):
    """FedSim-style similarity-guided cluster aggregation baseline."""

    def __init__(
        self,
        compression_dim: int = 256,
        compression_seed: int = 0,
        graph_mode: str = "knn",
        graph_source: str = "update",
        knn_k: int = 2,
        edge_threshold: float = 0.0,
        graph_scale_sigma: float = 1.0,
        learned_graph_lambda: float = 1.0,
        graph_seed: int = 0,
        cluster_weighting: str = "uniform",
        log_w_matrix_max_clients: int = 20,
        **kwargs,
    ) -> None:
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        self.compression_dim = int(compression_dim)
        self.compression_seed = int(compression_seed)
        self.graph_mode = str(graph_mode)
        self.graph_source = str(graph_source)
        self.knn_k = int(knn_k)
        self.edge_threshold = float(edge_threshold)
        self.graph_scale_sigma = float(graph_scale_sigma)
        self.learned_graph_lambda = float(learned_graph_lambda)
        self.graph_seed = int(graph_seed)
        self.cluster_weighting = str(cluster_weighting)
        self.log_w_matrix_max_clients = int(log_w_matrix_max_clients)
        self._current_global: Optional[NDArrays] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self.round_logs: List[Dict[str, Any]] = []

    @staticmethod
    def _norm_key(value: str) -> str:
        return normalize_key(value)

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
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        ema_updates: Optional[List[NDArrays]] = None,
    ) -> Tuple[List[np.ndarray], str]:
        return graph_vectors_for_fedsim(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            source=self.graph_source,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._current_global = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)

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

        ordered_results = sort_fit_results_by_cid(results)
        local_weights: List[NDArrays] = []
        n_examples: List[int] = []
        cids: List[str] = []
        for proxy, fit_res in ordered_results:
            metrics = dict(fit_res.metrics or {})
            cids.append(str(metrics.get("cid", getattr(proxy, "cid", "?"))))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(int(fit_res.num_examples))

        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        graph_vectors, graph_source_used = self._graph_vectors(
            local_weights=local_weights,
            local_updates=local_updates,
        )
        z_mat = np.stack([self._project(g) for g in graph_vectors], axis=0)
        graph_rng = np.random.default_rng(
            int(self.graph_seed) * 1009 + int(server_round) * 13
        )
        w_mat = build_client_graph(
            z_mat=z_mat,
            mode=self.graph_mode,
            knn_k=self.knn_k,
            edge_threshold=self.edge_threshold,
            rng=graph_rng,
            graph_scale_sigma=self.graph_scale_sigma,
            learned_graph_lambda=self.learned_graph_lambda,
        )
        graph_diag = compute_graph_diagnostics(w_mat)
        components = _connected_components(w_mat)

        n_examples_arr = np.asarray(n_examples, dtype=np.float64)
        cluster_updates: List[NDArrays] = []
        cluster_sizes: List[int] = []
        cluster_examples: List[float] = []
        for comp in components:
            comp_weights = n_examples_arr[comp]
            comp_updates = [local_updates[i] for i in comp]
            cluster_updates.append(
                weighted_average_by_alpha(comp_updates, comp_weights)
            )
            cluster_sizes.append(len(comp))
            cluster_examples.append(float(np.sum(comp_weights)))

        weighting = self._norm_key(self.cluster_weighting)
        if weighting in {"sample", "samples", "num_examples", "fedavg"}:
            cluster_alpha = np.asarray(cluster_examples, dtype=np.float64)
            cluster_weighting_used = "samples"
        else:
            cluster_alpha = np.ones(len(cluster_updates), dtype=np.float64)
            cluster_weighting_used = "uniform_clusters"

        agg_delta = weighted_average_by_alpha(cluster_updates, cluster_alpha)
        new_global = [gp + gd for gp, gd in zip(self._current_global, agg_delta)]

        w_matrix_log: Optional[List[List[float]]] = None
        if z_mat.shape[0] <= self.log_w_matrix_max_clients:
            w_matrix_log = [[float(v) for v in row] for row in w_mat.tolist()]

        round_log: Dict[str, Any] = {
            "round": int(server_round),
            "method": "fedsim",
            "cids": list(cids),
            "graph_mode": str(self.graph_mode),
            "graph_source": str(self.graph_source),
            "graph_source_used": graph_source_used,
            "knn_k": int(self.knn_k),
            "edge_threshold": float(self.edge_threshold),
            "graph_scale_sigma": float(self.graph_scale_sigma),
            "learned_graph_lambda": float(self.learned_graph_lambda),
            "graph_density": float(graph_diag["graph_density"]),
            "graph_degree_list": list(graph_diag["graph_degree_list"]),
            "number_of_edges": int(graph_diag["number_of_edges"]),
            "graph_empty": bool(graph_diag["graph_empty"]),
            "W_matrix": w_matrix_log,
            "fedsim_num_clusters": int(len(components)),
            "fedsim_cluster_sizes": [int(x) for x in cluster_sizes],
            "fedsim_cluster_num_examples": [float(x) for x in cluster_examples],
            "fedsim_cluster_weighting": cluster_weighting_used,
            "client_num_examples": [int(x) for x in n_examples],
        }
        self.round_logs.append(round_log)
        metrics = {
            "graph_density": float(graph_diag["graph_density"]),
            "graph_empty": float(graph_diag["graph_empty"]),
            "number_of_edges": float(graph_diag["number_of_edges"]),
            "fedsim_num_clusters": float(len(components)),
            "fedsim_mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        }
        return ndarrays_to_parameters(new_global), metrics
