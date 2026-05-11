"""Baseline strategy wrappers with evaluation and interaction diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitIns, FitRes, NDArrays, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from spectral_fl.projection import flatten_weights
from spectral_fl.strategies.baselines.ordering import sort_fit_results_by_cid


class _EvalTracer:
    """Mixin providing eval_logs[round] = [{cid, accuracy, loss, num_examples}]."""

    def __init__(self) -> None:
        self.eval_logs: Dict[int, List[Dict[str, Any]]] = {}

    def _record_eval(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> None:
        per_client = []
        for proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            per_client.append(
                {
                    "cid": str(getattr(proxy, "cid", "?")),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "loss": float(eval_res.loss),
                    "num_examples": int(eval_res.num_examples),
                }
            )
        self.eval_logs[int(server_round)] = per_client


class _InteractionDiagnostics:
    """Mixin providing per-round client-update interaction diagnostics."""

    def __init__(self) -> None:
        self.round_logs: List[Dict[str, Any]] = []
        self._current_global: Optional[NDArrays] = None
        self._diagnostics_eps = 1e-12

    def _cache_current_global(self, parameters: Parameters) -> None:
        self._current_global = parameters_to_ndarrays(parameters)

    @staticmethod
    def _safe_client_id(proxy: ClientProxy, fit_res: FitRes) -> str:
        metrics = dict(fit_res.metrics or {})
        return str(metrics.get("cid", getattr(proxy, "cid", "?")))

    @staticmethod
    def _extract_metric(client_metrics: List[Dict[str, Any]], key: str) -> Optional[List[float]]:
        out: List[float] = []
        for metrics in client_metrics:
            if key not in metrics:
                continue
            try:
                out.append(float(metrics[key]))
            except (TypeError, ValueError):
                continue
        return out if out else None

    def _record_interaction_diagnostics(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        method: str,
    ) -> None:
        if not results or self._current_global is None:
            return

        eps = float(self._diagnostics_eps)
        ordered_results = sort_fit_results_by_cid(list(results))
        cids: List[str] = []
        local_weights: List[NDArrays] = []
        client_metrics: List[Dict[str, Any]] = []
        n_examples: List[int] = []
        for proxy, fit_res in ordered_results:
            cids.append(self._safe_client_id(proxy, fit_res))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            client_metrics.append(dict(fit_res.metrics or {}))
            n_examples.append(int(fit_res.num_examples))

        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        flat_updates = [
            flatten_weights(delta).astype(np.float64, copy=False)
            for delta in local_updates
        ]
        update_norms = np.asarray(
            [float(np.linalg.norm(vec)) for vec in flat_updates],
            dtype=np.float64,
        )
        n_examples_arr = np.asarray(n_examples, dtype=np.float64)
        total_examples = float(np.sum(n_examples_arr))
        if total_examples <= 0.0:
            p = np.ones(len(flat_updates), dtype=np.float64) / float(len(flat_updates))
        else:
            p = n_examples_arr / total_examples

        weighted_delta = np.zeros_like(flat_updates[0], dtype=np.float64)
        for weight, vec in zip(p, flat_updates):
            weighted_delta += float(weight) * vec
        delta_norm = float(np.linalg.norm(weighted_delta))

        weighted_norm_sum = float(np.sum(p * update_norms))
        delta_alignment_ratio = float(delta_norm / (weighted_norm_sum + eps))
        cancellation_ratio = float(1.0 - delta_alignment_ratio)

        pairwise_cosines: List[float] = []
        negative_pairs = 0
        weighted_conflict = 0.0
        for i in range(len(flat_updates)):
            for j in range(i + 1, len(flat_updates)):
                denom = float(update_norms[i] * update_norms[j])
                if denom <= 0.0:
                    cos_ij = 0.0
                else:
                    cos_ij = float(np.dot(flat_updates[i], flat_updates[j]) / (denom + eps))
                cos_ij = float(np.clip(cos_ij, -1.0, 1.0))
                pairwise_cosines.append(cos_ij)
                if cos_ij < 0.0:
                    negative_pairs += 1
                    weighted_conflict += float(p[i] * p[j])

        pair_count = len(pairwise_cosines)
        if pair_count > 0:
            cosine_mean = float(np.mean(pairwise_cosines))
            cosine_min = float(np.min(pairwise_cosines))
            cosine_max = float(np.max(pairwise_cosines))
            cosine_std = float(np.std(pairwise_cosines))
            fraction_negative = float(negative_pairs / pair_count)
        else:
            cosine_mean = float("nan")
            cosine_min = float("nan")
            cosine_max = float("nan")
            cosine_std = float("nan")
            fraction_negative = float("nan")

        q_raw = p * update_norms
        q_denom = float(np.sum(q_raw))
        q_bar = q_raw / (q_denom + eps)
        dominance_ratio = float(np.max(q_bar))
        effective_num_clients = float(1.0 / (np.sum(np.square(q_bar)) + eps))

        train_losses = self._extract_metric(client_metrics, "train_loss")
        train_accuracies = self._extract_metric(client_metrics, "train_accuracy")

        self.round_logs.append(
            {
                "round": int(server_round),
                "method": str(method),
                "update_definition": "g_i = w_i_local - w_t",
                "cids": list(cids),
                "client_num_examples": [int(x) for x in n_examples],
                "client_weights": [float(x) for x in p.tolist()],
                "pair_count": int(pair_count),
                "pairwise_cosine_mean": cosine_mean,
                "pairwise_cosine_min": cosine_min,
                "pairwise_cosine_max": cosine_max,
                "pairwise_cosine_std": cosine_std,
                "pairwise_cosine_fraction_negative": fraction_negative,
                "conflict_ratio": fraction_negative,
                "conflict_ratio_weighted": float(weighted_conflict),
                "cancellation_ratio": cancellation_ratio,
                "dominance_ratio": dominance_ratio,
                "effective_num_clients": effective_num_clients,
                "client_update_norms": [float(x) for x in update_norms.tolist()],
                "client_contribution_raw": [float(x) for x in q_raw.tolist()],
                "client_contribution_normalized": [float(x) for x in q_bar.tolist()],
                "client_update_norm_mean": float(np.mean(update_norms)),
                "client_update_norm_max": float(np.max(update_norms)),
                "client_update_norm_std": float(np.std(update_norms)),
                "delta_norm": delta_norm,
                "delta_norm_over_weighted_client_norm": delta_alignment_ratio,
                "train_loss_mean": (
                    float(np.mean(train_losses))
                    if train_losses is not None
                    else float("nan")
                ),
                "train_accuracy_mean": (
                    float(np.mean(train_accuracies))
                    if train_accuracies is not None
                    else float("nan")
                ),
            }
        )


class TracingFedAvg(_InteractionDiagnostics, _EvalTracer, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        _InteractionDiagnostics.__init__(self)

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

    def aggregate_fit(self, server_round, results, failures):
        self._record_interaction_diagnostics(
            server_round=server_round,
            results=list(results),
            method="fedavg",
        )
        return super().aggregate_fit(server_round, results, failures)


class TracingFedAvgM(_InteractionDiagnostics, _EvalTracer, fl.server.strategy.FedAvgM):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAvgM.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        _InteractionDiagnostics.__init__(self)

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

    def aggregate_fit(self, server_round, results, failures):
        self._record_interaction_diagnostics(
            server_round=server_round,
            results=list(results),
            method="fedavgm",
        )
        return super().aggregate_fit(server_round, results, failures)


class TracingFedAdagrad(_EvalTracer, fl.server.strategy.FedAdagrad):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAdagrad.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedAdam(_EvalTracer, fl.server.strategy.FedAdam):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAdam.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedYogi(_EvalTracer, fl.server.strategy.FedYogi):
    def __init__(self, **kwargs):
        fl.server.strategy.FedYogi.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedMedian(_EvalTracer, fl.server.strategy.FedMedian):
    def __init__(self, **kwargs):
        fl.server.strategy.FedMedian.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedTrimmedAvg(_EvalTracer, fl.server.strategy.FedTrimmedAvg):
    def __init__(self, **kwargs):
        fl.server.strategy.FedTrimmedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedProx(_EvalTracer, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        fl.server.strategy.FedProx.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)
