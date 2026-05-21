"""FedNova-style normalized averaging baseline."""

from __future__ import annotations

from typing import Any, Dict

import flwr as fl
import numpy as np
from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays

from graphfl_lab.strategies.baselines.ordering import sort_fit_results_by_cid
from graphfl_lab.strategies.baselines.tracing import _EvalTracer


class TracingFedNova(_EvalTracer, fl.server.strategy.FedAvg):
    """FedNova-style normalized averaging baseline.

    Each client reports ``local_steps``. The server aggregates
    ``(w_i - w_t) / local_steps`` and scales the averaged normalized update by
    the sample-weighted average step count, which reduces to FedAvg when all
    clients take the same number of optimizer steps.
    """

    def __init__(self, *, server_learning_rate: float = 1.0, **kwargs):
        initial_parameters = kwargs.get("initial_parameters")
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        self.server_learning_rate = float(server_learning_rate)
        self.current_global: NDArrays | None = (
            parameters_to_ndarrays(initial_parameters)
            if initial_parameters is not None
            else None
        )

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}
        ordered = sort_fit_results_by_cid(list(results))
        local_weights = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in ordered
        ]
        if self.current_global is None:
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, ordered, failures
            )
            if parameters_aggregated is not None:
                self.current_global = parameters_to_ndarrays(parameters_aggregated)
            return parameters_aggregated, metrics_aggregated

        n_examples = np.array(
            [float(fit_res.num_examples) for _, fit_res in ordered],
            dtype=np.float64,
        )
        total_examples = float(n_examples.sum())
        if total_examples <= 0.0:
            alpha = np.ones(len(ordered), dtype=np.float64) / float(len(ordered))
        else:
            alpha = n_examples / total_examples
        local_steps = np.array(
            [
                max(1.0, float((fit_res.metrics or {}).get("local_steps", 1.0)))
                for _, fit_res in ordered
            ],
            dtype=np.float64,
        )
        tau_eff = float(np.dot(alpha, local_steps))
        normalized_update: NDArrays = [
            np.zeros_like(base) for base in self.current_global
        ]
        for weight, tau_i, local in zip(alpha, local_steps, local_weights):
            for idx, (base, arr) in enumerate(zip(self.current_global, local)):
                normalized_update[idx] += float(weight) * (arr - base) / float(tau_i)
        new_global: NDArrays = [
            base + self.server_learning_rate * tau_eff * upd
            for base, upd in zip(self.current_global, normalized_update)
        ]
        self.current_global = [np.array(x, copy=True) for x in new_global]

        metrics_aggregated: Dict[str, Any] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics) for _, fit_res in ordered]
            )
        metrics_aggregated.update(
            {
                "fednova_tau_eff": tau_eff,
                "fednova_tau_min": float(local_steps.min()),
                "fednova_tau_max": float(local_steps.max()),
                "fednova_server_learning_rate": float(self.server_learning_rate),
            }
        )
        return ndarrays_to_parameters(new_global), metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)
