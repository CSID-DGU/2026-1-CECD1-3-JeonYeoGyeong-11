"""Flower fit-result collection helpers for GraphFL strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from flwr.common import FitRes, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from graphfl_lab.strategies.baselines.ordering import sort_fit_results_by_cid


@dataclass(frozen=True)
class ClientFitBatch:
    cids: List[str]
    local_weights: List[NDArrays]
    n_examples: List[int]
    n_examples_arr: np.ndarray
    client_metrics: List[Dict[str, Any]]


def collect_client_fit_batch(
    results: List[Tuple[ClientProxy, FitRes]],
) -> ClientFitBatch:
    cids: List[str] = []
    local_weights: List[NDArrays] = []
    n_examples: List[int] = []
    client_metrics: List[Dict[str, Any]] = []
    for proxy, fit_res in sort_fit_results_by_cid(results):
        metrics = dict(fit_res.metrics or {})
        cids.append(str(metrics.get("cid", getattr(proxy, "cid", "?"))))
        local_weights.append(parameters_to_ndarrays(fit_res.parameters))
        n_examples.append(int(fit_res.num_examples))
        client_metrics.append(metrics)

    return ClientFitBatch(
        cids=cids,
        local_weights=local_weights,
        n_examples=n_examples,
        n_examples_arr=np.array(n_examples, dtype=np.float64),
        client_metrics=client_metrics,
    )


__all__ = ["ClientFitBatch", "collect_client_fit_batch"]
