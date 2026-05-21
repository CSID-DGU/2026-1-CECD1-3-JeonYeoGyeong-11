"""Stable client ordering for graph-aware strategy logs."""

from __future__ import annotations

from typing import List, Tuple

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy


def _fit_result_cid_key(item: Tuple[ClientProxy, FitRes]) -> Tuple[int, int | str]:
    """Return a stable client-id sort key for Flower fit results."""
    proxy, fit_res = item
    metrics = dict(fit_res.metrics or {})
    raw_cid = metrics.get("cid", getattr(proxy, "cid", ""))
    try:
        return (0, int(raw_cid))
    except (TypeError, ValueError):
        return (1, str(raw_cid))


def sort_fit_results_by_cid(
    results: List[Tuple[ClientProxy, FitRes]]
) -> List[Tuple[ClientProxy, FitRes]]:
    """Keep graph rows aligned with client ids across rounds and runs."""
    return sorted(results, key=_fit_result_cid_key)
