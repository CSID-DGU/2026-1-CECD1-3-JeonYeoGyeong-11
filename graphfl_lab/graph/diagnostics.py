"""Graph-level summary metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_graph_diagnostics(w: np.ndarray) -> Dict[str, Any]:
    """Return graph density, degree, entropy, edge count, and emptiness flag."""
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    upper = w[iu]
    n_edges = int(np.sum(upper > 0.0))
    n_possible = max(int(n * (n - 1) // 2), 1)
    density = float(n_edges / n_possible)
    degrees = np.sum(w > 0.0, axis=1).astype(int).tolist()
    edge_weights = upper[upper > 0.0].astype(np.float64, copy=False)
    if edge_weights.size <= 1:
        entropy = 0.0
    else:
        p = edge_weights / (float(np.sum(edge_weights)) + 1e-12)
        entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(p.size))
        entropy = float(np.clip(entropy, 0.0, 1.0))
    return {
        "graph_num_nodes": int(n),
        "graph_density": float(density),
        "graph_entropy": float(entropy),
        "graph_degree_list": [int(x) for x in degrees],
        "graph_degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "graph_degree_min": int(min(degrees)) if degrees else 0,
        "graph_degree_max": int(max(degrees)) if degrees else 0,
        "number_of_edges": int(n_edges),
        "graph_empty": bool(n_edges == 0),
    }
