"""Graph-level summary metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_graph_diagnostics(w: np.ndarray) -> Dict[str, Any]:
    """Return density, degree list, edge count, and emptiness flag."""
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    upper = w[iu]
    n_edges = int(np.sum(upper > 0.0))
    n_possible = max(int(n * (n - 1) // 2), 1)
    density = float(n_edges / n_possible)
    degrees = np.sum(w > 0.0, axis=1).astype(int).tolist()
    return {
        "graph_density": float(density),
        "graph_degree_list": [int(x) for x in degrees],
        "number_of_edges": int(n_edges),
        "graph_empty": bool(n_edges == 0),
    }
