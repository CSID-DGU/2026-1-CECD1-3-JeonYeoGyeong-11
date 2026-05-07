"""Small tracing helpers for spectral strategy round logs."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def matrix_log_if_small(
    matrix: np.ndarray,
    max_clients: int,
) -> Optional[List[List[float]]]:
    """Return a JSON-friendly matrix only when the client count is small."""
    if int(matrix.shape[0]) > int(max_clients):
        return None
    return [[float(v) for v in row] for row in matrix.tolist()]


__all__ = ["matrix_log_if_small"]
