"""Backward-compatible facade for Cora graph-ablation orchestration."""

from spectral_fl.experiments.cora import graph_ablation as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
