"""Model definitions."""

from spectral_fl.models.cora import GCN
from spectral_fl.models.vision import SmallCNN, SmallCNN3ch, SmallMLP, build_model

__all__ = ["GCN", "SmallCNN", "SmallCNN3ch", "SmallMLP", "build_model"]
