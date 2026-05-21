"""Model definitions."""

from graphfl_lab.models.cora import GCN
from graphfl_lab.models.vision import SmallCNN, SmallCNN3ch, SmallMLP, build_model

__all__ = ["GCN", "SmallCNN", "SmallCNN3ch", "SmallMLP", "build_model"]
