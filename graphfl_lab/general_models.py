"""Backward-compatible facade for vision models."""

from graphfl_lab.models.vision import SmallCNN, SmallCNN3ch, SmallMLP, build_model

__all__ = ["SmallCNN", "SmallCNN3ch", "SmallMLP", "build_model"]
