"""Small vision models for vision FL benchmarks."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Light CNN for 1x28x28 inputs such as FashionMNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SmallMLP(nn.Module):
    """Flatten + MLP baseline."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN3ch(nn.Module):
    """Small CNN for 3x32x32 inputs such as CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 10, in_channels: int = 1) -> nn.Module:
    n = name.strip().lower()
    if n == "cnn":
        if in_channels == 3:
            return SmallCNN3ch(num_classes=num_classes)
        return SmallCNN(num_classes=num_classes)
    if n == "mlp":
        return SmallMLP(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
