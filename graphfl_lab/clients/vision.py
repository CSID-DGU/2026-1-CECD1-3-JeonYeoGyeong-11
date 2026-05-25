"""Flower NumPyClient for torchvision-backed vision FL."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from graphfl_lab.models.vision import build_model


def _seed_for(seed: int, server_round: int, cid: int) -> int:
    return (int(seed) * 100003 + int(server_round) * 1009 + int(cid)) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed) & 0x7FFFFFFF)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)


class VisionFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_dataset: Dataset,
        train_indices: np.ndarray,
        test_dataset: Dataset,
        model_name: str,
        num_classes: int,
        in_channels: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        batch_size: int,
        local_epochs: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.cid = int(cid)
        self.batch_size = int(batch_size)
        self.local_epochs = int(local_epochs)
        self.device = device
        self.seed = int(seed)
        self.train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, train_indices.tolist()),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
        )
        self.model = build_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.optim = self._build_optimizer()
        self.num_examples = int(len(train_indices))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def get_parameters(self, config):  # noqa: ARG002
        return get_parameters(self.model)

    def fit(self, parameters, config):
        seed = int(config.get("seed", self.seed)) if isinstance(config, dict) else self.seed
        server_round = int(config.get("server_round", 0)) if isinstance(config, dict) else 0
        seed_everything(_seed_for(seed=seed, server_round=server_round, cid=self.cid))
        set_parameters(self.model, parameters)
        # Local optimizer state should not leak across FL rounds after the
        # server replaces the model parameters.
        self.optim = self._build_optimizer()
        proximal_mu = (
            float(config.get("proximal_mu", 0.0))
            if isinstance(config, dict)
            else 0.0
        )
        global_params = [
            p.detach().clone()
            for p in self.model.parameters()
        ] if proximal_mu > 0.0 else []
        self.model.train()
        loss_sum = 0.0
        n_seen = 0
        correct = 0
        total = 0
        local_steps = 0
        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optim.zero_grad()
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                if proximal_mu > 0.0:
                    prox = torch.zeros((), device=self.device)
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((param - global_param) ** 2)
                    loss = loss + 0.5 * proximal_mu * prox
                loss.backward()
                self.optim.step()
                local_steps += 1
                loss_sum += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += xb.size(0)
        train_loss = loss_sum / max(n_seen, 1)
        train_acc = correct / max(total, 1)
        return (
            get_parameters(self.model),
            self.num_examples,
            {
                "cid": int(self.cid),
                "local_steps": int(local_steps),
                "train_accuracy": float(train_acc),
                "train_loss": float(train_loss),
            },
        )

    @torch.no_grad()
    def evaluate(self, parameters, config):  # noqa: ARG002
        # Single-client eval avoids repeating full test for every FL client.
        if self.cid != 0:
            return 0.0, 0, {"accuracy": 0.0}
        set_parameters(self.model, parameters)
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        for xb, yb in self.test_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
        loss = loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        n_eval = total
        return float(loss), n_eval, {"accuracy": float(acc)}
