"""Flower NumPyClient for Cora graph FL."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F

from spectral_fl.data import ClientGraph
from spectral_fl.models.cora import GCN


def _seed_for(seed: int, server_round: int, cid: int) -> int:
    """Deterministic per-(seed, round, client) RNG key."""
    return (int(seed) * 100003 + int(server_round) * 1009 + int(cid)) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed) & 0x7FFFFFFF)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)


def get_parameters(model: torch.nn.Module) -> List:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List) -> None:
    keys = list(model.state_dict().keys())
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_one_client(
    model: GCN,
    data,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    proximal_mu: float = 0.0,
    global_params: List[torch.Tensor] | None = None,
) -> None:
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    data = data.to(device)
    for _ in range(epochs):
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        if proximal_mu > 0.0 and global_params is not None:
            prox = torch.zeros((), device=device)
            for param, global_param in zip(model.parameters(), global_params):
                prox = prox + torch.sum((param - global_param) ** 2)
            loss = loss + 0.5 * proximal_mu * prox
        loss.backward()
        optim.step()


@torch.no_grad()
def evaluate(model: GCN, data, device: torch.device) -> Dict[str, float]:
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    mask = data.test_mask
    if int(mask.sum()) == 0:
        return {"accuracy": 0.0, "loss": 0.0}
    acc = float((pred[mask] == data.y[mask]).float().mean().item())
    loss = float(F.cross_entropy(out[mask], data.y[mask]).item())
    return {"accuracy": acc, "loss": loss}


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_graph: ClientGraph,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        lr: float,
        weight_decay: float,
        local_epochs: int,
        device: torch.device,
        cid: int = 0,
    ) -> None:
        self.cid = int(cid)
        self.model = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
        self.graph = client_graph
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        seed = int(config.get("seed", 0)) if isinstance(config, dict) else 0
        server_round = int(config.get("server_round", 0)) if isinstance(config, dict) else 0
        seed_everything(_seed_for(seed=seed, server_round=server_round, cid=self.cid))
        set_parameters(self.model, parameters)
        proximal_mu = (
            float(config.get("proximal_mu", 0.0))
            if isinstance(config, dict)
            else 0.0
        )
        global_params = [
            p.detach().clone()
            for p in self.model.parameters()
        ] if proximal_mu > 0.0 else None
        train_one_client(
            model=self.model,
            data=self.graph.data,
            device=self.device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.local_epochs,
            proximal_mu=proximal_mu,
            global_params=global_params,
        )
        return (
            get_parameters(self.model),
            self.graph.num_examples,
            {
                "cid": int(self.cid),
                "local_steps": int(self.local_epochs),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        m = evaluate(self.model, self.graph.data, self.device)
        n_eval = int(self.graph.data.test_mask.sum().item())
        if n_eval <= 0:
            n_eval = self.graph.num_examples
        return m["loss"], n_eval, {"accuracy": m["accuracy"]}
