from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph


@dataclass
class ClientGraph:
    data: Data
    num_examples: int


def _build_local_data(
    full_data: Data,
    node_indices: np.ndarray,
    global_test_mask: torch.Tensor,
) -> ClientGraph:
    node_idx_t = torch.tensor(node_indices, dtype=torch.long)
    local_edge_index, _ = subgraph(
        subset=node_idx_t,
        edge_index=full_data.edge_index,
        relabel_nodes=True,
    )

    x = full_data.x[node_idx_t]
    y = full_data.y[node_idx_t]
    train_mask = full_data.train_mask[node_idx_t]
    val_mask = full_data.val_mask[node_idx_t]

    # Keep evaluation stable across clients by reusing the original test split.
    test_mask = global_test_mask[node_idx_t]

    data = Data(
        x=x,
        edge_index=local_edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return ClientGraph(data=data, num_examples=int(train_mask.sum().item()))


def _iid_partition(rng: np.random.Generator, n_nodes: int, num_clients: int):
    perm = rng.permutation(n_nodes)
    return np.array_split(perm, num_clients)


def _dirichlet_partition(
    rng: np.random.Generator,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
) -> List[np.ndarray]:
    """Label-Dirichlet partition (non-IID).

    For each class c, draw client mixture proportions p ~ Dirichlet(alpha) and
    split that class's node indices accordingly. Smaller alpha ⇒ stronger
    label skew across clients.
    """
    n_classes = int(labels.max()) + 1
    client_buckets: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
        # Convert proportions to absolute split sizes that sum to len(idx_c).
        cuts = (np.cumsum(proportions) * idx_c.size).astype(int)
        cuts[-1] = idx_c.size
        prev = 0
        for k, end in enumerate(cuts):
            client_buckets[k].extend(idx_c[prev:end].tolist())
            prev = end
    return [np.array(bucket, dtype=np.int64) for bucket in client_buckets]


def load_cora_clients(
    root: str,
    num_clients: int,
    seed: int,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
) -> List[ClientGraph]:
    dataset = Planetoid(root=root, name="Cora")
    full_data = dataset[0]

    rng = np.random.default_rng(seed)
    if partition == "iid":
        shards = _iid_partition(rng, full_data.num_nodes, num_clients)
    elif partition == "dirichlet":
        shards = _dirichlet_partition(
            rng=rng,
            labels=full_data.y.cpu().numpy(),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
        )
    else:
        raise ValueError(f"Unknown partition mode: {partition}")

    clients: List[ClientGraph] = []
    for shard in shards:
        clients.append(
            _build_local_data(
                full_data=full_data,
                node_indices=shard,
                global_test_mask=full_data.test_mask,
            )
        )
    return clients
