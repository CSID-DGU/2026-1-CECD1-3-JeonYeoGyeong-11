"""Torchvision datasets + IID / Dirichlet partitioning for general FL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class VisionClientShard:
    """Per-client train shard and metadata for logging."""

    cid: int
    train_indices: np.ndarray
    label_hist: List[int]


def _iid_partition(rng: np.random.Generator, n_samples: int, num_clients: int) -> List[np.ndarray]:
    perm = rng.permutation(n_samples)
    return [x.astype(np.int64) for x in np.array_split(perm, num_clients)]


def _dirichlet_partition(
    rng: np.random.Generator,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
) -> List[np.ndarray]:
    n_classes = int(labels.max()) + 1
    client_buckets: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
        cuts = (np.cumsum(proportions) * idx_c.size).astype(int)
        cuts[-1] = idx_c.size
        prev = 0
        for k, end in enumerate(cuts):
            client_buckets[k].extend(idx_c[prev:end].tolist())
            prev = end
    return [np.array(bucket, dtype=np.int64) for bucket in client_buckets]


def _ensure_min_samples_per_client(
    rng: np.random.Generator,
    shards: Sequence[np.ndarray],
    min_samples: int,
) -> List[np.ndarray]:
    """Move samples from large shards so every client can build a DataLoader."""
    buckets = [list(map(int, shard.tolist())) for shard in shards]
    min_n = max(int(min_samples), 0)
    if min_n <= 0:
        return [np.array(bucket, dtype=np.int64) for bucket in buckets]
    if sum(len(bucket) for bucket in buckets) < len(buckets) * min_n:
        raise ValueError(
            "Not enough training samples to satisfy min_samples_per_client="
            f"{min_n} for {len(buckets)} clients"
        )

    for cid in range(len(buckets)):
        while len(buckets[cid]) < min_n:
            donor = max(range(len(buckets)), key=lambda j: len(buckets[j]))
            if donor == cid or len(buckets[donor]) <= min_n:
                raise ValueError(
                    "Could not rebalance client shards to satisfy "
                    f"min_samples_per_client={min_n}"
                )
            take_pos = int(rng.integers(0, len(buckets[donor])))
            buckets[cid].append(buckets[donor].pop(take_pos))
    return [np.array(bucket, dtype=np.int64) for bucket in buckets]


def _take_subset(indices: np.ndarray, max_size: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_size is None or max_size <= 0 or indices.size <= max_size:
        return indices
    rng.shuffle(indices)
    return indices[:max_size]


def _build_torchvision_train(
    name: str,
    root: str,
    train: bool,
    download: bool,
):
    name_l = name.strip().lower()
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor()])
    if name_l == "mnist":
        return datasets.MNIST(root=root, train=train, download=download, transform=tf)
    if name_l == "fashionmnist":
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=tf)
    if name_l == "cifar10":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        return datasets.CIFAR10(root=root, train=train, download=download, transform=tf)
    raise ValueError(f"Unknown dataset: {name}")


def load_vision_clients(
    dataset_name: str,
    root: str,
    num_clients: int,
    seed: int,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
    train_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    min_samples_per_client: int = 1,
) -> Tuple[List[VisionClientShard], Dataset, Dataset]:
    """Return client shards, full (possibly subset) train dataset, test dataset."""

    rng = np.random.default_rng(seed)
    train_full = _build_torchvision_train(dataset_name, root, train=True, download=True)
    test_full = _build_torchvision_train(dataset_name, root, train=False, download=True)

    n_train = len(train_full)
    labels = np.array([train_full[i][1] for i in range(n_train)], dtype=np.int64)

    train_indices = np.arange(n_train, dtype=np.int64)
    train_indices = _take_subset(train_indices, train_subset_size, rng)

    labels_sub = labels[train_indices]
    idx_map = train_indices

    if partition == "iid":
        rel_shards = _iid_partition(rng, len(idx_map), num_clients)
    elif partition == "dirichlet":
        rel_shards = _dirichlet_partition(rng, labels_sub, num_clients, dirichlet_alpha)
    else:
        raise ValueError(f"Unknown partition: {partition}")
    rel_shards = _ensure_min_samples_per_client(
        rng=rng,
        shards=rel_shards,
        min_samples=min_samples_per_client,
    )

    n_cls = int(labels_sub.max()) + 1 if labels_sub.size else 0
    shards: List[VisionClientShard] = []
    for cid, rel in enumerate(rel_shards):
        absolute = idx_map[rel]
        hist = np.bincount(labels[absolute], minlength=n_cls).tolist() if absolute.size else [0] * n_cls
        shards.append(VisionClientShard(cid=cid, train_indices=absolute, label_hist=[int(x) for x in hist]))

    if test_subset_size is not None and test_subset_size > 0 and test_subset_size < len(test_full):
        test_ix = rng.permutation(len(test_full))[:test_subset_size]
        test_ds: Dataset = Subset(test_full, test_ix.tolist())
    else:
        test_ds = test_full

    train_ds: Dataset = train_full
    return shards, train_ds, test_ds


def vision_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    n = dataset_name.strip().lower()
    if n == "cifar10":
        return (3, 32, 32)
    return (1, 28, 28)


def vision_num_classes(dataset_name: str) -> int:
    n = dataset_name.strip().lower()
    if n == "cifar10":
        return 10
    return 10
