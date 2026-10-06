"""Seller-local item-item relations for text_relation (model.md §3).

From one snapshot of a seller's visible purchases, three relations between items:

- customer: R^X = X^T X with X[u, c] = 1 if customer u bought c (quantity ignored)
- basket:   R^B = B^T B with B[b, c] = 1 if basket b holds c
- time:     item c in a visit and item j in the same customer's next visit, d days
            later. c->j and j->c stay apart. Two items of one basket are never a time
            pair; that is the basket relation.

Strength is R_cj / sqrt(R_cc R_jj) and support log1p(R_cj); a zero co-count means
not observed. Neighbours of c are the top NEIGHBOR_K of each relation by support
(ties: catalog row), merged round-robin customer, basket, time up to N_MAX, never c
itself. An item with no neighbour gets no relation, and the model sets l = 0 for it.

Time gaps fall into TIME_BINS bins so that mean_d time_mlp(d) is a weighted sum
over bins: whole days 0..29, bin 30 "30 or more, observed", bin 31 "capped at 30,
censored" (Instacart). Instacart gaps are whole days, so its bins are exact;
absolute-time sources are floored to days. These rules and constants are part of
preprocessing_version.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from commerce.packages.data_adapters.baskets import Visit
from commerce.packages.recommender.replay import visible_prefix

NEIGHBOR_K = 8
N_MAX = 16
TIME_BINS = 32
CENSORED_BIN = 31
OPEN_BIN = 30
# Pair features, in order: customer strength, customer support, basket strength, basket
# support, log1p customers of j, log1p baskets of j, has c->j, has j->c, log1p #c->j, log1p #j->c.
REL_FEATURES = 10


def gap_bin(gap_days: float, censored: bool) -> int:
    if censored:
        return CENSORED_BIN
    return OPEN_BIN if gap_days >= OPEN_BIN else int(gap_days)


def bin_inputs() -> torch.Tensor:
    """time_mlp input per bin: [log1p(min(d, 30)) / log1p(30), censored bit]."""
    days = [min(b, 30) for b in range(TIME_BINS - 1)] + [30]
    censored = [0.0] * (TIME_BINS - 1) + [1.0]
    return torch.tensor([[math.log1p(d) / math.log1p(30), c] for d, c in zip(days, censored)])


@dataclass
class RelationTensors:
    neighbor: torch.Tensor  # (C, N_MAX) catalog rows, -1 for padding
    features: torch.Tensor  # (C, N_MAX, REL_FEATURES) float32
    time_forward: torch.Tensor  # (C, N_MAX, TIME_BINS) float16, share of c->j pairs per bin
    time_backward: torch.Tensor  # (C, N_MAX, TIME_BINS) float16, share of j->c pairs per bin
    has_neighbor: torch.Tensor  # (C,) bool

    @property
    def neighbors_per_item(self) -> float:
        return float((self.neighbor >= 0).sum()) / max(1, len(self.neighbor))


def build_relations(visits_by_customer: Mapping[str, Sequence[Visit]], row_of: Mapping[str, int],
                    n_items: int) -> RelationTensors:
    """Relations from exactly the visits given; the caller cuts each history to its snapshot."""
    x_self, b_self = Counter(), Counter()
    x_pair, b_pair = Counter(), Counter()
    time_pairs: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for visits in visits_by_customer.values():
        bought = set()
        for i, visit in enumerate(visits):
            rows = sorted({row_of[item] for item in visit.basket.item_ids})
            bought.update(rows)
            b_self.update(rows)
            b_pair.update(combinations(rows, 2))
            if i + 1 < len(visits):
                nxt = visits[i + 1]
                bin_ = gap_bin(nxt.gap_days, nxt.gap_censored)
                later = {row_of[item] for item in nxt.basket.item_ids}
                for c in rows:
                    for j in later:
                        if c != j:
                            time_pairs[(c, j)][bin_] += 1
        rows = sorted(bought)
        x_self.update(rows)
        x_pair.update(combinations(rows, 2))

    def adjacency(pairs):
        adj = defaultdict(dict)
        for (a, b), n in pairs.items():
            adj[a][b] = n
            adj[b][a] = n
        return adj
    x_adj, b_adj = adjacency(x_pair), adjacency(b_pair)
    t_adj = defaultdict(dict)
    for (c, j), bins in time_pairs.items():
        n = sum(bins.values())
        t_adj[c][j] = t_adj[c].get(j, 0) + n
        t_adj[j][c] = t_adj[j].get(c, 0) + n

    neighbor = np.full((n_items, N_MAX), -1, dtype=np.int64)
    features = np.zeros((n_items, N_MAX, REL_FEATURES), dtype=np.float32)
    forward = np.zeros((n_items, N_MAX, TIME_BINS), dtype=np.float16)
    backward = np.zeros((n_items, N_MAX, TIME_BINS), dtype=np.float16)
    for c in range(n_items):
        ranked = [sorted(adj.get(c, {}).items(), key=lambda kv: (-kv[1], kv[0]))[:NEIGHBOR_K]
                  for adj in (x_adj, b_adj, t_adj)]
        chosen = []
        for depth in range(NEIGHBOR_K):
            for lst in ranked:
                if depth < len(lst) and lst[depth][0] not in chosen and len(chosen) < N_MAX:
                    chosen.append(lst[depth][0])
        for slot, j in enumerate(chosen):
            xcj, bcj = x_adj[c].get(j, 0), b_adj[c].get(j, 0)
            fwd, bwd = time_pairs.get((c, j), Counter()), time_pairs.get((j, c), Counter())
            n_fwd, n_bwd = sum(fwd.values()), sum(bwd.values())
            neighbor[c, slot] = j
            features[c, slot] = [
                xcj / math.sqrt(x_self[c] * x_self[j]) if xcj else 0.0, math.log1p(xcj),
                bcj / math.sqrt(b_self[c] * b_self[j]) if bcj else 0.0, math.log1p(bcj),
                math.log1p(x_self[j]), math.log1p(b_self[j]),
                float(n_fwd > 0), float(n_bwd > 0), math.log1p(n_fwd), math.log1p(n_bwd),
            ]
            for hist, out in ((fwd, forward), (bwd, backward)):
                total = sum(hist.values())
                for bin_, n in hist.items():
                    out[c, slot, bin_] = n / total
    return RelationTensors(torch.from_numpy(neighbor), torch.from_numpy(features),
                           torch.from_numpy(forward), torch.from_numpy(backward),
                           torch.from_numpy((neighbor >= 0).any(axis=1)))


def replay_relations(visits_by_customer: Mapping[str, Sequence[Visit]], bucket: int, row_of: Mapping[str, int],
                     n_items: int) -> RelationTensors:
    """Instacart snapshot at a progress bucket: each customer's first floor(t n / 10) visits only."""
    visible = {c: v[:visible_prefix(bucket, len(v))] for c, v in visits_by_customer.items()}
    return build_relations(visible, row_of, n_items)
