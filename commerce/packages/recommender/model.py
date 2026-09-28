"""text_only recommender core (model.md §3·§6, comparison.md §2).

    e[c] = fusion(z[c], l = 0)                      l is a fixed zero for text_only
    b[v] = basket_encoder(mean of e over visit v's items)
    h[u] = sequence(b + seq_time_pos(recency, gap bits))   at the most recent visit
    score[u, c] = query_proj(h[u]) . scorer(e[c])

Module names are the shared groups; text_only has six of the nine. There is no
item-ID embedding or item bias: an item is its text vector. Candidate e keeps
its gradient because e is recomputed from z for the whole seller catalog.
"""
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from commerce.packages.recommender.examples import Example

GAP_CAP_DAYS = 30.0


@dataclass(frozen=True)
class ModelConfig:
    architecture_version: str
    model_variant: str
    d_text: int = 384
    d_relation: int = 64  # width of l in the fusion input; all zero for text_only
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ffn: int = 256
    mlp_hidden: int = 128
    dropout: float = 0.1
    max_visits: int = 10
    max_items: int = 32


# Immutable registry (model-lab.md §6): a version never changes meaning once used.
ARCHITECTURES = {
    "text_only.v1": ModelConfig("text_only.v1", "text_only"),
}


def architecture(version: str) -> ModelConfig:
    return ARCHITECTURES[version]


class TextOnlyRecommender(nn.Module):
    GROUPS = ("fusion", "basket_encoder", "seq_time_pos", "sequence", "query_proj", "scorer")

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.model_variant != "text_only":
            raise ValueError("this module is the text_only variant")
        self.config = config
        d = config.d_model
        self.fusion = nn.Sequential(nn.Linear(config.d_text + config.d_relation, d), nn.LayerNorm(d))
        self.basket_encoder = nn.Sequential(nn.Linear(d, config.mlp_hidden), nn.GELU(),
                                            nn.Linear(config.mlp_hidden, d), nn.LayerNorm(d))
        self.seq_time_pos = nn.ModuleDict({"position": nn.Embedding(config.max_visits, d),
                                           "time": nn.Linear(3, d)})
        layer = nn.TransformerEncoderLayer(d, config.n_heads, config.d_ffn, config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.query_proj = nn.Linear(d, d)
        # No bias: q . b would add the same constant to every candidate of a query, so it has no
        # gradient and would only drift with rounding noise.
        self.scorer = nn.Linear(d, d, bias=False)

    def item_repr(self, z: torch.Tensor) -> torch.Tensor:
        zero_l = z.new_zeros(*z.shape[:-1], self.config.d_relation)
        return self.fusion(torch.cat([z, zero_l], dim=-1))

    def query(self, e: torch.Tensor, batch: "HistoryBatch") -> torch.Tensor:
        items = e[batch.item_index.clamp(min=0)] * batch.item_mask.unsqueeze(-1)
        counts = batch.item_mask.sum(-1, keepdim=True).clamp(min=1)
        baskets = self.basket_encoder(items.sum(-2) / counts)
        steps = baskets + self.seq_time_pos["position"](batch.recency) + self.seq_time_pos["time"](batch.time)
        hidden = self.sequence(steps, src_key_padding_mask=~batch.visit_mask)
        return self.query_proj(hidden[:, -1])  # the most recent visit sits last

    def score(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ self.scorer(e).T / math.sqrt(self.config.d_model)


@dataclass
class HistoryBatch:
    item_index: torch.Tensor  # (B, L, K) catalog rows, -1 for padding
    item_mask: torch.Tensor  # (B, L, K) float
    visit_mask: torch.Tensor  # (B, L) bool, True for a real visit; visits are right-aligned
    recency: torch.Tensor  # (B, L) 0 for the most recent visit
    time: torch.Tensor  # (B, L, 3): log1p(min(gap, 30)) / log1p(30), gap missing, gap censored


def history_batch(examples: Sequence[Example], row_of: dict[str, int], config: ModelConfig) -> HistoryBatch:
    b, length, k = len(examples), config.max_visits, config.max_items
    index = torch.full((b, length, k), -1, dtype=torch.long)
    time = torch.zeros((b, length, 3))
    visit_mask = torch.zeros((b, length), dtype=torch.bool)
    for i, example in enumerate(examples):
        offset = length - len(example.history)
        for j, visit in enumerate(example.history):
            rows = [row_of[item] for item in visit.items]
            index[i, offset + j, :len(rows)] = torch.tensor(rows, dtype=torch.long)
            visit_mask[i, offset + j] = True
            if visit.gap_days is None:
                time[i, offset + j, 1] = 1.0
            else:
                time[i, offset + j, 0] = math.log1p(min(visit.gap_days, GAP_CAP_DAYS)) / math.log1p(GAP_CAP_DAYS)
                time[i, offset + j, 2] = float(visit.gap_censored)
    recency = torch.arange(length - 1, -1, -1).expand(b, length).clone()
    # Padding slots still need a valid position id; the key padding mask hides them.
    return HistoryBatch(index, (index >= 0).float(), visit_mask, recency, time)


def sampled_softmax_loss(positive: torch.Tensor, negatives: torch.Tensor,
                         negative_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """OQ01: one positive per example against shared negatives, mean over usable examples.

    positive (B,), negatives (B, N), negative_mask (B, N) True where a negative is
    allowed, i.e. not in that example's target set. An example with no allowed
    negative is skipped and not counted.
    """
    logits = torch.cat([positive.unsqueeze(1), negatives.masked_fill(~negative_mask, float("-inf"))], dim=1)
    usable = negative_mask.any(dim=1)
    if not usable.any():
        return positive.sum() * 0.0, 0
    per_example = -F.log_softmax(logits[usable], dim=1)[:, 0]
    return per_example.mean(), int(usable.sum())


def config_record(config: ModelConfig) -> dict:
    return asdict(config)
