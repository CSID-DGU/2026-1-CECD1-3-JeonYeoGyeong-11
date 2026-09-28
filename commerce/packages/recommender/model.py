"""Recommender core for both variants (model.md §3·§6, comparison.md §2).

    e[c] = fusion(z[c], l[c])        text_only: l = 0 from the start
    l[c] = 0 if c has no neighbour else relation_pool(mean_j relation_mlp(R[c, j], q[c->j], q[j->c], z[j]))
    q[c->j] = sum over gap bins of share(bin) * time_mlp(bin)       = mean_d time_mlp(d)
    b[v] = basket_encoder(mean of e over visit v's items)
    h[u] = sequence(b + seq_time_pos(recency, gap bits))   at the most recent visit
    score[u, c] = query_proj(h[u]) . scorer(e[c])

Module names are the shared groups: text_relation has nine, text_only the six
without time_mlp, relation_mlp and relation_pool. There is no item-ID embedding
or item bias: an item is its text vector plus, for text_relation, what its local
neighbours look like. Candidate e keeps its gradient because e is recomputed for
the whole seller catalog.
"""
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from commerce.packages.recommender.examples import Example
from commerce.packages.recommender.relations import REL_FEATURES, RelationTensors, bin_inputs
from commerce.packages.recommender.transfer import to_device

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
    d_time: int | None = None  # time_mlp width; text_relation only


# Immutable registry (model-lab.md §6): a version never changes meaning once used.
ARCHITECTURES = {
    "text_only.v1": ModelConfig("text_only.v1", "text_only"),
    "text_relation.v1": ModelConfig("text_relation.v1", "text_relation", d_time=16),
}


def architecture(version: str) -> ModelConfig:
    return ARCHITECTURES[version]


def build_model(config: ModelConfig) -> "Recommender":
    return {"text_only": TextOnlyRecommender, "text_relation": TextRelationRecommender}[config.model_variant](config)


class Recommender(nn.Module):
    """The six groups both variants share; subclasses define item_repr."""
    GROUPS = ("fusion", "basket_encoder", "seq_time_pos", "sequence", "query_proj", "scorer")
    VARIANT = ""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.model_variant != self.VARIANT:
            raise ValueError("config is for %s, this module is %s" % (config.model_variant, self.VARIANT))
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

    def item_repr(self, z: torch.Tensor, relations: RelationTensors | None = None) -> torch.Tensor:
        raise NotImplementedError

    def query(self, e: torch.Tensor, batch: "HistoryBatch") -> torch.Tensor:
        items = e[batch.item_index.clamp(min=0)] * batch.item_mask.unsqueeze(-1)
        counts = batch.item_mask.sum(-1, keepdim=True).clamp(min=1)
        baskets = self.basket_encoder(items.sum(-2) / counts)
        steps = baskets + self.seq_time_pos["position"](batch.recency) + self.seq_time_pos["time"](batch.time)
        hidden = self.sequence(steps, src_key_padding_mask=~batch.visit_mask)
        return self.query_proj(hidden[:, -1])  # the most recent visit sits last

    def score(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ self.score_side(e).T / self.score_scale

    def score_side(self, e: torch.Tensor) -> torch.Tensor:
        return self.scorer(e)

    @property
    def score_scale(self) -> float:
        return math.sqrt(self.config.d_model)

    # The interface training.py uses, shared with the HAREX-style backbone (harex.py).
    def encode_items(self, seller) -> torch.Tensor:
        return self.item_repr(seller.z, seller.relations)

    def encode_queries(self, e: torch.Tensor, examples, seller) -> torch.Tensor:
        return self.query(e, history_batch(examples, seller.row_of, self.config))


class TextOnlyRecommender(Recommender):
    VARIANT = "text_only"

    def item_repr(self, z: torch.Tensor, relations: RelationTensors | None = None) -> torch.Tensor:
        if relations is not None:
            raise ValueError("text_only takes no relations")
        zero_l = z.new_zeros(*z.shape[:-1], self.config.d_relation)
        return self.fusion(torch.cat([z, zero_l], dim=-1))


class TextRelationRecommender(Recommender):
    GROUPS = Recommender.GROUPS + ("time_mlp", "relation_mlp", "relation_pool")
    VARIANT = "text_relation"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        t, h, r = config.d_time, config.mlp_hidden, config.d_relation
        self.time_mlp = nn.Sequential(nn.Linear(2, t), nn.GELU(), nn.Linear(t, t))
        # One MLP over [pair features, q[c->j], q[j->c], z[j]]; its first layer is split so the
        # z[j] part is computed once per item instead of once per pair.
        self.relation_mlp = nn.ModuleDict({"pair": nn.Linear(REL_FEATURES + 2 * t, h),
                                           "text": nn.Linear(config.d_text, h, bias=False),
                                           "out": nn.Linear(h, r)})
        self.relation_pool = nn.Sequential(nn.Linear(r, r), nn.GELU(), nn.Linear(r, r))
        self.register_buffer("time_bins", bin_inputs(), persistent=False)

    def relation_repr(self, z: torch.Tensor, relations: RelationTensors) -> torch.Tensor:
        times = self.time_mlp(self.time_bins)  # (bins, d_time)
        forward = relations.time_forward.float() @ times  # mean over that pair's gaps
        backward = relations.time_backward.float() @ times
        pair = self.relation_mlp["pair"](torch.cat([relations.features, forward, backward], dim=-1))
        text = self.relation_mlp["text"](z)[relations.neighbor.clamp(min=0)]
        per_pair = self.relation_mlp["out"](F.gelu(pair + text))
        mask = (relations.neighbor >= 0).unsqueeze(-1).to(per_pair.dtype)
        pooled = self.relation_pool((per_pair * mask).sum(1) / mask.sum(1).clamp(min=1))
        # No neighbour means l = 0 by an explicit branch, not by hoping the MLP maps 0 to 0.
        return torch.where(relations.has_neighbor.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    def item_repr(self, z: torch.Tensor, relations: RelationTensors | None = None) -> torch.Tensor:
        if relations is None:
            raise ValueError("text_relation needs the seller's relations")
        return self.fusion(torch.cat([z, self.relation_repr(z, relations)], dim=-1))


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

    The mask may stay on the CPU where it was built: the count then needs no wait
    on the device. A skipped row keeps only its positive, so its loss is exactly 0.
    """
    usable = negative_mask.any(dim=1)
    used = int(usable.sum())
    if used == 0:
        return positive.sum() * 0.0, 0
    mask = to_device(negative_mask, negatives.device)
    logits = torch.cat([positive.unsqueeze(1), negatives.masked_fill(~mask, float("-inf"))], dim=1)
    per_example = -F.log_softmax(logits, dim=1)[:, 0] * to_device(usable, logits.device).to(logits.dtype)
    return per_example.sum() / used, used


def config_record(config: ModelConfig) -> dict:
    return asdict(config)
