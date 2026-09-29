"""HAREX (GCI, Lee et al. AAAI-24) style backbone shared by the D0022 variants.

    input = the customer's most recent purchased items, oldest first (at most max_items)
    h     = TransformerEncoder(1 layer, d_model 128, 4 heads, FFN 256, dropout 0.2)(e[items] + position)
    q     = query_proj(h at the most recent item)
    score = q . scorer(e[c]) / sqrt(d_model) over the seller's whole catalog

GCI generates the next product name and matches it to a real one; here the
output scores the real candidates instead (D0017). There is no customer or item
ID. The four variants differ only in the item representation e:

    hx: mean of the item's word-token embeddings (GCI space tokenization). The
        table follows the seller's own vocabulary, so it is seller-local and, as
        in GCI's glocalization, never part of an FL export ("local_tokens").
    lm: text_proj(z) of the frozen pretrained encoder.
    R:  e = fusion([base; l]) with l pooled over the item's local relations (model.md §3);
    T:  the same fusion with l = 0.

Shared modules are created before the relation modules, so T and R start from
the same shared weights under the same seed (comparison.md §4).
"""
from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from commerce.packages.data_adapters.text import normalize_field
from commerce.packages.recommender.examples import Example
from commerce.packages.recommender.relations import REL_FEATURES, RelationTensors, bin_inputs
from commerce.packages.recommender.transfer import to_device


@dataclass(frozen=True)
class HarexConfig:
    architecture_version: str
    text: str  # "hx" word tokens or "lm" pretrained encoder
    relation: bool
    d_model: int = 128  # GCI: D_MODEL
    n_heads: int = 4  # GCI: NUM_HEADS
    d_ffn: int = 256  # GCI: UNITS
    dropout: float = 0.2  # GCI: DROPOUT
    n_layers: int = 1  # GCI: NUM_LAYERS
    max_items: int = 50  # recent purchased items in the input
    max_tokens: int = 32  # word tokens kept per item text (hx)
    d_text: int = 384  # pretrained encoder width (lm)
    d_relation: int = 64
    mlp_hidden: int = 128
    d_time: int = 16
    # D0023 service configuration: a repeat-purchase path next to the item scores (not the
    # HAREX comparison, which keeps it off as GCI did, D0022).
    repeat: bool = False
    repeat_hidden: int = 32


# Immutable registry (model-lab.md §6).
HAREX_ARCHITECTURES = {version: HarexConfig(version, text, relation, repeat=repeat)
                       for version, text, relation, repeat in (
    ("harex.T_hx.v1", "hx", False, False), ("harex.R_hx.v1", "hx", True, False),
    ("harex.T_lm.v1", "lm", False, False), ("harex.R_lm.v1", "lm", True, False),
    ("harex.T_hx_rep.v1", "hx", False, True), ("harex.R_hx_rep.v1", "hx", True, True),
    ("harex.T_lm_rep.v1", "lm", False, True), ("harex.R_lm_rep.v1", "lm", True, True),
)}
# Per (example, item the customer bought before): log1p(count), count over earlier visits,
# and the recency bin: the last L visits one by one, then "earlier" (L = examples.L_VISITS).
RECENCY_BINS = 11
REPEAT_FEATURES = 2 + RECENCY_BINS


def repeat_features(examples: Sequence[Example], row_of: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, M) catalog rows of each example's earlier purchases (-1 pads) and their (B, M, F) features.

    Read from the example alone: prior_counts covers every earlier visit, history the last ones.
    """
    rows, feats = [], []
    for example in examples:
        n_prior = max(1, example.target_position - 1)
        last = {}
        for back, visit in enumerate(reversed(example.history), start=1):
            for item in visit.items:
                last.setdefault(item, back)
        r, f = [], []
        for item, count in sorted(example.prior_counts.items()):
            if item not in row_of:
                continue
            recency = [0.0] * RECENCY_BINS
            recency[min(last.get(item, RECENCY_BINS), RECENCY_BINS) - 1] = 1.0
            r.append(row_of[item])
            f.append([math.log1p(count), count / n_prior] + recency)
        rows.append(r)
        feats.append(f)
    width = max(1, max(len(r) for r in rows))
    index = torch.full((len(examples), width), -1, dtype=torch.long)
    values = torch.zeros(len(examples), width, REPEAT_FEATURES)
    for i, (r, f) in enumerate(zip(rows, feats)):
        if r:
            index[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            values[i, :len(r)] = torch.tensor(f)
    return index, values
PAD, UNKNOWN = 0, 1


def word_tokens(text: str | None) -> list[str]:
    """GCI space tokenization on the normalized text: whitespace-separated words, case kept."""
    normalized = normalize_field(text)
    return normalized.split(" ") if normalized else []


class WordVocabulary:
    """One seller's word vocabulary. Token 0 pads and 1 stands for a word the seller never listed."""

    def __init__(self, texts: Iterable[str]):
        words = sorted(Counter(w for text in texts for w in word_tokens(text)))
        self.index = {word: i + 2 for i, word in enumerate(words)}

    def __len__(self) -> int:
        return len(self.index) + 2

    def encode(self, texts: Sequence[str], max_tokens: int) -> torch.Tensor:
        rows = [[self.index.get(w, UNKNOWN) for w in word_tokens(text)][:max_tokens] for text in texts]
        out = torch.zeros(len(rows), max(1, max((len(r) for r in rows), default=1)), dtype=torch.long)
        for i, row in enumerate(rows):
            out[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        return out


def item_sequences(examples: Sequence[Example], row_of: dict[str, int], max_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N) catalog rows of each example's most recent items, oldest first and right-aligned; -1 pads."""
    index = torch.full((len(examples), max_items), -1, dtype=torch.long)
    for i, example in enumerate(examples):
        rows = [row_of[item] for visit in example.history for item in visit.items][-max_items:]
        if rows:
            index[i, max_items - len(rows):] = torch.tensor(rows, dtype=torch.long)
    return index, index >= 0


class HarexRecommender(nn.Module):
    GROUPS = ("fusion", "position", "sequence", "query_proj", "scorer")

    def __init__(self, config: HarexConfig, *, vocab_size: int | None = None):
        super().__init__()
        self.config = config
        d = config.d_model
        if config.text == "hx":
            if not vocab_size:
                raise ValueError("hx needs the seller's vocabulary size")
            self.local_tokens = nn.Embedding(vocab_size, d, padding_idx=PAD)
        elif config.text == "lm":
            self.text_proj = nn.Linear(config.d_text, d)
        else:
            raise ValueError("text must be hx or lm")
        self.fusion = nn.Sequential(nn.Linear(d + config.d_relation, d), nn.LayerNorm(d))
        self.position = nn.Embedding(config.max_items, d)
        layer = nn.TransformerEncoderLayer(d, config.n_heads, config.d_ffn, config.dropout, batch_first=True)
        self.sequence = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.query_proj = nn.Linear(d, d)
        # No bias: q . b adds one constant to every candidate of a query (see model.py).
        self.scorer = nn.Linear(d, d, bias=False)
        if config.relation:
            t, h, r = config.d_time, config.mlp_hidden, config.d_relation
            self.time_mlp = nn.Sequential(nn.Linear(2, t), nn.GELU(), nn.Linear(t, t))
            self.relation_mlp = nn.ModuleDict({"pair": nn.Linear(REL_FEATURES + 2 * t, h),
                                               "text": nn.Linear(d, h, bias=False),
                                               "out": nn.Linear(h, r)})
            self.relation_pool = nn.Sequential(nn.Linear(r, r), nn.GELU(), nn.Linear(r, r))
            self.register_buffer("time_bins", bin_inputs(), persistent=False)
        if config.repeat:
            # Made last, so every other group starts from the weights the same seed gives without it.
            self.repeat_mlp = nn.Sequential(nn.Linear(REPEAT_FEATURES, config.repeat_hidden), nn.GELU(),
                                            nn.Linear(config.repeat_hidden, 1))
            self.repeat_gate = nn.Linear(d, 1)

    @property
    def has_extra(self) -> bool:
        return self.config.repeat

    def extra_scores(self, q: torch.Tensor, examples: Sequence[Example], seller) -> torch.Tensor:
        """(B, C) repeat scores: gate(q) * mlp(features) on the customer's earlier purchases, 0 elsewhere."""
        index, values = repeat_features(examples, seller.row_of)
        index, values = to_device(index, q.device), to_device(values, q.device)
        mask = (index >= 0).to(q.dtype)
        per_item = self.repeat_mlp(values).squeeze(-1) * mask
        gate = F.softplus(self.repeat_gate(q))  # (B, 1): how much this query leans on repeats
        out = torch.zeros(q.shape[0], len(seller.items), device=q.device, dtype=q.dtype)
        return out.scatter_add(1, index.clamp(min=0), gate * per_item)

    @property
    def device(self) -> torch.device:
        return self.fusion[0].weight.device

    def shared_state(self) -> dict[str, torch.Tensor]:
        """Everything an FL round may exchange: the seller-local token table is left out."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("local_tokens.")}

    def base_repr(self, seller) -> torch.Tensor:
        if self.config.text == "hx":
            tokens = seller.tokens.to(self.device)
            mask = (tokens != PAD).unsqueeze(-1).to(torch.float32)
            return (self.local_tokens(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.text_proj(seller.z.to(self.device))

    def relation_repr(self, base: torch.Tensor, relations: RelationTensors) -> torch.Tensor:
        dev = base.device
        neighbor = relations.neighbor.to(dev)
        times = self.time_mlp(self.time_bins)
        forward = relations.time_forward.to(dev).float() @ times
        backward = relations.time_backward.to(dev).float() @ times
        pair = self.relation_mlp["pair"](torch.cat([relations.features.to(dev), forward, backward], dim=-1))
        # F.embedding, not [neighbor]: popular items are thousands of items' neighbours, and the
        # indexing backward adds their gradients one atomic at a time (27x slower on the lab GPU).
        text = F.embedding(neighbor.clamp(min=0), self.relation_mlp["text"](base))
        per_pair = self.relation_mlp["out"](F.gelu(pair + text))
        mask = (neighbor >= 0).unsqueeze(-1).to(per_pair.dtype)
        pooled = self.relation_pool((per_pair * mask).sum(1) / mask.sum(1).clamp(min=1))
        # No neighbour means l = 0 by an explicit branch (model.md §3).
        return torch.where(relations.has_neighbor.to(dev).unsqueeze(-1), pooled, torch.zeros_like(pooled))

    def encode_items(self, seller) -> torch.Tensor:
        base = self.base_repr(seller)
        if self.config.relation:
            if seller.relations is None:
                raise ValueError("an R variant needs the seller's relations")
            l = self.relation_repr(base, seller.relations)
        else:
            if seller.relations is not None:
                raise ValueError("a T variant takes no relations")
            l = base.new_zeros(base.shape[0], self.config.d_relation)
        return self.fusion(torch.cat([base, l], dim=-1))

    def encode_queries(self, e: torch.Tensor, examples: Sequence[Example], seller) -> torch.Tensor:
        index, _ = item_sequences(examples, seller.row_of, self.config.max_items)
        index = to_device(index, e.device)
        mask = index >= 0
        n = self.config.max_items
        recency = torch.arange(n - 1, -1, -1, device=e.device).expand(len(examples), n)
        steps = F.embedding(index.clamp(min=0), e) * mask.unsqueeze(-1) + self.position(recency)
        hidden = self.sequence(steps, src_key_padding_mask=~mask)
        return self.query_proj(hidden[:, -1])

    def score_side(self, e: torch.Tensor) -> torch.Tensor:
        return self.scorer(e)

    @property
    def score_scale(self) -> float:
        return math.sqrt(self.config.d_model)

    def score(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ self.score_side(e).T / self.score_scale
