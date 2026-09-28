"""Training step, fixed validation loss and full-catalog scores (model.md §6).

A batch always comes from one seller and one relation snapshot: candidates and
negatives are that seller's catalog, and e is computed once for the batch. train() on one seller's data is the local_only mode of
model-lab.md §4. Passing several sellers pools them into one optimizer, which is
none of the lab modes and never an FL result; FL keeps sellers apart and merges
their deltas through C's aggregation core.
"""
from dataclasses import asdict, dataclass, field
import random
from typing import Sequence

import torch

from commerce.packages.recommender.examples import Example
from commerce.packages.recommender.model import Recommender, sampled_softmax_loss
from commerce.packages.recommender.relations import RelationTensors


@dataclass
class SellerData:
    seller_id: str
    items: tuple[str, ...]  # catalog; row i of z belongs to items[i]
    z: torch.Tensor  # (C, d_text) frozen text vectors
    examples: list[Example]
    relations: RelationTensors | None = None  # text_relation: this snapshot's relations
    tokens: torch.Tensor | None = None  # (C, T) word-token ids for GCI-style (hx) item text, 0 = pad
    row_of: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.row_of = {item: row for row, item in enumerate(self.items)}


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 40
    batch_size: int = 64
    n_negatives: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    seed: int = 0


def batch_loss(model: Recommender, seller: SellerData, examples: Sequence[Example],
               n_negatives: int, rng: random.Random) -> tuple[torch.Tensor, int]:
    e = model.encode_items(seller)
    q = model.encode_queries(e, examples, seller)
    device = e.device
    positives = torch.tensor([seller.row_of[rng.choice(sorted(ex.target_items))] for ex in examples], device=device)
    negatives = torch.tensor(rng.sample(range(len(seller.items)), min(n_negatives, len(seller.items))))
    targets = [{seller.row_of[i] for i in ex.target_items} for ex in examples]
    allowed = torch.tensor([[int(n) not in t for n in negatives.tolist()] for t in targets], dtype=torch.bool)
    negatives = negatives.to(device)
    pos_scores = (q * model.score_side(e[positives])).sum(-1) / model.score_scale
    return sampled_softmax_loss(pos_scores, model.score(q, e[negatives]), allowed)


def train(model: Recommender, sellers: Sequence[SellerData], config: TrainConfig,
          optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Run config.steps optimizer steps. Without an optimizer every call starts a fresh AdamW, as
    every FL round does; a local_only run passes its own so early-stopping checks keep its state."""
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    pool = [s for s in sellers if s.examples]
    if not pool:
        return {"steps": 0, "skipped_batches": 0, "loss_mean": None, "grad_norm_mean": None, "config": asdict(config)}
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train()
    losses, norms, skipped = [], [], 0
    weights = [len(s.examples) for s in pool]
    for _ in range(config.steps):
        seller = rng.choices(pool, weights=weights)[0]
        examples = rng.sample(seller.examples, min(config.batch_size, len(seller.examples)))
        loss, used = batch_loss(model, seller, examples, config.n_negatives, rng)
        if used == 0:
            skipped += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm))
        optimizer.step()
        losses.append(loss.detach())
    model.eval()
    # Read back once at the end: a read per step makes every step wait for the device.
    losses = torch.stack(losses).tolist() if losses else []
    norms = torch.stack(norms).tolist() if norms else []
    return {"steps": len(losses), "skipped_batches": skipped,
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "grad_norm_mean": sum(norms) / len(norms) if norms else None, "config": asdict(config)}


@torch.no_grad()
def validation_loss(model: Recommender, sellers: Sequence[SellerData], *, n_negatives: int = 200,
                    batch_size: int = 256, seed: int = 0) -> float | None:
    """Mean loss over fixed validation examples. The seed never holds a round id,
    so every round sees the same positives and negatives (model.md §6)."""
    model.eval()
    total, count = 0.0, 0
    for seller in sellers:
        rng = random.Random("%d:%s" % (seed, seller.seller_id))
        for start in range(0, len(seller.examples), batch_size):
            loss, used = batch_loss(model, seller, seller.examples[start:start + batch_size], n_negatives, rng)
            total = total + loss * used
            count += used
    return float(total) / count if count else None


@torch.no_grad()
def catalog_scores(model: Recommender, seller: SellerData, examples: Sequence[Example],
                   batch_size: int = 256) -> torch.Tensor:
    """(len(examples), C) scores over the seller's whole catalog, rows in seller.items order."""
    model.eval()
    e = model.encode_items(seller)
    rows = [model.score(model.encode_queries(e, examples[s:s + batch_size], seller), e).cpu()
            for s in range(0, len(examples), batch_size)]
    return torch.cat(rows) if rows else torch.zeros((0, len(seller.items)))


def seller_on(seller: SellerData, device: torch.device) -> SellerData:
    """A copy of the seller's tensors on the given device; examples and ids are shared."""
    relations = seller.relations
    if relations is not None:
        relations = RelationTensors(*(getattr(relations, f).to(device) for f in (
            "neighbor", "features", "time_forward", "time_backward", "has_neighbor")))
    moved = SellerData(seller.seller_id, seller.items, seller.z.to(device), seller.examples, relations,
                       None if seller.tokens is None else seller.tokens.to(device))
    return moved
