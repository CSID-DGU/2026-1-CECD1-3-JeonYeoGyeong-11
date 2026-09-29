"""GCI's own model (Lee et al., AAAI-24) on Instacart: an added-scope reproduction (evaluation.md §4).

    python -m commerce.evaluation.gci_original --instacart-dir fedcommerce/data/instacart --mode fl

GCI does not score candidates. A Transformer encoder reads the names of the last
four product-merchant pairs and a decoder writes the next one's name token by
token; each written name is replaced by the genuine name with the highest
Jaccard similarity, and HR@10 counts the label among ten such names. Word
embeddings and the output layer stay with each client (glocalization); the
encoder and decoder layers are averaged by FedAvg; training stops on the
validation loss with patience 20. Units, split and clients follow
gci_protocol.py; this module swaps our scoring backbone for GCI's generator so
the paper's numbers can be read against the same data.

From the paper: 1 encoder and 1 decoder layer, D_MODEL 128, 4 heads, UNITS 256,
DROPOUT 0.2, 1 epoch per round, BATCH_SIZE 128, up to 500 rounds, patience 20,
space tokenization per client, units of five shifted by two, Jaccard matching,
local / glocal FL / data-shared comparison. The shared parameter counts of
Table 3 (encoder 132,480, decoder 198,784) are checked by a test.

Assumed, because the paper does not say: the TensorFlow Transformer tutorial its
hyperparameter names come from (Adam 0.9/0.98/1e-9, warm-up 4000 learning rate,
sinusoidal positions, post-norm layers); ten names from a width-10 beam search;
Instacart's aisle standing in for the merchant name; clients of five of our
sellers each (the paper's clients hold 20-34k units); FedAvg weighted by
training units as in McMahan et al.; a 10% random test share.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.encoder_probe import peak_memory_mb
from commerce.evaluation.gci_protocol import menu_items, seller_units
from commerce.evaluation.harex_compare import code_version
from commerce.evaluation.scoring import MacroAverager, expected_metrics, p_topfreq_scores, popularity_scores
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.instacart import cart_orders, load_instacart
from commerce.packages.data_adapters.text import normalize_field

PAD, START, END, SEP, UNKNOWN = 0, 1, 2, 3, 4
SPECIAL = 5
MAX_SOURCE, MAX_TARGET = 64, 16
EVAL_BATCH = 64  # 640 written names against a client's catalog at a time: a few hundred MB on the GPU
LOCAL = ("enc_embed.", "dec_embed.", "out.")  # GCI keeps these with each client


def pair_text(item: dict) -> str:
    """GCI's product-merchant name; Instacart has no merchant name, so its aisle stands in."""
    path = item.get("category_path") or []
    aisle = path[1] if len(path) == 2 else None
    return " ".join(t for t in (normalize_field(item["title_text"]), normalize_field(aisle)) if t)


class Vocabulary:
    """GCI's space tokenization, one vocabulary per client."""

    def __init__(self, texts):
        words = sorted({w for t in texts for w in t.split(" ") if w})
        self.index = {w: i + SPECIAL for i, w in enumerate(words)}

    def __len__(self):
        return len(self.index) + SPECIAL

    def words(self, text):
        return [self.index.get(w, UNKNOWN) for w in text.split(" ") if w]


def source_tokens(vocab, texts):
    out = []
    for i, text in enumerate(texts):
        if i:
            out.append(SEP)
        out += vocab.words(text)
    return out[-MAX_SOURCE:]  # keep the most recent words


def target_tokens(vocab, text):
    return [START] + vocab.words(text)[:MAX_TARGET - 2] + [END]


def pad(rows, width=None):
    width = width or max(len(r) for r in rows)
    out = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r[:width], dtype=torch.long)
    return out


def positions(length, d):
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class GCIModel(nn.Module):
    """TensorFlow-tutorial Transformer: 1 encoder layer, 1 decoder layer, post-norm."""

    def __init__(self, vocab_size, d=128, heads=4, units=256, dropout=0.2):
        super().__init__()
        self.d = d
        self.enc_embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.dec_embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.encoder = nn.TransformerEncoderLayer(d, heads, units, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoderLayer(d, heads, units, dropout, batch_first=True)
        self.out = nn.Linear(d, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("pe", positions(MAX_SOURCE + MAX_TARGET, d), persistent=False)

    def shared_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(LOCAL)}

    def _embed(self, table, tokens):
        return self.drop(table(tokens) * math.sqrt(self.d) + self.pe[:tokens.shape[1]])

    def encode(self, source):
        pad_mask = source == PAD
        return self.encoder(self._embed(self.enc_embed, source), src_key_padding_mask=pad_mask), pad_mask

    def decode(self, memory, memory_pad, target_in):
        n = target_in.shape[1]
        causal = torch.triu(torch.full((n, n), float("-inf"), device=target_in.device), diagonal=1)
        h = self.decoder(self._embed(self.dec_embed, target_in), memory, tgt_mask=causal,
                         tgt_key_padding_mask=target_in == PAD, memory_key_padding_mask=memory_pad)
        return self.out(h)

    def loss(self, source, target):
        """Mean token cross-entropy of the target given the source (teacher forcing), and its token count."""
        memory, memory_pad = self.encode(source)
        logits = self.decode(memory, memory_pad, target[:, :-1])
        gold = target[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1), ignore_index=PAD), \
            int((gold != PAD).sum())


def warmup_schedule(d=128, warmup=4000):
    """The tutorial's rate: d^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    return lambda step: d ** -0.5 * min((step + 1) ** -0.5, (step + 1) * warmup ** -1.5)


def optimizer_for(model):
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, warmup_schedule())


@torch.no_grad()
def beam_search(model, source, beam=10, max_len=MAX_TARGET):
    """The ten best written names per source row, best first: (B, beam, T) tokens after START."""
    model.eval()
    b = source.shape[0]
    memory, memory_pad = model.encode(source)
    memory, memory_pad = memory.repeat_interleave(beam, 0), memory_pad.repeat_interleave(beam, 0)
    seqs = torch.full((b * beam, 1), START, dtype=torch.long, device=source.device)
    scores = torch.full((b, beam), float("-inf"), device=source.device)
    scores[:, 0] = 0.0  # one live beam at the start
    done = torch.zeros(b * beam, dtype=torch.bool, device=source.device)
    lengths = torch.zeros(b * beam, device=source.device)
    for _ in range(max_len - 1):
        logp = F.log_softmax(model.decode(memory, memory_pad, seqs)[:, -1], dim=-1)
        logp[done] = float("-inf")
        logp[done, PAD] = 0.0  # a finished name only pads
        vocab = logp.shape[-1]
        total = (scores.reshape(-1, 1) + logp).reshape(b, beam * vocab)
        scores, flat = total.topk(beam, dim=-1)
        parent = (flat // vocab) + torch.arange(b, device=source.device).unsqueeze(1) * beam
        token = (flat % vocab).reshape(-1)
        parent = parent.reshape(-1)
        seqs = torch.cat([seqs[parent], token.unsqueeze(1)], dim=1)
        done, lengths = done[parent], lengths[parent]
        lengths = lengths + (~done).float()
        done = done | (token == END)
        if bool(done.all()):
            break
    normalized = scores.reshape(-1) / lengths.clamp(min=1)
    order = normalized.reshape(b, beam).argsort(dim=-1, descending=True)
    seqs = seqs.reshape(b, beam, -1)[:, :, 1:]
    return torch.gather(seqs, 1, order.unsqueeze(-1).expand(-1, -1, seqs.shape[-1]))


class Catalog:
    """Genuine names of one client as token sets, for GCI's Jaccard replacement."""

    def __init__(self, items, texts, vocab, device):
        self.items = list(items)
        sets = [sorted(set(vocab.words(t))) for t in texts]
        rows = [r for r, s in enumerate(sets) for _ in s]
        cols = [w for s in sets for w in s]
        self.matrix = torch.sparse_coo_tensor([rows, cols], torch.ones(len(rows)), (len(sets), len(vocab)),
                                              device=device, check_invariants=False).coalesce()
        self.sizes = torch.tensor([len(s) for s in sets], dtype=torch.float32, device=device)

    def nearest(self, written, k=10):
        """For each written name (N, T tokens), the k genuine rows with the highest Jaccard similarity."""
        n, vocab = written.shape[0], self.matrix.shape[1]
        g = torch.zeros(n, vocab, device=written.device)
        keep = written >= SPECIAL
        g[torch.arange(n, device=written.device).unsqueeze(1).expand_as(written)[keep], written[keep]] = 1.0
        inter = torch.sparse.mm(self.matrix, g.T)  # (items, N)
        union = self.sizes.unsqueeze(1) + g.sum(1).unsqueeze(0) - inter
        jaccard = inter / union.clamp(min=1)
        return jaccard.T.topk(min(k, len(self.items)), dim=-1).indices


def ten_items(candidates, k=10):
    """Walk the written names best first: each takes its nearest genuine name not taken yet;
    where names collide, their next-nearest names fill the list in the same order."""
    chosen = []
    for rank in range(candidates.shape[1]):
        for beam in range(candidates.shape[0]):
            row = int(candidates[beam, rank])
            if row not in chosen:
                chosen.append(row)
                if len(chosen) == k:
                    return chosen
    return chosen


def build_clients(args, record):
    """GCI-style clients: each joins sellers_per_client of our sellers, with one vocabulary and catalog."""
    started = time.perf_counter()
    split = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=args.split_seed)
    chosen = sorted(STAND_IN_TARGETS)[:args.clients * args.sellers_per_client]
    users = {u: c for u, c in split.clients.items() if c in chosen}
    sample = load_instacart(args.instacart_dir, users)
    carts = cart_orders(args.instacart_dir, [int(e["basket_id_local"].rsplit("-", 1)[1]) for e in sample.events])
    by_seller, catalogs = {}, {}
    for event in sample.events:
        b = basket_from_event(event)
        by_seller.setdefault(b.seller_id, {}).setdefault(b.customer_id_local, []).append(b)
    for item in sample.catalog_items:
        catalogs.setdefault(item["seller_id"], []).append(item)
    all_visits = {seller: {cust: customer_visits(bs) for cust, bs in customers.items()}
                  for seller, customers in by_seller.items()}
    menu = menu_items(all_visits, carts, args.menu_size) if args.menu_size else None
    clients = []
    for k in range(args.clients):
        members = ["ic-client-%d" % c for c in chosen[k * args.sellers_per_client:(k + 1) * args.sellers_per_client]]
        text_of, grouped = {}, {"train": [], "validation": [], "test": []}
        for seller in members:
            for item in catalogs[seller]:
                if menu is None or item["item_id_local"] in menu:
                    text_of.setdefault(item["item_id_local"], pair_text(item))
            units, _, _ = seller_units(seller, all_visits[seller], carts, args.split_seed, menu)
            for (role, _), examples in units.items():
                grouped[role] += examples
        items = sorted(text_of)
        clients.append({"name": "client-%d" % (k + 1), "sellers": members, "items": items,
                        "row_of": {item: row for row, item in enumerate(items)},
                        "texts": [text_of[i] for i in items], "text_of": text_of, "grouped": grouped})
    record["data"] = {"assignment": split.record, "adapter_report": sample.report,
                      "menu_size": len(menu) if menu else None,
                      "clients": [{"name": c["name"], "sellers": c["sellers"], "catalog_items": len(c["items"]),
                                   "units": {r: len(x) for r, x in c["grouped"].items()}} for c in clients],
                      "seconds": round(time.perf_counter() - started, 1)}
    return clients


def tensors(client, vocab, role):
    examples = client["grouped"][role]
    source = pad([source_tokens(vocab, [client["text_of"][v.items[0]] for v in e.history]) for e in examples],
                 MAX_SOURCE)
    target = pad([target_tokens(vocab, client["text_of"][next(iter(e.target_items))]) for e in examples], MAX_TARGET)
    return source, target


def train_epoch(model, optimizer, schedule, data, batch_size, rng):
    model.train()
    source, target = data
    order = torch.from_numpy(rng.permutation(len(source)))
    for start in range(0, len(source), batch_size):
        idx = order[start:start + batch_size].to(source.device)
        loss, _ = model.loss(source[idx], target[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()


@torch.no_grad()
def loss_sum(model, data, batch_size=512):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(data[0]), batch_size):
        loss, n = model.loss(data[0][start:start + batch_size], data[1][start:start + batch_size])
        total, count = total + float(loss) * n, count + n
    return total, count


def evaluate(model, client, vocab, data, arms, arm, beam, device):
    """HR@10 and NDCG@10 of GCI's ten names, next to popularity and P-TopFreq over the same catalog."""
    catalog = Catalog(client["items"], client["texts"], vocab, device)
    examples = client["grouped"]["test"]
    popular = Counter(next(iter(e.target_items)) for e in client["grouped"]["train"])
    pop_scores = popularity_scores(client["items"], popular)
    for start in range(0, len(examples), EVAL_BATCH):
        written = beam_search(model, data[0][start:start + EVAL_BATCH], beam)
        b, w, t = written.shape
        nearest = catalog.nearest(written.reshape(b * w, t), k=beam).reshape(b, w, -1).cpu()
        for i, example in enumerate(examples[start:start + EVAL_BATCH]):
            label = client["row_of"][next(iter(example.target_items))]
            picked = ten_items(nearest[i])
            rank = picked.index(label) if label in picked else None
            arms[arm].add(client["name"], example.customer_id_local, {
                "hr@10": float(rank is not None), "ndcg@10": 1.0 / math.log2(rank + 2) if rank is not None else 0.0,
                "precision@10": float(rank is not None) / 10})
            for name, scores in (("popularity", pop_scores),
                                 ("P-TopFreq", p_topfreq_scores(client["items"], example.prior_counts, popular))):
                m = expected_metrics(scores, {label})
                arms[name].add(client["name"], example.customer_id_local,
                               {"hr@10": m["hr@10"], "ndcg@10": m["ndcg@10"], "precision@10": m["hr@10"] / 10})


def fit_until_patience(step, check, rounds, patience, snapshot):
    """Train one round at a time; keep the lowest validation loss; stop after `patience` rounds without it."""
    best, history = None, []
    for rnd in range(1, rounds + 1):
        step(rnd)
        val = check()
        history.append([rnd, val])
        if best is None or val < best["val"]:
            best = {"round": rnd, "val": val, "state": snapshot()}
        elif rnd - best["round"] >= patience:
            break
        if rnd % 10 == 0:
            print("round %d val %.4f best %d (%.4f)" % (rnd, val, best["round"], best["val"]), flush=True)
    return best, history, rnd


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("local", "fl", "shared"))
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--sellers-per-client", type=int, default=5)
    # BBQ-like: one small shared menu (gci_protocol.menu_items); 0 keeps every product.
    parser.add_argument("--menu-size", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/gci_original"))
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "GCI original reproduction", "mode": args.mode, "started_at": stamp, "code": code_version(),
              "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "machine": {"os": platform.platform(), "torch": torch.__version__, "device": str(device)},
              "labels": ["HAREX reproduction (added scope, evaluation.md §4)", "GCI's own model on Instacart",
                         "item-level units, random split", "early stopping as in GCI"]
              + (["BBQ-like menu of the %d most bought products" % args.menu_size] if args.menu_size else [])
              + (["data-shared training is not FL"] if args.mode == "shared" else [])
              + (["unprotected FL simulation"] if args.mode == "fl" else [])}
    clients = build_clients(args, record)
    rng = np.random.default_rng(args.seed)
    arm = "GCI %s" % args.mode
    arms = {name: MacroAverager() for name in (arm, "popularity", "P-TopFreq")}
    started = time.perf_counter()
    training = {}

    def new_model(vocab_size):
        torch.manual_seed(args.seed)
        return GCIModel(vocab_size).to(device)

    if args.mode == "shared":
        vocab = Vocabulary([t for c in clients for t in c["texts"]])
        data = {r: [tensors(c, vocab, r) for c in clients] for r in ("train", "validation", "test")}
        joined = {r: tuple(torch.cat([d[i] for d in data[r]]).to(device) for i in range(2)) for r in ("train", "validation")}
        model = new_model(len(vocab))
        opt, sched = optimizer_for(model)
        best, history, stop = fit_until_patience(
            lambda rnd: train_epoch(model, opt, sched, joined["train"], args.batch_size, rng),
            lambda: (lambda t, n: t / n)(*loss_sum(model, joined["validation"])), args.rounds, args.patience,
            lambda: {k: v.detach().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best["state"])
        for c, test in zip(clients, data["test"]):
            evaluate(model, c, vocab, tuple(x.to(device) for x in test), arms, arm, args.beam, device)
        training["shared"] = {"best_round": best["round"], "rounds_run": stop, "history": history, "vocab": len(vocab)}
    else:
        vocabs = [Vocabulary(c["texts"]) for c in clients]
        data = [{r: tuple(x.to(device) for x in tensors(c, v, r)) for r in ("train", "validation", "test")}
                for c, v in zip(clients, vocabs)]
        models = [new_model(len(v)) for v in vocabs]
        optimizers = [optimizer_for(m) for m in models]
        if args.mode == "local":
            for c, v, d, model, (opt, sched) in zip(clients, vocabs, data, models, optimizers):
                best, history, stop = fit_until_patience(
                    lambda rnd: train_epoch(model, opt, sched, d["train"], args.batch_size, rng),
                    lambda: (lambda t, n: t / n)(*loss_sum(model, d["validation"])), args.rounds, args.patience,
                    lambda: {k: x.detach().clone() for k, x in model.state_dict().items()})
                model.load_state_dict(best["state"])
                evaluate(model, c, v, d["test"], arms, arm, args.beam, device)
                training[c["name"]] = {"best_round": best["round"], "rounds_run": stop, "history": history,
                                       "vocab": len(v)}
                print("%s best round %d of %d" % (c["name"], best["round"], stop), flush=True)
        else:
            weights = [len(d["train"][0]) for d in data]
            shared = {k: x.detach().clone() for k, x in models[0].shared_state().items()}

            def fl_round(rnd):
                states = []
                for d, model, (opt, sched) in zip(data, models, optimizers):
                    model.load_state_dict(shared, strict=False)
                    train_epoch(model, opt, sched, d["train"], args.batch_size, rng)
                    states.append(model.shared_state())
                total = sum(weights)
                for k in shared:  # FedAvg weighted by training units (McMahan et al.)
                    shared[k] = sum(s[k].float() * w for s, w in zip(states, weights)) / total
                for model in models:
                    model.load_state_dict(shared, strict=False)

            def fl_check():
                sums = [loss_sum(model, d["validation"]) for model, d in zip(models, data)]
                return sum(t for t, _ in sums) / sum(n for _, n in sums)  # an aggregate, never per client

            best, history, stop = fit_until_patience(
                fl_round, fl_check, args.rounds, args.patience,
                lambda: [{k: x.detach().clone() for k, x in m.state_dict().items()} for m in models])
            for c, v, d, model, state in zip(clients, vocabs, data, models, best["state"]):
                model.load_state_dict(state)
                evaluate(model, c, v, d["test"], arms, arm, args.beam, device)
            training["fl"] = {"best_round": best["round"], "rounds_run": stop, "history": history,
                              "vocab": [len(v) for v in vocabs]}
    probe = GCIModel(64)
    record["parameters"] = {"encoder_layer": sum(p.numel() for p in probe.encoder.parameters()),
                            "decoder_layer": sum(p.numel() for p in probe.decoder.parameters())}
    record["training"] = training
    record["seconds"] = round(time.perf_counter() - started, 1)
    record["metrics"] = {name: a.result() for name, a in arms.items()}
    record["peak_memory_mb"] = peak_memory_mb()
    out = args.out_dir / ("%s_%s%s_s%d_%d" % (stamp, args.mode, "_menu%d" % args.menu_size if args.menu_size else "",
                                               args.seed, os.getpid()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {name: {"macro": {k: round(v, 4) for k, v in r["macro"].items()},
                      "per_client": [round(s["hr@10"], 4) for s in r["per_seller"]]}
               for name, r in record["metrics"].items() if r.get("examples")}
    print(json.dumps(summary, indent=2))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
