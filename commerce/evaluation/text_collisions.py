"""OQ04: how often products cannot be told apart by their text (data.md §3, model.md §1).

    python -m commerce.evaluation.text_collisions --instacart-dir fedcommerce/data/instacart \\
        --dunnhumby-dir fedcommerce/data \\
        --encoder-dir commerce/evaluation/cache/encoders/<model>/<revision>

The model has no item ID or bias, so two products with the same encoder input get the
same z, and without purchase relations the same score. For each source this counts the
share of products whose normalized text, and whose token IDs after the service
encoder's truncation, equal another product's: over the whole source and within each
seller's catalog, which is what one ranking competes over. Instacart sellers are the
experiments' stand-in split (alpha 0.25, seed 0), Dunnhumby sellers the roster, live
the hand-made live fixture. Prints counts and shares only.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.encoder_probe import CANDIDATES
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.dunnhumby import load_dunnhumby
from commerce.packages.data_adapters.instacart import load_instacart
from commerce.packages.data_adapters.text import catalog_item_text

ENCODER = CANDIDATES["minilm-l12"]  # the service encoder (nlp-encoder.md)
LIVE_FIXTURE = Path(__file__).resolve().parents[1] / "packages" / "data_adapters" / "tests" / "fixtures" / "live" \
    / "catalog_items.json"

Tokenize = Callable[[Sequence[str]], list[list[int]]]


def collision_shares(texts: Mapping[str, str], tokenize: Tokenize, max_length: int) -> dict:
    """Share of items whose text, or whose truncated token IDs, equal another item's."""
    items = sorted(texts)
    text = [texts[i] for i in items]
    tokens = [tuple(t) for t in tokenize(text)]
    n = len(items)
    if not n:
        return {"items": 0, "text_duplicate": 0.0, "token_duplicate": 0.0, "truncated": 0.0}
    by_text, by_tokens = Counter(text), Counter(tokens)
    return {"items": n,
            "text_duplicate": sum(by_text[t] > 1 for t in text) / n,
            "token_duplicate": sum(by_tokens[t] > 1 for t in tokens) / n,
            "truncated": sum(len(t) >= max_length for t in tokens) / n}


def per_seller(catalog_items: Sequence[Mapping], tokenize: Tokenize, max_length: int) -> dict:
    by_seller: dict[str, dict[str, str]] = defaultdict(dict)
    for item in catalog_items:
        by_seller[item["seller_id"]][item["item_id_local"]] = catalog_item_text(item)
    rows = [collision_shares(t, tokenize, max_length) for t in by_seller.values()]
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    return {"sellers": len(rows), "text_duplicate_macro": mean("text_duplicate"),
            "token_duplicate_macro": mean("token_duplicate"),
            "token_duplicate_max": max(r["token_duplicate"] for r in rows),
            "truncated_macro": mean("truncated")}


def whole_source(catalog_items: Sequence[Mapping], tokenize: Tokenize, max_length: int) -> dict:
    """Each product once, whatever sellers list it."""
    return collision_shares({i["item_id_local"]: catalog_item_text(i) for i in catalog_items}, tokenize, max_length)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--dunnhumby-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from commerce.packages.recommender.text_encoder import FrozenTextEncoder
    encoder = FrozenTextEncoder(args.encoder_dir, ENCODER)
    tokenize = lambda texts: encoder.token_ids(texts, max_length=ENCODER.max_length)
    out = {"encoder": ENCODER.model_id, "max_length": ENCODER.max_length}

    split = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=0)
    ic = load_instacart(args.instacart_dir, split.clients).catalog_items
    out["instacart"] = {"products": whole_source(ic, tokenize, ENCODER.max_length),
                        "within_seller": per_seller(ic, tokenize, ENCODER.max_length)}
    dh = load_dunnhumby(args.dunnhumby_dir).catalog_items
    out["dunnhumby"] = {"products": whole_source(dh, tokenize, ENCODER.max_length),
                        "within_seller": per_seller(dh, tokenize, ENCODER.max_length)}
    live = json.loads(LIVE_FIXTURE.read_text(encoding="utf-8"))
    out["live_fixture"] = {"within_seller": per_seller(live, tokenize, ENCODER.max_length)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
