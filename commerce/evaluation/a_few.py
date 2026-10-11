"""The 2x2 of comparison.md at held-out Instacart sellers: A-0 (T-G, R-G) and A-few (T-P, R-P).

evaluation.md §2 and §4.1. Each seller runs B's SellerRuntime on its own data, as a store
would. Every customer's visits but the last are the seller's ledger and the adaptation
prefix; each last visit is the test answer and never trains or chooses anything.

- T-G, R-G: the first releases as they are (no local weight update, A-0).
- T-P, R-P: personalize_local on the same base, with the service's settings
  (DEFAULT_PERSONAL unless --personal-lr), trained and judged on the prefix only.
  A seller whose personalization is not accepted has no P cell; its reason is counted.
- T-auto, R-auto: what the service shows, P where accepted and G otherwise
  (evaluation.md §4.1, the main comparison).

The sellers default to held-out sellers 11-20: sellers 1-10 chose the personalization
settings (model.md §8), so they are not reused for this report (evaluation.md §4.1).

    python -m commerce.evaluation.a_few --instacart-dir fedcommerce/data/instacart \
        --encoder-dir <installed frozen encoder> --releases <folder holding the release folders> \
        --out commerce/evaluation/runs/a_few/<name>.json
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from commerce.evaluation.d0022_report import paired_bootstrap
from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.harex_compare import HELD_OUT_TARGETS
from commerce.evaluation.metrics.ranking import MacroAverager, expected_metrics, p_topfreq_scores, popularity_scores
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.instacart import load_instacart
from commerce.packages.recommender.seller_runtime import DEFAULT_PERSONAL, SellerRuntime
from commerce.packages.recommender.serving import FrozenText, load_bundle

RELEASES = {"text_only": "ic100-T_lm-r500", "text_relation": "ic100-R_lm-r500"}
SHORT = {"text_only": "T", "text_relation": "R"}
ARMS = ("T-G", "R-G", "T-P", "R-P", "T-auto", "R-auto", "popularity", "P-TopFreq")


def split(events):
    """(ledger events, {customer: answer item set}): every visit but each customer's last is ledger."""
    by_customer = defaultdict(list)
    for e in events:
        by_customer[e["customer_id_local"]].append(e)
    ledger, answers = [], {}
    for customer, own in by_customer.items():
        own.sort(key=lambda e: e["order_rank"])
        if len(own) >= 2:
            answers[customer] = {i["item_id_local"] for i in own[-1]["items"]}
            own = own[:-1]
        ledger.extend(own)
    return ledger, answers


def run_seller(seller, sample, text, releases, config, seen):
    events = [e for e in sample.events if e["seller_id"] == seller]
    catalog = [i for i in sample.catalog_items if i["seller_id"] == seller]
    ledger, answers = split(events)
    seller_counts = Counter(i["item_id_local"] for e in ledger for i in e["items"])
    prior = defaultdict(Counter)
    for e in ledger:
        prior[e["customer_id_local"]].update(i["item_id_local"] for i in e["items"])
    out = {"seller": seller, "ledger_events": len(ledger), "items": len(catalog), "test_customers": len(answers),
           "personalization": {}}
    with tempfile.TemporaryDirectory() as tmp:
        rt = SellerRuntime(seller, Path(tmp) / "f.sqlite", Path(tmp) / "m", text=text)
        try:
            for seq, item in enumerate(catalog, 1):
                rt.upsert_catalog_item(item, seq)
            for e in ledger:
                rt.ingest_purchase_event(e)
            for variant, folder in releases.items():
                release, manifest, tensors = load_bundle(folder)
                rt.install_release(release, manifest, tensors, model_variant=variant)
            ref = rt.get_local_data_ref()
            for variant in releases:
                started = time.perf_counter()
                result = rt.personalize_local(ref, config, model_variant=variant)
                out["personalization"][variant] = {"status": result.status, "reason": result.reason,
                                                   "seconds": round(time.perf_counter() - started, 2)}
            snap = rt.store.snapshot()
            before = list(snap.baskets)
            for customer, answer in sorted(answers.items()):
                scored = {}
                for variant in releases:
                    arch, base = rt._arch(variant), rt._handles[variant]
                    personal = rt._personal.get(variant)
                    g = rt._model_scores(arch, base, base.model, snap, before, customer)
                    if g is None:
                        break
                    scored["%s-G" % SHORT[variant]] = g
                    if personal is not None:
                        scored["%s-P" % SHORT[variant]] = rt._model_scores(arch, base, personal.model, snap, before,
                                                                           customer)
                if "T-G" not in scored or "R-G" not in scored:
                    continue  # nothing to read for this customer (no earlier visit with a catalog item)
                items = sorted(scored["T-G"])
                relevant = [k for k, item in enumerate(items) if item in answer]
                if not relevant:
                    continue  # the answer holds no catalog item
                for arm, scores in scored.items():
                    seen[arm].add(seller, customer, expected_metrics(np.array([scores[i] for i in items]), relevant))
                for v in ("T", "R"):
                    arm = "%s-P" % v if "%s-P" % v in scored else "%s-G" % v
                    seen["%s-auto" % v].add(seller, customer,
                                            expected_metrics(np.array([scored[arm][i] for i in items]), relevant))
                seen["popularity"].add(seller, customer,
                                       expected_metrics(popularity_scores(items, seller_counts), relevant))
                seen["P-TopFreq"].add(seller, customer,
                                      expected_metrics(p_topfreq_scores(items, prior[customer], seller_counts), relevant))
        finally:
            rt.close()
    return out


def contrasts(result, metric="ndcg@10"):
    """Seller-level paired differences with a bootstrap 95% interval (d0022_report.paired_bootstrap)."""
    rows = []
    for name, a, b in (("R-auto − T-auto", "R-auto", "T-auto"), ("R-G − T-G", "R-G", "T-G"),
                       ("T-P − T-G", "T-P", "T-G"), ("R-P − R-G", "R-P", "R-G"), ("R-P − T-P", "R-P", "T-P")):
        pa, pb = result[a].get("per_seller", {}), result[b].get("per_seller", {})
        both = sorted(set(pa) & set(pb))
        if not both:
            continue
        mean, low, high = paired_bootstrap(np.array([pa[s][metric] for s in both]),
                                           np.array([pb[s][metric] for s in both]))
        rows.append({"contrast": name, "sellers": len(both), "mean": round(mean, 4),
                     "ci95": [round(low, 4), round(high, 4)]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="T-G/R-G/T-P/R-P at held-out Instacart sellers (A-0, A-few).")
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True, help="installed frozen encoder files")
    parser.add_argument("--releases", type=Path, required=True, help="folder holding %s" % ", ".join(RELEASES.values()))
    parser.add_argument("--first", type=int, default=10, help="index of the first held-out seller (0-based)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--personal-lr", type=float, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    config = {} if args.personal_lr is None else {"lr": args.personal_lr}
    settings = dict(DEFAULT_PERSONAL, **config)

    cohort = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=0)
    held = assign_clients(args.instacart_dir, HELD_OUT_TARGETS, alpha=0.25, seed=0, exclude=set(cohort.clients))
    chosen = sorted(HELD_OUT_TARGETS)[args.first:args.first + args.count]
    sample = load_instacart(args.instacart_dir, {u: c for u, c in held.clients.items() if c in chosen})
    text = FrozenText(args.encoder_dir)
    releases = {v: args.releases / f for v, f in RELEASES.items()}
    seen = {arm: MacroAverager() for arm in ARMS}
    sellers = []
    started = time.perf_counter()
    for client in chosen:
        sellers.append(run_seller("ic-client-%d" % client, sample, text, releases, config, seen))
        print("seller %s done %.0fs" % (client, time.perf_counter() - started), file=sys.stderr, flush=True)

    result = {arm: seen[arm].result() for arm in ARMS}
    availability = {v: Counter(s["personalization"][v]["status"] + (":" + s["personalization"][v]["reason"]
                                                                     if s["personalization"][v]["reason"] else "")
                               for s in sellers) for v in RELEASES}
    record = {
        "run": "a_few", "labels": ["held-out Instacart sellers (A-0, A-few)", "unprotected FL simulation releases",
                                   "last visit per customer is the test answer"],
        "sellers": [s["seller"] for s in sellers], "first": args.first, "personal_settings": settings,
        "releases": {v: str(p.name) for v, p in releases.items()},
        "availability": {v: dict(c) for v, c in availability.items()},
        "personal_seconds_median": {v: statistics.median(s["personalization"][v]["seconds"] for s in sellers)
                                    for v in RELEASES},
        "metrics": {arm: {k: v for k, v in result[arm].items() if k != "per_seller"} for arm in ARMS},
        "per_seller": {arm: result[arm].get("per_seller", {}) for arm in ARMS},
        "contrasts_ndcg@10": contrasts(result),
        "seller_rows": sellers, "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    for arm in ARMS:
        m = result[arm].get("macro")
        if m:
            print("%-10s sellers %2d  NDCG@10 %.4f  HR@10 %.4f  Recall@20 %.4f"
                  % (arm, result[arm]["sellers"], m["ndcg@10"], m["hr@10"], m["recall@20"]))
    print(json.dumps({"availability": record["availability"], "contrasts_ndcg@10": record["contrasts_ndcg@10"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
