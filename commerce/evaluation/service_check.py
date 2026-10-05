"""The service path on real data: held-out Instacart sellers through SellerRuntime.

    python -m commerce.evaluation.service_check --instacart-dir fedcommerce/data/instacart \\
        --encoder-dir commerce/evaluation/cache/encoders/<model>/<revision> \\
        --text-only commerce/evaluation/outputs/releases/ic100-T_lm-r500 \\
        --text-relation commerce/evaluation/outputs/releases/ic100-R_lm-r500 --sellers 3

Each held-out seller (evaluation.md §3 A-0: none of its customers is in the FL
cohort) gets a fresh runtime in a temporary folder: its catalog through
upsert_catalog_item and every customer's visits but the last through
ingest_purchase_event, exactly as A would deliver them. Each customer's last
visit is then the answer to one predict_local call, first with no release
installed (local popularity fallback), then after install_release of each
variant. HR@10 and Recall@20 are macro over the customers.

The runtime opens with warm=True as open_runtime does: after the ledger is in
and after each install, the check waits for the background warm-up and records
how long it took, so latency_ms_first is the first request a customer would see.

This checks the whole service path on real data; it is not the D0022 table: one
answer per customer, the last visit, relations from the whole ledger. Latency
is wall-clock per call on this machine.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import tempfile
import time

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.harex_compare import HELD_OUT_TARGETS
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.instacart import load_instacart
from commerce.packages.recommender.seller_runtime import SellerRuntime
from commerce.packages.recommender.serving import FrozenText, load_bundle

AS_OF = "2026-10-01T00:00:00Z"  # relative-time visits all count as earlier (seller_runtime.py)


def check_seller(seller: str, events: list, catalog: list, text: FrozenText, releases: dict) -> dict:
    by_customer = defaultdict(list)
    for event in events:
        by_customer[event["customer_id_local"]].append(event)
    answers, ledger = {}, []
    for customer, own in by_customer.items():
        own.sort(key=lambda e: e["order_rank"])
        if len(own) < 2:
            continue
        answers[customer] = {i["item_id_local"] for i in own[-1]["items"]}
        ledger.extend(own[:-1])
    out = {"seller": seller, "customers": len(answers), "ledger_events": len(ledger), "catalog_items": len(catalog)}
    with tempfile.TemporaryDirectory() as tmp:
        runtime = SellerRuntime(seller, Path(tmp) / "features.sqlite", Path(tmp) / "models", text=text, warm=True)
        started = time.perf_counter()
        for seq, item in enumerate(catalog, start=1):
            runtime.upsert_catalog_item(item, seq)
        for event in ledger:
            runtime.ingest_purchase_event(event)
        out["ingest_seconds"] = round(time.perf_counter() - started, 1)
        started = time.perf_counter()
        runtime.wait_warm()
        out["warm_after_ingest_seconds"] = round(time.perf_counter() - started, 1)
        arms = [("popularity", None)] + [(variant, folder) for variant, folder in releases.items()]
        for name, folder in arms:
            warm_seconds = None
            if folder is not None:
                release, manifest, tensors = load_bundle(folder)
                runtime.install_release(release, manifest, tensors, model_variant=name)
                started = time.perf_counter()
                runtime.wait_warm()
                warm_seconds = round(time.perf_counter() - started, 1)
            variant = name if folder is not None else "text_only"
            hits, recalls, waits, fallbacks = [], [], [], defaultdict(int)
            for customer, answer in sorted(answers.items()):
                request = {"schema_version": "recommendation_request.v1", "seller_id": seller,
                           "customer_id_local": customer, "as_of": AS_OF, "candidate_item_ids": None, "top_n": 20}
                started = time.perf_counter()
                served = runtime.predict_local(request, model_variant=variant, mode="global")
                waits.append(time.perf_counter() - started)
                fallbacks[served["fallback_reason"]] += 1
                ranked = [i["item_id_local"] for i in served["items"]]
                hits.append(float(bool(answer & set(ranked[:10]))))
                recalls.append(len(answer & set(ranked[:20])) / len(answer))
            out[name] = {"hr@10": round(statistics.fmean(hits), 4), "recall@20": round(statistics.fmean(recalls), 4),
                         "fallback": dict(fallbacks),
                         "latency_ms_mean": round(1000 * statistics.fmean(waits), 1),
                         "latency_ms_first": round(1000 * waits[0], 1),
                         "latency_ms_p95": round(1000 * sorted(waits)[int(0.95 * (len(waits) - 1))], 1),
                         "warm_after_install_seconds": warm_seconds}
        runtime.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--text-only", type=Path, required=True)
    parser.add_argument("--text-relation", type=Path, required=True)
    parser.add_argument("--sellers", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)
    cohort = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=args.split_seed)
    held = assign_clients(args.instacart_dir, HELD_OUT_TARGETS, alpha=0.25, seed=args.split_seed,
                          exclude=set(cohort.clients))
    chosen = sorted(HELD_OUT_TARGETS)[:args.sellers]
    sample = load_instacart(args.instacart_dir, {u: c for u, c in held.clients.items() if c in chosen})
    text = FrozenText(args.encoder_dir)
    releases = {"text_only": args.text_only, "text_relation": args.text_relation}
    results = []
    for client in chosen:
        seller = "ic-client-%d" % client
        events = [e for e in sample.events if e["seller_id"] == seller]
        catalog = [i for i in sample.catalog_items if i["seller_id"] == seller]
        results.append(check_seller(seller, events, catalog, text, releases))
        print(json.dumps(results[-1], ensure_ascii=False))
    summary = {name: {m: round(statistics.fmean(r[name][m] for r in results), 4) for m in ("hr@10", "recall@20")}
               for name in ("popularity", "text_only", "text_relation")}
    print(json.dumps({"as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sellers": chosen,
                      "macro_over_sellers": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
