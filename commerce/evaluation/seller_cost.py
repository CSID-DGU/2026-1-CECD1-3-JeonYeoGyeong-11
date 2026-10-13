"""One seller's computing cost on this machine (OQ11, model-lab.md §5): the service path on CPU.

    python -m commerce.evaluation.seller_cost --instacart-dir fedcommerce/data/instacart \
        --encoder-dir commerce/evaluation/cache/encoders/<model>/<revision> \
        --releases commerce/evaluation/outputs/releases --threads 8

The first held-out Instacart seller (A-0, no customer in the FL cohort) goes into a fresh
SellerRuntime with warm=True, as open_runtime opens it. Measured: ingesting its catalog and
history, the background text vectors, then for each variant installing the first release,
one recommendation, one train_round at model.md §6 starting values (local_steps 40, batch
64, max_local_epochs 6, n_neg 200) and personalize_local with DEFAULT_PERSONAL; and the
process's peak memory. Wall-clock on this machine only.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import torch

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.encoder_probe import peak_memory_mb
from commerce.evaluation.harex_compare import HELD_OUT_TARGETS
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.instacart import load_instacart
from commerce.packages.recommender.seller_runtime import SellerRuntime
from commerce.packages.recommender.serving import FrozenText, load_bundle

RELEASES = {"text_only": "ic100-T_lm-r500", "text_relation": "ic100-R_lm-r500"}
AS_OF = "2026-10-01T00:00:00Z"  # relative-time Instacart visits all count as earlier (seller_runtime.py)


def seconds_since(start: float) -> float:
    return round(time.perf_counter() - start, 1)


def round_config(release, manifest) -> dict:
    """model.md §6 starting values."""
    return {"schema_version": "round_config.v1", "round_id": "round-cost", "model_version": release["model_version"],
            "manifest_hash": manifest["manifest_hash"], "architecture_version": manifest["architecture_version"],
            "min_clients": 1, "deadline": "2026-10-10T00:00:00Z", "local_steps": 40, "max_local_epochs": 6,
            "n_neg": 200, "batch_size": 64, "learning_rate": 0.001, "seed": 0}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True, help="the folder holding the release folders")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    cohort = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=0)
    held = assign_clients(args.instacart_dir, HELD_OUT_TARGETS, alpha=0.25, seed=0, exclude=set(cohort.clients))
    client = sorted(HELD_OUT_TARGETS)[0]
    sample = load_instacart(args.instacart_dir, {u: c for u, c in held.clients.items() if c == client})
    seller = "ic-client-%d" % client
    events = [e for e in sample.events if e["seller_id"] == seller]
    catalog = [i for i in sample.catalog_items if i["seller_id"] == seller]
    out = {"machine": {"os": platform.platform(), "cpu_count": os.cpu_count(), "torch_threads": args.threads,
                       "torch": torch.__version__},
           "seller": {"events": len(events), "customers": len({e["customer_id_local"] for e in events}),
                      "catalog_items": len(catalog)}}
    with tempfile.TemporaryDirectory() as tmp:
        runtime = SellerRuntime(seller, Path(tmp) / "features.sqlite", Path(tmp) / "models",
                                text=FrozenText(args.encoder_dir), warm=True)
        start = time.perf_counter()
        for seq, item in enumerate(catalog, start=1):
            runtime.upsert_catalog_item(item, seq)
        for event in events:
            runtime.ingest_purchase_event(event)
        out["ingest_s"] = seconds_since(start)
        start = time.perf_counter()
        runtime.wait_warm()
        out["text_vectors_s"] = seconds_since(start)
        request = {"schema_version": "recommendation_request.v1", "seller_id": seller,
                   "customer_id_local": events[0]["customer_id_local"], "as_of": AS_OF,
                   "candidate_item_ids": None, "top_n": 10}
        for variant, folder in RELEASES.items():
            row = {}
            release, manifest, tensors = load_bundle(args.releases / folder)
            runtime.install_release(release, manifest, tensors, model_variant=variant)
            start = time.perf_counter()
            runtime.wait_warm()
            row["warm_after_install_s"] = seconds_since(start)
            start = time.perf_counter()
            runtime.predict_local(request, model_variant=variant)
            row["recommend_ms"] = round(1000 * (time.perf_counter() - start), 1)
            start = time.perf_counter()
            result = runtime.train_round(runtime.get_local_data_ref(), round_config(release, manifest),
                                         model_variant=variant)
            row["train_round_s"], row["train_completed"] = seconds_since(start), result.completed
            start = time.perf_counter()
            personal = runtime.personalize_local(runtime.get_local_data_ref(), {}, model_variant=variant)
            row["personalize_s"], row["personalize"] = seconds_since(start), [personal.status, personal.reason]
            out[variant] = row
        runtime.close()
    out["peak_memory_mb"] = peak_memory_mb()
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
