"""Compare frozen encoder candidates on product text (B card step 3, E-G0 input check).

    python -m commerce.evaluation.encoder_probe --cache-dir commerce/evaluation/cache/encoders \
        [--instacart-dir fedcommerce/data/instacart]

Model files are downloaded beforehand into <cache-dir>/<org>__<name>/<revision>/.
Only aggregate numbers are written, to commerce/evaluation/outputs/encoder_probe/;
no raw product name leaves the run.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

from commerce.packages.data_adapters.instacart import MISSING_LABEL
from commerce.packages.data_adapters.text import (
    audit_texts, build_product_text, catalog_item_text, instacart_fields, live_fields,
)
from commerce.packages.recommender.text_encoder import EncoderSpec, FrozenTextEncoder, artifact_hash

CANDIDATES = {
    "minilm-l12": EncoderSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                              "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", max_length=64),
    "e5-small": EncoderSpec("intfloat/multilingual-e5-small",
                            "614241f622f53c4eeff9890bdc4f31cfecc418b3", max_length=64, prefix="query: "),
}
LENGTHS = (32, 48, 64)
ARTIFACT_PATTERNS = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "sentencepiece.bpe.model", "modules.json",
                     "sentence_bert_config.json", "1_Pooling/config.json"]
LIVE_FIXTURE = Path("commerce/packages/data_adapters/tests/fixtures/live/catalog_items.json")


def live(title, description=None, path=None):
    return build_product_text(live_fields(title, description, path))


def ic(name, aisle, department):
    return build_product_text(instacart_fields(name, aisle, department))


# Hand-written probe pairs; none of these strings come from raw data.
PAIRS = [
    ("size_only_ko", live("통밀 식빵 450g", None, ["식품", "베이커리", "식빵"]),
     live("통밀 식빵 900g", None, ["식품", "베이커리", "식빵"])),
    ("organic_vs_regular_ko", live("유기농 우유 1L", None, ["식품", "유제품", "우유"]),
     live("일반 우유 1L", None, ["식품", "유제품", "우유"])),
    ("sugar_ko", live("무가당 두유 950ml", None, ["식품", "음료", "두유"]),
     live("가당 두유 950ml", None, ["식품", "음료", "두유"])),
    ("size_only_en", ic("Oat Drink Original 1 L", "plant drinks", "chilled goods"),
     ic("Oat Drink Original 2 L", "plant drinks", "chilled goods")),
    ("sugar_en", ic("Almond Drink Unsweetened 946 ml", "plant drinks", "chilled goods"),
     ic("Almond Drink Sweetened 946 ml", "plant drinks", "chilled goods")),
    ("cross_lingual", live("유기농 우유 1L", None, ["식품", "유제품", "우유"]),
     live("Organic Milk 1 L", None, ["Food", "Dairy", "Milk"])),
    ("same_aisle_other_product_ko", live("유기농 우유 1L", None, ["식품", "유제품", "우유"]),
     live("플레인 요거트 450g", None, ["식품", "유제품", "요거트"])),
    ("unrelated_ko", live("유기농 우유 1L", None, ["식품", "유제품", "우유"]),
     live("약산성 샴푸 500ml", None, ["생활용품", "헤어", "샴푸"])),
    ("unrelated_en", ic("Oat Drink Original 1 L", "plant drinks", "chilled goods"),
     ic("Dish Soap Lemon 750 ml", "dish detergents", "household")),
]


def peak_memory_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t)] + [
                (name, ctypes.c_size_t) for name in (
                    "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                    "PeakPagefileUsage")]
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # the default int truncates the handle
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return round(counters.PeakWorkingSetSize / 2**20)
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)  # Linux reports KiB


def instacart_texts(data_dir: Path) -> list[str]:
    def table(name):
        with open(data_dir / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def label(value):
        return None if value == MISSING_LABEL else value
    aisles = {r["aisle_id"]: label(r["aisle"]) for r in table("aisles.csv")}
    departments = {r["department_id"]: label(r["department"]) for r in table("departments.csv")}
    return [build_product_text(instacart_fields(r["product_name"], aisles[r["aisle_id"]],
                                                departments[r["department_id"]]))
            for r in table("products.csv")]


def token_report(encoder: FrozenTextEncoder, texts: list[str]) -> dict:
    full = encoder.token_ids(texts)
    lengths = [len(ids) for ids in full]
    unk = encoder.tokenizer.unk_token_id
    tokens = sum(lengths)
    distinct = sorted(set(texts))
    report = {
        "texts": len(texts), "distinct_texts": len(distinct),
        "tokens_p50": statistics.median(lengths), "tokens_p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "tokens_max": max(lengths), "unk_share": sum(ids.count(unk) for ids in full) / tokens,
        "round_trip_exact": sum(
            encoder.tokenizer.decode(ids, skip_special_tokens=True) == encoder.spec.prefix + text
            for ids, text in zip(full, texts)) / len(texts),
    }
    for length in LENGTHS:
        cut = encoder.token_ids(distinct, max_length=length)
        groups = {}
        for ids in cut:
            groups[tuple(ids)] = groups.get(tuple(ids), 0) + 1
        report["truncated_share@%d" % length] = sum(n > length for n in lengths) / len(lengths)
        # Distinct texts that become the same model input once cut to this length.
        report["collision_share@%d" % length] = sum(n for n in groups.values() if n > 1) / len(distinct)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("commerce/evaluation/cache/encoders"))
    parser.add_argument("--instacart-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/outputs/encoder_probe"))
    parser.add_argument("--only", choices=sorted(CANDIDATES),
                        help="one candidate per process gives a clean peak-memory number")
    parser.add_argument("--download", action="store_true",
                        help="fetch the pinned revisions (about 450 MiB each) into --cache-dir first")
    args = parser.parse_args(argv)
    if args.download:
        from huggingface_hub import snapshot_download
        for name, spec in CANDIDATES.items():
            if not args.only or name == args.only:
                snapshot_download(spec.model_id, revision=spec.revision, allow_patterns=ARTIFACT_PATTERNS,
                                  local_dir=args.cache_dir / spec.model_id.replace("/", "__") / spec.revision)

    live_texts = [catalog_item_text(item) for item in json.loads(LIVE_FIXTURE.read_text(encoding="utf-8"))]
    korean = live_texts + [text for _, a, b in PAIRS for text in (a, b) if any("가" <= c <= "힣" for c in text)]
    corpora = {"korean_probe": korean}
    if args.instacart_dir:
        corpora["instacart_products"] = instacart_texts(args.instacart_dir)
        audit = audit_texts({i: (("[TEXT]", t),) for i, t in enumerate(corpora["instacart_products"])})
        instacart_audit = {"items": audit.items, "duplicate_text": audit.duplicate_text}
    else:
        instacart_audit = None

    result = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": {"os": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "cpu": platform.processor(), "cpu_count": os.cpu_count()},
        "instacart_text_audit": instacart_audit, "candidates": {},
    }
    for name, spec in CANDIDATES.items():
        if args.only and name != args.only:
            continue
        model_dir = args.cache_dir / spec.model_id.replace("/", "__") / spec.revision
        started = time.perf_counter()
        encoder = FrozenTextEncoder(model_dir, spec)
        load_s = time.perf_counter() - started
        entry = {"spec": spec.__dict__, "artifact_hash": artifact_hash(model_dir, spec), "dim": encoder.dim,
                 "parameters": sum(p.numel() for p in encoder.model.parameters()),
                 "trainable_parameters": sum(p.numel() for p in encoder.model.parameters() if p.requires_grad),
                 "load_seconds": round(load_s, 2),
                 "lowercases": encoder.token_ids(["MILK"]) == encoder.token_ids(["milk"]),
                 "tokens": {k: token_report(encoder, v) for k, v in corpora.items()}}
        texts = [t for _, a, b in PAIRS for t in (a, b)]
        z = encoder.encode(texts)
        entry["pair_cosine"] = {label: round(float(z[2 * i] @ z[2 * i + 1]), 4)
                                for i, (label, _, _) in enumerate(PAIRS)}
        entry["repeat_identical"] = bool((encoder.encode(texts) == z).all())
        if "instacart_products" in corpora:
            started = time.perf_counter()
            encoder.encode(corpora["instacart_products"])
            seconds = time.perf_counter() - started
            entry["instacart_encode"] = {"seconds": round(seconds, 1),
                                         "texts_per_second": round(len(corpora["instacart_products"]) / seconds)}
        entry["process_peak_memory_mb"] = peak_memory_mb()  # includes earlier candidates unless --only
        result["candidates"][name] = entry
        del encoder

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ("probe-%s.json" % result["run_at"].replace(":", ""))
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
