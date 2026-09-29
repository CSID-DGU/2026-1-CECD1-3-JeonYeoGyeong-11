"""D0022 results as Markdown tables, from the Git-ignored run records.

    python -m commerce.evaluation.d0022_report --runs commerce/evaluation/runs

Reads fl_lab records (the last round is the result, evaluation.md §5),
harex_compare records (local_only) and cold_start records. The first contrast
(evaluation.md §4) is R_hx - T_hx NDCG@10 over FL sellers: per seller the mean
over the seeds both variants ran, then a seller-level paired bootstrap 95%
interval. An interval that holds 0 is written "차이를 확인하지 못했다". Runs
under GCI's conditions (--protocol gci) get their own table next to the paper's
HR@10 and never enter the main ones. Tables hold aggregates only, so they can
go into a PR once C has checked them.
"""
import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("ndcg@10", "recall@20", "hr@10")
# GCI Table 4, HR@10: (local, glocal FL) per client.
GCI_PAPER = {"BBQ 1": (0.443, 0.478), "BBQ 2": (0.455, 0.522), "Ulsan Pedal 1": (0.110, 0.147),
             "Ulsan Pedal 2": (0.117, 0.140)}


def per_seller(result: dict, metric: str) -> list[float]:
    """Per-seller macro values in seller order; every run lists sellers in sorted order."""
    return [s[metric] for s in result.get("per_seller", [])]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over sellers and its 95% percentile interval from resampling sellers."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(low: float, high: float) -> str:
    return "차이를 확인하지 못했다" if low <= 0 <= high else ("R_hx가 높다" if low > 0 else "T_hx가 높다")


def load(root: Path):
    fl, local, cold = [], [], []
    for path in sorted(root.rglob("record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        run = record.get("run", "")
        if run == "D0022 federated_lab_sim":
            fl.append(record)
        elif run == "D0022 harex_compare":
            local.append(record)
        elif run == "A-0 cold start":
            cold.append(record)
    return fl, local, cold


def fmt(value) -> str:
    return "" if value is None else "%.4f" % value


def macro(result: dict, metric: str):
    return result.get("macro", {}).get(metric) if result.get("examples") else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("commerce/evaluation/runs"))
    args = parser.parse_args(argv)
    fl, local, cold = load(args.runs)
    gci_fl = [r for r in fl if r["settings"].get("protocol") == "gci"]
    gci_local = [r for r in local if r["settings"].get("protocol") == "gci"]
    fl = [r for r in fl if r["settings"].get("protocol", "next_visit") == "next_visit"]
    local = [r for r in local if r["settings"].get("protocol", "next_visit") == "next_visit"]
    out = []

    # FL runs by (target, holdout, variant) -> {seed: record}
    runs = {}
    for record in fl:
        s = record["settings"]
        key = (s["target"], s.get("holdout_frac", 0.0), s["variant"])
        runs.setdefault(key, {})[s["seed"]] = record
    locals_ = {}
    for record in local:
        s = record["settings"]
        for target, variants in record["results"].items():
            for variant, res in variants.items():
                key = (target, s.get("holdout_frac", 0.0), variant)
                locals_.setdefault(key, {})[s["seed"]] = res

    for target in ("basket", "next_item"):
        keys = sorted(k for k in runs if k[0] == target and k[1] == 0.0)
        if not keys:
            continue
        out.append("### %s — 판매자 100곳, 판매자 macro, 마지막 라운드(seed 평균)\n" % target)
        out.append("| 모델 | 방식 | seed | NDCG@10 | Recall@20 | HR@10 | 재구매 NDCG@10 | 첫 구매 NDCG@10 | 최적 라운드 | 수렴 |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        baselines = None
        for key in keys:
            variant = key[2]
            seeds = runs[key]
            arm = "%s FL" % variant
            rows = [r["metrics"]["final_round"] for r in seeds.values()]
            mean = {m: np.mean([macro(r[arm]["all"], m) for r in rows]) for m in METRICS}
            parts = {p: np.mean([macro(r[arm][p], "ndcg@10") for r in rows]) for p in ("repeat", "explore")}
            best = ", ".join(str(r["training"]["best_round"]) for r in seeds.values())
            plateau = all(r["training"]["plateaued"] for r in seeds.values())
            out.append("| %s | FL | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                variant, len(seeds), fmt(mean["ndcg@10"]), fmt(mean["recall@20"]), fmt(mean["hr@10"]),
                fmt(parts["repeat"]), fmt(parts["explore"]), best, "예" if plateau else "아니오(더 길게)"))
            baselines = baselines or rows[0]
            if key in locals_:
                lrows = list(locals_[key].values())
                larm = next(n for n in lrows[0]["metrics"] if n.startswith(variant))
                lm = {m: np.mean([macro(r["metrics"][larm]["all"], m) for r in lrows]) for m in METRICS}
                lp = {p: np.mean([macro(r["metrics"][larm][p], "ndcg@10") for r in lrows]) for p in ("repeat", "explore")}
                conv = sum(t["converged"] for r in lrows for t in r["training"].values())
                total = sum(len(r["training"]) for r in lrows)
                out.append("| %s | local_only | %s | %s | %s | %s | %s | %s | | %d/%d 판매자 |" % (
                    variant, len(lrows), fmt(lm["ndcg@10"]), fmt(lm["recall@20"]), fmt(lm["hr@10"]),
                    fmt(lp["repeat"]), fmt(lp["explore"]), conv, total))
        for name in ("popularity", "P-TopFreq"):
            b = baselines[name]
            out.append("| %s | 기준선 | | %s | %s | %s | %s | %s | | |" % (
                name, fmt(macro(b["all"], "ndcg@10")), fmt(macro(b["all"], "recall@20")), fmt(macro(b["all"], "hr@10")),
                fmt(macro(b["repeat"], "ndcg@10")), fmt(macro(b["explore"], "ndcg@10"))))
        out.append("")

        # First contrast and the other pairs, seller-level paired bootstrap over seed-mean values.
        pairs = [("R_hx", "T_hx", "1차 대비"), ("R_hx", "R_hx_shuffled", "관계 셔플 대조"),
                 ("T_lm", "T_hx", "보조"), ("R_lm", "T_lm", "보조")]
        lines = []
        for a, b, label in pairs:
            ka, kb = (target, 0.0, a), (target, 0.0, b)
            if ka not in runs or kb not in runs:
                continue
            seeds = sorted(set(runs[ka]) & set(runs[kb]))
            va = np.mean([per_seller(runs[ka][s]["metrics"]["final_round"]["%s FL" % a]["all"], "ndcg@10") for s in seeds], axis=0)
            vb = np.mean([per_seller(runs[kb][s]["metrics"]["final_round"]["%s FL" % b]["all"], "ndcg@10") for s in seeds], axis=0)
            diff, low, high = paired_bootstrap(va, vb)
            judged = verdict(low, high) if label == "1차 대비" else ""
            lines.append("| %s | %s − %s | %d | %+.4f | [%+.4f, %+.4f] | %s |" % (
                label, a, b, len(seeds), diff, low, high, judged))
        if lines:
            out.append("| 비교 | 차이 | seed | NDCG@10 차이 | 판매자 bootstrap 95% | 판정 |")
            out.append("| --- | --- | --- | --- | --- | --- |")
            out += lines
            out.append("")

    # Cold start: C-new at the cohort (FL runs trained with a holdout) and A-0 at held-out sellers.
    cnew = sorted(k for k in runs if k[1] > 0)
    if cnew or cold:
        out.append("### 콜드스타트 — NDCG@10 (Recall@20), 판매자 macro\n")
        out.append("| 모델 | 기존 매장 전체 | 기존 매장·처음 보는 상품(C-new) | 신규 매장(A-0) | 신규 매장 단독 학습 | 신규 매장·처음 보는 상품 |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        cold_runs, cold_local = {}, {}
        for record in cold:
            for group in record["groups"]:
                for r in group["runs"]:
                    cold_runs[(r["target"], r["holdout_frac"], r["variant"])] = r
                for r in group["local"]:
                    cold_local[(r["target"], group["holdout_frac"], r["variant"])] = r

        def cell(result):
            n, r = macro(result, "ndcg@10"), macro(result, "recall@20")
            return "" if n is None else "%.4f (%.4f)" % (n, r)
        for variant in ("T_hx", "R_hx", "T_lm", "R_lm"):
            base = runs.get(("basket", 0.0, variant), {}).get(0)
            held = next((runs[k][0] for k in cnew if k[0] == "basket" and k[2] == variant and 0 in runs[k]), None)
            a0 = cold_runs.get(("basket", 0.0, variant))
            a0_new = next((v for k, v in cold_runs.items() if k[0] == "basket" and k[1] > 0 and k[2] == variant), None)
            loc = cold_local.get(("basket", 0.0, variant))
            arm = "%s FL" % variant
            out.append("| %s | %s | %s | %s | %s | %s |" % (
                variant,
                cell(base["metrics"]["final_round"][arm]["all"]) if base else "",
                cell(held["metrics"]["final_round"][arm]["cnew"]) if held else "",
                cell(a0["final_round"]["%s FL A-0" % variant]["all"]) if a0 else "",
                cell(loc["metrics"]["all"]) if loc else "",
                cell(a0_new["final_round"]["%s FL A-0" % variant]["cnew"]) if a0_new else ""))
        out.append("")
    if gci_fl or gci_local:
        out += gci_table(gci_fl, gci_local)
    print("\n".join(out))


def gci_table(fl, local):
    out = ["### HAREX 재현 조건(추가 범위) — 상품 단위 5개 창·무작위 분할, FL은 GCI처럼 조기 종료한 모델, 판매자 macro, seed 평균\n",
           "| 모델 | FL HR@10 | FL NDCG@10 | 단독 학습 HR@10 | 단독 학습 NDCG@10 |",
           "| --- | --- | --- | --- | --- |"]

    def mean(rows, metric):
        return fmt(float(np.mean([macro(x, metric) for x in rows]))) if rows else ""
    for variant in ("T_hx", "R_hx", "T_lm", "R_lm"):
        f = [r["metrics"][r["metrics"].get("primary", "final_round")]["%s FL" % variant]["all"]
             for r in fl if r["settings"]["variant"] == variant]
        l = [res["metrics"][arm]["all"] for r in local for v, res in r["results"].get("basket", {}).items()
             if v == variant for arm in res["metrics"] if arm.startswith(variant)]
        if f or l:
            out.append("| %s | %s | %s | %s | %s |" % (variant, mean(f, "hr@10"), mean(f, "ndcg@10"),
                                                      mean(l, "hr@10"), mean(l, "ndcg@10")))
    base = fl[0]["metrics"]["final_round"] if fl else None
    if base:
        for name in ("popularity", "P-TopFreq"):
            out.append("| %s | %s | %s | | |" % (name, fmt(macro(base[name]["all"], "hr@10")),
                                                fmt(macro(base[name]["all"], "ndcg@10"))))
    out.append("")
    out.append("GCI 논문 Table 4의 HR@10(단독 학습 / Glocal FL): " + ", ".join(
        "%s %.3f / %.3f" % (name, a, b) for name, (a, b) in GCI_PAPER.items()))
    out.append("")
    return out


if __name__ == "__main__":
    main()
