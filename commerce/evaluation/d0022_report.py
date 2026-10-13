"""D0022 results as Markdown tables, from the Git-ignored run records.

    python -m commerce.evaluation.d0022_report --runs commerce/evaluation/runs

Reads fl_lab records (the last round is the result, evaluation.md §5),
harex_compare records (local_only) and cold_start records. The first contrast
(evaluation.md §4) is R_hx - T_hx NDCG@10 over FL sellers: per seller the mean
over the seeds both variants ran, then a seller-level paired bootstrap 95%
interval. An interval that holds 0 is written "차이를 확인하지 못했다". Runs
under GCI's conditions (--protocol gci) get their own table next to the paper's
HR@10 and never enter the main ones. Runs under different data or loss settings
(seller count, seller size, menu, negatives) get separate tables, so a small-seller
run never joins a 1,040-order seed mean. Tables hold aggregates only, so they can
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


def condition(settings: dict) -> tuple:
    """(sellers, train orders per seller, menu size, negatives): what one table holds fixed.
    Records written before a flag existed ran at its default."""
    return (settings.get("sellers", 100), settings.get("seller_size", 0), settings.get("menu_size", 0),
            settings.get("negatives", 200))


MAIN = condition({})


def condition_label(cond: tuple) -> str:
    sellers, size, menu, negatives = cond
    parts = ["판매자 %d곳" % sellers, "train 주문 %s건" % ("{:,}".format(size) if size else "1,040")]
    if menu:
        parts.append("메뉴 %d개" % menu)
    if negatives != 200:
        parts.append("전체 상품 softmax" if negatives == 0 else "음성 %d개" % negatives)
    return ", ".join(parts)


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
    dh_fl = [r for r in fl if r["settings"].get("protocol") == "dunnhumby"]
    dh_local = [r for r in local if r["settings"].get("protocol") == "dunnhumby"]
    fl = [r for r in fl if r["settings"].get("protocol", "next_visit") == "next_visit"]
    local = [r for r in local if r["settings"].get("protocol", "next_visit") == "next_visit"]
    out = []

    # FL runs by (condition, target, holdout, variant) -> {seed: record}
    runs = {}
    for record in fl:
        s = record["settings"]
        key = (condition(s), s["target"], s.get("holdout_frac", 0.0), s["variant"])
        runs.setdefault(key, {})[s["seed"]] = record
    locals_ = {}
    for record in local:
        s = record["settings"]
        for target, variants in record["results"].items():
            for variant, res in variants.items():
                key = (condition(s), target, s.get("holdout_frac", 0.0), variant)
                locals_.setdefault(key, {})[s["seed"]] = res

    # The pre-registered condition first, then the others (small sellers, pilots) in their own tables.
    conditions = sorted({k[0] for k in runs}, key=lambda c: (c != MAIN, c))
    for cond, target in ((c, t) for c in conditions for t in ("basket", "next_item")):
        keys = sorted(k for k in runs if k[0] == cond and k[1] == target and k[2] == 0.0)
        if not keys:
            continue
        out.append("### %s — %s, 판매자 macro, 마지막 라운드(seed 평균)\n" % (target, condition_label(cond)))
        out.append("| 모델 | 방식 | seed | NDCG@10 | Recall@20 | HR@10 | 재구매 NDCG@10 | 첫 구매 NDCG@10 | 최적 라운드 | 수렴 |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        baselines = None
        for key in keys:
            variant = key[3]
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
            ka, kb = (cond, target, 0.0, a), (cond, target, 0.0, b)
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
    cnew = sorted(k for k in runs if k[0] == MAIN and k[2] > 0)
    if cnew or cold:
        out.append("### 콜드스타트 — %s, NDCG@10 (Recall@20), 판매자 macro, seed 평균\n" % condition_label(MAIN))
        out.append("| 모델 | 기존 매장 전체 | 기존 매장·처음 보는 상품(C-new) | 신규 매장(A-0) | 신규 매장 단독 학습 | 신규 매장·처음 보는 상품 |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        cold_runs, cold_local = {}, {}
        for record in cold:
            for group in record["groups"]:
                for r in group["runs"]:
                    cold_runs.setdefault((r["target"], r["holdout_frac"], r["variant"]), {})[r.get("seed", 0)] = r
                for r in group["local"]:
                    cold_local[(r["target"], group["holdout_frac"], r["variant"])] = r

        def cell(results):
            results = [x for x in results if macro(x, "ndcg@10") is not None]
            if not results:
                return ""
            return "%.4f (%.4f)" % tuple(np.mean([macro(x, m) for x in results]) for m in ("ndcg@10", "recall@20"))
        for variant in ("T_hx", "R_hx", "T_lm", "R_lm"):
            arm = "%s FL" % variant
            base = runs.get((MAIN, "basket", 0.0, variant), {}).values()
            held = [r for k in cnew if k[1] == "basket" and k[3] == variant for r in runs[k].values()]
            a0 = cold_runs.get(("basket", 0.0, variant), {}).values()
            a0_new = [r for k, v in cold_runs.items() if k[0] == "basket" and k[1] > 0 and k[2] == variant
                      for r in v.values()]
            loc = cold_local.get(("basket", 0.0, variant))
            out.append("| %s | %s | %s | %s | %s | %s |" % (
                variant,
                cell(r["metrics"]["final_round"][arm]["all"] for r in base),
                cell(r["metrics"]["final_round"][arm]["cnew"] for r in held),
                cell(r["final_round"]["%s FL A-0" % variant]["all"] for r in a0),
                cell([loc["metrics"]["all"]] if loc else []),
                cell(r["final_round"]["%s FL A-0" % variant]["cnew"] for r in a0_new)))
        out.append("")
    for cond in sorted({condition(r["settings"]) for r in gci_fl + gci_local}):
        out += gci_table([r for r in gci_fl if condition(r["settings"]) == cond],
                         [r for r in gci_local if condition(r["settings"]) == cond], cond)
    for sellers in sorted({r["settings"]["sellers"] for r in dh_fl + dh_local}):
        out += dh_table([r for r in dh_fl if r["settings"]["sellers"] == sellers],
                        [r for r in dh_local if r["settings"]["sellers"] == sellers], sellers)
    print("\n".join(out))


def dh_table(fl, local, sellers):
    """The auxiliary Dunnhumby cohort: answers split by whether their text is unique in the store (evaluation.md §4)."""
    out = ["### Dunnhumby 보조 cohort — 점포 %d곳, 주 기준 분할, 판매자 macro, seed 평균\n" % sellers,
           "| 모델 | 방식 | seed | NDCG@10 | HR@10 | Recall@20 | 정답 텍스트 고유 NDCG@10 | 같은 텍스트 정답 NDCG@10 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    fl_by, local_by, base = {}, {}, None
    for r in fl:
        fl_by.setdefault(r["settings"]["variant"], {})[r["settings"]["seed"]] = r["metrics"]["final_round"]
    for r in local:
        for variant, res in r["results"].get("basket", {}).items():
            local_by.setdefault(variant, {})[r["settings"]["seed"]] = res["metrics"]

    def row(name, mode, results, arm, seeds=True):
        res = list(results.values())

        def mean(part, metric):
            return np.mean([macro(x[arm][part], metric) for x in res])
        return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, mode, len(res) if seeds else "", fmt(mean("all", "ndcg@10")), fmt(mean("all", "hr@10")),
            fmt(mean("all", "recall@20")), fmt(mean("text_unique", "ndcg@10")), fmt(mean("text_shared", "ndcg@10")))
    for variant in sorted(set(fl_by) | set(local_by)):
        if variant in fl_by:
            out.append(row(variant, "FL", fl_by[variant], "%s FL" % variant))
            base = base or next(iter(fl_by[variant].values()))
        if variant in local_by:
            arm = "%s (HAREX baseline)" % variant if variant == "T_hx" else variant
            out.append(row(variant, "local_only", local_by[variant], arm))
    if base:
        for name in ("popularity", "P-TopFreq"):
            out.append(row(name, "기준선", {0: base}, name, seeds=False))
    out.append("")
    contrasts = []
    for a, b in (("R_lm", "T_lm"), ("R_hx", "T_hx")):
        both = sorted(set(fl_by.get(a, {})) & set(fl_by.get(b, {})))
        if not both:
            continue
        for part, label in (("all", "전체"), ("text_unique", "정답 텍스트 고유")):
            pa = [fl_by[a][seed]["%s FL" % a][part].get("per_seller", {}) for seed in both]
            pb = [fl_by[b][seed]["%s FL" % b][part].get("per_seller", {}) for seed in both]
            stores = sorted(set.intersection(*(set(x) for x in pa + pb)))
            if not stores:
                continue
            va = np.array([np.mean([x[st]["ndcg@10"] for x in pa]) for st in stores])
            vb = np.array([np.mean([x[st]["ndcg@10"] for x in pb]) for st in stores])
            mean, low, high = paired_bootstrap(va, vb)
            judged = "차이를 확인하지 못했다" if low <= 0 <= high else "%s가 높다" % (a if low > 0 else b)
            contrasts.append("| %s − %s FL (%s) | %d | %+.4f | [%+.4f, %+.4f] | %s |" % (
                a, b, label, len(both), mean, low, high, judged))
    if contrasts:
        out += ["| 비교 | seed | NDCG@10 차이 | 판매자 bootstrap 95% | 판정 |", "| --- | --- | --- | --- | --- |"]
        out += contrasts + [""]
    out.append("DH는 상품명이 없어 점포 상품의 약 3분의 2가 다른 상품과 텍스트가 같다(OQ04). 그래서 R이 T보다 높아도 "
               "관계 덕분인지 상품 구별 덕분인지 가를 수 없어, 정답 텍스트가 고유한 줄을 함께 본다(evaluation.md §4).")
    out.append("")
    return out


def gci_table(fl, local, cond=MAIN):
    kind = "BBQ 유사" if cond[2] else "울산페달식(전체 상품)"
    out = ["### HAREX 재현 조건(추가 범위), %s — %s, 상품 단위 5개 창·무작위 분할, FL은 GCI처럼 조기 종료한 모델, "
           "판매자 macro, seed 평균\n" % (kind, condition_label(cond)),
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
