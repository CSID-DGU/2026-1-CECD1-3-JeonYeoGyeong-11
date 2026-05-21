"""Family-level parsers for vision suite variant tokens."""

from __future__ import annotations

import re
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_helpers import token_float

ParsedVariant = Tuple[str, str, List[str]]


def parse_baseline_variant(v: str, default_knn_k: int) -> ParsedVariant | None:
    if v == "fedavg":
        return "fedavg", "fedavg", []
    if v == "fedavgm":
        return "fedavgm", "fedavgm", []
    if v == "fedadagrad":
        return "fedadagrad", "fedadagrad", []
    if v == "fedadam":
        return "fedadam", "fedadam", []
    if v == "fedyogi":
        return "fedyogi", "fedyogi", []
    if v == "fednova":
        return "fednova", "fednova", []
    m = re.match(r"^(fedadagrad|fedadam|fedyogi)_eta([0-9][0-9p.]*)$", v)
    if m:
        return m.group(1), v, ["--fedopt-eta", token_float(m.group(2))]
    m = re.match(r"^(fedadagrad|fedadam|fedyogi)_etal([0-9][0-9p.]*)$", v)
    if m:
        return m.group(1), v, ["--fedopt-eta-l", token_float(m.group(2))]
    m = re.match(
        r"^(fedadam|fedyogi)_eta([0-9][0-9p.]*)_etal([0-9][0-9p.]*)$",
        v,
    )
    if m:
        return (
            m.group(1),
            v,
            [
                "--fedopt-eta",
                token_float(m.group(2)),
                "--fedopt-eta-l",
                token_float(m.group(3)),
            ],
        )
    m = re.match(r"^fednova_slr([0-9][0-9p.]*)$", v)
    if m:
        return "fednova", v, ["--server-learning-rate", token_float(m.group(1))]
    if v == "fedprox":
        return "fedprox", "fedprox", []
    m = re.match(r"^fedprox_mu([0-9][0-9p.]*)$", v)
    if m:
        return "fedprox", v, ["--fedprox-mu", token_float(m.group(1))]
    if v == "fedmedian":
        return "fedmedian", "fedmedian", []
    if v == "fedtrimmedavg":
        return "fedtrimmedavg", "fedtrimmedavg", []
    m = re.match(r"^fedtrimmedavg_beta([0-9][0-9p.]*)$", v)
    if m:
        return "fedtrimmedavg", v, ["--trimmed-beta", token_float(m.group(1))]
    if v == "fedsim":
        return (
            "fedsim",
            "fedsim",
            ["--graph-mode", "knn", "--knn-k", str(default_knn_k)],
        )
    m = re.match(r"^fedsim_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "knn", "--knn-k", k]
    m = re.match(r"^fedsim_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "magnitude_knn", "--knn-k", k]
    m = re.match(r"^fedsim_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "rbf_knn", "--knn-k", k]
    return None
