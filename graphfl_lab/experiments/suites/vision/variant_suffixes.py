"""Recursive suffix handling for vision suite variant tokens."""

from __future__ import annotations

import argparse
import re
from typing import Callable, List, Tuple

from graphfl_lab.experiments.suites.vision.variant_helpers import token_float

ParsedVariant = Tuple[str, str, List[str]]
VariantParser = Callable[[str, argparse.Namespace], ParsedVariant]


def parse_suffix_variant(
    variant: str,
    args: argparse.Namespace,
    parser: VariantParser,
) -> ParsedVariant | None:
    v = variant.strip().lower()
    tau_suffix_args = {
        "fixed_tau": ["--disable-adaptive-tau", "true"],
        "norm_tau": ["--tau-source", "h_spec_normalized"],
        "normalized_tau": ["--tau-source", "h_spec_normalized"],
        "estd_tau": ["--tau-source", "e_std"],
        "e_std_tau": ["--tau-source", "e_std"],
        "norm_estd_tau": ["--tau-source", "h_spec_normalized_times_e_std"],
        "norm_e_std_tau": ["--tau-source", "h_spec_normalized_times_e_std"],
    }
    if v != "ours_fixed_tau":
        for suffix, tau_args in sorted(
            tau_suffix_args.items(), key=lambda item: -len(item[0])
        ):
            marker = f"_{suffix}"
            if v.endswith(marker):
                base = v[: -len(marker)]
                try:
                    method, _, extras = parser(base, args)
                except ValueError:
                    continue
                if method != "ours":
                    continue
                return "ours", v, extras + tau_args

    for suffix in ("graph_filter_only", "filter_only", "spectral_only", "speconly"):
        marker = f"_{suffix}"
        if v.endswith(marker):
            base = v[: -len(marker)]
            method, _, extras = parser(base, args)
            if method != "ours":
                raise ValueError(
                    f"Graph-filter-only suffix is only supported for Ours variants: {variant!r}"
                )
            return (
                "ours",
                v,
                extras
                + [
                    "--conflict-mix",
                    "0.0",
                    "--min-client-weight",
                    "0.0",
                    "--ours-server-learning-rate",
                    "1.0",
                    "--ours-server-momentum",
                    "0.0",
                ],
            )

    m = re.match(r"^(?P<base>.+)_lp(?P<strength>[0-9][0-9p.]*)$", v)
    if m:
        method, _, extras = parser(m.group("base"), args)
        if method != "ours":
            raise ValueError(f"Low-pass suffix is only supported for Ours variants: {variant!r}")
        return (
            "ours",
            v,
            extras + ["--graph-filter-strength", token_float(m.group("strength"))],
        )

    if v.endswith("_serverm"):
        base = v[: -len("_serverm")]
        method, _, extras = parser(base, args)
        if method != "ours":
            raise ValueError(
                f"Server momentum suffix is only supported for Ours variants: {variant!r}"
            )
        return (
            "ours",
            v,
            extras
            + [
                "--ours-server-learning-rate",
                str(args.ours_server_learning_rate),
                "--ours-server-momentum",
                str(args.server_momentum),
            ],
        )

    return None
