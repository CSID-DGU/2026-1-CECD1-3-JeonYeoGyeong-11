"""Plot round-wise convergence curves from vision FL suite result JSONs.

The script intentionally writes SVG with the Python standard library only, so
experiment plotting does not depend on matplotlib being installed.  It can plot
one suite or overlay multiple matched suites, for example warmup vs no-warmup.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MetricPoint = Tuple[int, float]


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#a16207",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create convergence SVG/CSV plots from run_vision_suite.py outputs."
        )
    )
    p.add_argument(
        "--suite-dir",
        type=Path,
        nargs="*",
        default=[],
        help="One or more suite output directories.",
    )
    p.add_argument(
        "--result",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Optional individual result_vision_*.json or result_general_*.json files. Multiple files "
            "with the same label and variant are aggregated across seeds."
        ),
    )
    p.add_argument(
        "--label",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels matching --suite-dir order.",
    )
    p.add_argument(
        "--result-label",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels matching --result order.",
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional variant filter, e.g. fedavg ours_knn_k2.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy", "loss"],
        choices=["accuracy", "loss"],
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--title", type=str, default="Vision FL convergence")
    p.add_argument(
        "--show-seeds",
        action="store_true",
        help="Draw faint per-seed lines behind the mean curve.",
    )
    p.add_argument(
        "--no-std-band",
        action="store_true",
        help="Do not draw mean +/- std bands.",
    )
    return p.parse_args()


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def pair_series(raw: Any) -> List[MetricPoint]:
    out: List[MetricPoint] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        round_num = as_float(item[0])
        value = as_float(item[1])
        if round_num is None or value is None:
            continue
        out.append((int(round_num), float(value)))
    return out


def load_variant_tokens(suite_dir: Path) -> List[str]:
    tokens = set()
    for name in ("general_suite_rows.json", "suite_rows.json"):
        path = suite_dir / name
        if not path.is_file():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("variant"):
                tokens.add(str(row["variant"]))
    return sorted(tokens, key=len, reverse=True)


def infer_variant(
    path: Path,
    obj: Dict[str, Any],
    method: str,
    seed: int,
    variant_tokens: Sequence[str],
) -> str:
    if method == "fedavg":
        return "fedavg"

    run_tag = str(obj.get("meta", {}).get("run_tag") or "")
    for token in variant_tokens:
        if token != "fedavg" and run_tag == token:
            return token
        if token != "fedavg" and run_tag == f"{token}_seed{seed}":
            return token
        if token != "fedavg" and run_tag.endswith("_" + token + f"_seed{seed}"):
            return token
        if token != "fedavg" and run_tag.endswith("_" + token):
            return token

    if method != "ours":
        return method

    name = path.stem
    matches = re.findall(r"_(ours_[A-Za-z0-9_]+?)_seed\d+", name)
    if matches:
        return max((m.lower() for m in matches), key=len)

    graph = obj.get("meta", {}).get("graph", {})
    mode = str(graph.get("graph_mode", "ours")).lower()
    knn_k = graph.get("knn_k")
    if mode == "knn" and knn_k is not None:
        return f"ours_knn_k{knn_k}"
    if mode == "random" and knn_k is not None:
        return f"ours_random_matched_k{knn_k}"
    return f"ours_{mode}"


def result_method_and_seed(path: Path, obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    results = obj.get("results", {})
    if not isinstance(results, dict) or not results:
        return None, None
    method = next(
        (
            candidate
            for candidate in (
                "fedavg",
                "fedavgm",
                "fedprox",
                "fedmedian",
                "fedtrimmedavg",
                "fedsim",
                "ours",
            )
            if candidate in results
        ),
        next(iter(results.keys())),
    )

    seed = obj.get("meta", {}).get("experiment", {}).get("seed")
    if seed is None:
        seed = obj.get("meta", {}).get("seed")
    if seed is None:
        match = re.search(r"_seed(\d+)", path.name)
        seed = int(match.group(1)) if match else None
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        return method, None
    return method, seed_i


def metric_series(result: Dict[str, Any], method: str, metric: str) -> List[MetricPoint]:
    method_obj = result.get("results", {}).get(method, {})
    if metric == "accuracy":
        return pair_series(method_obj.get("metrics_distributed", {}).get("accuracy"))
    if metric == "loss":
        return pair_series(method_obj.get("losses_distributed"))
    raise ValueError(metric)


def warmup_rounds(result: Dict[str, Any]) -> Optional[int]:
    value = result.get("meta", {}).get("aggregation", {}).get("warmup_rounds")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def pstdev(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / len(values)))


def load_suite_records(
    suite_dir: Path,
    suite_label: str,
    variant_filter: Optional[set[str]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    variant_tokens = load_variant_tokens(suite_dir)
    result_paths: dict[str, Path] = {}
    for path in sorted(suite_dir.glob("result_general_*.json")):
        result_paths[path.name.replace("result_general_", "", 1)] = path
    for path in sorted(suite_dir.glob("result_vision_*.json")):
        result_paths[path.name.replace("result_vision_", "", 1)] = path
    for path in sorted(result_paths.values()):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        method, seed = result_method_and_seed(path, obj)
        if method is None or seed is None:
            continue
        variant = infer_variant(path, obj, method, seed, variant_tokens)
        if variant_filter is not None and variant not in variant_filter:
            continue
        records.append(
            {
                "suite": suite_label,
                "suite_dir": str(suite_dir),
                "path": path,
                "result": obj,
                "method": method,
                "variant": variant,
                "seed": seed,
                "warmup_rounds": warmup_rounds(obj),
            }
        )
    return records


def load_result_record(
    result_path: Path,
    result_label: str,
    variant_filter: Optional[set[str]],
) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    method, seed = result_method_and_seed(result_path, obj)
    if method is None or seed is None:
        return []
    variant = infer_variant(
        result_path,
        obj,
        method,
        seed,
        load_variant_tokens(result_path.parent),
    )
    if variant_filter is not None and variant not in variant_filter:
        return []
    return [
        {
            "suite": result_label,
            "suite_dir": str(result_path.parent),
            "path": result_path,
            "result": obj,
            "method": method,
            "variant": variant,
            "seed": seed,
            "warmup_rounds": warmup_rounds(obj),
        }
    ]


def aggregate_metric_rows(
    records: Sequence[Dict[str, Any]], metric: str
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], Dict[int, Dict[int, float]]] = {}
    warmups: Dict[Tuple[str, str], set[int]] = {}
    seed_series: Dict[Tuple[str, str], Dict[int, List[MetricPoint]]] = {}

    for rec in records:
        key = (str(rec["suite"]), str(rec["variant"]))
        series = metric_series(rec["result"], str(rec["method"]), metric)
        if not series:
            continue
        grouped.setdefault(key, {}).setdefault(int(rec["seed"]), {})
        seed_series.setdefault(key, {})[int(rec["seed"])] = series
        for round_num, value in series:
            grouped[key][int(rec["seed"])][int(round_num)] = float(value)
        wr = rec.get("warmup_rounds")
        if wr is not None:
            warmups.setdefault(key, set()).add(int(wr))

    rows: List[Dict[str, Any]] = []
    plot_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(grouped.keys()):
        suite, variant = key
        rounds = sorted({r for by_seed in grouped[key].values() for r in by_seed})
        agg_points: List[Dict[str, Any]] = []
        for round_num in rounds:
            values = [
                by_round[round_num]
                for by_round in grouped[key].values()
                if round_num in by_round
            ]
            if not values:
                continue
            row = {
                "suite": suite,
                "variant": variant,
                "metric": metric,
                "round": int(round_num),
                "n_seeds": len(values),
                "mean": mean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            rows.append(row)
            agg_points.append(row)
        plot_groups[key] = {
            "points": agg_points,
            "seed_series": seed_series.get(key, {}),
            "warmup_rounds": sorted(warmups.get(key, set())),
        }
    return rows, plot_groups


def nice_ticks(lo: float, hi: float, count: int = 6) -> List[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if abs(hi - lo) < 1e-12:
        pad = abs(lo) * 0.1 if abs(lo) > 1e-12 else 1.0
        lo -= pad
        hi += pad
    step = (hi - lo) / max(1, count - 1)
    return [lo + i * step for i in range(count)]


def polyline(points: Iterable[Tuple[float, float]], **attrs: Any) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attr_s = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f"<polyline points=\"{pts}\" {attr_s} />"


def polygon(points: Iterable[Tuple[float, float]], **attrs: Any) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attr_s = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f"<polygon points=\"{pts}\" {attr_s} />"


def render_svg(
    metric: str,
    groups: Dict[Tuple[str, str], Dict[str, Any]],
    out_path: Path,
    title: str,
    show_seeds: bool,
    show_std_band: bool,
) -> None:
    width, height = 1100, 660
    left, right, top, bottom = 86, 310, 58, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_points = [p for group in groups.values() for p in group["points"]]
    if not all_points:
        out_path.write_text("", encoding="utf-8")
        return

    x_min = min(int(p["round"]) for p in all_points)
    x_max = max(int(p["round"]) for p in all_points)
    y_lows = [float(p["mean"]) - float(p["std"]) for p in all_points]
    y_highs = [float(p["mean"]) + float(p["std"]) for p in all_points]
    y_min = min(y_lows)
    y_max = max(y_highs)
    if metric == "accuracy":
        y_min = max(0.0, y_min - 0.03)
        y_max = min(1.0, y_max + 0.03)
    else:
        pad = max(0.02, (y_max - y_min) * 0.08)
        y_min -= pad
        y_max += pad

    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def sx(round_num: float) -> float:
        return left + (float(round_num) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - float(value)) / (y_max - y_min) * plot_h

    lines: List[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Segoe UI, Arial, sans-serif; fill: #111827; }",
        ".axis { stroke: #111827; stroke-width: 1.2; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".tick { fill: #4b5563; font-size: 12px; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { fill: #4b5563; font-size: 13px; }",
        ".legend { font-size: 12px; }",
        ".warmup { stroke: #6b7280; stroke-width: 1.2; stroke-dasharray: 5 5; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text class="title" x="{left}" y="30">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="50">metric: {html.escape(metric)}; line = seed mean, band = +/- std</text>',
    ]

    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" />')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.3f}</text>')

    x_ticks = list(range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1))
    for tick in x_ticks:
        x = sx(tick)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" />')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{tick}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" />')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" />')
    lines.append(f'<text class="tick" x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle">round</text>')
    y_label = "distributed accuracy" if metric == "accuracy" else "distributed loss"
    lines.append(
        f'<text class="tick" x="24" y="{top + plot_h / 2:.2f}" '
        f'text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.2f})">{y_label}</text>'
    )

    unique_warmups = sorted(
        {
            wr
            for group in groups.values()
            for wr in group.get("warmup_rounds", [])
            if wr and wr >= x_min and wr <= x_max
        }
    )
    for wr in unique_warmups:
        x = sx(wr)
        lines.append(f'<line class="warmup" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" />')
        lines.append(f'<text class="tick" x="{x + 5:.2f}" y="{top + 15}" transform="rotate(90 {x + 5:.2f} {top + 15})">warmup={wr}</text>')

    for idx, (key, group) in enumerate(sorted(groups.items())):
        color = PALETTE[idx % len(PALETTE)]
        points = group["points"]
        mean_pts = [(sx(p["round"]), sy(p["mean"])) for p in points]
        if show_std_band:
            upper = [(sx(p["round"]), sy(float(p["mean"]) + float(p["std"]))) for p in points]
            lower = [(sx(p["round"]), sy(float(p["mean"]) - float(p["std"]))) for p in reversed(points)]
            lines.append(polygon(upper + lower, fill=color, opacity="0.10", stroke="none"))
        if show_seeds:
            for seed, series in sorted(group.get("seed_series", {}).items()):
                seed_pts = [(sx(r), sy(v)) for r, v in series]
                lines.append(polyline(seed_pts, fill="none", stroke=color, stroke_width="1", opacity="0.22"))
        lines.append(polyline(mean_pts, fill="none", stroke=color, stroke_width="2.6", stroke_linejoin="round", stroke_linecap="round"))
        for x, y in mean_pts:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}" />')

    legend_x = left + plot_w + 32
    legend_y = top + 8
    lines.append(f'<text class="legend" x="{legend_x}" y="{legend_y - 18}" font-weight="700">Series</text>')
    for idx, ((suite, variant), _group) in enumerate(sorted(groups.items())):
        color = PALETTE[idx % len(PALETTE)]
        y = legend_y + idx * 24
        label = f"{suite}: {variant}"
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<text class="legend" x="{legend_x + 30}" y="{y + 4}">{html.escape(label)}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = ["suite", "variant", "metric", "round", "n_seeds", "mean", "std", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    args = parse_args()
    if not args.suite_dir and not args.result:
        raise SystemExit("Provide at least one --suite-dir or --result")
    if args.label is not None and len(args.label) not in {0, len(args.suite_dir)}:
        raise SystemExit("--label count must match --suite-dir count")
    if args.result_label is not None and len(args.result_label) not in {0, len(args.result)}:
        raise SystemExit("--result-label count must match --result count")

    labels = (
        list(args.label)
        if args.label
        else [p.name for p in args.suite_dir]
    )
    result_labels = (
        list(args.result_label)
        if args.result_label
        else [p.parent.name for p in args.result]
    )
    default_out_root = args.suite_dir[0] if args.suite_dir else args.result[0].parent
    out_dir = args.out_dir or (default_out_root / "convergence_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_filter = {v.strip().lower() for v in args.variants} if args.variants else None

    records: List[Dict[str, Any]] = []
    for suite_dir, label in zip(args.suite_dir, labels):
        records.extend(load_suite_records(suite_dir=suite_dir, suite_label=label, variant_filter=variant_filter))
    for result_path, label in zip(args.result, result_labels):
        records.extend(
            load_result_record(
                result_path=result_path,
                result_label=label,
                variant_filter=variant_filter,
            )
        )
    if not records:
        raise SystemExit("No matching result_vision_*.json or result_general_*.json records found")

    manifest_lines = [
        "# Convergence Plots",
        "",
        f"- title: `{args.title}`",
        f"- suites: {', '.join(f'`{label}`={path}' for label, path in zip(labels, args.suite_dir)) or 'none'}",
        f"- result files: {', '.join(f'`{label}`={path}' for label, path in zip(result_labels, args.result)) or 'none'}",
        f"- variants: `{', '.join(args.variants)}`" if args.variants else "- variants: all",
        "",
        "## Files",
        "",
    ]

    all_rows: List[Dict[str, Any]] = []
    for metric in args.metrics:
        rows, groups = aggregate_metric_rows(records, metric)
        all_rows.extend(rows)
        svg_path = out_dir / f"convergence_{metric}.svg"
        render_svg(
            metric=metric,
            groups=groups,
            out_path=svg_path,
            title=args.title,
            show_seeds=bool(args.show_seeds),
            show_std_band=not bool(args.no_std_band),
        )
        manifest_lines.append(f"- `{svg_path.name}`")

    csv_path = out_dir / "convergence_round_summary.csv"
    write_csv(all_rows, csv_path)
    manifest_lines.append(f"- `{csv_path.name}`")
    (out_dir / "README.md").write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved convergence plots under {out_dir}")


if __name__ == "__main__":
    main()
