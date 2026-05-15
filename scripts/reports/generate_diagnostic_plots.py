"""Generate simple diagnostic plots from suite CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_placeholder(path: Path, message: str) -> None:
    path.write_text(message + "\n", encoding="utf-8")


def _write_svg_bar_chart(path: Path, variants: List[str], values: List[float], title: str) -> None:
    width = 900
    height = 420
    margin = 48
    n = max(len(variants), 1)
    bar_w = max((width - 2 * margin) // (2 * n), 12)
    vmax = max([abs(v) for v in values if v == v] or [1.0])
    scale = (height - 2 * margin) / (2 * vmax if vmax > 0 else 1.0)
    baseline = height // 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{margin}" y="28" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{baseline}" x2="{width - margin}" y2="{baseline}" stroke="#444" />',
    ]
    for i, (name, value) in enumerate(zip(variants, values)):
        x = margin + i * (2 * bar_w)
        h = 0 if value != value else int(abs(value) * scale)
        y = baseline - h if value >= 0 else baseline
        color = "#2f7ed8" if value >= 0 else "#d9534f"
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" />')
        parts.append(
            f'<text x="{x}" y="{height - 14}" font-size="10" transform="rotate(30 {x},{height - 14})">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def generate_plots(input_dir: Path) -> List[Path]:
    out_dir = input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = input_dir / "diagnostic_summary.csv"
    rows = _load_rows(diagnostic_path)
    if not rows:
        p = out_dir / "diagnostic_plot_placeholder.txt"
        _write_placeholder(p, "No diagnostic_summary.csv rows available.")
        return [p]

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        variants = [r.get("variant", "") for r in rows]
        delta = [_to_float(r.get("mean_delta_vs_fedavg", "")) for r in rows]
        p = out_dir / "variant_delta_bar.svg"
        _write_svg_bar_chart(p, variants, delta, "Mean Delta vs FedAvg")
        note = out_dir / "diagnostic_plot_note.txt"
        _write_placeholder(note, "matplotlib unavailable; generated SVG fallback.")
        return [p, note]

    variants = [r.get("variant", "") for r in rows]
    delta = [_to_float(r.get("mean_delta_vs_fedavg", "")) for r in rows]
    di_drop = [_to_float(r.get("mean_di_drop", "")) for r in rows]
    neff_gain = [_to_float(r.get("mean_neff_gain", "")) for r in rows]

    generated: List[Path] = []
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.bar(variants, delta)
    ax.set_title("Mean Delta vs FedAvg")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    p1 = out_dir / "variant_delta_bar.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    generated.append(p1)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(variants, di_drop, marker="o", label="DI drop")
    ax.plot(variants, neff_gain, marker="s", label="N_eff gain")
    ax.set_title("Diagnostic Gains by Variant")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "diagnostic_gain_lines.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    generated.append(p2)

    return generated


def main() -> None:
    p = argparse.ArgumentParser(description="Generate diagnostic plot bundle.")
    p.add_argument("--input-dir", type=str, required=True)
    args = p.parse_args()
    generated = generate_plots(Path(args.input_dir))
    for path in generated:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
