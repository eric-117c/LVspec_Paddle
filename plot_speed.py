"""Plot the decoding throughput comparison saved by compare_speed.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot(source, output):
    report = json.loads(Path(source).read_text())
    blue, teal = "#526885", "#008F87"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172B43",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.subplots_adjust(left=.075, right=.975, bottom=.23, top=.72, wspace=.33)
    fig.text(.075, .94, "LVSpec vs. full autoregressive decoding", fontsize=19, weight="bold")
    fig.text(.075, .875, "Qwen2.5-VL-32B target + 7B draft  |  4 GPUs  |  64-frame setting  |  5 videos", color="#64748B", fontsize=10.5)
    ax = axes[0]
    values = [report["full_ar"]["tokens_per_second"], report["accelerated"]["tokens_per_second"]]
    bars = ax.bar(["Full (AR)", "LVSpec"], values, color=[blue, teal], width=.53, zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=7, fontsize=13, weight="bold")
    ax.set_ylim(0, max(values)*1.32)
    ax.set_ylabel("Decoding throughput (tokens/s)")
    ax.set_title("Aggregate throughput", loc="left", fontsize=12, pad=13, weight="bold")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.text(.97, .95, f'{report["decoding_throughput_speedup"]:.2f}× faster', transform=ax.transAxes,
            ha="right", va="top", fontsize=13, color=teal, weight="bold")
    ax = axes[1]
    rows = sorted(report["per_sample"], key=lambda row: row["sample_index"])
    values = [row["throughput_speedup"] for row in rows]
    bars = ax.bar(range(len(rows)), values, width=.58, color=teal, alpha=.9, zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}×" for v in values], padding=6, fontsize=10)
    ax.axhline(1, color=blue, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(len(rows)-.45, 1.04, "Full (AR) = 1×", fontsize=9, ha="right", color=blue)
    aggregate = report["decoding_throughput_speedup"]
    ax.axhline(aggregate, color="#B07824", linewidth=1.2, linestyle=":", zorder=4,
               label=f"Aggregate: {aggregate:.2f}×")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xticks(range(len(rows)), [str(row["sample_index"] + 1) for row in rows])
    ax.set_xlabel("Sample (1–5)")
    ax.set_ylabel("Throughput speedup (×)")
    ax.set_ylim(0, max(values)*1.3)
    ax.set_title("Speedup by video", loc="left", fontsize=12, pad=13, weight="bold")
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.7)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="both", length=0, pad=7)
    fig.text(.075, .095, "Speedup = (LVSpec tokens / decoding time) ÷ (AR tokens / decoding time).", fontsize=9, color="#64748B")
    fig.text(.075, .055, "Decode only; excludes prefill, video preprocessing and model loading. Output lengths differ; repository token counters used.", fontsize=8.5, color="#64748B")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=220)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/speed_comparison.json")
    parser.add_argument("--output", default="results/speed_comparison")
    args = parser.parse_args()
    plot(args.input, args.output)
