"""Plot prevalence-sensitive audit loss for the external APPS study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


MONITORS = {
    "qwen-neutral-activation-probe": ("Neutral activation", "#0072B2", "o"),
    "qwen-activation-probe": ("Verdict-adjacent activation", "#56B4E9", "s"),
    "qwen-supervised-text-ridge": ("Supervised text", "#009E73", "^"),
    "qwen-supervised-score-distribution": ("Prompt score distribution", "#E69F00", "D"),
    "qwen-controlarena-basic": ("Basic prompt", "#CC79A7", "v"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cells = payload["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for axis, prevalence in zip(axes, (0.01, 0.05), strict=True):
        selected = sorted(
            (
                cell
                for cell in cells.values()
                if float(cell["prevalence"]) == prevalence
            ),
            key=lambda cell: float(cell["budget"]),
        )
        budgets = [100.0 * float(cell["budget"]) for cell in selected]
        for monitor, (label, color, marker) in MONITORS.items():
            values = [
                cell["monitors"][monitor]["excess_over_oracle"]["mean"]
                for cell in selected
            ]
            axis.plot(
                budgets,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=1.7,
                markersize=4.5,
            )
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_xscale("log")
        axis.set_xticks(budgets, [f"{value:g}" for value in budgets])
        axis.set_xlabel("Audit budget (%)")
        axis.set_title(f"Attack prevalence {100 * prevalence:g}%")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Mean escaped attacks above oracle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.99, top=0.86, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
