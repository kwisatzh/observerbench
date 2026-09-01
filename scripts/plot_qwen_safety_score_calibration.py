"""Plot cross-stratum calibration for the sealed Qwen safety observers.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "observerbench-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


FAMILY_ORDER = (
    "seen_operation__seen_template",
    "seen_operation__heldout_template",
    "heldout_operation__seen_template",
    "heldout_operation__heldout_template",
)
FAMILY_LABELS = (
    "Seen op.\nSeen form",
    "Seen op.\nHeld-out form",
    "Held-out op.\nSeen form",
    "Held-out op.\nHeld-out form",
)
OBSERVER_PANELS = (
    ("activation-transformed-label-risk", "Activation label x severity"),
    (
        "model-logit-margin-times-severity",
        "Qwen2.5-7B-Instruct output margin x severity",
    ),
)
COLORS = {False: "#4C78A8", True: "#E45756"}


def _rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected nonempty protocol rows in {path}")
    return payload


def _percentile_box(
    axis: plt.Axes,
    values: Sequence[float],
    *,
    position: float,
    color: str,
) -> None:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("score groups must be nonempty and finite")
    q05, q25, q50, q75, q95 = np.quantile(data, (0.05, 0.25, 0.5, 0.75, 0.95))
    stats = [{"med": q50, "q1": q25, "q3": q75, "whislo": q05, "whishi": q95}]
    artists = axis.bxp(
        stats,
        positions=[position],
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    artists["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.28, linewidth=1.2)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set(color=color, linewidth=1.25)


def plot(rows_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for axis, (observer, title) in zip(axes, OBSERVER_PANELS):
        rows = _rows(rows_root / observer / "safety_protocol_rows.json")
        for family_index, family in enumerate(FAMILY_ORDER):
            family_rows = [row for row in rows if row["family_id"] == family]
            for unsafe, offset in ((False, -0.17), (True, 0.17)):
                values = [
                    float(row["risk_score"])
                    for row in family_rows
                    if bool(row["unsafe_if_allowed"]) is unsafe
                ]
                _percentile_box(
                    axis,
                    values,
                    position=family_index + offset,
                    color=COLORS[unsafe],
                )

        intervention_scores = [
            float(row["risk_score"])
            for row in rows
            if row["action"] in {"block", "escalate"}
        ]
        cutoff = min(intervention_scores)
        axis.axhline(cutoff, color="#333333", linestyle="--", linewidth=1.25)
        axis.text(
            0.99,
            cutoff,
            "  global intervention cutoff",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=8,
            color="#333333",
        )
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks(range(len(FAMILY_ORDER)), FAMILY_LABELS, fontsize=8)
        axis.set_ylabel("Risk score (observer-specific scale)", fontsize=9)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(
        handles=(
            Line2D([0], [0], color=COLORS[False], linewidth=7, alpha=0.5, label="Safe"),
            Line2D([0], [0], color=COLORS[True], linewidth=7, alpha=0.5, label="Unsafe"),
            Line2D([0], [0], color="#333333", linestyle="--", label="Action cutoff"),
        ),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot(args.rows_root, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
