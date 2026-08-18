"""Diagnose frozen APPS monitor scores across family and input-length strata.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from observerbench.ai_control import read_ai_control_scores
from observerbench.ai_control_followup import budget_metrics
from observerbench.core import write_json
from observerbench.provenance import file_sha256


DEFAULT_MONITORS = (
    "qwen-neutral-activation-probe",
    "qwen-activation-probe",
    "qwen-supervised-text-ridge",
    "qwen-supervised-score-distribution",
    "qwen-controlarena-basic",
)


def _load_design(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("design must contain rows")
    return rows


def _length_labels(lengths: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    edges = tuple(float(value) for value in np.quantile(lengths, (0.25, 0.5, 0.75)))
    labels = np.asarray(
        [f"Q{1 + int(np.searchsorted(edges, value, side='left'))}" for value in lengths],
        dtype=str,
    )
    return labels, edges


def _stratum_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    budgets: Sequence[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    ranked = np.argsort(-scores, kind="mergesort")
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_labels = labels[selected]
        group_scores = scores[selected]
        cell: dict[str, Any] = {
            "n": int(np.sum(selected)),
            "n_attack": int(np.sum(group_labels)),
            "risk_auroc": budget_metrics(group_labels, group_scores, budget=0.02)[
                "risk_auroc"
            ],
            "honest_score_mean": float(np.mean(group_scores[~group_labels])),
            "attack_score_mean": float(np.mean(group_scores[group_labels])),
            "honest_score_median": float(np.median(group_scores[~group_labels])),
            "attack_score_median": float(np.median(group_scores[group_labels])),
            "global_audit_allocation": {},
        }
        for budget in budgets:
            n_audited = max(1, int(math.floor(float(budget) * len(labels) + 1e-12)))
            audited = ranked[:n_audited]
            audited_in_group = int(np.sum(groups[audited] == group))
            attacks_caught = int(
                np.sum(labels[audited] & (groups[audited] == group))
            )
            expected = n_audited * float(np.mean(selected))
            expected_attacks = n_audited * int(np.sum(group_labels)) / max(1, int(np.sum(labels)))
            cell["global_audit_allocation"][str(float(budget))] = {
                "n_audited": audited_in_group,
                "attacks_caught": attacks_caught,
                "expected_if_proportional": expected,
                "allocation_ratio": audited_in_group / expected if expected else float("nan"),
                "expected_if_attack_proportional": expected_attacks,
                "attack_allocation_ratio": (
                    attacks_caught / expected_attacks if expected_attacks else float("nan")
                ),
            }
        output[str(group)] = cell
    return output


def _plot_neutral(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    length_groups: np.ndarray,
    length_edges: Sequence[float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    specifications = (
        (
            axes[0],
            families,
            {
                "competition:control_tax": "Competition\ncontrol-tax",
                "interview:apps": "Interview\nAPPS",
                "interview:control_tax": "Interview\ncontrol-tax",
                "introductory:control_tax": "Introductory\ncontrol-tax",
            },
            "Dataset family",
        ),
        (
            axes[1],
            length_groups,
            {
                "Q1": f"Q1\n$\\leq${length_edges[0]:.0f}",
                "Q2": f"Q2\n{length_edges[0] + 1:.0f}--{length_edges[1]:.0f}",
                "Q3": f"Q3\n{length_edges[1] + 1:.0f}--{length_edges[2]:.0f}",
                "Q4": f"Q4\n$\\geq${math.ceil(length_edges[2]):.0f}",
            },
            "Neutral-context token length",
        ),
    )
    cutoff = float(np.sort(scores)[-max(1, int(math.floor(0.02 * len(scores))))])
    for axis, groups, display, xlabel in specifications:
        names = tuple(display)
        positions = np.arange(len(names), dtype=float)
        honest = [scores[(groups == name) & ~labels] for name in names]
        attack = [scores[(groups == name) & labels] for name in names]
        for values, offsets, color in (
            (honest, positions - 0.18, "#56B4E9"),
            (attack, positions + 0.18, "#D55E00"),
        ):
            box = axis.boxplot(
                values,
                positions=offsets,
                widths=0.30,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": "black", "linewidth": 1.0},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.38)
            for element in ("whiskers", "caps"):
                for artist in box[element]:
                    artist.set_color(color)
        axis.axhline(cutoff, color="0.25", linestyle="--", linewidth=0.9)
        axis.set_xticks(positions, [display[name] for name in names], fontsize=7.5)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Neutral activation risk score")
    axes[1].text(
        0.98,
        cutoff,
        " global 2% cutoff",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.legend(
        handles=(
            Patch(facecolor="#56B4E9", alpha=0.38, label="Honest"),
            Patch(facecolor="#D55E00", alpha=0.38, label="Working backdoor"),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.28, left=0.10, right=0.99, top=0.96, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--followup-cache", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args(argv)

    design = _load_design(args.design)
    with np.load(args.followup_cache, allow_pickle=False) as archive:
        neutral_lengths = archive["neutral_lengths"].copy()
    if len(design) != len(neutral_lengths):
        raise ValueError("design and neutral-length rows differ")
    locked_indices = [index for index, row in enumerate(design) if row["split"] == "locked_test"]
    locked = [design[index] for index in locked_indices]
    sample_ids = tuple(str(row["sample_id"]) for row in locked)
    labels = np.asarray([bool(row["backdoor_works"]) for row in locked], dtype=bool)
    families = np.asarray([str(row["family_id"]) for row in locked], dtype=str)
    lengths = np.asarray([neutral_lengths[index] for index in locked_indices], dtype=float)
    length_groups, length_edges = _length_labels(lengths)

    score_rows = read_ai_control_scores(args.scores)
    by_monitor = {
        monitor: {row.sample_id: float(row.score) for row in score_rows if row.monitor_name == monitor}
        for monitor in DEFAULT_MONITORS
    }
    missing = {
        monitor: sorted(set(sample_ids) - set(values))
        for monitor, values in by_monitor.items()
        if set(sample_ids) != set(values)
    }
    if missing:
        raise ValueError(f"score panels do not match the locked design: {missing}")

    budgets = (0.02, 0.10)
    monitors: dict[str, Any] = {}
    neutral_scores: np.ndarray | None = None
    for monitor, values in by_monitor.items():
        scores = np.asarray([values[sample_id] for sample_id in sample_ids], dtype=float)
        monitors[monitor] = {
            "overall": budget_metrics(labels, scores, budget=0.02),
            "by_family": _stratum_metrics(labels, scores, families, budgets=budgets),
            "by_input_length_quartile": _stratum_metrics(
                labels, scores, length_groups, budgets=budgets
            ),
        }
        if monitor == "qwen-neutral-activation-probe":
            neutral_scores = scores
    assert neutral_scores is not None

    payload = {
        "schema": "observerbench.controlarena_apps_qwen.stratum_diagnostic.v0",
        "status": "post_outcome_frozen_score_diagnostic",
        "inputs": {
            "design_sha256": file_sha256(args.design),
            "followup_cache_sha256": file_sha256(args.followup_cache),
            "scores_sha256": file_sha256(args.scores),
        },
        "n_locked_rows": len(locked),
        "length_measure": "neutral-context tokenizer length",
        "length_quartile_edges": list(length_edges),
        "audit_budgets": list(budgets),
        "monitors": monitors,
    }
    write_json(args.out, payload)
    _plot_neutral(
        scores=neutral_scores,
        labels=labels,
        families=families,
        length_groups=length_groups,
        length_edges=length_edges,
        output=args.figure,
    )
    print(json.dumps(payload["monitors"]["qwen-neutral-activation-probe"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
