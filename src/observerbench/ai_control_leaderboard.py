"""Bundled checked rows for the external AI-control monitor study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from importlib.resources import files
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def bundled_ai_control_leaderboard_path() -> Path:
    return Path(str(files("observerbench").joinpath("data/ai_control_leaderboard_v0.json")))


def load_ai_control_leaderboard(
    path: str | Path | None = None,
    *,
    task_id: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    source = Path(path) if path is not None else bundled_ai_control_leaderboard_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "observerbench.ai_control_leaderboard.v0":
        raise ValueError("unsupported AI-control leaderboard schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("AI-control leaderboard must contain rows")
    identities = {
        (str(row.get("task_id", "")), str(row.get("observer_name", "")))
        for row in rows
    }
    if len(identities) != len(rows) or any(not all(identity) for identity in identities):
        raise ValueError("AI-control leaderboard contains duplicate or unnamed rows")
    selected = tuple(row for row in rows if task_id is None or row["task_id"] == task_id)
    if task_id is not None and not selected:
        known = ", ".join(sorted({str(row["task_id"]) for row in rows}))
        raise KeyError(f"unknown AI-control task {task_id!r}; choose one of: {known}")
    return selected


def _task_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    return tuple((task_id, tuple(group)) for task_id, group in grouped.items())


def format_ai_control_leaderboard(
    rows: Sequence[Mapping[str, Any]], *, output_format: str = "text"
) -> str:
    if not rows:
        raise ValueError("at least one AI-control row is required")
    groups = _task_groups(rows)
    if len(groups) > 1 and output_format in {"text", "markdown"}:
        prefix = "### " if output_format == "markdown" else "Task: "
        return "\n\n".join(
            prefix
            + task_id
            + "\n"
            + format_ai_control_leaderboard(group, output_format=output_format).rstrip()
            for task_id, group in groups
        ) + "\n"
    flat = [
        {
            "task_id": row["task_id"],
            "monitor": row["display_name"],
            "status": row["result_status"],
            "access": row["access_regime"],
            "auroc": row["metrics"]["risk_auroc"],
            "mean_violations_p01_b02": row["metrics"].get(
                "mean_realized_violations_at_p01_b02"
            ),
            "violations_at_2pct": row["metrics"]["violations_at_0.02"],
            "audit_precision_at_2pct": row["metrics"]["audit_precision_at_0.02"],
            "violations_at_10pct": row["metrics"]["violations_at_0.1"],
            "audit_precision_at_10pct": row["metrics"]["audit_precision_at_0.1"],
        }
        for row in rows
    ]
    if output_format == "json":
        return json.dumps(flat, indent=2, sort_keys=True) + "\n"
    if output_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=tuple(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
        return output.getvalue()
    if output_format == "markdown":
        lines = [
            "| Monitor | Status | Access | AUROC | Mean violations (1% attacks, 2% audit) | Violations @ 2% | Audit precision @ 2% | Violations @ 10% |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in flat:
            lines.append(
                f"| {row['monitor']} | {row['status']} | {row['access']} | {row['auroc']:.3f} | "
                f"{row['mean_violations_p01_b02']:.3f} | "
                f"{int(row['violations_at_2pct'])} | "
                f"{row['audit_precision_at_2pct']:.3f} | "
                f"{int(row['violations_at_10pct'])} |"
            )
        return "\n".join(lines) + "\n"
    if output_format != "text":
        raise ValueError("output_format must be text, json, csv, or markdown")
    header = (
        f"{'Monitor':42} {'Status':28} {'AUROC':>7} {'MeanViol@1%/2%':>15} {'Viol@2%':>9} {'Prec@2%':>8} "
        f"{'Viol@10%':>10}  Access"
    )
    lines = [header, "-" * len(header)]
    for row in flat:
        lines.append(
            f"{row['monitor'][:42]:42} {row['status'][:28]:28} "
            f"{row['auroc']:7.3f} "
            f"{row['mean_violations_p01_b02']:15.3f} "
            f"{int(row['violations_at_2pct']):9d} "
            f"{row['audit_precision_at_2pct']:8.3f} "
            f"{int(row['violations_at_10pct']):10d}  {row['access']}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "bundled_ai_control_leaderboard_path",
    "format_ai_control_leaderboard",
    "load_ai_control_leaderboard",
]
