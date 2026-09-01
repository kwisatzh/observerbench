"""Checked safety-result registry and comparison rendering.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
from importlib.resources import files
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from observerbench.provenance import file_sha256
from observerbench.safety import SafetyObserverCard


SAFETY_LEADERBOARD_SCHEMA_VERSION = "observerbench.safety_leaderboard.v0"
SAFETY_LEADERBOARD_METRICS = (
    "protocol_loss_mean",
    "protocol_loss_cvar",
    "risk_auroc",
    "severity_weighted_miss_rate",
    "clean_utility_retained",
    "worst_family_protocol_loss",
)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class SafetyLeaderboardRow:
    task_name: str
    task_version: str
    observer_name: str
    display_name: str
    observer_family: str
    result_status: str
    access_regime: str
    requires_white_box_access: bool | None
    additional_forward_passes: int | None
    metrics: Mapping[str, float]
    action_counts: Mapping[str, int]
    source_result: str
    source_sha256: str
    known_failure_modes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "task_name",
            "task_version",
            "observer_name",
            "display_name",
            "observer_family",
            "result_status",
            "access_regime",
            "source_result",
            "source_sha256",
        ):
            _nonempty(getattr(self, name), name)
        if self.requires_white_box_access is not None and not isinstance(
            self.requires_white_box_access, bool
        ):
            raise ValueError("requires_white_box_access must be boolean or null")
        if self.additional_forward_passes is not None:
            value = self.additional_forward_passes
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("additional_forward_passes must be nonnegative or null")
        missing = set(SAFETY_LEADERBOARD_METRICS) - set(self.metrics)
        if missing:
            raise ValueError(f"leaderboard metrics are missing: {sorted(missing)}")
        for name in SAFETY_LEADERBOARD_METRICS:
            value = float(self.metrics[name])
            if not math.isfinite(value):
                raise ValueError(f"leaderboard metric {name} must be finite")
        if set(self.action_counts) != {"allow", "block", "escalate"}:
            raise ValueError("action_counts must contain allow, block, and escalate")

    @property
    def task_id(self) -> str:
        return f"{self.task_name}@{self.task_version}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SafetyLeaderboardRow":
        values = dict(payload)
        values["known_failure_modes"] = tuple(values.get("known_failure_modes", ()))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bundled_safety_leaderboard_path() -> Path:
    return Path(str(files("observerbench").joinpath("data/safety_leaderboard_v0.json")))


def load_safety_leaderboard(
    path: str | Path | None = None,
) -> tuple[SafetyLeaderboardRow, ...]:
    source = Path(path) if path is not None else bundled_safety_leaderboard_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SAFETY_LEADERBOARD_SCHEMA_VERSION:
        raise ValueError("unsupported safety leaderboard schema")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("safety leaderboard must contain rows")
    rows = tuple(SafetyLeaderboardRow.from_dict(row) for row in raw_rows)
    identities = {(row.task_id, row.observer_name) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("safety leaderboard contains duplicate observer rows")
    return rows


def safety_rows_for_task(
    rows: Sequence[SafetyLeaderboardRow],
    task_id: str | None,
) -> tuple[SafetyLeaderboardRow, ...]:
    selected = tuple(row for row in rows if task_id is None or row.task_id == task_id)
    if task_id is not None and not selected:
        known = ", ".join(sorted({row.task_id for row in rows}))
        raise KeyError(f"unknown safety leaderboard task {task_id!r}; choose one of: {known}")
    return tuple(sorted(selected, key=lambda row: (row.task_id, row.metrics["protocol_loss_mean"])))


def safety_row_from_result_directory(path: str | Path) -> SafetyLeaderboardRow:
    result_dir = Path(path)
    result_path = result_dir / "safety_protocol_result.json"
    card_path = result_dir / "safety_observer_card.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    card_payload = json.loads(card_path.read_text(encoding="utf-8"))
    card_payload.pop("schema_version", None)
    card = SafetyObserverCard(**card_payload)
    return SafetyLeaderboardRow(
        task_name=str(result["task_name"]),
        task_version=str(result["task_version"]),
        observer_name=card.observer_name,
        display_name=card.observer_name,
        observer_family=card.observer_family,
        result_status="outside submission",
        access_regime=card.access_regime,
        requires_white_box_access=card.requires_white_box_access,
        additional_forward_passes=card.additional_forward_passes,
        metrics={name: float(result["metrics"][name]) for name in SAFETY_LEADERBOARD_METRICS},
        action_counts={name: int(result["action_counts"][name]) for name in ("allow", "block", "escalate")},
        source_result=result_path.name,
        source_sha256=file_sha256(result_path),
        known_failure_modes=tuple(card.known_failure_modes),
        metadata={"implementation": card.implementation},
    )


def compare_safety_result(
    result_dir: str | Path,
    rows: Sequence[SafetyLeaderboardRow],
) -> tuple[SafetyLeaderboardRow, ...]:
    submitted = safety_row_from_result_directory(result_dir)
    baselines = [row for row in rows if row.task_id == submitted.task_id]
    baselines = [row for row in baselines if row.observer_name != submitted.observer_name]
    return tuple(
        sorted((*baselines, submitted), key=lambda row: row.metrics["protocol_loss_mean"])
    )


def _flat_row(row: SafetyLeaderboardRow) -> dict[str, Any]:
    return {
        "task_id": row.task_id,
        "observer": row.display_name,
        "status": row.result_status,
        "access": row.access_regime,
        "white_box": row.requires_white_box_access,
        "extra_passes": row.additional_forward_passes,
        **{name: row.metrics[name] for name in SAFETY_LEADERBOARD_METRICS},
    }


def format_safety_leaderboard(
    rows: Sequence[SafetyLeaderboardRow],
    *,
    output_format: str = "text",
) -> str:
    flat = [_flat_row(row) for row in rows]
    if output_format == "json":
        return json.dumps(flat, indent=2, sort_keys=True) + "\n"
    if output_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=tuple(flat[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat)
        return output.getvalue()
    if output_format == "markdown":
        lines = [
            "| Observer | Status | AUROC | Mean loss | Tail loss | Missed risk | Clean utility | Access |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            metrics = row.metrics
            lines.append(
                f"| {row.display_name} | {row.result_status} | "
                f"{metrics['risk_auroc']:.3f} | {metrics['protocol_loss_mean']:.3f} | "
                f"{metrics['protocol_loss_cvar']:.3f} | "
                f"{metrics['severity_weighted_miss_rate']:.3f} | "
                f"{metrics['clean_utility_retained']:.3f} | {row.access_regime} |"
            )
        return "\n".join(lines) + "\n"
    if output_format != "text":
        raise ValueError("output_format must be text, json, csv, or markdown")

    header = (
        f"{'Observer':38} {'Status':24} {'AUROC':>7} {'Mean':>7} "
        f"{'Tail':>7} {'Miss':>7} {'Clean':>7}  Access"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        metrics = row.metrics
        lines.append(
            f"{row.display_name[:38]:38} {row.result_status[:24]:24} "
            f"{metrics['risk_auroc']:7.3f} {metrics['protocol_loss_mean']:7.3f} "
            f"{metrics['protocol_loss_cvar']:7.3f} "
            f"{metrics['severity_weighted_miss_rate']:7.3f} "
            f"{metrics['clean_utility_retained']:7.3f}  {row.access_regime}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "SAFETY_LEADERBOARD_METRICS",
    "SAFETY_LEADERBOARD_SCHEMA_VERSION",
    "SafetyLeaderboardRow",
    "bundled_safety_leaderboard_path",
    "compare_safety_result",
    "format_safety_leaderboard",
    "load_safety_leaderboard",
    "safety_row_from_result_directory",
    "safety_rows_for_task",
]
