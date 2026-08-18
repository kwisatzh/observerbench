"""Portable, inference-free safety task packs composed from the core contract.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from observerbench.core import write_json
from observerbench.provenance import json_sha256
from observerbench.safety import (
    SafetyMeasurement,
    SafetyPolicy,
    SafetyQuery,
    SafetyTarget,
    SafetyTask,
    SafetyTaskCard,
)


SAFETY_TASK_PACK_SCHEMA_VERSION = "observerbench.safety_task_pack.v0"
SAFETY_TASK_TARGETS_SCHEMA_VERSION = "observerbench.safety_task_targets.v0"


def _read_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _records(payload: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    rows = payload.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{field} must be a nonempty JSON list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"every {field} entry must be a JSON object")
    return rows


def _task_card(payload: Mapping[str, Any]) -> SafetyTaskCard:
    card_payload = payload.get("task_card")
    if not isinstance(card_payload, dict):
        raise ValueError("task_card must be a JSON object")
    return SafetyTaskCard(**card_payload)


def _policy(payload: Mapping[str, Any]) -> SafetyPolicy:
    policy_payload = payload.get("policy")
    if not isinstance(policy_payload, dict):
        raise ValueError("policy must be a JSON object")
    return SafetyPolicy(**policy_payload)


def _public_components(
    payload: Mapping[str, Any],
) -> tuple[
    str,
    str,
    tuple[SafetyMeasurement[Any], ...],
    tuple[SafetyQuery[Any], ...],
    SafetyPolicy,
    SafetyTaskCard,
]:
    if payload.get("schema_version") != SAFETY_TASK_PACK_SCHEMA_VERSION:
        raise ValueError("unsupported public safety task-pack schema")
    if "targets" in payload:
        raise ValueError("held-out targets must not appear in the public task pack")
    name = payload.get("task_name")
    version = payload.get("task_version")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("task_name must be a nonempty string")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("task_version must be a nonempty string")
    measurements = tuple(SafetyMeasurement(**row) for row in _records(payload, "measurements"))
    queries = tuple(SafetyQuery(**row) for row in _records(payload, "queries"))
    measurement_ids = [row.measurement_id for row in measurements]
    query_ids = [row.query_id for row in queries]
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("measurement IDs must be unique")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    policy = _policy(payload)
    card = _task_card(payload)
    if card.task_name != name or card.task_version != version:
        raise ValueError("task-pack identity differs from its task card")
    return name, version, measurements, queries, policy, card


def validate_public_safety_task_pack(path: str | Path) -> dict[str, Any]:
    """Validate the shareable fit/query pack without opening evaluator targets."""

    payload = _read_object(path)
    name, version, measurements, queries, policy, card = _public_components(payload)
    return {
        "schema_version": SAFETY_TASK_PACK_SCHEMA_VERSION,
        "task_id": f"{name}@{version}",
        "public_task_sha256": json_sha256(payload),
        "n_measurements": len(measurements),
        "n_queries": len(queries),
        "families": sorted({query.family_id for query in queries}),
        "block_budget_fraction": policy.block_budget_fraction,
        "escalation_budget_fraction": policy.escalation_budget_fraction,
        "threat_model": card.threat_model,
    }


def load_safety_task_pack(
    public_task_path: str | Path,
    targets_path: str | Path,
) -> SafetyTask[Any]:
    """Compose a core ``SafetyTask`` from public and evaluator-only JSON files."""

    public_payload = _read_object(public_task_path)
    name, version, measurements, queries, policy, card = _public_components(public_payload)
    target_payload = _read_object(targets_path)
    if target_payload.get("schema_version") != SAFETY_TASK_TARGETS_SCHEMA_VERSION:
        raise ValueError("unsupported evaluator-target schema")
    if target_payload.get("task_name") != name or target_payload.get("task_version") != version:
        raise ValueError("evaluator targets refer to a different safety task")
    expected_hash = target_payload.get("public_task_sha256")
    actual_hash = json_sha256(public_payload)
    if expected_hash != actual_hash:
        raise ValueError("public safety task does not match the evaluator-target seal")
    targets = tuple(SafetyTarget(**row) for row in _records(target_payload, "targets"))
    return SafetyTask(
        name=name,
        version=version,
        measurements=measurements,
        queries=queries,
        targets=targets,
        policy=policy,
        card=card,
    )


def write_safety_task_pack(
    task: SafetyTask[Any],
    outdir: str | Path,
) -> tuple[Path, Path]:
    """Write a shareable task and separately sealed evaluator-target file."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    public_path = output / "safety_task.json"
    public_payload = {
        "schema_version": SAFETY_TASK_PACK_SCHEMA_VERSION,
        "task_name": task.name,
        "task_version": task.version,
        "task_card": asdict(task.card),
        "policy": asdict(task.policy),
        "measurements": [asdict(row) for row in task.measurements],
        "queries": [asdict(row) for row in task.queries],
    }
    write_json(public_path, public_payload)
    canonical_public = _read_object(public_path)
    targets_path = output / "evaluator_targets.json"
    write_json(
        targets_path,
        {
            "schema_version": SAFETY_TASK_TARGETS_SCHEMA_VERSION,
            "task_name": task.name,
            "task_version": task.version,
            "public_task_sha256": json_sha256(canonical_public),
            "targets": [asdict(row) for row in task.targets],
        },
    )
    return public_path, targets_path


__all__ = [
    "SAFETY_TASK_PACK_SCHEMA_VERSION",
    "SAFETY_TASK_TARGETS_SCHEMA_VERSION",
    "load_safety_task_pack",
    "validate_public_safety_task_pack",
    "write_safety_task_pack",
]
