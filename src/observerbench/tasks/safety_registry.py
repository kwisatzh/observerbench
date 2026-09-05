"""Closed registry for versioned, inference-free safety tasks.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from observerbench.safety import SafetyTask
from observerbench.tasks.evidence_integrity import EvidenceIntegrityConfig, evidence_task_version, make_evidence_integrity_task
from observerbench.tasks.safety_interlock import (
    SAFETY_INTERLOCK_TASK_NAME,
    SAFETY_INTERLOCK_TASK_VERSION,
    SafetyInterlockConfig,
    make_safety_interlock_task,
)


@dataclass(frozen=True)
class SafetyTaskSpec:
    name: str
    version: str
    summary: str

    @property
    def task_id(self) -> str:
        return f"{self.name}@{self.version}"


_SAFETY_TASK_SPECS = (
    SafetyTaskSpec(
        name=SAFETY_INTERLOCK_TASK_NAME,
        version=SAFETY_INTERLOCK_TASK_VERSION,
        summary="Inert paired-scope authorization interlock with exact policy labels.",
    ),
    SafetyTaskSpec(
        name="evidence_integrity_pending_operation",
        version=evidence_task_version(EvidenceIntegrityConfig(measurement_fraction=0.0)),
        summary="Open report-only pending-operation task; buy observations through the evidence-integrity example.",
    ),
    SafetyTaskSpec(
        name="evidence_integrity_cross_agent_origin",
        version=evidence_task_version(EvidenceIntegrityConfig(variant="cross-agent-origin", measurement_fraction=0.0)),
        summary="Open report-only cross-agent origin task; scripted linked events, no autonomous execution.",
    ),
)
_SAFETY_TASK_BY_ID = {spec.task_id: spec for spec in _SAFETY_TASK_SPECS}


def safety_task_specs() -> tuple[SafetyTaskSpec, ...]:
    return tuple(sorted(_SAFETY_TASK_SPECS, key=lambda item: item.task_id))


def safety_task_ids() -> tuple[str, ...]:
    return tuple(spec.task_id for spec in safety_task_specs())


def load_safety_task(task_id: str) -> SafetyTask:
    try:
        spec = _SAFETY_TASK_BY_ID[task_id]
    except KeyError as error:
        known = ", ".join(safety_task_ids())
        raise KeyError(f"unknown safety task ID {task_id!r}; choose one of: {known}") from error
    if spec.name == SAFETY_INTERLOCK_TASK_NAME:
        task = make_safety_interlock_task(SafetyInterlockConfig())
    elif spec.name.startswith("evidence_integrity_"):
        variant = spec.name.removeprefix("evidence_integrity_").replace("_", "-")
        task, _ = make_evidence_integrity_task(EvidenceIntegrityConfig(variant=variant, measurement_fraction=0.0))
    else:  # pragma: no cover
        raise KeyError(f"no loader is available for safety task {task_id!r}")
    if task.name != spec.name or task.version != spec.version:
        raise ValueError("loaded safety task identity differs from the registry")
    return task


__all__ = ["SafetyTaskSpec", "load_safety_task", "safety_task_ids", "safety_task_specs"]
