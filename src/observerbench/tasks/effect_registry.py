"""Closed registry for versioned, table-backed finite-effect tasks.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from observerbench.effect_prediction import FiniteEffectPredictionTask
from observerbench.tasks.ioi.effect_task import (
    IOI_EFFECT_MEASUREMENT_BUDGETS,
    IOI_EFFECT_TASK_NAME,
    ioi_effect_task_version,
    load_ioi_effect_prediction_task,
)
from observerbench.tasks.qwen_induction.effect_task import (
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_EFFECT_TASK_NAME,
    load_qwen_induction_effect_prediction_task,
    qwen_induction_effect_task_version,
)


@dataclass(frozen=True)
class FiniteEffectTaskSpec:
    """Discoverable identity and budget for one immutable task version."""

    name: str
    version: str
    measurement_budget: int
    summary: str

    @property
    def task_id(self) -> str:
        return f"{self.name}@{self.version}"


_IOI_FINITE_EFFECT_TASK_SPECS: tuple[FiniteEffectTaskSpec, ...] = tuple(
    FiniteEffectTaskSpec(
        name=IOI_EFFECT_TASK_NAME,
        version=ioi_effect_task_version(budget),
        measurement_budget=budget,
        summary=(
            "Checked GPT-2-small IOI finite-effect prediction from cached "
            "template-conditioned mean-ablation tables."
        ),
    )
    for budget in IOI_EFFECT_MEASUREMENT_BUDGETS
)
_QWEN_INDUCTION_FINITE_EFFECT_TASK_SPECS: tuple[FiniteEffectTaskSpec, ...] = tuple(
    FiniteEffectTaskSpec(
        name=QWEN_INDUCTION_EFFECT_TASK_NAME,
        version=qwen_induction_effect_task_version(budget),
        measurement_budget=budget,
        summary=(
            "Checked Qwen2.5-7B induction-copy finite-effect prediction from "
            "cached exact-label mean-ablation tables."
        ),
    )
    for budget in QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
)
_FINITE_EFFECT_TASK_SPECS = (
    *_IOI_FINITE_EFFECT_TASK_SPECS,
    *_QWEN_INDUCTION_FINITE_EFFECT_TASK_SPECS,
)
_FINITE_EFFECT_TASK_BY_ID = {
    spec.task_id: spec for spec in _FINITE_EFFECT_TASK_SPECS
}


def finite_effect_task_specs() -> tuple[FiniteEffectTaskSpec, ...]:
    """Return all built-in versions in deterministic task-ID order."""

    return tuple(sorted(_FINITE_EFFECT_TASK_SPECS, key=lambda spec: spec.task_id))


def finite_effect_task_ids() -> tuple[str, ...]:
    """Return canonical ``name@version`` identifiers in deterministic order."""

    return tuple(spec.task_id for spec in finite_effect_task_specs())


def finite_effect_task_versions(task_name: str) -> tuple[str, ...]:
    """Return the registered versions of one finite-effect task."""

    versions = tuple(
        spec.version
        for spec in finite_effect_task_specs()
        if spec.name == task_name
    )
    if not versions:
        known = ", ".join(sorted({spec.name for spec in _FINITE_EFFECT_TASK_SPECS}))
        raise KeyError(f"unknown finite-effect task {task_name!r}; choose one of: {known}")
    return versions


def finite_effect_measurement_budgets(task_name: str) -> tuple[int, ...]:
    """Return the registered measurement budgets of one task."""

    finite_effect_task_versions(task_name)
    return tuple(
        spec.measurement_budget
        for spec in finite_effect_task_specs()
        if spec.name == task_name
    )


def get_finite_effect_task_spec(task_id: str) -> FiniteEffectTaskSpec:
    """Look up one exact task version; implicit mutable defaults are not used."""

    try:
        return _FINITE_EFFECT_TASK_BY_ID[task_id]
    except KeyError as error:
        known = ", ".join(finite_effect_task_ids())
        raise KeyError(
            f"unknown finite-effect task ID {task_id!r}; choose one of: {known}"
        ) from error


def load_finite_effect_task(
    task_id: str,
    *,
    artifacts_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> FiniteEffectPredictionTask:
    """Load a registered task from frozen tables without running model inference."""

    spec = get_finite_effect_task_spec(task_id)
    if spec.name == IOI_EFFECT_TASK_NAME:
        task = load_ioi_effect_prediction_task(
            artifacts_root,
            measurement_budget=spec.measurement_budget,
            verify_hashes=verify_hashes,
        )
    elif spec.name == QWEN_INDUCTION_EFFECT_TASK_NAME:
        task = load_qwen_induction_effect_prediction_task(
            artifacts_root,
            measurement_budget=spec.measurement_budget,
            verify_hashes=verify_hashes,
        )
    else:  # pragma: no cover - every closed-registry spec has a loader
        raise KeyError(f"no loader is available for finite-effect task {task_id!r}")
    if task.name != spec.name or task.version != spec.version:
        raise ValueError("loaded finite-effect task identity differs from the registry")
    return task


__all__ = [
    "FiniteEffectTaskSpec",
    "finite_effect_measurement_budgets",
    "finite_effect_task_ids",
    "finite_effect_task_specs",
    "finite_effect_task_versions",
    "get_finite_effect_task_spec",
    "load_finite_effect_task",
]
