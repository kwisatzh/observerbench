"""Paper-task registry for the ObserverBench reproduction CLI."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from observerbench.tasks.ctl1_adapter import (
    evaluate_ctl1_analytic_observer,
    make_ctl1_analytic_controller,
    make_ctl1_analytic_task,
)
from observerbench.tasks.effect_registry import (
    FiniteEffectTaskSpec,
    finite_effect_measurement_budgets,
    finite_effect_task_ids,
    finite_effect_task_specs,
    finite_effect_task_versions,
    get_finite_effect_task_spec,
    load_finite_effect_task,
)
from observerbench.tasks.registry import (
    TASKS,
    TaskSpec,
    get_task_spec,
    make_observer_task,
    observer_task_names,
    task_names,
    task_specs,
)
from observerbench.tasks.safety_registry import (
    SafetyTaskSpec,
    load_safety_task,
    safety_task_ids,
    safety_task_specs,
)

__all__ = [
    "TASKS",
    "FiniteEffectTaskSpec",
    "SafetyTaskSpec",
    "TaskSpec",
    "evaluate_ctl1_analytic_observer",
    "finite_effect_measurement_budgets",
    "finite_effect_task_ids",
    "finite_effect_task_specs",
    "finite_effect_task_versions",
    "get_finite_effect_task_spec",
    "get_task_spec",
    "load_finite_effect_task",
    "load_safety_task",
    "make_observer_task",
    "make_ctl1_analytic_controller",
    "make_ctl1_analytic_task",
    "observer_task_names",
    "task_names",
    "task_specs",
    "safety_task_ids",
    "safety_task_specs",
]
