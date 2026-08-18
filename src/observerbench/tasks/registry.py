"""Internal registry for paper reproduction tasks.

This registry is intentionally small. It names the current paper tasks and
dispatches migrated Ctl implementations. It is not a plugin API.
"""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

from observerbench.core import API_CONTRACT_VERSION, ObserverTask


@dataclass(frozen=True)
class TaskSpec:
    name: str
    summary: str
    config: str
    status: str = "stub"
    supports_external_observer: bool = False
    external_observer_contract: str | None = None
    external_observer_factory: str | None = None


TASKS: dict[str, TaskSpec] = {
    "ctl1_analytic": TaskSpec(
        name="ctl1_analytic",
        summary="Analytic Ctl-1 collateral geometry reproduction task.",
        config="configs/ctl1_analytic.yaml",
        status="migrated",
        supports_external_observer=True,
        external_observer_contract=API_CONTRACT_VERSION,
        external_observer_factory="observerbench.tasks.make_observer_task",
    ),
    "trained_ctl1": TaskSpec(
        name="trained_ctl1",
        summary="Trained-transformer Ctl-1 one-shot observer geometry task.",
        config="configs/trained_ctl1.yaml",
        status="migrated",
    ),
    "trained_ctl2": TaskSpec(
        name="trained_ctl2",
        summary="Trained-transformer Ctl-2 closed-loop observer-control task.",
        config="configs/trained_ctl2.yaml",
        status="migrated",
    ),
    "safety_interlock_analytic": TaskSpec(
        name="safety_interlock_analytic",
        summary="Inert paired-scope authorization interlock with exact policy labels.",
        config="configs/safety_interlock_analytic.yaml",
        status="migrated",
    ),
    "ioi_stage1": TaskSpec(
        name="ioi_stage1",
        summary="IOI Stage 1 whole-group self-repair diagnostic.",
        config="configs/ioi_stage1.yaml",
        status="wired",
    ),
    "ioi_stage2b": TaskSpec(
        name="ioi_stage2b",
        summary="IOI Stage 2b random head-subset prediction diagnostic.",
        config="configs/ioi_stage2b.yaml",
        status="wired",
    ),
    "ioi_stage2c": TaskSpec(
        name="ioi_stage2c",
        summary="IOI Stage 2c primary-stratified head-subset diagnostic.",
        config="configs/ioi_stage2c.yaml",
        status="wired",
    ),
    "ioi_stage2d": TaskSpec(
        name="ioi_stage2d",
        summary="IOI Stage 2d per-pair decomposition diagnostic.",
        config="configs/ioi_stage2d.yaml",
        status="wired",
    ),
}


def _dataclass_kwargs(config: dict[str, Any], cls: type) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in config.items() if key in names}


def _apply_quick_ctl1(cfg: Any) -> None:
    cfg.n_train = min(cfg.n_train, 800)
    cfg.n_test = min(cfg.n_test, 400)
    cfg.train_steps = min(cfg.train_steps, 80)
    cfg.d_model = min(cfg.d_model, 16)
    cfg.n_layers = min(cfg.n_layers, 1)
    cfg.n_heads = min(cfg.n_heads, 2)
    cfg.d_mlp = min(cfg.d_mlp, 48)
    cfg.batch_size = min(cfg.batch_size, 128)


def _apply_quick_ctl2(cfg: Any) -> None:
    _apply_quick_ctl1(cfg)
    cfg.loop_steps = min(cfg.loop_steps, 8)
    cfg.n_calibration = min(cfg.n_calibration, 128)


def _run_ctl1_analytic(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.ctl1_analytic import CollateralTaskConfig, run_task

    cfg = CollateralTaskConfig(**_dataclass_kwargs(config, CollateralTaskConfig))
    return run_task(cfg, outdir)


def _run_trained_ctl1(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.trained_ctl1 import TrainedTransformerCtl1Config, run_trained_transformer_ctl1

    cfg = TrainedTransformerCtl1Config(**_dataclass_kwargs(config, TrainedTransformerCtl1Config))
    if config.get("quick", False):
        _apply_quick_ctl1(cfg)
    return run_trained_transformer_ctl1(cfg, outdir)


def _run_trained_ctl2(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.trained_ctl2 import TrainedTransformerCtl2Config, run_trained_transformer_ctl2

    cfg = TrainedTransformerCtl2Config(**_dataclass_kwargs(config, TrainedTransformerCtl2Config))
    if config.get("quick", False):
        _apply_quick_ctl2(cfg)
    return run_trained_transformer_ctl2(cfg, outdir)


def _run_safety_interlock(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.safety_interlock import (
        SafetyInterlockConfig,
        run_safety_interlock,
    )

    cfg = SafetyInterlockConfig(**_dataclass_kwargs(config, SafetyInterlockConfig))
    if config.get("quick", False):
        cfg = SafetyInterlockConfig(
            **{
                **asdict(cfg),
                "n_train_pairs": min(cfg.n_train_pairs, 64),
                "n_test_pairs": min(cfg.n_test_pairs, 128),
            }
        )
    return run_safety_interlock(cfg, outdir)


def _run_ioi_stage1(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.ioi.stage1 import IOIStage1Config, run_ioi_stage1

    cfg = IOIStage1Config(**_dataclass_kwargs(config, IOIStage1Config))
    return run_ioi_stage1(cfg, outdir)


def _run_ioi_stage2b(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.ioi.stage2b import IOIStage2bConfig, run_ioi_stage2b

    cfg = IOIStage2bConfig(**_dataclass_kwargs(config, IOIStage2bConfig))
    return run_ioi_stage2b(cfg, outdir)


def _run_ioi_stage2c(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.ioi.stage2c import IOIStage2cConfig, run_ioi_stage2c

    cfg = IOIStage2cConfig(**_dataclass_kwargs(config, IOIStage2cConfig))
    return run_ioi_stage2c(cfg, outdir)


def _run_ioi_stage2d(config: dict[str, Any], outdir: Path) -> Any:
    from observerbench.tasks.ioi.stage2d import IOIStage2dConfig, run_ioi_stage2d

    cfg = IOIStage2dConfig(**_dataclass_kwargs(config, IOIStage2dConfig))
    return run_ioi_stage2d(cfg, outdir)


RUNNERS: dict[str, Callable[[dict[str, Any], Path], Any]] = {
    "ctl1_analytic": _run_ctl1_analytic,
    "trained_ctl1": _run_trained_ctl1,
    "trained_ctl2": _run_trained_ctl2,
    "safety_interlock_analytic": _run_safety_interlock,
    "ioi_stage1": _run_ioi_stage1,
    "ioi_stage2b": _run_ioi_stage2b,
    "ioi_stage2c": _run_ioi_stage2c,
    "ioi_stage2d": _run_ioi_stage2d,
}


def task_names() -> tuple[str, ...]:
    """Return built-in paper-task names in deterministic order."""

    return tuple(sorted(TASKS))


def get_task_spec(task_name: str) -> TaskSpec:
    """Return one built-in paper-task specification."""

    try:
        return TASKS[task_name]
    except KeyError as exc:
        known = ", ".join(task_names())
        raise KeyError(f"Unknown ObserverBench task {task_name!r}; choose one of: {known}") from exc


def task_specs() -> tuple[TaskSpec, ...]:
    """Return built-in paper-task specifications in deterministic order."""

    return tuple(get_task_spec(name) for name in task_names())


def observer_task_names() -> tuple[str, ...]:
    """Return tasks with a public external-observer composition adapter."""

    return tuple(name for name in task_names() if TASKS[name].supports_external_observer)


def make_observer_task(task_name: str) -> ObserverTask:
    """Construct one discoverable external-observer task without registry mutation."""

    if task_name == "ctl1_analytic":
        from observerbench.tasks.ctl1_adapter import make_ctl1_analytic_task

        return make_ctl1_analytic_task()
    supported = ", ".join(observer_task_names()) or "none"
    raise KeyError(
        f"Task {task_name!r} has no external-observer adapter; supported tasks: {supported}"
    )


def _write_stub_task(task_name: str, config: dict[str, Any], config_path: Path, outdir: Path) -> Path:
    """Write run metadata for a registered task without scientific execution."""

    spec = get_task_spec(task_name)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "observerbench.task_run.stub.v0",
        "task": asdict(spec),
        "config_path": str(config_path),
        "config": config,
        "status": "stub",
        "note": (
            "Experiment implementation has not been migrated yet. This file "
            "only verifies the reproduction CLI wiring."
        ),
    }
    metadata_path = outdir / "run_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def run_registered_task(
    task_name: str,
    config: dict[str, Any],
    config_path: Path,
    outdir: Path,
) -> Any:
    """Run a paper task, using a stub only for tasks not yet migrated."""

    get_task_spec(task_name)
    runner = RUNNERS.get(task_name)
    if runner is None:
        return _write_stub_task(task_name, config, config_path, outdir)
    return runner(config, outdir)
