"""Internal registry for paper reproduction tasks.

This registry is intentionally small. It names the current paper tasks so the
CLI can be installed and smoke-tested before experiment code is migrated.
It is not a plugin API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    summary: str
    config: str
    status: str = "stub"


TASKS: dict[str, TaskSpec] = {
    "ctl1_analytic": TaskSpec(
        name="ctl1_analytic",
        summary="Analytic Ctl-1 collateral geometry reproduction task.",
        config="configs/ctl1_analytic.yaml",
    ),
    "trained_ctl1": TaskSpec(
        name="trained_ctl1",
        summary="Trained-transformer Ctl-1 one-shot observer geometry task.",
        config="configs/trained_ctl1.yaml",
    ),
    "trained_ctl2": TaskSpec(
        name="trained_ctl2",
        summary="Trained-transformer Ctl-2 closed-loop observer-control task.",
        config="configs/trained_ctl2.yaml",
    ),
    "ioi_stage1": TaskSpec(
        name="ioi_stage1",
        summary="IOI Stage 1 whole-group self-repair diagnostic.",
        config="configs/ioi_stage1.yaml",
    ),
    "ioi_stage2b": TaskSpec(
        name="ioi_stage2b",
        summary="IOI Stage 2b random head-subset prediction diagnostic.",
        config="configs/ioi_stage2b.yaml",
    ),
    "ioi_stage2c": TaskSpec(
        name="ioi_stage2c",
        summary="IOI Stage 2c primary-stratified head-subset diagnostic.",
        config="configs/ioi_stage2c.yaml",
    ),
    "ioi_stage2d": TaskSpec(
        name="ioi_stage2d",
        summary="IOI Stage 2d per-pair decomposition diagnostic.",
        config="configs/ioi_stage2d.yaml",
    ),
}


def run_stub_task(
    task_name: str,
    config: dict[str, Any],
    config_path: Path,
    outdir: Path,
) -> Path:
    """Write run metadata for a registered task without scientific execution."""

    spec = TASKS[task_name]
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
