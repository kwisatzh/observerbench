#!/usr/bin/env python3
"""Run full ObserverBench paper reproduction commands where feasible.

Expensive tasks are opt-in. By default this script refuses any selected command
that trains a model or may require TransformerLens/GPT-2. Passing
--yes-run-expensive acknowledges those costs.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "paper_full"


@dataclass(frozen=True)
class FullTask:
    key: str
    command: tuple[str, ...]
    trains: bool
    uses_transformerlens: bool
    uses_gpt2: bool
    requires_gpu_mps: bool
    feasible: bool
    note: str

    @property
    def expensive(self) -> bool:
        return self.trains or self.uses_transformerlens or self.uses_gpt2 or self.requires_gpu_mps


def _python() -> str:
    return sys.executable


def task_plan(outdir: Path) -> dict[str, FullTask]:
    return {
        "ctl1_analytic": FullTask(
            key="ctl1_analytic",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "ctl1_analytic",
                "--config",
                "configs/ctl1_analytic.yaml",
                "--outdir",
                str(outdir / "ctl1_analytic"),
            ),
            trains=False,
            uses_transformerlens=False,
            uses_gpt2=False,
            requires_gpu_mps=False,
            feasible=True,
            note="CPU analytic reproduction; no model training.",
        ),
        "trained_ctl1": FullTask(
            key="trained_ctl1",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "trained_ctl1",
                "--config",
                "configs/trained_ctl1.yaml",
                "--outdir",
                str(outdir / "trained_ctl1"),
            ),
            trains=True,
            uses_transformerlens=False,
            uses_gpt2=False,
            requires_gpu_mps=False,
            feasible=True,
            note="Trains the small Ctl-1 transformer; GPU/MPS recommended for paper-scale configs.",
        ),
        "trained_ctl2": FullTask(
            key="trained_ctl2",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "trained_ctl2",
                "--config",
                "configs/trained_ctl2.yaml",
                "--outdir",
                str(outdir / "trained_ctl2"),
            ),
            trains=True,
            uses_transformerlens=False,
            uses_gpt2=False,
            requires_gpu_mps=False,
            feasible=True,
            note="Trains the small Ctl-2 transformer and runs closed-loop rollouts.",
        ),
        "ioi_stage1": FullTask(
            key="ioi_stage1",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "ioi_stage1",
                "--config",
                "configs/ioi_stage1.yaml",
                "--outdir",
                str(outdir / "ioi_stage1"),
            ),
            trains=False,
            uses_transformerlens=True,
            uses_gpt2=True,
            requires_gpu_mps=False,
            feasible=False,
            note="Full GPT-2/TransformerLens runner is not migrated into the base package yet.",
        ),
        "ioi_stage2b": FullTask(
            key="ioi_stage2b",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "ioi_stage2b",
                "--config",
                "configs/ioi_stage2b.yaml",
                "--outdir",
                str(outdir / "ioi_stage2b"),
            ),
            trains=False,
            uses_transformerlens=True,
            uses_gpt2=True,
            requires_gpu_mps=True,
            feasible=False,
            note="Full GPT-2/TransformerLens runner is not migrated into the base package yet.",
        ),
        "ioi_stage2c": FullTask(
            key="ioi_stage2c",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "ioi_stage2c",
                "--config",
                "configs/ioi_stage2c.yaml",
                "--outdir",
                str(outdir / "ioi_stage2c"),
            ),
            trains=False,
            uses_transformerlens=True,
            uses_gpt2=True,
            requires_gpu_mps=True,
            feasible=False,
            note="Full GPT-2/TransformerLens runner is not migrated into the base package yet.",
        ),
        "ioi_stage2d": FullTask(
            key="ioi_stage2d",
            command=(
                _python(),
                "-m",
                "observerbench.cli",
                "run",
                "ioi_stage2d",
                "--config",
                "configs/ioi_stage2d.yaml",
                "--input-run",
                "results/frozen/ioi/stage2c_primary_stratified_mean_end",
                "--outdir",
                str(outdir / "ioi_stage2d"),
            ),
            trains=False,
            uses_transformerlens=False,
            uses_gpt2=False,
            requires_gpu_mps=False,
            feasible=True,
            note="CPU postprocess over frozen Stage 2c outputs.",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--only", nargs="*", default=None, help="Task keys to run; default is all.")
    parser.add_argument("--yes-run-expensive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_task(task: FullTask) -> None:
    flags = []
    if task.trains:
        flags.append("training")
    if task.uses_transformerlens:
        flags.append("TransformerLens")
    if task.uses_gpt2:
        flags.append("GPT-2")
    if task.requires_gpu_mps:
        flags.append("GPU/MPS recommended")
    label = ", ".join(flags) if flags else "CPU/base"
    print(f"{task.key}: {label}")
    print(f"  {' '.join(_display_arg(arg) for arg in task.command)}")
    print(f"  {task.note}")


def _display_arg(arg: str) -> str:
    if arg == sys.executable:
        return "python"
    try:
        path = Path(arg)
        if path.is_absolute() and path.is_relative_to(REPO_ROOT):
            return str(path.relative_to(REPO_ROOT))
    except ValueError:
        pass
    return arg


def main() -> int:
    args = build_parser().parse_args()
    plan = task_plan(args.outdir)
    selected_keys = args.only if args.only else list(plan)
    unknown = sorted(set(selected_keys) - set(plan))
    if unknown:
        raise SystemExit(f"Unknown task key(s): {', '.join(unknown)}")
    selected = [plan[key] for key in selected_keys]

    expensive = [task for task in selected if task.expensive]
    if expensive and not args.yes_run_expensive:
        print("Refusing to run expensive full reproduction tasks without --yes-run-expensive.")
        print("Selected expensive tasks:")
        for task in expensive:
            _print_task(task)
        print("Fast reproduction remains available with: python scripts/reproduce_paper_fast.py")
        return 2

    infeasible = [task for task in selected if not task.feasible]
    if infeasible:
        print("Some selected full reruns are documented but not yet feasible in the migrated base package:")
        for task in infeasible:
            _print_task(task)
        selected = [task for task in selected if task.feasible]
        if not selected:
            return 3

    for task in selected:
        _print_task(task)
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(task.command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
