#!/usr/bin/env python3
"""Build checked Qwen Copy-v2 effect-prediction leaderboard panels."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (16, 40, 64, 128)
TASK_NAME = "induction-qwen2-5-7b-finite-effects"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _row(
    *,
    name: str,
    display_name: str,
    family: str,
    access: str,
    status: str,
    budget: int,
    mae: float,
    rmse: float,
    source: Path,
    white_box: bool,
) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "task_version": f"copy-v2-b{budget:03d}",
        "observer_name": name,
        "display_name": display_name,
        "observer_family": family,
        "access_regime": access,
        "requires_white_box_access": white_box,
        "result_status": status,
        "measurement_budget": budget,
        "metrics": {"mae": mae, "rmse": rmse},
        "source_result": _source_label(source),
        "source_sha256": _sha256(source),
    }


def build(
    *,
    output_root: Path,
    finite_path: Path | None = None,
    atp_path: Path | None = None,
) -> None:
    finite_path = finite_path or (
        ROOT
        / "results"
        / "revision"
        / "phase10"
        / "qwen_induction_copy_v2_complete"
        / "copy_v2"
        / "work"
        / "evaluation"
        / "mean_effect_metrics.csv"
    )
    atp_path = atp_path or (
        ROOT
        / "results"
        / "revision"
        / "published_baselines"
        / "qwen_induction_attribution_patching_v1"
        / "summary.csv"
    )
    finite = _read_rows(finite_path)
    atp = _read_rows(atp_path)

    for budget in BUDGETS:
        task_version = f"copy-v2-b{budget:03d}"
        finite_rows = {
            row["model"]: row
            for row in finite
            if int(row["measurement_budget"]) == budget
        }
        atp_rows = {
            row["observer"]: row
            for row in atp
            if int(row["measurement_budget"]) == budget
        }
        required_finite = {"additive", "quadratic"}
        required_atp = {
            "qwen-induction-attribution-patching-raw",
            "qwen-induction-attribution-patching-scalar-calibrated",
        }
        if not required_finite.issubset(finite_rows):
            raise ValueError(f"Qwen finite rows are incomplete at budget {budget}")
        if not required_atp.issubset(atp_rows):
            raise ValueError(f"Qwen attribution rows are incomplete at budget {budget}")
        additive = finite_rows["additive"]
        quadratic = finite_rows["quadratic"]
        raw = atp_rows["qwen-induction-attribution-patching-raw"]
        calibrated = atp_rows[
            "qwen-induction-attribution-patching-scalar-calibrated"
        ]
        rows = [
            _row(
                name="qwen-induction-quadratic-ridge",
                display_name="Quadratic finite-effect ridge",
                family="interaction-aware finite-effect predictor",
                access="frozen forward intervention measurements",
                status="prespecified bundled baseline",
                budget=budget,
                mae=float(quadratic["mae"]),
                rmse=float(quadratic["rmse"]),
                source=finite_path,
                white_box=False,
            ),
            _row(
                name="qwen-induction-additive-ridge",
                display_name="Additive finite-effect ridge",
                family="first-order finite-effect predictor",
                access="frozen forward intervention measurements",
                status="prespecified bundled baseline",
                budget=budget,
                mae=float(additive["mae"]),
                rmse=float(additive["rmse"]),
                source=finite_path,
                white_box=False,
            ),
            _row(
                name="qwen-induction-attribution-patching-scalar-calibrated",
                display_name="Scalar-calibrated attribution patching",
                family="published attribution-patching baseline",
                access=(
                    "white-box gradients plus frozen finite calibration measurements"
                ),
                status="post-outcome published-method baseline",
                budget=budget,
                mae=float(calibrated["mae"]),
                rmse=float(calibrated["rmse"]),
                source=atp_path,
                white_box=True,
            ),
            _row(
                name="qwen-induction-attribution-patching-raw",
                display_name="Raw attribution patching",
                family="published attribution-patching baseline",
                access="white-box gradients over 256 public training prompts",
                status="post-outcome published-method baseline",
                budget=budget,
                mae=float(raw["mae"]),
                rmse=float(raw["rmse"]),
                source=atp_path,
                white_box=True,
            ),
        ]
        panel = output_root / f"{TASK_NAME}-b{budget:03d}"
        panel.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "observerbench.effect_leaderboard.v0",
            "task_id": f"{TASK_NAME}@{task_version}",
            "metric_direction": {"mae": "lower", "rmse": "lower"},
            "rows": rows,
        }
        (panel / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (panel / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("observer_name", "access_regime", "mae", "rmse", "result_status")
            )
            for row in rows:
                writer.writerow(
                    (
                        row["observer_name"],
                        row["access_regime"],
                        row["metrics"]["mae"],
                        row["metrics"]["rmse"],
                        row["result_status"],
                    )
                )
        (panel / "README.md").write_text(
            (
                f"# Qwen2.5-7B induction-copy finite effects — budget {budget}\n\n"
                "Checked held-out prediction results for the frozen "
                f"`{task_version}` task. Compare methods within this panel; "
                "white-box gradient and forward-only rows have different access costs.\n"
            ),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "leaderboards" / "effect",
    )
    parser.add_argument("--finite-path", type=Path, default=None)
    parser.add_argument("--atp-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        output_root=args.output_root,
        finite_path=args.finite_path,
        atp_path=args.atp_path,
    )
