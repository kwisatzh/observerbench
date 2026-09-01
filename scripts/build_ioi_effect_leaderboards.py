#!/usr/bin/env python3
"""Build checked GPT-2 IOI effect-prediction leaderboard panels."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (20, 40, 80, 160)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], name: str) -> float:
    return float(row[name])


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
        "task_name": "ioi-gpt2-small-finite-effects",
        "task_version": f"phase5-test-v1-b{budget:03d}",
        "observer_name": name,
        "display_name": display_name,
        "observer_family": family,
        "access_regime": access,
        "requires_white_box_access": white_box,
        "result_status": status,
        "measurement_budget": budget,
        "metrics": {"mae": mae, "rmse": rmse},
        "source_result": source.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(source),
    }


def build(*, output_root: Path) -> None:
    finite_path = ROOT / "results/revision/phase05/ioi_confirmatory/prediction_metrics.csv"
    atp_path = ROOT / "results/revision/published_baselines/ioi_attribution_patching_v1/summary.csv"
    finite = _read_rows(finite_path)
    atp = _read_rows(atp_path)

    for budget in BUDGETS:
        task_version = f"phase5-test-v1-b{budget:03d}"
        finite_rows = {
            row["model"]: row
            for row in finite
            if row["split"] == "test" and int(row["measurement_budget"]) == budget
        }
        atp_rows = {
            row["observer"]: row
            for row in atp
            if int(row["measurement_budget"]) == budget
        }
        additive = finite_rows["additive_head"]
        pairs = finite_rows["count_plus_all_bin4"]
        raw = atp_rows["ioi-attribution-patching-raw"]
        calibrated = atp_rows["ioi-attribution-patching-scalar-calibrated"]
        rows = [
            _row(
                name="ioi-count-plus-all-pairs-ridge",
                display_name="All-pairs finite-effect ridge",
                family="interaction-aware finite-effect predictor",
                access="frozen forward intervention measurements",
                status="prespecified bundled baseline",
                budget=budget,
                mae=_float(pairs, "mae"),
                rmse=_float(pairs, "rmse"),
                source=finite_path,
                white_box=False,
            ),
            _row(
                name="ioi-additive-ridge",
                display_name="Additive finite-effect ridge",
                family="first-order finite-effect predictor",
                access="frozen forward intervention measurements",
                status="prespecified bundled baseline",
                budget=budget,
                mae=_float(additive, "mae"),
                rmse=_float(additive, "rmse"),
                source=finite_path,
                white_box=False,
            ),
            _row(
                name="ioi-attribution-patching-scalar-calibrated",
                display_name="Scalar-calibrated attribution patching",
                family="published attribution-patching baseline",
                access="white-box gradients plus frozen finite calibration measurements",
                status="post-outcome published-method baseline",
                budget=budget,
                mae=_float(calibrated, "mae"),
                rmse=_float(calibrated, "rmse"),
                source=atp_path,
                white_box=True,
            ),
            _row(
                name="ioi-attribution-patching-raw",
                display_name="Raw attribution patching",
                family="published attribution-patching baseline",
                access="white-box gradients over 192 public training prompts",
                status="post-outcome published-method baseline",
                budget=budget,
                mae=_float(raw, "mae"),
                rmse=_float(raw, "rmse"),
                source=atp_path,
                white_box=True,
            ),
        ]
        panel = output_root / f"ioi-gpt2-small-finite-effects-b{budget:03d}"
        panel.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "observerbench.effect_leaderboard.v0",
            "task_id": f"ioi-gpt2-small-finite-effects@{task_version}",
            "metric_direction": {"mae": "lower", "rmse": "lower"},
            "rows": rows,
        }
        (panel / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (panel / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("observer_name", "access_regime", "mae", "rmse", "result_status"))
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
                f"# GPT-2 IOI finite effects — budget {budget}\n\n"
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
        default=ROOT / "leaderboards/effect",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(output_root=parse_args().output_root)
