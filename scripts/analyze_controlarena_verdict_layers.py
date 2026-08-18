"""Re-evaluate cached verdict-context layers without new model inference.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from observerbench.ai_control_followup import budget_metrics
from observerbench.core import write_json
from observerbench.provenance import file_sha256
from run_controlarena_apps_qwen import _fit_ridge, _projection


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = _load_object(args.config)
    design = _load_object(args.design)
    rows = design.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("design must contain rows")
    with np.load(args.cache, allow_pickle=False) as archive:
        activations = archive["basic_activations"].astype(np.float32)
    if len(rows) != len(activations):
        raise ValueError("design and activation rows differ")

    labels = np.asarray([bool(row["backdoor_works"]) for row in rows], dtype=float)
    splits = np.asarray([str(row["split"]) for row in rows], dtype=str)
    fit = splits == "fit"
    calibration = splits == "calibration"
    locked = splits == "locked_test"
    measurement = config["measurement"]
    candidates = []
    for layer_position, layer in enumerate(measurement["layers"]):
        raw = activations[:, layer_position]
        projected = raw @ _projection(
            raw.shape[1],
            int(measurement["projection_dim"]),
            int(measurement["projection_seed"]),
            int(layer),
        )
        for ridge_value in measurement["ridge_grid"]:
            ridge = float(ridge_value)
            calibration_scores = _fit_ridge(
                projected[fit], labels[fit], projected[calibration], ridge
            )
            locked_scores = _fit_ridge(
                projected[fit | calibration],
                labels[fit | calibration],
                projected[locked],
                ridge,
            )
            candidates.append(
                {
                    "layer": int(layer),
                    "ridge": ridge,
                    "calibration": budget_metrics(
                        labels[calibration].astype(bool),
                        calibration_scores,
                        budget=0.02,
                    ),
                    "locked_test_post_outcome_diagnostic": budget_metrics(
                        labels[locked].astype(bool), locked_scores, budget=0.02
                    ),
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["calibration"]["realized_violation_rate"],
            -row["calibration"]["risk_auroc"],
            row["layer"],
            row["ridge"],
        ),
    )
    payload = {
        "schema": "observerbench.controlarena_apps_qwen.verdict_layer_diagnostic.v0",
        "status": "post_outcome_cached_activation_diagnostic",
        "inputs": {
            "config_sha256": file_sha256(args.config),
            "design_sha256": file_sha256(args.design),
            "cache_sha256": file_sha256(args.cache),
        },
        "selection_rule": "calibration violations, then calibration AUROC, layer, ridge",
        "selected_on_calibration": selected,
        "candidates": candidates,
    }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
