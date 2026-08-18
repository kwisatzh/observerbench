#!/usr/bin/env python3
"""Reanalyze frozen nonlinear-suffix rows without retraining the fixture.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from observerbench.core import write_json
from observerbench.tasks.trained_ctl2_suffix import (
    NonlinearSuffixConfig,
    analyze_suffix_results,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/revision/ctl2_phase05_nonlinear_suffix_v2.json"
DEFAULT_RESULTS = ROOT / "results/revision/phase05/nonlinear_suffix_v3_support"


def _load_config(path: Path) -> NonlinearSuffixConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = NonlinearSuffixConfig.__dataclass_fields__
    values = {key: value for key, value in payload.items() if key in fields}
    for key in (
        "seeds",
        "gamma_values_natural",
        "rho_targets",
        "bias_targets",
        "relative_offsets",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return NonlinearSuffixConfig(**values)


def analyze_existing(config_path: Path, results_dir: Path) -> dict[str, object]:
    cfg = _load_config(config_path)
    controlled = pd.read_csv(results_dir / "controlled_conditions.csv")
    natural = pd.read_csv(results_dir / "natural_factorial_conditions.csv")
    failures = pd.read_csv(results_dir / "manipulability_failures.csv")
    analysis = analyze_suffix_results(controlled, natural, failures, cfg)
    write_json(results_dir / "analysis.json", analysis)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    analysis = analyze_existing(args.config.resolve(), args.results_dir.resolve())
    print(json.dumps({
        "analysis_schema": analysis["schema"],
        "results_dir": str(args.results_dir),
        "training_rerun": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
