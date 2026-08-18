#!/usr/bin/env python3
"""Run the frozen Phase-5 nonlinear suffix-loop experiment.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.trained_ctl2_suffix import (
    NonlinearSuffixConfig,
    run_nonlinear_suffix_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/revision/ctl2_phase05_nonlinear_suffix_v2.json"
DEFAULT_OUTDIR = ROOT / "results/revision/phase05/nonlinear_suffix_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.config.read_text())
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
    cfg = NonlinearSuffixConfig(**values)
    analysis = run_nonlinear_suffix_experiment(
        cfg,
        args.outdir,
        protocol_path=args.config,
    )
    print(json.dumps({
        "outdir": str(args.outdir),
        "all_gates_pass": analysis["all_gates_pass"],
        "gates": analysis["gates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

