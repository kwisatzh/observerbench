"""Run the explicitly post-outcome Phase-6 IOI direct-risk exploration.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase5_analysis import IOIPhase5AnalysisConfig
from observerbench.tasks.ioi.phase6_risk import (
    IOIRiskExploratoryConfig,
    run_risk_exploratory_analysis,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase06_risk_exploratory_v1.json",
    )
    parser.add_argument(
        "--phase5-protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase05_confirmatory_v2.json",
    )
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/design",
    )
    parser.add_argument(
        "--effects-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/ioi_effects",
    )
    parser.add_argument(
        "--mean-fit-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/ioi_fit",
    )
    parser.add_argument(
        "--confirmatory-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/ioi_confirmatory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_risk_exploratory",
    )
    parser.add_argument("--bootstrap-repeats", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exploratory_mapping = json.loads(args.config.read_text(encoding="utf-8"))
    phase5_mapping = json.loads(args.phase5_protocol.read_text(encoding="utf-8"))
    exploratory_config = IOIRiskExploratoryConfig.from_mapping(
        exploratory_mapping,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    phase5_config = IOIPhase5AnalysisConfig.from_protocol(
        phase5_mapping,
        bootstrap_repeats=exploratory_config.bootstrap_repeats,
    )
    digest = run_risk_exploratory_analysis(
        args.design_dir,
        args.effects_dir,
        args.mean_fit_dir,
        args.confirmatory_dir,
        args.outdir,
        config=exploratory_config,
        phase5_config=phase5_config,
        config_path=args.config,
    )
    print(json.dumps(digest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

