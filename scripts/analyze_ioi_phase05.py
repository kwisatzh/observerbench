"""Fit and evaluate the frozen Phase-5 IOI effect observers in two stages.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.phase5_analysis import (
    IOIPhase5AnalysisConfig,
    IOIPhase5EvaluationConfig,
    evaluate_phase5_observers,
    fit_phase5_observers,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase05_confirmatory_v2.json",
    )
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase05_evaluation_v3.json",
    )
    parser.add_argument("--design-dir", type=Path, default=ROOT / "results/revision/phase05/design")
    parser.add_argument("--effects-dir", type=Path, default=ROOT / "results/revision/phase05/ioi_effects")
    parser.add_argument("--fit-dir", type=Path, default=ROOT / "results/revision/phase05/ioi_fit")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/revision/phase05/ioi_confirmatory")
    parser.add_argument("--bootstrap-repeats", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    design_manifest = json.loads(
        (args.design_dir / "design_manifest.json").read_text(encoding="utf-8")
    )
    if design_manifest.get("protocol_hash") != file_sha256(args.protocol):
        raise ValueError("analysis protocol does not match the frozen design protocol")
    config = IOIPhase5AnalysisConfig.from_protocol(
        protocol,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    if args.stage == "fit":
        fit_phase5_observers(
            args.design_dir,
            args.effects_dir,
            args.fit_dir,
            config=config,
        )
    else:
        evaluation_protocol = json.loads(
            args.evaluation_protocol.read_text(encoding="utf-8")
        )
        frozen_paths = {
            "design_protocol": args.protocol,
            "design_manifest": args.design_dir / "design_manifest.json",
            "effect_manifest": args.effects_dir / "effect_manifest.json",
            "fit_manifest": args.fit_dir / "fit_manifest.json",
            "candidate_predictions": args.fit_dir / "candidate_predictions.csv",
        }
        frozen_inputs = evaluation_protocol.get("frozen_inputs")
        if not isinstance(frozen_inputs, dict):
            raise ValueError("evaluation protocol lacks frozen input hashes")
        for name, path in frozen_paths.items():
            record = frozen_inputs.get(name)
            if not isinstance(record, dict) or record.get("sha256") != file_sha256(path):
                raise ValueError(f"evaluation input differs from v3 freeze: {name}")
            expected_path = (ROOT / str(record.get("path"))).resolve()
            if expected_path != path.resolve():
                raise ValueError(f"evaluation input path differs from v3 freeze: {name}")
        evaluation_config = IOIPhase5EvaluationConfig.from_protocol(
            evaluation_protocol,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        evaluate_phase5_observers(
            args.design_dir,
            args.effects_dir,
            args.fit_dir,
            args.outdir,
            config=config,
            evaluation_config=evaluation_config,
            evaluation_protocol_sha256=file_sha256(args.evaluation_protocol),
        )


if __name__ == "__main__":
    main()
