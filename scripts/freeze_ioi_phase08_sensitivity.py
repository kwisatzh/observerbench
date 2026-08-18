"""Freeze all Phase-8 target-sensitivity predictions and actions.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase8_sensitivity import Phase8Paths, freeze_phase8_sensitivity


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"
PHASE8 = ROOT / "results/revision/phase08/ioi_target_sensitivity"


def phase8_paths() -> Phase8Paths:
    return Phase8Paths(
        phase7_protocol=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
        phase7_design=PHASE7 / "design",
        phase7_pretest=PHASE7 / "clean_pretest",
        phase7_freeze=PHASE7 / "prediction_freeze",
        phase7_audit=PHASE7 / "preoutcome_audit",
        phase7_measurement=PHASE7 / "selected_measurement",
        phase5_design=PHASE5 / "design",
        phase5_effects=PHASE5 / "ioi_effects",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze post-confirmatory IOI target-sensitivity actions without new inference."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json",
    )
    parser.add_argument("--outdir", type=Path, default=PHASE8 / "prediction_freeze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = freeze_phase8_sensitivity(
        phase8_paths(),
        args.outdir,
        protocol_path=args.protocol,
        repository_root=ROOT,
    )
    manifest = json.loads(
        (output / "prediction_action_manifest.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()

