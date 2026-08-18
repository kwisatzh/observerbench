"""Fit and freeze Phase-7 actions after the clean stop gate passes.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase7_confirmation import fit_and_freeze_phase7_actions


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit on sealed Phase-5 train/calibration rows and freeze Phase-7 actions."
    )
    parser.add_argument("--design-dir", type=Path, default=PHASE7 / "design")
    parser.add_argument("--pretest-dir", type=Path, default=PHASE7 / "clean_pretest")
    parser.add_argument("--phase5-design-dir", type=Path, default=PHASE5 / "design")
    parser.add_argument("--phase5-effects-dir", type=Path, default=PHASE5 / "ioi_effects")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "prediction_freeze")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = fit_and_freeze_phase7_actions(
        args.design_dir,
        args.pretest_dir,
        args.phase5_design_dir,
        args.phase5_effects_dir,
        args.outdir,
        protocol_path=args.protocol,
    )
    manifest = json.loads(
        (output / "prediction_action_manifest.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"output": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
