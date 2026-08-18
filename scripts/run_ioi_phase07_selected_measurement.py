"""Measure the frozen Phase-7 selected non-noop action union.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase7_measurement import (
    Phase7MeasurementConfig,
    run_phase7_selected_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="After all seals pass, measure only Phase-7 selected non-noop masks."
    )
    parser.add_argument("--design-dir", type=Path, default=PHASE7 / "design")
    parser.add_argument("--pretest-dir", type=Path, default=PHASE7 / "clean_pretest")
    parser.add_argument("--freeze-dir", type=Path, default=PHASE7 / "prediction_freeze")
    parser.add_argument("--audit-dir", type=Path, default=PHASE7 / "preoutcome_audit")
    parser.add_argument("--phase5-design-dir", type=Path, default=PHASE5 / "design")
    parser.add_argument("--phase5-effects-dir", type=Path, default=PHASE5 / "ioi_effects")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "selected_measurement")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_phase7_selected_measurement(
        args.design_dir,
        args.pretest_dir,
        args.freeze_dir,
        args.audit_dir,
        args.phase5_design_dir,
        args.phase5_effects_dir,
        args.outdir,
        protocol_path=args.protocol,
        config=Phase7MeasurementConfig(
            device=args.device, pair_batch_size=args.pair_batch_size
        ),
    )
    manifest = json.loads(
        (output / "measurement_manifest.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"output": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
