"""Measure only the frozen Phase-8 new selected-mask set difference.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_ioi_phase08_sensitivity import PHASE8, ROOT, phase8_paths
from observerbench.tasks.ioi.phase8_measurement import (
    Phase8MeasurementConfig,
    run_phase8_selected_measurement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="After the audit passes, measure exactly 148 new selected IOI masks."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json",
    )
    parser.add_argument("--freeze-dir", type=Path, default=PHASE8 / "prediction_freeze")
    parser.add_argument("--audit-dir", type=Path, default=PHASE8 / "preoutcome_audit")
    parser.add_argument("--outdir", type=Path, default=PHASE8 / "new_measurement")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_phase8_selected_measurement(
        phase8_paths(),
        args.freeze_dir,
        args.audit_dir,
        args.outdir,
        protocol_path=args.protocol,
        repository_root=ROOT,
        config=Phase8MeasurementConfig(
            device=args.device,
            pair_batch_size=args.pair_batch_size,
        ),
    )
    manifest = json.loads(
        (output / "measurement_manifest.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()

