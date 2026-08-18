"""Evaluate every frozen Phase-8 target and reference as secondary analyses.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_ioi_phase08_sensitivity import PHASE8, ROOT, phase8_paths
from observerbench.tasks.ioi.phase8_evaluation import evaluate_phase8_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate post-confirmatory IOI target and transformed-mean sensitivities."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json",
    )
    parser.add_argument("--freeze-dir", type=Path, default=PHASE8 / "prediction_freeze")
    parser.add_argument("--audit-dir", type=Path, default=PHASE8 / "preoutcome_audit")
    parser.add_argument("--measurement-dir", type=Path, default=PHASE8 / "new_measurement")
    parser.add_argument("--outdir", type=Path, default=PHASE8 / "evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_phase8_sensitivity(
        phase8_paths(),
        args.freeze_dir,
        args.audit_dir,
        args.measurement_dir,
        args.outdir,
        protocol_path=args.protocol,
        repository_root=ROOT,
    )
    digest = json.loads((output / "result_digest.json").read_text(encoding="utf-8"))
    print(json.dumps(digest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

