"""Apply the frozen Phase-7 joint primary evaluation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase7_evaluation import evaluate_phase7_confirmation


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the sealed Phase-7 joint primary test.")
    parser.add_argument("--design-dir", type=Path, default=PHASE7 / "design")
    parser.add_argument("--pretest-dir", type=Path, default=PHASE7 / "clean_pretest")
    parser.add_argument("--freeze-dir", type=Path, default=PHASE7 / "prediction_freeze")
    parser.add_argument("--audit-dir", type=Path, default=PHASE7 / "preoutcome_audit")
    parser.add_argument("--phase5-design-dir", type=Path, default=PHASE5 / "design")
    parser.add_argument("--phase5-effects-dir", type=Path, default=PHASE5 / "ioi_effects")
    parser.add_argument("--measurement-dir", type=Path, default=PHASE7 / "selected_measurement")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "evaluation")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_phase7_confirmation(
        args.design_dir,
        args.pretest_dir,
        args.freeze_dir,
        args.audit_dir,
        args.phase5_design_dir,
        args.phase5_effects_dir,
        args.measurement_dir,
        args.outdir,
        protocol_path=args.protocol,
    )
    digest = json.loads((output / "result_digest.json").read_text(encoding="utf-8"))
    print(json.dumps(digest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
