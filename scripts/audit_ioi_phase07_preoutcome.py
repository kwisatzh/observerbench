"""Independently recompute and seal the repaired Phase-7 v2 freeze.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase7_freeze_audit import (
    AUDIT_FILENAME,
    run_phase7_preoutcome_audit,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the Phase-7 v2 freeze and seal post-freeze source code."
    )
    parser.add_argument("--design-dir", type=Path, default=PHASE7 / "design")
    parser.add_argument("--pretest-dir", type=Path, default=PHASE7 / "clean_pretest")
    parser.add_argument("--freeze-dir", type=Path, default=PHASE7 / "prediction_freeze")
    parser.add_argument("--phase5-design-dir", type=Path, default=PHASE5 / "design")
    parser.add_argument("--phase5-effects-dir", type=Path, default=PHASE5 / "ioi_effects")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "preoutcome_audit")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_phase7_preoutcome_audit(
        args.design_dir,
        args.pretest_dir,
        args.freeze_dir,
        args.phase5_design_dir,
        args.phase5_effects_dir,
        args.outdir,
        protocol_path=args.protocol,
    )
    audit = json.loads((output / AUDIT_FILENAME).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "status": audit["status"],
                "counts": audit["counts"],
                "source_code_hashes": audit["source_code_hashes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
