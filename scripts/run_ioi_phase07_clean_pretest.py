"""Run the clean-only Phase-7 stop gate.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observerbench.tasks.ioi.phase7_pretest import (
    Phase7PretestConfig,
    rebind_frozen_phase7_clean_pretest,
    run_phase7_clean_pretest,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE7_V1 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score only clean Phase-7 prompts; candidate masks cannot run here."
    )
    parser.add_argument("--design-dir", type=Path, default=PHASE7 / "design")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "clean_pretest")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    parser.add_argument(
        "--reuse-frozen-pretest-dir",
        type=Path,
        default=PHASE7_V1 / "clean_pretest",
        help="Rebind the v1 clean-only scores, which predate the aborted v1 shard.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_frozen_pretest_dir is not None:
        output = rebind_frozen_phase7_clean_pretest(
            args.design_dir,
            args.reuse_frozen_pretest_dir,
            args.outdir,
            protocol_path=args.protocol,
        )
    else:
        output = run_phase7_clean_pretest(
            args.design_dir,
            args.outdir,
            protocol_path=args.protocol,
            config=Phase7PretestConfig(device=args.device, batch_size=args.batch_size),
        )
    manifest = json.loads((output / "pretest_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"], "gate": manifest["gate"]}, indent=2))


if __name__ == "__main__":
    main()
