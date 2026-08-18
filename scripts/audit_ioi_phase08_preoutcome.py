"""Recompute Phase-8 actions and bind downstream code before inference.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_ioi_phase08_sensitivity import PHASE8, ROOT, phase8_paths
from observerbench.tasks.ioi.phase8_sensitivity import (
    REQUIRED_DOWNSTREAM_SOURCE_FILES,
    audit_phase8_freeze,
)


DOWNSTREAM_SOURCE_FILES = REQUIRED_DOWNSTREAM_SOURCE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Phase-8 freeze and seal measurement/evaluation sources."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json",
    )
    parser.add_argument("--freeze-dir", type=Path, default=PHASE8 / "prediction_freeze")
    parser.add_argument("--outdir", type=Path, default=PHASE8 / "preoutcome_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = audit_phase8_freeze(
        phase8_paths(),
        args.freeze_dir,
        args.outdir,
        protocol_path=args.protocol,
        repository_root=ROOT,
        downstream_source_files=DOWNSTREAM_SOURCE_FILES,
    )
    audit = json.loads((output / "preoutcome_audit.json").read_text(encoding="utf-8"))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
