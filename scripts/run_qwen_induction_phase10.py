#!/usr/bin/env python3
"""Run the sealed Qwen induction Copy-v2 stage chain.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from observerbench.tasks.qwen_induction.copy_v2_runner import (
    COPY_V2_STAGES,
    CopyV2Runner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one sealed stage of the prospective Qwen Copy-v2 study."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--stage", choices=(*COPY_V2_STAGES, "all"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from hash-checked completed stages and checkpoints.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = CopyV2Runner(
        args.config,
        args.artifacts_root,
        device=args.device,
        local_files_only=args.local_files_only,
        resume=args.resume,
    )
    result = runner.run(args.stage)
    if isinstance(result, tuple):
        payload = [asdict(item) for item in result]
        succeeded = all(item.status == "complete" for item in result)
    else:
        payload = asdict(result)
        succeeded = result.status == "complete"
    print(json.dumps(payload, default=str, indent=2))
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
