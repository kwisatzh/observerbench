"""Run the secondary Qwen safety checks over the sealed activation artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from observerbench.tasks.qwen_safety.followup import run_qwen_safety_followup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-dimension", type=int, default=2048)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_qwen_safety_followup(
        config_path=args.config,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        text_dimension=args.text_dimension,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(payload["status"])
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
