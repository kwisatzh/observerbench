#!/usr/bin/env python3
"""Create or verify the local source seal for the Qwen3.5 APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from observerbench.tasks.qwen35_apps.registration import (
    seal_qwen35_apps_sources,
    verify_qwen35_apps_source_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        verify_qwen35_apps_source_manifest(args.config, args.manifest)
        if args.verify
        else seal_qwen35_apps_sources(args.config, args.manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
