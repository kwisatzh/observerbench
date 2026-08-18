"""Colab wrapper for the post-outcome Qwen safety decision-margin extraction.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


ARCHIVE = Path("/content/observerbench_qwen_safety_bundle.tar.gz")
REPO = Path("/content/ObserverBench")
OUTPUT = Path("/content/qwen_safety_decision_margin")
CONFIG = REPO / "configs/revision/qwen_safety/qwen2_5_7b_instruct_paired_scope_v0.json"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    run("tar", "-xzf", str(ARCHIVE), "-C", "/content")
    run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[qwen]")
    run(
        sys.executable,
        str(REPO / "scripts/extract_qwen_safety_decision_margin.py"),
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--device",
        "cuda",
    )
    shutil.make_archive("/content/qwen_safety_decision_margin", "gztar", OUTPUT)
    print("ARTIFACT_BUNDLE=/content/qwen_safety_decision_margin.tar.gz", flush=True)


if __name__ == "__main__":
    main()
