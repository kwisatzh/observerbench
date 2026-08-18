"""Run the sealed Qwen safety study inside a Colab accelerator session.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


ARCHIVE = Path("/content/observerbench_qwen_safety_bundle.tar.gz")
REPO = Path("/content/ObserverBench")
ARTIFACTS = Path("/content/qwen_safety_artifacts")
CONFIG = REPO / "configs/revision/qwen_safety/qwen2_5_7b_instruct_paired_scope_v0.json"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    completed = subprocess.run(args, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    completed.check_returncode()


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"missing uploaded source bundle: {ARCHIVE}")
    if REPO.exists():
        shutil.rmtree(REPO)
    run("tar", "-xzf", str(ARCHIVE), "-C", "/content")
    for sidecar in REPO.rglob("._*"):
        sidecar.unlink()
    run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[qwen]")
    run(
        sys.executable,
        str(REPO / "scripts/run_qwen_safety.py"),
        "--config",
        str(CONFIG),
        "--artifacts-root",
        str(ARTIFACTS),
        "--stage",
        "all",
        "--device",
        "cuda",
        "--resume",
    )
    result_path = ARTIFACTS / "evaluation/qwen_safety_results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    shutil.make_archive("/content/qwen_safety_artifacts", "gztar", ARTIFACTS)
    print("ARTIFACT_BUNDLE=/content/qwen_safety_artifacts.tar.gz", flush=True)


if __name__ == "__main__":
    main()
