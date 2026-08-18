"""Execute the external APPS monitor study in a Colab accelerator session.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


ARCHIVE = Path("/content/observerbench_controlarena_bundle.tar.gz")
REPO = Path("/content/ObserverBench")
ARTIFACTS = Path("/content/controlarena_apps_qwen_artifacts")
CONFIG = REPO / "configs/revision/ai_control/controlarena_apps_qwen2_5_7b_v0.json"


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
    run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[ai-control]")
    run(
        sys.executable,
        str(REPO / "scripts/run_controlarena_apps_qwen.py"),
        "--config",
        str(CONFIG),
        "--outdir",
        str(ARTIFACTS),
        "--device",
        "cuda",
        "--resume",
    )
    result = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(result["budget_results"]["0.02"], indent=2), flush=True)
    shutil.make_archive("/content/controlarena_apps_qwen_artifacts", "gztar", ARTIFACTS)
    print("ARTIFACT_BUNDLE=/content/controlarena_apps_qwen_artifacts.tar.gz", flush=True)


if __name__ == "__main__":
    main()
