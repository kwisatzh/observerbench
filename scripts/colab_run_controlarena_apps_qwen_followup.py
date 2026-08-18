"""Execute the ControlArena APPS follow-up controls in Colab.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ARCHIVE = Path("/content/observerbench_controlarena_followup_bundle.tar.gz")
REPO = Path("/content/ObserverBench")
BASE = REPO / "results/revision/ai_control/controlarena_apps_qwen_v0"
OUT = Path("/content/controlarena_apps_qwen_followup_artifacts")
BASE_CONFIG = REPO / "configs/revision/ai_control/controlarena_apps_qwen2_5_7b_v0.json"
FOLLOWUP_CONFIG = REPO / "configs/revision/ai_control/controlarena_apps_qwen_followup_v1.json"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    process = subprocess.Popen(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, args)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"missing uploaded source bundle: {ARCHIVE}")
    if REPO.exists():
        shutil.rmtree(REPO)
    run("tar", "-xzf", str(ARCHIVE), "-C", "/content")
    import sys

    run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[ai-control]")
    run(
        sys.executable,
        str(REPO / "scripts/run_controlarena_apps_qwen_followup.py"),
        "--base-config",
        str(BASE_CONFIG),
        "--followup-config",
        str(FOLLOWUP_CONFIG),
        "--base-outdir",
        str(BASE),
        "--outdir",
        str(OUT),
        "--device",
        "cuda",
        "--resume",
    )
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(result["budget_results"]["0.02"], indent=2), flush=True)
    shutil.make_archive("/content/controlarena_apps_qwen_followup_artifacts", "gztar", OUT)
    print(
        "ARTIFACT_BUNDLE=/content/controlarena_apps_qwen_followup_artifacts.tar.gz",
        flush=True,
    )


if __name__ == "__main__":
    main()
