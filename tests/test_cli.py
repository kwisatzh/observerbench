from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "observerbench.cli", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_cli_starts() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "ObserverBench paper reproduction CLI" in result.stdout


def test_list_tasks_works() -> None:
    result = run_cli("list-tasks")

    assert result.returncode == 0
    assert "ctl1_analytic" in result.stdout
    assert "ioi_stage2d" in result.stdout


def test_list_effect_tasks_includes_both_model_families() -> None:
    result = run_cli("list-effect-tasks")

    assert result.returncode == 0, result.stderr
    assert "ioi-gpt2-small-finite-effects@phase5-test-v1-b020" in result.stdout
    assert "induction-qwen2-5-7b-finite-effects@copy-v1-b016" in result.stdout
    assert "budget:128" in result.stdout


def test_installed_console_script_lists_tasks() -> None:
    executable = Path(sys.executable).with_name("observerbench")
    if not executable.exists():
        return
    result = subprocess.run(
        [str(executable), "list-tasks"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ctl1_analytic" in result.stdout
    assert "trained_ctl2" in result.stdout
