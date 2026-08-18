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


def test_list_safety_tasks_includes_paired_interlock() -> None:
    result = run_cli("list-safety-tasks")

    assert result.returncode == 0, result.stderr
    assert "safety_interlock_analytic@paired-scope-v0" in result.stdout


def test_list_safety_results_includes_checked_qwen_observers() -> None:
    result = run_cli("list-safety-results")

    assert result.returncode == 0, result.stderr
    assert "Activation label x severity" in result.stdout
    assert "Qwen output margin x severity" in result.stdout


def test_list_ai_control_results_includes_external_monitors() -> None:
    result = run_cli("list-ai-control-results")

    assert result.returncode == 0, result.stderr
    assert "Qwen verdict-adjacent activation probe" in result.stdout
    assert "Qwen neutral-context activation probe" in result.stdout
    assert "Qwen + ControlArena basic prompt" in result.stdout
    assert "0.953" in result.stdout


def test_make_figures_runs_checked_fast_reproduction(tmp_path: Path) -> None:
    outdir = tmp_path / "figures"
    result = run_cli(
        "make-figures",
        "--only",
        "revision_ioi_mobius",
        "--outdir",
        str(outdir),
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "revision_ioi_mobius_intervals.png").exists()
    assert (outdir / "reproduction_manifest.json").exists()
    assert not (outdir / "figures_manifest.json").exists()


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
