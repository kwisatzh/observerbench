from __future__ import annotations

import subprocess
import sys


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
