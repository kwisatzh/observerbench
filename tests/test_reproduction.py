from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_fast_reproduction_script_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "generated_figures"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl-cache")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper_fast.py",
            "--only",
            "figure_01",
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "figure_01_ctl1_analytic_interaction_sweep.png").exists()
    assert (outdir / "reproduction_manifest.json").exists()
