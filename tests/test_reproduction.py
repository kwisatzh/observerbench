from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import json
from hashlib import sha256
import os
from pathlib import Path
import re
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


def test_fast_reproduction_includes_checked_revision_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "generated_figures"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl-cache")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper_fast.py",
            "--only",
            "revision_ioi_stage2c",
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
    assert (outdir / "revision_ioi_stage2c_capacity_matched.png").exists()
    manifest = json.loads((outdir / "reproduction_manifest.json").read_text())
    assert manifest["schema"] == "observerbench.paper_reproduction_fast.v1"
    assert manifest["generated"][0]["result_id"] == "revision_ioi_stage2c"
    assert manifest["downloads_models"] is False
    assert manifest["trains_models"] is False


def test_fast_reproduction_default_matches_current_paper(tmp_path: Path) -> None:
    outdir = tmp_path / "generated_figures"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl-cache")
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper_fast.py",
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
    manifest = json.loads((outdir / "reproduction_manifest.json").read_text())
    generated_ids = [item["result_id"] for item in manifest["generated"]]
    assert generated_ids == manifest["current_default_ids"]
    assert not {
        "figure_05",
        "figure_06",
        "figure_07",
        "figure_09",
        "figure_10",
        "table_01",
        "table_02",
    }.intersection(generated_ids)
    assert (outdir / "revision_ctl2_factorial_table.tex").exists()
    assert (outdir / "revision_ioi_capacity_contrasts.tex").exists()
    assert (outdir / "observer_cards/ctl2_observer_card.tex").exists()
    for item in manifest["generated"]:
        for relative, expected in item["sha256"].items():
            actual = sha256((outdir / relative).read_bytes()).hexdigest()
            assert actual == expected

    source = ROOT / "paper/observerbench_v15_source/sections"
    current_sections = [
        source / "ctl1_phase01.tex",
        source / "trained_ctl1_phase01.tex",
        source / "ctl2_phase01.tex",
        source / "ioi.tex",
    ]
    figure_names = {
        match
        for path in current_sections
        for match in re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
            path.read_text(encoding="utf-8"),
        )
    }
    assert figure_names
    assert all((outdir / name).exists() for name in figure_names)
