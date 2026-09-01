# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_ioi_effect_leaderboards",
    ROOT / "scripts" / "build_ioi_effect_leaderboards.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_builder_emits_budget_matched_published_method_panels(tmp_path: Path) -> None:
    BUILDER.build(output_root=tmp_path)

    panels = sorted(path.name for path in tmp_path.iterdir())
    assert panels == [
        "ioi-gpt2-small-finite-effects-b020",
        "ioi-gpt2-small-finite-effects-b040",
        "ioi-gpt2-small-finite-effects-b080",
        "ioi-gpt2-small-finite-effects-b160",
    ]

    payload = json.loads(
        (tmp_path / panels[0] / "results.json").read_text(encoding="utf-8")
    )
    rows = {row["observer_name"]: row for row in payload["rows"]}
    assert rows["ioi-attribution-patching-scalar-calibrated"]["metrics"]["mae"] < rows[
        "ioi-attribution-patching-raw"
    ]["metrics"]["mae"]
    assert rows["ioi-attribution-patching-scalar-calibrated"][
        "requires_white_box_access"
    ]
