# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gemma_cost_summary", ROOT / "scripts/summarize_gemma_observer_costs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RESULTS = ROOT / "results/revision/gemma_observer_costs_20260904"


def test_published_summary_reproduces_from_recorded_stages():
    report = module.summarize(json.loads((RESULTS / "results.json").read_text()))
    assert report == json.loads((RESULTS / "summary.json").read_text())
    rows = {row["observer"]: row for row in report["rows"]}
    assert rows["dense_neutral"]["median_seconds"] == pytest.approx(5.982810646)
    assert rows["sae_neutral"]["median_seconds"] == pytest.approx(6.037870141)


@pytest.mark.parametrize("duplicate", [False, True])
def test_incomplete_or_duplicate_stages_are_not_silently_summed(duplicate):
    result = json.loads((RESULTS / "results.json").read_text())
    stage = next(r for r in result["timings"] if r["stage"] == "warm_sae_readout")
    if duplicate:
        result["timings"].append(copy.deepcopy(stage))
    else:
        result["timings"].remove(stage)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        module.summarize(result)
