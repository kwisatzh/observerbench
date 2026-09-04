"""Tests of the submitted-effect to action-score composition.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.cli import main
from observerbench.core import write_json
from observerbench.effect_prediction import EffectObserverCard, FiniteEffectPrediction, write_effect_predictions
from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.decision_submission import (
    DEFAULT_PACK, NOOP, PACK_SCHEMA, PREDICTION_TASK_ID, TASK_ID, TARGETS,
    evaluate_ioi_decision_csv, load_decision_pack, select_decision_actions,
)


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    queries = pd.DataFrame({"query_id": ["q0", "q1", "q2", "q3"],
        "mask_bits": ["1".zfill(13), "10".zfill(13), "100".zfill(13), "1000".zfill(13)],
        "pool_id": ["p0", "p0", "p1", "p1"], "n_heads": [1] * 4})
    queries.to_csv(root / "queries.csv", index=False)
    pd.DataFrame({"measurement_id": [f"m{i}" for i in range(160)],
                  "mask_bits": ["0" * 13] * 160, "observed_effect": [0.0] * 160}).to_csv(root / "calibration_measurements.csv", index=False)
    pd.DataFrame({"prompt_id": ["a", "b", "c", "d"], "cluster_id": ["ab", "ab", "cd", "cd"]}).to_csv(root / "prompt_clusters.csv", index=False)
    np.savez_compressed(root / "responses.npz", responses=np.array([[0, .8, 0, .8], [2, .8, 2, .8]] * 2),
                        mask_ids=np.array(queries.query_id.tolist()), prompt_ids=np.array(["a", "b", "c", "d"]))
    write_effect_predictions(root / "submission.csv", [FiniteEffectPrediction(q, p) for q, p in zip(queries.query_id, [1, .8, 1, .8])])
    write_effect_predictions(root / "reference.csv", [FiniteEffectPrediction(q, p) for q, p in zip(queries.query_id, [-3, 1, -3, 1])])
    observer = EffectObserverCard(observer_name="test", observer_version="1", observer_family="test",
        access_regime="test", measurement_basis="test", fit_procedure="test", implementation="test")
    write_json(root / "observer.json", observer.to_dict())
    card = {"schema": PACK_SCHEMA, "task_id": TASK_ID, "prediction_task_id": PREDICTION_TASK_ID,
        "targets": list(TARGETS), "primary_target": 1.0, "include_noop": True, "participation_class": "open_replay",
        "selector": "absolute_mean_error_then_head_count_then_mask_id", "model": "fixture", "information_boundary": "fixture",
        "n_queries": 4, "n_prompts": 4, "pool_sizes": {"p0": 2, "p1": 2},
        "references": {"reference": {"predictions": "reference.csv"}}, "bootstrap_repeats": 20,
        "bootstrap_seed": 1, "uncertainty": "test", "scope": "test",
        "files": {p.name: file_sha256(p) for p in root.iterdir() if p.name not in {"observer.json", "submission.csv"}}}
    write_json(root / "task_card.json", card)
    return root


def evaluate(pack, tmp_path):
    return evaluate_ioi_decision_csv(TASK_ID, pack_root=pack, predictions_path=pack / "submission.csv",
                                     observer_card_path=pack / "observer.json", outdir=tmp_path / "out")


def test_exact_mean_can_choose_worse_action_and_reports_paired_loss(pack, tmp_path):
    result = evaluate(pack, tmp_path)
    assert result["prediction_metrics"]["mae"] == 0
    target = result["by_target"][1]
    assert target["action_loss"] == 1
    assert target["references"]["reference"]["action_loss"] == pytest.approx(.2)
    assert target["references"]["reference"]["paired_interval_95"] == pytest.approx([.8, .8])
    assert target["references"][NOOP]["submission_minus_reference"] == 0
    assert result["sealed_score"] is False
    assert (tmp_path / "out/selected_actions.csv").is_file()
    assert (tmp_path / "out/effect_evaluation.json").is_file()


def test_selection_is_order_invariant_and_noop_wins_exact_tie(pack):
    _, queries, _ = load_decision_pack(pack)
    predictions = {q: 1.0 for q in queries.query_id}
    first = select_decision_actions(queries, predictions)
    second = select_decision_actions(queries.iloc[::-1], predictions)
    assert first == second
    assert first[("p0", .5)]["selected_mask_id"] == NOOP
    assert first[("p0", 1.0)]["selected_mask_id"] == "q0"


@pytest.mark.parametrize("bad", [{"q0": 1}, {"q0": float("nan"), "q1": 1, "q2": 1, "q3": 1}])
def test_incomplete_or_nonfinite_predictions_fail(pack, bad):
    _, queries, _ = load_decision_pack(pack)
    with pytest.raises(ValueError, match="every query"):
        select_decision_actions(queries, bad)


def test_duplicate_submission_fails(pack, tmp_path):
    table = pd.read_csv(pack / "submission.csv")
    pd.concat([table, table.iloc[:1]]).to_csv(pack / "submission.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pack, tmp_path)


def test_wrong_estimand_fails_before_scoring(pack, tmp_path):
    observer = json.loads((pack / "observer.json").read_text())
    observer["metadata"]["prediction_estimand"] = "expected_action_loss"
    write_json(pack / "observer.json", observer)
    with pytest.raises(ValueError, match="estimand"):
        evaluate(pack, tmp_path)


@pytest.mark.parametrize("change", ["target", "hash", "overlap", "missing_response", "unhashed_reference"])
def test_broken_contract_rejected(pack, tmp_path, change):
    card = json.loads((pack / "task_card.json").read_text())
    if change == "target":
        card["targets"] = [1.0]
    elif change == "hash":
        card["files"]["responses.npz"] = "wrong"
    elif change == "overlap":
        calibration = pd.read_csv(pack / "calibration_measurements.csv", dtype={"mask_bits": str})
        calibration.loc[0, "measurement_id"] = "q0"
        calibration.to_csv(pack / "calibration_measurements.csv", index=False)
        card["files"]["calibration_measurements.csv"] = file_sha256(pack / "calibration_measurements.csv")
    elif change == "missing_response":
        with np.load(pack / "responses.npz") as data:
            modified = {k: data[k].copy() for k in data.files}
        modified["responses"][0, 0] = np.nan
        np.savez_compressed(pack / "responses.npz", **modified)
        card["files"]["responses.npz"] = file_sha256(pack / "responses.npz")
    else:
        del card["files"]["reference.csv"]
    write_json(pack / "task_card.json", card)
    with pytest.raises(ValueError):
        evaluate(pack, tmp_path)


def test_cli_returns_the_scorecard(pack, tmp_path, capsys):
    result = main(["evaluate-ioi-decision-csv", TASK_ID, "--pack-dir", str(pack),
                   "--predictions", str(pack / "submission.csv"), "--observer-card", str(pack / "observer.json"),
                   "--outdir", str(tmp_path / "cli")])
    assert result == 0
    assert "Target 1: action loss 1.0000" in capsys.readouterr().out


def test_released_pack_and_attribution_table_work_without_original_archives(tmp_path):
    card, queries, calibration = load_decision_pack(DEFAULT_PACK)
    assert len(queries) == 320
    assert len(calibration) == 160
    assert card["n_source_clusters"] == 107
    result = evaluate_ioi_decision_csv(TASK_ID, pack_root=DEFAULT_PACK,
        predictions_path=DEFAULT_PACK / "submissions/atp_calibrated.csv",
        observer_card_path=DEFAULT_PACK / "submissions/atp_calibrated.json", outdir=tmp_path)
    assert result["prediction_metrics"]["mae"] == pytest.approx(.6315642766701114)
    assert result["by_target"][1]["action_loss"] == pytest.approx(.9995, abs=.0001)
    assert "atp_raw" in result["by_target"][1]["references"]
    assert result["scorer_source_hashes"]


def test_wrong_task_rejected(pack, tmp_path):
    with pytest.raises(ValueError, match="IOI decision task"):
        evaluate_ioi_decision_csv("wrong-task", pack_root=pack,
            predictions_path=pack / "submission.csv", observer_card_path=pack / "observer.json", outdir=tmp_path)
