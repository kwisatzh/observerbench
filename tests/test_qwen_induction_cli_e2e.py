"""End-to-end test for the inference-free Qwen effect-prediction CLI.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from observerbench import FiniteEffectPrediction, write_effect_predictions
from observerbench.core import write_json
from observerbench.tasks.qwen_induction.artifacts import (
    load_phase09_config,
    write_effect_artifacts,
    write_frozen_design_artifacts,
    write_preselection_artifacts,
)
from observerbench.tasks.qwen_induction.design import (
    SequenceBankCounts,
    build_mask_design,
    build_sequence_design,
)
from observerbench.tasks.qwen_induction.effect_task import (
    QWEN_INDUCTION_EFFECT_TASK_NAME,
    qwen_induction_additive_baseline_card,
    qwen_induction_effect_task_version,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = (
    ROOT / "configs/revision/phase09/qwen2_5_7b_induction_full.json"
)
HEAD_LABELS = tuple(f"L{4 + 2 * index}H{index}" for index in range(8))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _effect(mask_bits: str) -> float:
    return sum(
        0.02 * (index + 1) * int(bit)
        for index, bit in enumerate(mask_bits)
    )


def _write_checked_fixture(root: Path) -> None:
    config = deepcopy(load_phase09_config(FULL_CONFIG, require_scientific=True))
    config["token_pool"]["per_split_size"] = 64
    for bank in config["sequence_design"]["prompts_per_family"]:
        config["sequence_design"]["prompts_per_family"][bank] = 1

    counts = SequenceBankCounts(
        reference=1,
        discovery=1,
        head_fit=1,
        head_test=1,
        calibration=1,
        locked_test=1,
    )
    sequence = build_sequence_design(
        range(10_000, 11_000),
        bank_counts=counts,
        per_split_size=64,
        seed=config["sequence_design"]["seed"],
    )
    masks = build_mask_design(
        HEAD_LABELS,
        seed=config["mask_design"]["seed"],
    )
    selected_heads = [
        {
            "component_index": index,
            "head_label": label,
            "layer": 4 + 2 * index,
            "head": index,
            "kv_group": index % 4,
        }
        for index, label in enumerate(HEAD_LABELS)
    ]

    write_preselection_artifacts(
        sequence,
        config,
        root,
        exact_scientific_config=False,
    )
    proof_dir = root / "gate_proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    discovery_gate = proof_dir / "discovery_gate.json"
    confirmation_gate = proof_dir / "confirmation_gate.json"
    write_json(discovery_gate, {"passed": True})
    write_json(confirmation_gate, {"passed": True})
    write_frozen_design_artifacts(
        masks,
        selected_heads,
        config,
        root,
        all_design_gates_pass=True,
        gate_artifacts={
            "discovery_gate": discovery_gate,
            "confirmation_gate": confirmation_gate,
        },
        exact_scientific_config=False,
    )

    prompts = _read_csv(root / "design/prompts.csv")
    masks_by_split = {
        "train": _read_csv(root / "design/calibration_masks.csv"),
        "test": _read_csv(root / "design/test_masks.csv"),
    }
    measurement_rows = []
    for prompt_index, prompt in enumerate(prompts):
        clean_margin = 3.0 + 0.01 * prompt_index
        for mask in masks_by_split[prompt["split"]]:
            effect = _effect(mask["mask_bits"])
            measurement_rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "mask_id": mask["mask_id"],
                    "mask_bits": mask["mask_bits"],
                    "clean_margin": clean_margin,
                    "ablated_margin": clean_margin - effect,
                    "drop_from_clean": effect,
                }
            )
    write_effect_artifacts(
        measurement_rows,
        config,
        root,
        shard_size=512,
        exact_scientific_config=False,
    )


def test_qwen_evaluate_effect_csv_cli_end_to_end(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_checked_fixture(artifacts)

    predictions = tmp_path / "submission.csv"
    test_masks = _read_csv(artifacts / "design/test_masks.csv")
    write_effect_predictions(
        predictions,
        [
            FiniteEffectPrediction(
                query_id=row["mask_id"],
                predicted_effect=_effect(row["mask_bits"]),
            )
            for row in test_masks
        ],
    )
    observer_card = tmp_path / "observer_card.json"
    write_json(
        observer_card,
        qwen_induction_additive_baseline_card().to_dict(),
    )

    version = qwen_induction_effect_task_version(16)
    task_id = f"{QWEN_INDUCTION_EFFECT_TASK_NAME}@{version}"
    outdir = tmp_path / "evaluation"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "observerbench.cli",
            "evaluate-effect-csv",
            task_id,
            "--artifacts-root",
            str(artifacts),
            "--predictions",
            str(predictions),
            "--observer-card",
            str(observer_card),
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(outdir / "effect_evaluation.json")
    evaluation = json.loads(
        (outdir / "effect_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["task_name"] == QWEN_INDUCTION_EFFECT_TASK_NAME
    assert evaluation["task_version"] == version
    assert evaluation["measurement_budget"] == 16
    assert evaluation["n_queries"] == 128
    assert evaluation["metrics"] == pytest.approx(
        {
            "mae": 0.0,
            "rmse": 0.0,
            "mean_error": 0.0,
            "max_absolute_error": 0.0,
        },
        abs=1e-12,
    )
    assert {
        path.name for path in outdir.iterdir()
    } == {
        "effect_predictions.csv",
        "effect_evaluation.json",
        "task_card.json",
        "observer_card.json",
    }
