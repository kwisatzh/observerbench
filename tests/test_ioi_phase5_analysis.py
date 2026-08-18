"""Tests for confirmatory IOI fitting and target-matching selection.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.provenance import file_sha256
from observerbench.tasks.ioi import phase5_analysis
from observerbench.tasks.ioi.phase5_analysis import (
    IOIPhase5AnalysisConfig,
    IOIPhase5EvaluationConfig,
    _choose,
    _contrast_table,
    _exact_mean_effect_oracle,
    _two_way_draws,
    _validate_effect_manifest,
    _validate_split_cartesian,
    evaluate_phase5_observers,
    fit_phase5_observers,
)


def test_selection_tie_breaks_by_size_then_stable_id() -> None:
    candidates = pd.DataFrame(
        {
            "mask_id": ["m2", "m1", "m0"],
            "n_heads": [5, 3, 3],
        }
    )
    result = _choose(
        candidates,
        prediction=np.array([1.0, 1.0, 1.0]),
        actual=np.array([1.1, 1.2, 0.9]),
        target=1.0,
        head_cost_penalty=0.02,
        target_tolerance=0.25,
    )
    assert result["selected_mask_id"] == "m0"
    assert result["fixed_action_target_loss"] == pytest.approx(0.1)
    assert "oracle_regret" not in result
    assert "cost_aware_regret" not in result


def test_two_way_bootstrap_resamples_both_axes_deterministically() -> None:
    values = pd.DataFrame(
        [
            {"prompt_id": prompt, "pool_id": pool, "value": value}
            for prompt, row in (("p0", (1.0, 2.0)), ("p1", (3.0, 4.0)))
            for pool, value in zip(("a", "b"), row)
        ]
    )
    first = _two_way_draws(values, repeats=20, seed=3)
    second = _two_way_draws(values, repeats=20, seed=3)
    assert np.array_equal(first, second)
    assert len(set(first)) > 1


def test_exact_mean_oracle_exposes_mean_estimand_action_gap() -> None:
    candidates = pd.DataFrame(
        {
            "mask_id": ["mean_match", "risk_match"],
            "pool_id": ["pool_0", "pool_0"],
            "n_heads": [1, 2],
        }
    )
    effects = pd.DataFrame(
        [
            {"prompt_id": "p0", "mask_id": "mean_match", "drop_from_clean": 0.0},
            {"prompt_id": "p1", "mask_id": "mean_match", "drop_from_clean": 2.0},
            {"prompt_id": "p0", "mask_id": "risk_match", "drop_from_clean": 0.8},
            {"prompt_id": "p1", "mask_id": "risk_match", "drop_from_clean": 0.8},
        ]
    )
    best_fixed = pd.DataFrame(
        {
            "pool_id": ["pool_0"],
            "target": [1.0],
            "best_fixed_action_mask_id": ["risk_match"],
            "best_fixed_action_head_count": [2],
            "best_fixed_action_mean_target_loss": [0.2],
        }
    )

    result = _exact_mean_effect_oracle(
        candidates,
        effects,
        best_fixed,
        targets=(1.0,),
    ).iloc[0]

    assert result["exact_mean_mask_id"] == "mean_match"
    assert result["exact_mean_effect"] == pytest.approx(1.0)
    assert result["exact_mean_to_target_error"] == pytest.approx(0.0)
    assert result["exact_mean_fixed_action_loss"] == pytest.approx(1.0)
    assert result["same_mask_as_best_fixed"] == np.False_
    assert result["fixed_action_loss_gap"] == pytest.approx(0.8)


def test_fixed_action_loss_is_primary_denominator_with_name_pair_sensitivity() -> None:
    rows = []
    losses = {
        "additive_head": 1.0,
        "count_additive": 0.8,
        "count_plus_PE_bin4": 0.6,
        "count_plus_all_bin4": 0.5,
    }
    for model, loss in losses.items():
        for prompt in ("p0", "p1", "p2", "p3"):
            for pool in ("a", "b"):
                rows.append(
                    {
                        "model": model,
                        "measurement_budget": 160,
                        "prompt_id": prompt,
                        "pool_id": pool,
                        "target": 1.0,
                        "fixed_action_target_loss": loss,
                        "within_tolerance": int(loss <= 0.6),
                        "cost_aware_fixed_action_loss": loss + 0.1,
                    }
                )
    selection = pd.DataFrame(rows)
    size_selection = selection.assign(size_match_cell="n4")
    clusters = pd.DataFrame(
        {
            "prompt_id": ["p0", "p1", "p2", "p3"],
            "ordered_name_pair_id": ["a::b", "b::a", "c::d", "d::c"],
            "unordered_name_pair_id": ["a::b", "a::b", "c::d", "c::d"],
        }
    )
    contrasts = _contrast_table(
        selection,
        size_selection,
        clusters,
        primary_budget=160,
        evaluation_config=IOIPhase5EvaluationConfig(bootstrap_repeats=40),
    )
    row = contrasts[
        (contrasts["reference"] == "additive_head")
        & (contrasts["candidate"] == "count_plus_all_bin4")
    ].iloc[0]
    assert row["fixed_action_target_loss_reduction_mean"] == pytest.approx(0.5)
    assert row["fixed_action_target_loss_reduction_fraction"] == pytest.approx(0.5)
    assert row["fixed_action_target_loss_reduction_q025"] > 0.0
    assert row[
        "fixed_action_target_loss_reduction_ordered_name_pair_q025"
    ] > 0.0
    assert row[
        "fixed_action_target_loss_reduction_unordered_name_pair_q025"
    ] > 0.0


def _mask_row(mask_id: str, bits: str, **extra: object) -> dict[str, object]:
    mask = np.fromiter((int(bit) for bit in bits), dtype=int)
    return {
        "mask_id": mask_id,
        "mask_bits": bits,
        "n_heads": int(mask.sum()),
        "n_P": int(mask[:3].sum()),
        "n_B": int(mask[3:11].sum()),
        "n_E": int(mask[11:].sum()),
        **extra,
    }


def _write_synthetic_phase5_fixture(root: Path) -> tuple[Path, Path]:
    design = root / "design"
    effects = root / "effects"
    design.mkdir()

    prompts = pd.DataFrame(
        [
            {
                "prompt_id": f"prompt_{split}",
                "split": split,
                "template_id": "synthetic_template",
                "structure": "ABBA",
                "io_name": f"IO{index}",
                "s_name": f"S{index}",
                "prompt": f"synthetic prompt {split}",
            }
            for index, split in enumerate(("train", "validation", "test"))
        ]
    )
    calibration_bits = [
        "".join("1" if position == head else "0" for position in range(13))
        for head in range(13)
    ]
    calibration_bits.extend(
        [
            "1100000000000",
            "1010000000000",
            "1001000000000",
            "1000000000010",
            "0100100000000",
            "0010000000001",
            "0001100000000",
        ]
    )
    calibration = [
        _mask_row(
            f"cal_{index:02d}",
            bits,
            bank="calibration",
            pool_id="",
            measurement_order=index,
            size_match_cell=f"n{bits.count('1')}",
        )
        for index, bits in enumerate(calibration_bits)
    ]
    excluded = set(calibration_bits)
    candidate_bits = [
        format(value, "013b")
        for value in range(1, 2**13)
        if 3 <= format(value, "013b").count("1") <= 6
        and format(value, "013b") not in excluded
    ][:8]
    candidates = [
        _mask_row(
            f"candidate_{index:02d}",
            bits,
            bank="candidate",
            pool_id=f"pool_{index // 4}",
            measurement_order=-1,
            size_match_cell=f"n{bits.count('1')}",
        )
        for index, bits in enumerate(candidate_bits)
    ]
    masks = pd.DataFrame([*calibration, *candidates])
    prompts.to_csv(design / "prompts.csv", index=False)
    masks.to_csv(design / "masks.csv", index=False)
    manifest = {
        "schema": "observerbench.ioi_phase5_design_manifest.v1",
        "status": "frozen_before_outcomes",
        "design_id": "synthetic-phase5-design",
        "source_hashes": {
            name: file_sha256(design / name)
            for name in ("prompts.csv", "masks.csv")
        },
    }
    (design / "design_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weights = np.linspace(0.04, 0.16, 13)
    for prompt_index, split in enumerate(("train", "validation", "test")):
        split_dir = effects / "shards" / split
        split_dir.mkdir(parents=True)
        rows = []
        for mask in masks.itertuples(index=False):
            bits = np.fromiter((int(bit) for bit in str(mask.mask_bits)), dtype=float)
            interaction = 0.04 * float(mask.n_P * mask.n_E)
            drop = float(bits @ weights + interaction + 0.01 * prompt_index)
            rows.append(
                {
                    "split": split,
                    "prompt_id": f"prompt_{split}",
                    "mask_id": str(mask.mask_id),
                    "mask_bits": str(mask.mask_bits),
                    "bank": str(mask.bank),
                    "pool_id": str(mask.pool_id),
                    "clean_ld": 3.0,
                    "ablated_ld": 3.0 - drop,
                    "drop_from_clean": drop,
                }
            )
        pd.DataFrame(rows).to_csv(split_dir / "effects_0000_0028.csv", index=False)
    (effects / "template_head_means.npz").write_bytes(b"synthetic-cache")
    for split in ("train", "validation", "test"):
        (effects / f"clean_scores_{split}.csv").write_text(
            "prompt_id,clean_ld\n",
            encoding="utf-8",
        )
    artifact_paths = [
        effects / "template_head_means.npz",
        *[effects / f"clean_scores_{split}.csv" for split in ("train", "validation", "test")],
        *[
            effects / "shards" / split / "effects_0000_0028.csv"
            for split in ("train", "validation", "test")
        ],
    ]
    effect_manifest = {
        "schema": "observerbench.ioi_effect_run.v1",
        "status": "complete_unopened_confirmatory_outcomes",
        "design_manifest_sha256": file_sha256(design / "design_manifest.json"),
        "model": {
            "requested_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        },
        "counts": {
            "reference_prompts": 0,
            "outcome_prompts": 3,
            "masks": 28,
            "effect_cells": 84,
            "shards": 3,
        },
        "artifacts": {
            path.relative_to(effects).as_posix(): file_sha256(path)
            for path in artifact_paths
        },
    }
    (effects / "effect_manifest.json").write_text(
        json.dumps(effect_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return design, effects


def test_protocol_drives_tolerance_and_success_thresholds() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "configs/revision/ioi_phase05_confirmatory_v2.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = IOIPhase5AnalysisConfig.from_protocol(protocol, bootstrap_repeats=17)
    assert config.target_tolerance == 0.25
    assert config.regret_reduction_fraction_min == 0.15
    assert config.within_tolerance_improvement_min == 0.05
    assert config.bootstrap_repeats == 17
    evaluation_path = (
        Path(__file__).resolve().parents[1]
        / "configs/revision/ioi_phase05_evaluation_v3.json"
    )
    evaluation_protocol = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation = IOIPhase5EvaluationConfig.from_protocol(
        evaluation_protocol,
        bootstrap_repeats=19,
    )
    assert evaluation.primary_budget == 160
    assert evaluation.fixed_action_loss_reduction_fraction_min == 0.15
    assert evaluation.bootstrap_repeats == 19


def test_cartesian_validator_rejects_missing_cells_and_changed_mask_mapping(
    tmp_path: Path,
) -> None:
    design, effects = _write_synthetic_phase5_fixture(tmp_path)
    prompts = pd.read_csv(design / "prompts.csv", dtype={"prompt_id": str})
    masks = pd.read_csv(
        design / "masks.csv",
        dtype={"mask_id": str, "mask_bits": str},
    )
    train = pd.read_csv(
        effects / "shards/train/effects_0000_0028.csv",
        dtype={"prompt_id": str, "mask_id": str, "mask_bits": str},
    )
    with pytest.raises(ValueError, match="exact mask set"):
        _validate_split_cartesian(
            train.iloc[:-1].copy(),
            prompts,
            masks,
            split="train",
        )
    for column, value in (
        ("mask_bits", "1" * 13),
        ("bank", "wrong_bank"),
        ("pool_id", "wrong_pool"),
    ):
        changed = train.copy()
        changed.loc[changed.index[0], column] = value
        with pytest.raises(ValueError, match="mapping differs"):
            _validate_split_cartesian(changed, prompts, masks, split="train")


def test_effect_manifest_counts_must_match_frozen_design(tmp_path: Path) -> None:
    design, effects = _write_synthetic_phase5_fixture(tmp_path)
    manifest_path = effects / "effect_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["effect_cells"] -= 1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="effect_cells differs from the frozen design"):
        _validate_effect_manifest(effects, design)


def test_fit_freezes_train_only_before_evaluation_reads_heldout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, effects = _write_synthetic_phase5_fixture(tmp_path)
    fit_dir = tmp_path / "fit"
    evaluation_dir = tmp_path / "evaluation"
    config = IOIPhase5AnalysisConfig(
        budgets=(20,),
        targets=(0.5,),
        bootstrap_repeats=20,
    )
    evaluation_config = IOIPhase5EvaluationConfig(
        primary_budget=20,
        targets=(0.5,),
        bootstrap_repeats=20,
    )
    calls: list[str] = []
    original = phase5_analysis._load_split_effects

    def tracked_load(
        effects_dir: str | Path,
        split: str,
    ) -> tuple[pd.DataFrame, tuple[Path, ...]]:
        calls.append(split)
        if split in {"validation", "test"}:
            assert (fit_dir / "fit_manifest.json").is_file()
            frozen = json.loads(
                (fit_dir / "fit_manifest.json").read_text(encoding="utf-8")
            )
            prediction_path = fit_dir / frozen["frozen_prediction"]["relative_path"]
            assert file_sha256(prediction_path) == frozen["frozen_prediction"]["sha256"]
        return original(effects_dir, split)

    monkeypatch.setattr(phase5_analysis, "_load_split_effects", tracked_load)
    fit_phase5_observers(design, effects, fit_dir, config=config)
    assert calls == ["train"]
    fit_manifest = json.loads(
        (fit_dir / "fit_manifest.json").read_text(encoding="utf-8")
    )
    assert fit_manifest["schema"] == "observerbench.ioi_phase5_fit.v2"
    assert fit_manifest["frozen_prediction"]["sha256"] == file_sha256(
        fit_dir / "candidate_predictions.csv"
    )

    evaluate_phase5_observers(
        design,
        effects,
        fit_dir,
        evaluation_dir,
        config=config,
        evaluation_config=evaluation_config,
    )
    assert calls == ["train", "validation", "test"]
    assert (evaluation_dir / "evaluation_manifest.json").is_file()
    assert (evaluation_dir / "exact_mean_effect_oracle.csv").is_file()
    evaluation_manifest = json.loads(
        (evaluation_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation_manifest["outputs"]["exact_mean_effect_oracle.csv"] == file_sha256(
        evaluation_dir / "exact_mean_effect_oracle.csv"
    )

    calls.clear()
    manifest_path = fit_dir / "fit_manifest.json"
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_sources = json.loads(json.dumps(frozen_manifest))
    first_source = next(iter(changed_sources["train_effect_sources"]))
    changed_sources["train_effect_sources"][first_source] = "0" * 64
    manifest_path.write_text(
        json.dumps(changed_sources, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed train shard hashes"):
        evaluate_phase5_observers(
            design,
            effects,
            fit_dir,
            tmp_path / "changed_train_source_evaluation",
            config=config,
            evaluation_config=evaluation_config,
        )
    assert calls == []

    manifest_path.write_text(
        json.dumps(frozen_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete_manifest = dict(frozen_manifest)
    incomplete_manifest.pop("frozen_prediction")
    manifest_path.write_text(
        json.dumps(incomplete_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lacks the required frozen_prediction"):
        evaluate_phase5_observers(
            design,
            effects,
            fit_dir,
            tmp_path / "missing_hash_evaluation",
            config=config,
            evaluation_config=evaluation_config,
        )
    assert calls == []

    manifest_path.write_text(
        json.dumps(frozen_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_path = fit_dir / "candidate_predictions.csv"
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash changed after fit freeze"):
        evaluate_phase5_observers(
            design,
            effects,
            fit_dir,
            tmp_path / "tampered_evaluation",
            config=config,
            evaluation_config=evaluation_config,
        )
    assert calls == []
