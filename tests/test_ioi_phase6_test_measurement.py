"""Tests for the sealed Phase-6 held-out measurement boundary.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.ioi.phase6_confirmatory import PHASE6_STATUS
from observerbench.tasks.ioi.phase6_test_measurement import (
    Phase6TestInputs,
    Phase6TestMeasurementConfig,
    _clean_output_rows,
    _mask_shard_spans,
    build_phase6_test_measurement_spec,
    load_phase6_test_inputs,
    validate_candidate_effect_shard,
    validate_clean_test_scores,
)


def _prompts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_id": ["prompt_0", "prompt_1"],
            "split": "test",
            "template_id": ["template_a", "template_b"],
            "structure": ["ABBA", "BABA"],
            "unordered_name_pair_id": "pair_0",
            "pair_orientation": ["a_to_b", "b_to_a"],
            "io_name": ["Alice", "Bob"],
            "s_name": ["Bob", "Alice"],
            "prompt": ["first", "second"],
        }
    )


def _masks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mask_id": ["mask_0", "mask_1"],
            "mask_bits": ["1000000000000", "0100000000000"],
            "bank": "candidate",
            "pool_id": "pool_0",
        }
    )


def _effect_rows(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = _masks()
    for prompt in _prompts().itertuples(index=False):
        clean_ld = float(clean.loc[clean["prompt_id"] == prompt.prompt_id, "clean_ld"].iloc[0])
        for index, mask in enumerate(masks.itertuples(index=False), start=1):
            ablated = clean_ld - 0.1 * index
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "split": "test",
                    "mask_id": mask.mask_id,
                    "mask_bits": mask.mask_bits,
                    "bank": mask.bank,
                    "pool_id": mask.pool_id,
                    "clean_ld": clean_ld,
                    "ablated_ld": ablated,
                    "drop_from_clean": clean_ld - ablated,
                }
            )
    return pd.DataFrame(rows)


def test_invalid_seal_stops_before_any_source_is_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import observerbench.tasks.ioi.phase6_test_measurement as measurement

    def reject(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("seal rejected")

    monkeypatch.setattr(measurement, "validate_phase6_prediction_action_seal", reject)
    with pytest.raises(RuntimeError, match="seal rejected"):
        load_phase6_test_inputs(
            tmp_path / "missing_design",
            tmp_path / "missing_calibration",
            tmp_path / "missing_freeze",
            protocol_path=tmp_path / "missing_protocol.json",
            config=Phase6TestMeasurementConfig(),
        )


def test_config_pins_the_model_revision() -> None:
    with pytest.raises(ValueError, match="pinned"):
        Phase6TestMeasurementConfig(model_revision="moving-main")
    with pytest.raises(ValueError, match="GPT-2-small"):
        Phase6TestMeasurementConfig(model_name="other-model")
    with pytest.raises(ValueError, match="16-mask"):
        Phase6TestMeasurementConfig(mask_shard_size=32)


def test_deterministic_shard_spans_cover_every_mask_once() -> None:
    spans = _mask_shard_spans(1536, 16)
    assert len(spans) == 96
    assert spans[0] == (0, 16)
    assert spans[-1] == (1520, 1536)
    covered = [index for start, stop in spans for index in range(start, stop)]
    assert covered == list(range(1536))


def test_clean_and_candidate_validators_require_exact_cartesian_data() -> None:
    prompts = _prompts()
    clean = _clean_output_rows(prompts, [2.0, 1.5])
    validate_clean_test_scores(clean, prompts)
    effects = _effect_rows(clean)
    validate_candidate_effect_shard(
        effects,
        prompts=prompts,
        masks=_masks(),
        clean_scores=clean,
    )

    contaminated = effects.copy()
    contaminated.loc[0, "bank"] = "calibration"
    with pytest.raises(ValueError, match="non-candidate"):
        validate_candidate_effect_shard(
            contaminated,
            prompts=prompts,
            masks=_masks(),
            clean_scores=clean,
        )
    missing = effects.iloc[:-1].copy()
    with pytest.raises(ValueError, match="expected 4 effect cells"):
        validate_candidate_effect_shard(
            missing,
            prompts=prompts,
            masks=_masks(),
            clean_scores=clean,
        )


def test_run_spec_freezes_complete_surface_and_forbids_adaptation() -> None:
    prompts = pd.concat([_prompts()] * 256, ignore_index=True)
    prompts["prompt_id"] = [f"prompt_{index:03d}" for index in range(512)]
    masks = pd.concat([_masks()] * 768, ignore_index=True)
    masks["mask_id"] = [f"mask_{index:04d}" for index in range(1536)]
    masks["pool_id"] = [f"pool_{index // 32:02d}" for index in range(1536)]
    inputs = Phase6TestInputs(
        test_prompts=prompts,
        candidate_masks=masks,
        template_head_means=np.zeros((8, 13, 64), dtype=np.float32),
        templates=tuple(f"template_{index}" for index in range(8)),
        protocol={},
        design_manifest={"design_id": "design_0"},
        calibration_manifest={},
        freeze_manifest={},
    )
    spec = build_phase6_test_measurement_spec(
        inputs,
        config=Phase6TestMeasurementConfig(),
        source_hashes={"seal": "abc"},
    )
    assert spec["scientific_status"] == PHASE6_STATUS
    assert spec["counts"]["candidate_effect_cells"] == 512 * 1536
    assert spec["counts"]["mask_shards"] == 96
    assert all(spec["forbidden_operations"].values())
    assert spec["policy_hierarchy"] == {
        "primary": "target_loss",
        "secondary": "cost_aware",
        "secondary_cannot_rescue_primary": True,
    }
    assert spec["accessed_prompt_splits"] == ["test"]
    assert spec["accessed_mask_banks"] == ["candidate"]
