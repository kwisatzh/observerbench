"""Sealed held-out measurement for the prospective Phase-6 IOI study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This stage opens the frozen test prompts and candidate masks only after the
prediction/action seal passes validation.  It measures every candidate rather
than choosing, fitting, or filtering anything from held-out outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    _model_tokens,
    _score_clean,
    _score_mask_shard,
    validate_effect_rows,
)
from observerbench.tasks.ioi.phase6_confirmatory import (
    DESIGN_SCHEMA,
    DESIGN_STATUS,
    PHASE6_STATUS,
    load_phase6_protocol,
)
from observerbench.tasks.ioi.phase6_freeze import (
    FREEZE_SCHEMA,
    FREEZE_STATUS,
    validate_phase6_prediction_action_seal,
)
from observerbench.tasks.ioi.phase6_measurement import (
    CALIBRATION_SCHEMA,
    CALIBRATION_STATUS,
)


TEST_MEASUREMENT_SPEC_SCHEMA = "observerbench.ioi_phase06_test_measurement_spec.v1"
TEST_MEASUREMENT_SPEC_STATUS = "sealed_sources_verified_before_test_measurement"
TEST_MEASUREMENT_PROGRESS_SCHEMA = "observerbench.ioi_phase06_test_measurement_progress.v1"
TEST_MEASUREMENT_SCHEMA = "observerbench.ioi_phase06_test_measurement_run.v1"
TEST_MEASUREMENT_STATUS = "phase6_all_test_candidate_outcomes_measured"

EXPECTED_DESIGN_ARTIFACTS = {
    "templates.csv",
    "names.csv",
    "pair_clusters.csv",
    "prompts.csv",
    "calibration_masks.csv",
    "candidate_masks.csv",
    "masks.csv",
    "leakage_audit.json",
}
EXPECTED_FREEZE_ARTIFACTS = {
    "candidate_predictions.csv",
    "observer_coefficients.csv",
    "fit_diagnostics.csv",
    "fixed_actions.csv",
}
POLICY_HIERARCHY = {
    "primary": "target_loss",
    "secondary": "cost_aware",
    "secondary_cannot_rescue_primary": True,
}


@dataclass(frozen=True)
class Phase6TestMeasurementConfig:
    """Compute-only settings; none changes prompts, masks, fits, or actions."""

    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    pair_batch_size: int = 128
    clean_batch_size: int = 64
    mask_shard_size: int = 16
    seed: int = 26062

    def __post_init__(self) -> None:
        if self.model_name != "gpt2-small":
            raise ValueError("Phase 6 requires GPT-2-small")
        if self.model_revision != GPT2_SMALL_REVISION:
            raise ValueError("Phase 6 requires the pinned GPT-2-small revision")
        if min(self.pair_batch_size, self.clean_batch_size, self.mask_shard_size) <= 0:
            raise ValueError("batch and shard sizes must be positive")
        if self.mask_shard_size != 16:
            raise ValueError("Phase 6 uses deterministic 16-mask held-out shards")


@dataclass(frozen=True)
class Phase6TestInputs:
    """Verified held-out inputs and the sealed reference cache."""

    test_prompts: pd.DataFrame
    candidate_masks: pd.DataFrame
    template_head_means: np.ndarray
    templates: tuple[str, ...]
    protocol: Mapping[str, Any]
    design_manifest: Mapping[str, Any]
    calibration_manifest: Mapping[str, Any]
    freeze_manifest: Mapping[str, Any]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_hash_index(
    root: str | Path,
    index: Mapping[str, Any],
    *,
    label: str,
    exact_labels: set[str] | None = None,
) -> None:
    """Verify a root-relative, traversal-free artifact hash index."""

    if not isinstance(index, Mapping) or not index:
        raise ValueError(f"{label} has no artifact hash index")
    labels = set(map(str, index))
    if exact_labels is not None and labels != exact_labels:
        raise ValueError(f"{label} does not index the exact expected artifacts")
    base = Path(root)
    for raw_name, expected in index.items():
        name = str(raw_name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} contains a non-portable artifact path")
        path = base / relative
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"{label} artifact changed: {name}")


def _expected_calibration_labels(
    calibration_dir: str | Path,
    manifest: Mapping[str, Any],
) -> set[str]:
    root = Path(calibration_dir)
    shards = tuple(sorted((root / "shards" / "train").glob("effects_*.csv")))
    if len(shards) != int(manifest.get("counts", {}).get("shards", -1)):
        raise ValueError("calibration shard count differs from its manifest")
    return {
        "template_head_means.npz",
        "clean_scores_train.csv",
        *(path.relative_to(root).as_posix() for path in shards),
    }


def _validate_test_design_rows(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = prompts.loc[prompts["split"].astype(str) == "test"].copy().reset_index(drop=True)
    candidates = candidates.copy()
    candidates["mask_id"] = candidates["mask_id"].astype(str)
    candidates["mask_bits"] = candidates["mask_bits"].astype(str).str.zfill(13)
    if len(test) != 512 or test["prompt_id"].astype(str).nunique() != 512:
        raise ValueError("Phase-6 test bank must contain 512 unique prompts")
    if set(test["split"].astype(str)) != {"test"}:
        raise ValueError("held-out measurement exposed a non-test prompt")
    if test["prompt"].astype(str).duplicated().any():
        raise ValueError("Phase-6 test prompt text must be unique")
    if len(candidates) != 1536 or candidates["mask_id"].nunique() != 1536:
        raise ValueError("Phase-6 candidate bank must contain 1,536 unique masks")
    if candidates["mask_bits"].nunique() != 1536:
        raise ValueError("Phase-6 candidate mask bit strings must be unique")
    if not candidates["mask_bits"].str.fullmatch(r"[01]{13}").all():
        raise ValueError("every Phase-6 candidate mask must contain 13 binary digits")
    if set(candidates["bank"].astype(str)) != {"candidate"}:
        raise ValueError("held-out measurement exposed a non-candidate mask")
    if candidates["pool_id"].astype(str).nunique() != 48:
        raise ValueError("Phase-6 candidate bank must contain 48 pools")
    if not candidates.groupby("pool_id").size().eq(32).all():
        raise ValueError("every Phase-6 candidate pool must contain 32 masks")
    candidates = candidates.sort_values(
        ["pool_index", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)
    return test, candidates


def load_phase6_test_inputs(
    design_dir: str | Path,
    calibration_dir: str | Path,
    freeze_dir: str | Path,
    *,
    protocol_path: str | Path,
    config: Phase6TestMeasurementConfig,
) -> Phase6TestInputs:
    """Validate every source seal before exposing test prompts or candidates."""

    # This must remain the first operation: a runner may not inspect held-out
    # inputs unless the independent prediction/action seal itself validates.
    freeze_manifest = validate_phase6_prediction_action_seal(
        freeze_dir,
        design_dir=design_dir,
        calibration_dir=calibration_dir,
        protocol_path=protocol_path,
    )

    protocol = load_phase6_protocol(protocol_path)
    design_root = Path(design_dir)
    calibration_root = Path(calibration_dir)
    design_manifest_path = design_root / "design_manifest.json"
    calibration_manifest_path = calibration_root / "calibration_manifest.json"
    design_manifest = _read_json(design_manifest_path)
    calibration_manifest = _read_json(calibration_manifest_path)

    if design_manifest.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unexpected Phase-6 design manifest schema")
    if design_manifest.get("status") != DESIGN_STATUS:
        raise ValueError("Phase-6 design is not frozen")
    if design_manifest.get("scientific_status") != PHASE6_STATUS:
        raise ValueError("Phase-6 scientific status changed")
    if design_manifest.get("contains_model_outcomes") is not False:
        raise ValueError("Phase-6 design unexpectedly contains outcomes")
    if design_manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("protocol changed after design freeze")
    _verify_hash_index(
        design_root,
        design_manifest.get("artifact_hashes", {}),
        label="design manifest",
        exact_labels=EXPECTED_DESIGN_ARTIFACTS,
    )

    if calibration_manifest.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected Phase-6 calibration manifest schema")
    if calibration_manifest.get("status") != CALIBRATION_STATUS:
        raise ValueError("Phase-6 calibration stage is not complete")
    if calibration_manifest.get("design_manifest_sha256") != file_sha256(
        design_manifest_path
    ):
        raise ValueError("calibration run used a different design")
    if calibration_manifest.get("design_id") != design_manifest.get("design_id"):
        raise ValueError("calibration and design identifiers differ")
    if calibration_manifest.get("accessed_prompt_splits") != ["reference", "train"]:
        raise ValueError("calibration run accessed an unauthorized prompt split")
    if calibration_manifest.get("accessed_mask_banks") != ["calibration"]:
        raise ValueError("calibration run accessed an unauthorized mask bank")
    if int(calibration_manifest.get("test_prompt_forward_passes", -1)) != 0:
        raise ValueError("test prompts were opened before the action seal")
    if int(calibration_manifest.get("candidate_mask_forward_passes", -1)) != 0:
        raise ValueError("candidate masks were opened before the action seal")
    calibration_config = calibration_manifest.get("config", {})
    if calibration_config.get("model_name") != config.model_name:
        raise ValueError("test runner model differs from calibration")
    if calibration_config.get("model_revision") != config.model_revision:
        raise ValueError("test runner revision differs from calibration")
    calibration_labels = _expected_calibration_labels(
        calibration_root, calibration_manifest
    )
    _verify_hash_index(
        calibration_root,
        calibration_manifest.get("artifacts", {}),
        label="calibration manifest",
        exact_labels=calibration_labels,
    )

    if freeze_manifest.get("schema") != FREEZE_SCHEMA:
        raise ValueError("unexpected prediction/action seal schema")
    if freeze_manifest.get("status") != FREEZE_STATUS:
        raise ValueError("prediction/actions are not frozen")
    if freeze_manifest.get("scientific_status") != PHASE6_STATUS:
        raise ValueError("prediction/action scientific status changed")
    if freeze_manifest.get("design_id") != design_manifest.get("design_id"):
        raise ValueError("prediction/action seal used a different design")
    if freeze_manifest.get("source_seals", {}).get("design") != design_manifest.get(
        "artifact_hashes"
    ):
        raise ValueError("prediction/action seal does not contain the exact design seal")
    if freeze_manifest.get("source_seals", {}).get(
        "calibration"
    ) != calibration_manifest.get("artifacts"):
        raise ValueError("prediction/action seal does not contain the exact calibration seal")
    if set(map(str, freeze_manifest.get("artifact_hashes", {}))) != EXPECTED_FREEZE_ARTIFACTS:
        raise ValueError("prediction/action seal does not index the exact frozen artifacts")
    if freeze_manifest.get("accessed_prompt_splits") != ["reference", "train"]:
        raise ValueError("prediction/action fit accessed an unauthorized prompt split")
    if freeze_manifest.get("accessed_mask_banks_for_outcomes") != ["calibration"]:
        raise ValueError("prediction/action fit accessed an unauthorized mask bank")

    prompts = pd.read_csv(design_root / "prompts.csv", dtype={"prompt_id": str})
    candidates = pd.read_csv(
        design_root / "candidate_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    test, candidates = _validate_test_design_rows(prompts, candidates)
    cache_path = calibration_root / "template_head_means.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        means = np.asarray(cache["means"])
        templates = tuple(map(str, cache["templates"].tolist()))
    expected_templates = tuple(sorted(test["template_id"].astype(str).unique()))
    if means.shape != (8, 13, 64):
        raise ValueError("template-conditioned reference mean array has wrong shape")
    if templates != expected_templates:
        raise ValueError("sealed reference means do not match the test templates")
    if not np.isfinite(means).all():
        raise ValueError("sealed reference means contain non-finite values")
    return Phase6TestInputs(
        test_prompts=test,
        candidate_masks=candidates,
        template_head_means=means,
        templates=templates,
        protocol=protocol,
        design_manifest=design_manifest,
        calibration_manifest=calibration_manifest,
        freeze_manifest=freeze_manifest,
    )


def _mask_shard_spans(count: int, size: int) -> tuple[tuple[int, int], ...]:
    if count <= 0 or size <= 0:
        raise ValueError("mask count and shard size must be positive")
    return tuple((start, min(start + size, count)) for start in range(0, count, size))


def _clean_output_rows(prompts: pd.DataFrame, scores: Sequence[float]) -> pd.DataFrame:
    columns = [
        "prompt_id",
        "split",
        "template_id",
        "structure",
        "unordered_name_pair_id",
        "pair_orientation",
        "io_name",
        "s_name",
    ]
    rows = prompts[columns].copy()
    rows["clean_ld"] = np.asarray(scores, dtype=float)
    return rows


def validate_clean_test_scores(rows: pd.DataFrame, prompts: pd.DataFrame) -> None:
    """Require one finite clean score for every frozen test prompt."""

    required = {
        "prompt_id",
        "split",
        "template_id",
        "structure",
        "unordered_name_pair_id",
        "pair_orientation",
        "io_name",
        "s_name",
        "clean_ld",
    }
    if not required <= set(rows.columns):
        raise ValueError("clean test scores lack required columns")
    if len(rows) != len(prompts) or rows["prompt_id"].astype(str).duplicated().any():
        raise ValueError("clean test score rows are not one-to-one with prompts")
    expected = prompts[list(required - {"clean_ld"})].copy()
    observed = rows[list(required - {"clean_ld"})].copy()
    for frame in (expected, observed):
        frame["prompt_id"] = frame["prompt_id"].astype(str)
        frame.sort_values("prompt_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed.equals(expected):
        raise ValueError("clean test score metadata differs from the sealed prompts")
    if set(rows["split"].astype(str)) != {"test"}:
        raise ValueError("clean score artifact contains a non-test prompt")
    if not np.isfinite(rows["clean_ld"].to_numpy(float)).all():
        raise ValueError("clean test scores contain non-finite values")


def validate_candidate_effect_shard(
    rows: pd.DataFrame,
    *,
    prompts: pd.DataFrame,
    masks: pd.DataFrame,
    clean_scores: pd.DataFrame,
) -> None:
    """Validate one exact held-out prompt-by-candidate Cartesian shard."""

    prompt_ids = prompts["prompt_id"].astype(str).tolist()
    mask_ids = masks["mask_id"].astype(str).tolist()
    validate_effect_rows(rows, prompt_ids=prompt_ids, mask_ids=mask_ids)
    if set(rows["split"].astype(str)) != {"test"}:
        raise ValueError("candidate effect shard contains a non-test prompt")
    if set(rows["bank"].astype(str)) != {"candidate"}:
        raise ValueError("candidate effect shard contains a non-candidate mask")
    observed_masks = rows[["mask_id", "mask_bits", "pool_id"]].drop_duplicates().copy()
    expected_masks = masks[["mask_id", "mask_bits", "pool_id"]].copy()
    for frame in (observed_masks, expected_masks):
        frame["mask_id"] = frame["mask_id"].astype(str)
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(13)
        frame["pool_id"] = frame["pool_id"].astype(str)
        frame.sort_values("mask_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed_masks.equals(expected_masks):
        raise ValueError("candidate effect shard mask mapping differs from the sealed design")
    clean = clean_scores.set_index(clean_scores["prompt_id"].astype(str))["clean_ld"]
    expected_clean = rows["prompt_id"].astype(str).map(clean).to_numpy(float)
    if not np.allclose(rows["clean_ld"].to_numpy(float), expected_clean, atol=1e-6, rtol=0):
        raise ValueError("candidate effect shard does not reuse the frozen clean scores")


def _source_hashes(
    *,
    design_dir: str | Path,
    calibration_dir: str | Path,
    freeze_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, str]:
    return {
        "protocol": file_sha256(protocol_path),
        "design_manifest": file_sha256(Path(design_dir) / "design_manifest.json"),
        "calibration_manifest": file_sha256(
            Path(calibration_dir) / "calibration_manifest.json"
        ),
        "prediction_action_manifest": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "template_head_means": file_sha256(
            Path(calibration_dir) / "template_head_means.npz"
        ),
    }


def build_phase6_test_measurement_spec(
    inputs: Phase6TestInputs,
    *,
    config: Phase6TestMeasurementConfig,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Describe the immutable computation before held-out outcomes are opened."""

    spans = _mask_shard_spans(len(inputs.candidate_masks), config.mask_shard_size)
    ordered_ids = inputs.candidate_masks["mask_id"].astype(str).tolist()
    return {
        "schema": TEST_MEASUREMENT_SPEC_SCHEMA,
        "status": TEST_MEASUREMENT_SPEC_STATUS,
        "scientific_status": PHASE6_STATUS,
        "design_id": inputs.design_manifest["design_id"],
        "config": asdict(config),
        "source_hashes": dict(source_hashes),
        "candidate_order_sha256": json_sha256(ordered_ids),
        "template_order": list(inputs.templates),
        "accessed_prompt_splits": ["test"],
        "accessed_mask_banks": ["candidate"],
        "reused_calibration_artifacts": ["template_head_means.npz"],
        "counts": {
            "test_prompts": len(inputs.test_prompts),
            "candidate_masks": len(inputs.candidate_masks),
            "candidate_pools": int(inputs.candidate_masks["pool_id"].nunique()),
            "candidate_effect_cells": len(inputs.test_prompts)
            * len(inputs.candidate_masks),
            "mask_shards": len(spans),
        },
        "forbidden_operations": {
            "refit_observers": True,
            "reselect_actions": True,
            "filter_prompts_or_masks": True,
            "change_targets_or_models": True,
        },
        "policy_hierarchy": POLICY_HIERARCHY,
        "measurement_scope": (
            "One clean score for each frozen test prompt and the complete Cartesian "
            "product of all frozen test prompts and candidate masks."
        ),
    }


def _write_or_validate_spec(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(expected):
            raise ValueError("test measurement resume spec differs from the sealed run")
        return
    write_json(path, dict(expected))


def _write_progress(
    path: Path,
    *,
    clean_path: Path | None,
    shard_paths: Sequence[Path],
    output: Path,
    prompt_count: int,
    completed_mask_count: int,
) -> None:
    artifacts = [item for item in (clean_path, *shard_paths) if item is not None]
    write_json(
        path,
        {
            "schema": TEST_MEASUREMENT_PROGRESS_SCHEMA,
            "status": "complete" if clean_path is not None and completed_mask_count == 1536 else "partial",
            "accessed_prompt_splits": ["test"] if artifacts else [],
            "accessed_mask_banks": ["candidate"] if shard_paths else [],
            "logical_forward_evaluations": {
                "clean_test_prompts": prompt_count if clean_path is not None else 0,
                "test_prompt_candidate_mask_pairs": prompt_count * completed_mask_count,
            },
            "completed_candidate_masks": completed_mask_count,
            "completed_shards": len(shard_paths),
            "artifact_hashes": {
                item.relative_to(output).as_posix(): file_sha256(item) for item in artifacts
            },
        },
    )


def _validate_complete_manifest(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    expected_spec: Mapping[str, Any],
) -> None:
    if manifest.get("schema") != TEST_MEASUREMENT_SCHEMA:
        raise ValueError("unexpected Phase-6 test measurement schema")
    if manifest.get("status") != TEST_MEASUREMENT_STATUS:
        raise ValueError("Phase-6 test measurement is not complete")
    if manifest.get("source_hashes") != expected_spec.get("source_hashes"):
        raise ValueError("completed test measurement used different sealed sources")
    if manifest.get("measurement_spec_sha256") != file_sha256(
        output / "measurement_run_spec.json"
    ):
        raise ValueError("test measurement spec changed after the run")
    shard_count = int(expected_spec.get("counts", {}).get("mask_shards", -1))
    shard_size = int(expected_spec.get("config", {}).get("mask_shard_size", -1))
    mask_count = int(expected_spec.get("counts", {}).get("candidate_masks", -1))
    expected_labels = {
        "measurement_run_spec.json",
        "clean_scores_test.csv",
        "measurement_progress.json",
        *(
            f"shards/test/effects_{start:04d}_{stop:04d}.csv"
            for start, stop in _mask_shard_spans(mask_count, shard_size)
        ),
    }
    if len(expected_labels) != shard_count + 3:
        raise ValueError("test measurement shard count differs from the sealed spec")
    _verify_hash_index(
        output,
        manifest.get("artifact_hashes", {}),
        label="test measurement manifest",
        exact_labels=expected_labels,
    )


def run_phase6_test_measurement(
    design_dir: str | Path,
    calibration_dir: str | Path,
    freeze_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    config: Phase6TestMeasurementConfig,
) -> Path:
    """Measure the complete frozen test-by-candidate surface without fitting."""

    inputs = load_phase6_test_inputs(
        design_dir,
        calibration_dir,
        freeze_dir,
        protocol_path=protocol_path,
        config=config,
    )
    sources = _source_hashes(
        design_dir=design_dir,
        calibration_dir=calibration_dir,
        freeze_dir=freeze_dir,
        protocol_path=protocol_path,
    )
    spec = build_phase6_test_measurement_spec(inputs, config=config, source_hashes=sources)
    output = Path(outdir)
    shard_root = output / "shards" / "test"
    output.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    spec_path = output / "measurement_run_spec.json"
    _write_or_validate_spec(spec_path, spec)

    complete_path = output / "test_manifest.json"
    if complete_path.exists():
        _validate_complete_manifest(output, _read_json(complete_path), expected_spec=spec)
        return output

    spans = _mask_shard_spans(len(inputs.candidate_masks), config.mask_shard_size)
    expected_names = {f"effects_{start:04d}_{stop:04d}.csv" for start, stop in spans}
    observed_names = {path.name for path in shard_root.glob("effects_*.csv")}
    if not observed_names <= expected_names:
        raise ValueError("test measurement directory contains an unexpected shard")

    clean_path = output / "clean_scores_test.csv"
    clean_rows: pd.DataFrame | None = None
    if clean_path.exists():
        clean_rows = pd.read_csv(clean_path, dtype={"prompt_id": str})
        validate_clean_test_scores(clean_rows, inputs.test_prompts)

    completed: list[Path] = []
    completed_masks = 0
    for start, stop in spans:
        path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
        if not path.exists():
            continue
        if clean_rows is None:
            raise ValueError("candidate shards exist without a complete clean-score artifact")
        masks = inputs.candidate_masks.iloc[start:stop].reset_index(drop=True)
        rows = pd.read_csv(
            path,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        validate_candidate_effect_shard(
            rows,
            prompts=inputs.test_prompts,
            masks=masks,
            clean_scores=clean_rows,
        )
        completed.append(path)
        completed_masks += len(masks)

    progress_path = output / "measurement_progress.json"
    _write_progress(
        progress_path,
        clean_path=clean_path if clean_rows is not None else None,
        shard_paths=completed,
        output=output,
        prompt_count=len(inputs.test_prompts),
        completed_mask_count=completed_masks,
    )

    needs_model = clean_rows is None or len(completed) != len(spans)
    model: Any | None = None
    tokens: Sequence[Any] | None = None
    io_tokens: np.ndarray | None = None
    s_tokens: np.ndarray | None = None
    if needs_model:
        import torch
        from transformer_lens import HookedTransformer

        torch.manual_seed(config.seed)
        model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
            revision=config.model_revision,
        )
        model.eval()
        if int(model.cfg.n_layers) != 12 or int(model.cfg.n_heads) != 12:
            raise ValueError("loaded model is not the pinned GPT-2-small architecture")
        if int(model.cfg.d_head) != inputs.template_head_means.shape[2]:
            raise ValueError("model head width differs from the sealed reference cache")
        tokens, io_tokens, s_tokens = _model_tokens(model, inputs.test_prompts)
        expected_io = inputs.test_prompts["answer_token_id"].to_numpy(int)
        expected_s = inputs.test_prompts["counterfactual_token_id"].to_numpy(int)
        if not np.array_equal(io_tokens, expected_io) or not np.array_equal(
            s_tokens, expected_s
        ):
            raise ValueError("pinned-model tokenization differs from the sealed design")

    if clean_rows is None:
        assert model is not None and tokens is not None
        assert io_tokens is not None and s_tokens is not None
        scores = _score_clean(
            model,
            inputs.test_prompts,
            tokens,
            io_tokens,
            s_tokens,
            batch_size=config.clean_batch_size,
        )
        clean_rows = _clean_output_rows(inputs.test_prompts, scores)
        validate_clean_test_scores(clean_rows, inputs.test_prompts)
        temporary = clean_path.with_suffix(".tmp")
        clean_rows.to_csv(temporary, index=False)
        temporary.replace(clean_path)
        _write_progress(
            progress_path,
            clean_path=clean_path,
            shard_paths=completed,
            output=output,
            prompt_count=len(inputs.test_prompts),
            completed_mask_count=completed_masks,
        )

    template_to_index = {
        template: index for index, template in enumerate(inputs.templates)
    }
    completed_set = set(completed)
    for start, stop in spans:
        path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
        if path in completed_set:
            continue
        assert model is not None and tokens is not None
        assert io_tokens is not None and s_tokens is not None and clean_rows is not None
        masks = inputs.candidate_masks.iloc[start:stop].reset_index(drop=True)
        rows = _score_mask_shard(
            model,
            inputs.test_prompts,
            masks,
            tokens,
            io_tokens,
            s_tokens,
            clean_rows["clean_ld"].to_numpy(float),
            template_to_index,
            inputs.template_head_means,
            pair_batch_size=config.pair_batch_size,
        )
        validate_candidate_effect_shard(
            rows,
            prompts=inputs.test_prompts,
            masks=masks,
            clean_scores=clean_rows,
        )
        temporary = path.with_suffix(".tmp")
        rows.to_csv(temporary, index=False)
        temporary.replace(path)
        completed.append(path)
        completed_set.add(path)
        completed_masks += len(masks)
        _write_progress(
            progress_path,
            clean_path=clean_path,
            shard_paths=completed,
            output=output,
            prompt_count=len(inputs.test_prompts),
            completed_mask_count=completed_masks,
        )

    completed = sorted(completed)
    if len(completed) != len(spans) or completed_masks != len(inputs.candidate_masks):
        raise AssertionError("held-out candidate surface is incomplete")
    expected_cells = len(inputs.test_prompts) * len(inputs.candidate_masks)
    artifact_paths = [spec_path, clean_path, progress_path, *completed]
    manifest = {
        "schema": TEST_MEASUREMENT_SCHEMA,
        "status": TEST_MEASUREMENT_STATUS,
        "scientific_status": PHASE6_STATUS,
        "design_id": inputs.design_manifest["design_id"],
        "config": asdict(config),
        "source_hashes": sources,
        "measurement_spec_sha256": file_sha256(spec_path),
        "accessed_prompt_splits": ["test"],
        "accessed_mask_banks": ["candidate"],
        "reused_calibration_artifacts": ["template_head_means.npz"],
        "policy_hierarchy": POLICY_HIERARCHY,
        "test_prompt_forward_passes": len(inputs.test_prompts),
        "candidate_mask_forward_passes": expected_cells,
        "forward_pass_accounting": {
            "unit": "logical model-evaluated examples; packed calls contain multiple examples",
            "clean_test_prompt_evaluations": len(inputs.test_prompts),
            "test_prompt_candidate_mask_pair_evaluations": expected_cells,
            "total_logical_evaluations": len(inputs.test_prompts) + expected_cells,
        },
        "counts": {
            "test_prompts": len(inputs.test_prompts),
            "candidate_masks": len(inputs.candidate_masks),
            "candidate_pools": int(inputs.candidate_masks["pool_id"].nunique()),
            "candidate_effect_cells": expected_cells,
            "mask_shards": len(completed),
        },
        "fit_or_selection_changes": {
            "observers_refit": False,
            "predictions_recomputed": False,
            "actions_reselected": False,
            "prompts_filtered": False,
            "candidate_masks_filtered": False,
        },
        "model": {
            "requested_name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_name": model.cfg.model_name if model is not None else "resume_only",
            "n_layers": int(model.cfg.n_layers) if model is not None else 12,
            "n_heads": int(model.cfg.n_heads) if model is not None else 12,
            "d_head": int(model.cfg.d_head) if model is not None else 64,
            "dtype": str(model.cfg.dtype) if model is not None else "recorded_in_calibration",
            "device": str(model.cfg.device) if model is not None else config.device,
        },
        "artifact_hashes": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in artifact_paths
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "Evaluate the frozen observers and actions under the prespecified H1, H2, "
            "Jensen sensitivity, stress-target, and clean-task gates. Do not refit or reselect."
        ),
    }
    write_json(complete_path, manifest)
    return output
