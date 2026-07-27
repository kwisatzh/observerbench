"""Table-backed IOI finite-effect task and bundled additive baseline.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from observerbench.effect_prediction import (
    EffectObserverCard,
    EffectTaskCard,
    FiniteEffectMeasurement,
    FiniteEffectPredictionTask,
    FiniteEffectQuery,
    FiniteEffectTarget,
)
from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.heads import head_records


IOI_EFFECT_TASK_NAME = "ioi-gpt2-small-finite-effects"
IOI_EFFECT_DATA_VERSION = "phase5-test-v1"
IOI_EFFECT_MEASUREMENT_BUDGETS: tuple[int, ...] = (20, 40, 80, 160)
IOI_EFFECT_ROW_SCHEMA = "observerbench.ioi_effect_rows.v1"
IOI_EFFECT_DESIGN_SCHEMA = "observerbench.ioi_phase5_design_manifest.v1"
IOI_EFFECT_RUN_SCHEMA = "observerbench.ioi_effect_run.v1"
IOI_EFFECT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
IOI_MASK_FEATURE_SCHEMA = "observerbench.ioi_mask_features.v1"
IOI_ADDITIVE_BASELINE_NAME = "ioi-additive-ridge"
IOI_ADDITIVE_BASELINE_VERSION = "1.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IOI_EFFECT_ARTIFACT_ROOT = _REPO_ROOT / "results" / "revision" / "phase05"
_HEAD_ROWS = tuple(head_records())
_HEAD_LABELS = tuple(str(row["label"]) for row in _HEAD_ROWS)
_HEAD_GROUPS = tuple(str(row["group"]) for row in _HEAD_ROWS)


def ioi_effect_task_version(measurement_budget: int) -> str:
    """Return the stable task version for one nested measurement budget."""

    budget = int(measurement_budget)
    if budget not in IOI_EFFECT_MEASUREMENT_BUDGETS:
        supported = ", ".join(map(str, IOI_EFFECT_MEASUREMENT_BUDGETS))
        raise ValueError(f"unsupported IOI measurement budget {budget}; choose {supported}")
    return f"{IOI_EFFECT_DATA_VERSION}-b{budget:03d}"


@dataclass(frozen=True)
class IOIMaskFeatures:
    """Public feature record for one intervention over the 13 fixed IOI heads."""

    mask_id: str
    mask_bits: str
    head_mask: tuple[int, ...]
    selected_heads: tuple[str, ...]
    n_heads: int
    n_name_movers: int
    n_backup_name_movers: int
    n_negative_name_movers: int
    candidate_pool: str = ""
    sampling_stratum: str = ""
    schema_version: str = IOI_MASK_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if not self.mask_id.strip():
            raise ValueError("mask_id must be a non-empty string")
        if len(self.head_mask) != len(_HEAD_ROWS) or set(self.head_mask) - {0, 1}:
            raise ValueError("head_mask must contain exactly 13 binary entries")
        expected_bits = "".join(map(str, self.head_mask))
        if self.mask_bits != expected_bits:
            raise ValueError("mask_bits and head_mask disagree")
        if self.n_heads != sum(self.head_mask):
            raise ValueError("n_heads and head_mask disagree")
        expected_heads = tuple(
            label
            for label, included in zip(_HEAD_LABELS, self.head_mask)
            if included
        )
        if self.selected_heads != expected_heads:
            raise ValueError("selected_heads and head_mask disagree")
        expected_counts = tuple(
            sum(
                included
                for included, label in zip(self.head_mask, _HEAD_GROUPS)
                if label == group
            )
            for group in ("P", "B", "E")
        )
        supplied_counts = (
            self.n_name_movers,
            self.n_backup_name_movers,
            self.n_negative_name_movers,
        )
        if supplied_counts != expected_counts:
            raise ValueError("head-group counts and head_mask disagree")


class IOIAdditiveRidgeBaseline:
    """First-order ridge baseline over the 13 head indicators."""

    name = IOI_ADDITIVE_BASELINE_NAME

    def __init__(self, ridge: float = 1e-6) -> None:
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be a positive finite number")
        self.ridge = float(ridge)
        self.coefficients_: np.ndarray | None = None

    @staticmethod
    def _row(features: IOIMaskFeatures) -> np.ndarray:
        if not isinstance(features, IOIMaskFeatures):
            raise TypeError("IOIAdditiveRidgeBaseline requires IOIMaskFeatures")
        return np.asarray((1.0, *features.head_mask), dtype=float)

    def fit(
        self,
        measurements: Sequence[FiniteEffectMeasurement[IOIMaskFeatures]],
    ) -> None:
        if not measurements:
            raise ValueError("at least one IOI finite-effect measurement is required")
        design = np.stack([self._row(row.features) for row in measurements])
        outcomes = np.asarray([row.observed_effect for row in measurements], dtype=float)
        regularizer = self.ridge * np.eye(design.shape[1], dtype=float)
        regularizer[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ outcomes,
        )

    def predict(
        self,
        queries: Sequence[FiniteEffectQuery[IOIMaskFeatures]],
    ) -> Sequence[float]:
        if self.coefficients_ is None:
            raise RuntimeError("fit must be called before predict")
        if not queries:
            return ()
        design = np.stack([self._row(row.features) for row in queries])
        return tuple(float(value) for value in design @ self.coefficients_)


def ioi_additive_baseline_card(*, ridge: float = 1e-6) -> EffectObserverCard:
    """Return metadata for the bundled first-order comparison observer."""

    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be a positive finite number")
    return EffectObserverCard(
        observer_name=IOI_ADDITIVE_BASELINE_NAME,
        observer_version=IOI_ADDITIVE_BASELINE_VERSION,
        observer_family="first-order additive finite-effect predictor",
        access_regime="frozen forward-only intervention measurements",
        measurement_basis="intercept plus 13 binary IOI-head indicators",
        fit_procedure=f"ridge regression with unpenalized intercept and ridge={ridge:g}",
        implementation="observerbench.tasks.ioi.IOIAdditiveRidgeBaseline",
        known_failure_modes=(
            "Cannot represent interactions between ablated heads.",
            "Predicts a prompt-averaged effect and does not model effect dispersion.",
        ),
        metadata={
            "feature_schema": IOI_MASK_FEATURE_SCHEMA,
            "baseline_role": "bundled first-order reference",
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required frozen artifact is missing: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _safe_artifact_path(root: Path, label: str) -> Path:
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact manifest contains an unsafe path: {label}")
    return root / relative


def _verify_artifacts(
    root: Path,
    hashes: Mapping[str, Any],
    *,
    required: Iterable[str],
) -> None:
    required_labels = set(required)
    missing_labels = sorted(required_labels - set(map(str, hashes)))
    if missing_labels:
        raise ValueError(f"artifact manifest is missing entries: {missing_labels}")
    for raw_label, raw_digest in sorted(hashes.items(), key=lambda item: str(item[0])):
        label = str(raw_label)
        digest = str(raw_digest)
        path = _safe_artifact_path(root, label)
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {label}")
        if file_sha256(path) != digest:
            raise ValueError(f"frozen artifact hash mismatch: {label}")


def _checked_manifests(
    design_dir: Path,
    effects_dir: Path,
    *,
    verify_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design_path = design_dir / "design_manifest.json"
    design = _read_json(design_path)
    if design.get("schema") != IOI_EFFECT_DESIGN_SCHEMA:
        raise ValueError("unexpected IOI design manifest schema")
    if design.get("status") != "frozen_before_outcomes":
        raise ValueError("IOI design is not marked frozen before outcomes")
    if not design.get("all_design_gates_pass"):
        raise ValueError("IOI design did not pass its frozen design gates")
    design_hashes = design.get("artifact_hashes")
    if not isinstance(design_hashes, Mapping):
        raise ValueError("IOI design manifest has no artifact hashes")

    effect = _read_json(effects_dir / "effect_manifest.json")
    if effect.get("schema") != IOI_EFFECT_RUN_SCHEMA:
        raise ValueError("unexpected IOI effect manifest schema")
    if effect.get("status") != "complete_unopened_confirmatory_outcomes":
        raise ValueError("IOI effect table is not complete and sealed")
    if effect.get("design_manifest_sha256") != file_sha256(design_path):
        raise ValueError("IOI effect table does not match the frozen design")
    model = effect.get("model")
    if not isinstance(model, Mapping) or model.get("requested_revision") != IOI_EFFECT_MODEL_REVISION:
        raise ValueError("IOI effect table did not use the pinned GPT-2 revision")
    effect_hashes = effect.get("artifacts")
    if not isinstance(effect_hashes, Mapping):
        raise ValueError("IOI effect manifest has no artifact hashes")

    required_design = (
        "calibration_masks.csv",
        "candidate_masks.csv",
        "masks.csv",
        "prompts.csv",
    )
    required_effect = tuple(
        path.relative_to(effects_dir).as_posix()
        for split in ("train", "test")
        for path in sorted((effects_dir / "shards" / split).glob("effects_*.csv"))
    )
    if not required_effect:
        raise FileNotFoundError("no frozen train/test IOI effect shards were found")
    if verify_hashes:
        _verify_artifacts(design_dir, design_hashes, required=required_design)
        _verify_artifacts(effects_dir, effect_hashes, required=required_effect)
    else:
        for label in (*required_design, *required_effect):
            root = design_dir if label in required_design else effects_dir
            if not _safe_artifact_path(root, label).is_file():
                raise FileNotFoundError(f"frozen artifact is missing: {label}")
    return design, effect


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"invalid integer field {field!r}") from None
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"invalid integer field {field!r}")
    return int(value)


def _optional(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def _mask_features(row: Mapping[str, str]) -> IOIMaskFeatures:
    mask_id = str(row.get("mask_id", "")).strip()
    bits = str(row.get("mask_bits", "")).strip()
    if not mask_id or len(bits) != len(_HEAD_ROWS) or set(bits) - {"0", "1"}:
        raise ValueError(f"invalid IOI mask record: {mask_id!r}, {bits!r}")
    head_mask = tuple(int(value) for value in bits)
    selected = tuple(
        label for label, included in zip(_HEAD_LABELS, head_mask) if included
    )
    counts = {
        group: sum(value for value, label in zip(head_mask, _HEAD_GROUPS) if label == group)
        for group in ("P", "B", "E")
    }
    supplied = {
        "n_heads": sum(head_mask),
        "n_P": counts["P"],
        "n_B": counts["B"],
        "n_E": counts["E"],
    }
    for field, expected in supplied.items():
        if field in row and _optional(row[field]) and _integer(row, field) != expected:
            raise ValueError(f"mask {mask_id} has inconsistent {field}")
    return IOIMaskFeatures(
        mask_id=mask_id,
        mask_bits=bits,
        head_mask=head_mask,
        selected_heads=selected,
        n_heads=supplied["n_heads"],
        n_name_movers=counts["P"],
        n_backup_name_movers=counts["B"],
        n_negative_name_movers=counts["E"],
        candidate_pool=_optional(row.get("pool_id")),
        sampling_stratum=_optional(row.get("sampling_stratum")),
    )


def _load_masks(
    design_dir: Path,
) -> tuple[list[tuple[int, IOIMaskFeatures]], tuple[IOIMaskFeatures, ...]]:
    calibration_rows = _read_csv(design_dir / "calibration_masks.csv")
    candidate_rows = _read_csv(design_dir / "candidate_masks.csv")
    calibration = sorted(
        ((_integer(row, "measurement_order"), _mask_features(row)) for row in calibration_rows),
        key=lambda item: item[0],
    )
    if [order for order, _ in calibration] != list(range(1, len(calibration) + 1)):
        raise ValueError("calibration measurement_order must be contiguous from one")
    candidates = tuple(sorted((_mask_features(row) for row in candidate_rows), key=lambda row: row.mask_id))
    all_features = [features for _, features in calibration] + list(candidates)
    mask_ids = [features.mask_id for features in all_features]
    mask_bits = [features.mask_bits for features in all_features]
    if len(set(mask_ids)) != len(mask_ids) or len(set(mask_bits)) != len(mask_bits):
        raise ValueError("frozen IOI mask IDs and bit patterns must be unique")

    combined = _read_csv(design_dir / "masks.csv")
    combined_by_id = {str(row.get("mask_id", "")): row for row in combined}
    if len(combined_by_id) != len(combined) or set(combined_by_id) != set(mask_ids):
        raise ValueError("masks.csv does not contain the exact calibration and candidate banks")
    calibration_ids = {features.mask_id for _, features in calibration}
    for features in all_features:
        row = combined_by_id[features.mask_id]
        expected_bank = "calibration" if features.mask_id in calibration_ids else "candidate"
        if str(row.get("mask_bits", "")) != features.mask_bits or str(row.get("bank", "")) != expected_bank:
            raise ValueError(f"masks.csv mapping disagrees for {features.mask_id}")
        if expected_bank == "candidate" and _optional(row.get("pool_id")) != features.candidate_pool:
            raise ValueError(f"masks.csv candidate pool disagrees for {features.mask_id}")
    return calibration, candidates


def _load_prompts(design_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_split: dict[str, dict[str, dict[str, str]]] = {"train": {}, "test": {}}
    for row in _read_csv(design_dir / "prompts.csv"):
        split = str(row.get("split", ""))
        if split not in by_split:
            continue
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id or prompt_id in by_split[split]:
            raise ValueError(f"duplicate or empty {split} prompt ID")
        by_split[split][prompt_id] = {
            "prompt_id": prompt_id,
            "template_id": str(row.get("template_id", "")),
            "structure": str(row.get("structure", "")),
        }
    if not by_split["train"] or not by_split["test"]:
        raise ValueError("frozen design must contain train and test prompts")
    return by_split


def _load_effect_split(
    effects_dir: Path,
    *,
    split: str,
    prompts: Mapping[str, Mapping[str, str]],
    masks: Mapping[str, IOIMaskFeatures],
    calibration_ids: set[str],
) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    paths = sorted((effects_dir / "shards" / split).glob("effects_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no frozen {split} effect shards were found")
    for path in paths:
        for row in _read_csv(path):
            if row.get("schema_version") != IOI_EFFECT_ROW_SCHEMA:
                raise ValueError(f"unexpected effect row schema in {path.name}")
            if row.get("split") != split:
                raise ValueError(f"{path.name} contains a different split")
            prompt_id = str(row.get("prompt_id", ""))
            mask_id = str(row.get("mask_id", ""))
            key = (prompt_id, mask_id)
            if prompt_id not in prompts or mask_id not in masks or key in rows:
                raise ValueError(f"invalid or duplicate prompt-mask cell in {path.name}: {key}")
            features = masks[mask_id]
            expected_bank = "calibration" if mask_id in calibration_ids else "candidate"
            if str(row.get("mask_bits", "")) != features.mask_bits or str(row.get("bank", "")) != expected_bank:
                raise ValueError(f"effect row mask mapping disagrees for {mask_id}")
            if expected_bank == "candidate" and _optional(row.get("pool_id")) != features.candidate_pool:
                raise ValueError(f"effect row candidate pool disagrees for {mask_id}")
            try:
                value = float(row["drop_from_clean"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"invalid finite effect in {path.name}") from None
            if not math.isfinite(value):
                raise ValueError(f"non-finite effect in {path.name}")
            rows[key] = value
    expected_count = len(prompts) * len(masks)
    if len(rows) != expected_count:
        raise ValueError(
            f"{split} effects are not the complete prompt-by-mask table: "
            f"expected {expected_count}, got {len(rows)}"
        )
    return rows


def load_ioi_effect_prediction_task(
    artifacts_root: str | Path | None = None,
    *,
    measurement_budget: int = 160,
    verify_hashes: bool = True,
) -> FiniteEffectPredictionTask[IOIMaskFeatures]:
    """Load one checked IOI task from cached tables; never run model inference."""

    budget = int(measurement_budget)
    version = ioi_effect_task_version(budget)
    root = Path(artifacts_root) if artifacts_root is not None else DEFAULT_IOI_EFFECT_ARTIFACT_ROOT
    design_dir = root / "design"
    effects_dir = root / "ioi_effects"
    design_manifest, effect_manifest = _checked_manifests(
        design_dir,
        effects_dir,
        verify_hashes=verify_hashes,
    )
    calibration, candidates = _load_masks(design_dir)
    if len(calibration) != max(IOI_EFFECT_MEASUREMENT_BUDGETS):
        raise ValueError("frozen IOI calibration bank must contain 160 masks")
    if budget > len(calibration):
        raise ValueError("measurement budget exceeds the frozen calibration bank")
    prompts = _load_prompts(design_dir)
    all_features = {
        features.mask_id: features
        for features in ([row for _, row in calibration] + list(candidates))
    }
    calibration_ids = {features.mask_id for _, features in calibration}
    train_effects = _load_effect_split(
        effects_dir,
        split="train",
        prompts=prompts["train"],
        masks=all_features,
        calibration_ids=calibration_ids,
    )
    test_effects = _load_effect_split(
        effects_dir,
        split="test",
        prompts=prompts["test"],
        masks=all_features,
        calibration_ids=calibration_ids,
    )

    measurements: list[FiniteEffectMeasurement[IOIMaskFeatures]] = []
    for order, features in calibration[:budget]:
        values = [train_effects[(prompt_id, features.mask_id)] for prompt_id in sorted(prompts["train"])]
        measurements.append(
            FiniteEffectMeasurement(
                measurement_id=features.mask_id,
                features=features,
                observed_effect=float(sum(values) / len(values)),
                metadata={
                    "measurement_order": order,
                    "aggregation": "mean_over_train_prompts",
                    "n_train_prompts": len(values),
                },
            )
        )

    queries: list[FiniteEffectQuery[IOIMaskFeatures]] = []
    targets: list[FiniteEffectTarget] = []
    test_prompt_ids = sorted(prompts["test"])
    for features in candidates:
        values = [
            test_effects[(prompt_id, features.mask_id)]
            for prompt_id in test_prompt_ids
        ]
        queries.append(
            FiniteEffectQuery(
                query_id=features.mask_id,
                features=features,
                metadata={
                    "candidate_pool": features.candidate_pool,
                    "sampling_stratum": features.sampling_stratum,
                    "aggregation": "mean_over_heldout_test_prompts",
                    "n_test_prompts": len(values),
                },
            )
        )
        targets.append(
            FiniteEffectTarget(
                query_id=features.mask_id,
                observed_effect=float(sum(values) / len(values)),
            )
        )

    model = effect_manifest["model"]
    task_id = f"{IOI_EFFECT_TASK_NAME}@{version}"
    card = EffectTaskCard(
        task_name=IOI_EFFECT_TASK_NAME,
        task_version=version,
        summary=(
            "Predict the held-out mean finite change in the IOI logit difference after "
            "template-conditioned mean ablation of a fixed subset of 13 documented heads."
        ),
        model_or_substrate=(
            f"GPT-2 small at pinned revision {model['requested_revision']}"
        ),
        access_regime="cached forward-only finite intervention measurements",
        estimand=(
            "mean drop from the clean IO-versus-subject logit difference over the "
            "held-out prompt distribution under a fixed multi-head mean ablation"
        ),
        intervention_family=(
            "binary subsets of Name Movers, Backup Name Movers, and Negative Name Movers "
            "ablated at the final token"
        ),
        measurement_design=(
            f"first {budget} masks in the frozen nested calibration order; each observed "
            f"effect averages {len(prompts['train'])} disjoint train prompts"
        ),
        validation_target=(
            f"mean effects for {len(candidates)} held-out candidate masks, each averaged "
            f"over {len(prompts['test'])} disjoint test prompts"
        ),
        train_split=f"{design_manifest['design_id']}:train",
        evaluation_split=f"{design_manifest['design_id']}:test",
        primary_metrics=("mae", "rmse"),
        known_scope_limits=(
            "The task uses one pinned GPT-2-small checkpoint and the documented IOI head groups.",
            "Measurements and targets are prompt-averaged; the task does not expose prompt-level outcomes.",
            "The task evaluates effect prediction, not mechanism discovery or behavioral safety.",
        ),
        metadata={
            "task_id": task_id,
            "data_version": IOI_EFFECT_DATA_VERSION,
            "feature_schema": IOI_MASK_FEATURE_SCHEMA,
            "measurement_budget": budget,
            "supported_measurement_budgets": IOI_EFFECT_MEASUREMENT_BUDGETS,
            "measurement_unit": "distinct intervention mask",
            "underlying_train_effect_cells": budget * len(prompts["train"]),
            "underlying_test_effect_cells": len(candidates) * len(prompts["test"]),
            "n_queries": len(queries),
            "head_order": _HEAD_LABELS,
            "design_manifest_sha256": file_sha256(design_dir / "design_manifest.json"),
            "effect_manifest_sha256": file_sha256(effects_dir / "effect_manifest.json"),
            "model_revision": IOI_EFFECT_MODEL_REVISION,
            "model_inference_included": False,
            "hash_verification": bool(verify_hashes),
        },
    )
    return FiniteEffectPredictionTask(
        name=IOI_EFFECT_TASK_NAME,
        version=version,
        measurements=measurements,
        queries=queries,
        targets=targets,
        card=card,
    )


__all__ = [
    "DEFAULT_IOI_EFFECT_ARTIFACT_ROOT",
    "IOI_ADDITIVE_BASELINE_NAME",
    "IOI_ADDITIVE_BASELINE_VERSION",
    "IOI_EFFECT_DATA_VERSION",
    "IOI_EFFECT_MEASUREMENT_BUDGETS",
    "IOI_EFFECT_MODEL_REVISION",
    "IOI_EFFECT_TASK_NAME",
    "IOI_MASK_FEATURE_SCHEMA",
    "IOIAdditiveRidgeBaseline",
    "IOIMaskFeatures",
    "ioi_additive_baseline_card",
    "ioi_effect_task_version",
    "load_ioi_effect_prediction_task",
]
