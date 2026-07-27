"""Inference-free Qwen induction finite-effect task and additive baseline.

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


QWEN_INDUCTION_EFFECT_TASK_NAME = "induction-qwen2-5-7b-finite-effects"
QWEN_INDUCTION_EFFECT_DATA_VERSION = "copy-v1"
QWEN_INDUCTION_EFFECT_DATA_VERSIONS = ("copy-v1", "copy-v2")
QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS: tuple[int, ...] = (16, 40, 64, 128)
QWEN_INDUCTION_MODEL_NAME = "Qwen/Qwen2.5-7B"
QWEN_INDUCTION_MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
QWEN_INDUCTION_N_HEADS = 8

QWEN_INDUCTION_DESIGN_SCHEMA = "observerbench.qwen_induction_design_manifest.v1"
QWEN_INDUCTION_EFFECT_RUN_SCHEMA = "observerbench.qwen_induction_effect_run.v1"
QWEN_INDUCTION_EFFECT_ROW_SCHEMA = "observerbench.qwen_induction_effect_rows.v1"
QWEN_INDUCTION_MASK_FEATURE_SCHEMA = "observerbench.qwen_induction_mask_features.v1"

QWEN_INDUCTION_ADDITIVE_BASELINE_NAME = "qwen-induction-additive-ridge"
QWEN_INDUCTION_ADDITIVE_BASELINE_VERSION = "1.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOT = (
    _REPO_ROOT / "results" / "revision" / "qwen_induction" / "copy_v1"
)
DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOTS = {
    "copy-v1": DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOT,
    "copy-v2": _REPO_ROOT
    / "results"
    / "revision"
    / "phase10"
    / "qwen_induction_copy_v2",
}

_SELECTED_HEAD_COLUMNS = (
    "component_index",
    "head_label",
    "layer",
    "head",
    "kv_group",
)
_CALIBRATION_MASK_COLUMNS = (
    "measurement_order",
    "mask_id",
    "mask_bits",
    "n_heads",
    "bank",
    "pool_id",
)
_TEST_MASK_COLUMNS = (
    "mask_id",
    "mask_bits",
    "n_heads",
    "bank",
    "pool_id",
)
_PROMPT_COLUMNS = (
    "prompt_id",
    "split",
    "family_id",
    "cluster_id",
    "token_bank_id",
    "input_ids",
    "target_token_id",
    "distractor_token_id_1",
    "distractor_token_id_2",
    "source_key_position",
    "source_value_position",
    "query_position",
    "sequence_length",
    "repeat_gap",
)
_EFFECT_COLUMNS = (
    "schema_version",
    "prompt_id",
    "split",
    "family_id",
    "cluster_id",
    "mask_id",
    "mask_bits",
    "bank",
    "pool_id",
    "clean_margin",
    "ablated_margin",
    "drop_from_clean",
)


def qwen_induction_effect_task_version(
    measurement_budget: int,
    *,
    data_version: str = QWEN_INDUCTION_EFFECT_DATA_VERSION,
) -> str:
    """Return the stable task version for one nested measurement budget."""

    budget = int(measurement_budget)
    if budget not in QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS:
        supported = ", ".join(map(str, QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS))
        raise ValueError(
            f"unsupported Qwen induction measurement budget {budget}; choose {supported}"
        )
    if data_version not in QWEN_INDUCTION_EFFECT_DATA_VERSIONS:
        raise ValueError(f"unsupported Qwen induction data version: {data_version}")
    return f"{data_version}-b{budget:03d}"


@dataclass(frozen=True)
class InductionMaskFeatures:
    """Public feature record for one mask over eight frozen Qwen heads."""

    mask_id: str
    mask_bits: str
    head_mask: tuple[int, ...]
    head_labels: tuple[str, ...]
    head_layers: tuple[int, ...]
    head_indices: tuple[int, ...]
    head_kv_groups: tuple[int, ...]
    ablated_heads: tuple[str, ...]
    n_heads: int
    candidate_pool: str = ""
    schema_version: str = QWEN_INDUCTION_MASK_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.mask_id, str) or not self.mask_id.strip():
            raise ValueError("mask_id must be a non-empty string")
        if (
            len(self.head_mask) != QWEN_INDUCTION_N_HEADS
            or set(self.head_mask) - {0, 1}
        ):
            raise ValueError("head_mask must contain exactly 8 binary entries")
        if self.mask_bits != "".join(map(str, self.head_mask)):
            raise ValueError("mask_bits and head_mask disagree")
        metadata = (
            self.head_labels,
            self.head_layers,
            self.head_indices,
            self.head_kv_groups,
        )
        if any(len(values) != QWEN_INDUCTION_N_HEADS for values in metadata):
            raise ValueError("selected-head metadata must contain exactly 8 entries")
        if any(not isinstance(label, str) or not label.strip() for label in self.head_labels):
            raise ValueError("head_labels must contain non-empty strings")
        if len(set(self.head_labels)) != QWEN_INDUCTION_N_HEADS:
            raise ValueError("head_labels must be unique")
        coordinates = tuple(zip(self.head_layers, self.head_indices))
        if len(set(coordinates)) != QWEN_INDUCTION_N_HEADS:
            raise ValueError("selected-head layer/head coordinates must be unique")
        if any(value < 0 for values in metadata[1:] for value in values):
            raise ValueError("selected-head integer metadata must be non-negative")
        expected_ablated = tuple(
            label
            for label, included in zip(self.head_labels, self.head_mask)
            if included
        )
        if self.ablated_heads != expected_ablated:
            raise ValueError("ablated_heads and head_mask disagree")
        if self.n_heads != sum(self.head_mask):
            raise ValueError("n_heads and head_mask disagree")


class QwenInductionAdditiveRidgeBaseline:
    """First-order ridge baseline over the eight frozen head indicators."""

    name = QWEN_INDUCTION_ADDITIVE_BASELINE_NAME

    def __init__(self, ridge: float = 1e-6) -> None:
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be a positive finite number")
        self.ridge = float(ridge)
        self.coefficients_: np.ndarray | None = None

    @staticmethod
    def _row(features: InductionMaskFeatures) -> np.ndarray:
        if not isinstance(features, InductionMaskFeatures):
            raise TypeError(
                "QwenInductionAdditiveRidgeBaseline requires InductionMaskFeatures"
            )
        return np.asarray((1.0, *features.head_mask), dtype=float)

    def fit(
        self,
        measurements: Sequence[FiniteEffectMeasurement[InductionMaskFeatures]],
    ) -> None:
        if not measurements:
            raise ValueError("at least one Qwen induction measurement is required")
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
        queries: Sequence[FiniteEffectQuery[InductionMaskFeatures]],
    ) -> Sequence[float]:
        if self.coefficients_ is None:
            raise RuntimeError("fit must be called before predict")
        if not queries:
            return ()
        design = np.stack([self._row(row.features) for row in queries])
        return tuple(float(value) for value in design @ self.coefficients_)


def qwen_induction_additive_baseline_card(
    *, ridge: float = 1e-6
) -> EffectObserverCard:
    """Return metadata for the bundled first-order comparison observer."""

    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be a positive finite number")
    return EffectObserverCard(
        observer_name=QWEN_INDUCTION_ADDITIVE_BASELINE_NAME,
        observer_version=QWEN_INDUCTION_ADDITIVE_BASELINE_VERSION,
        observer_family="first-order additive finite-effect predictor",
        access_regime="frozen forward-only intervention measurements",
        measurement_basis="intercept plus eight binary induction-head indicators",
        fit_procedure=(
            f"ridge regression with unpenalized intercept and ridge={ridge:g}"
        ),
        implementation=(
            "observerbench.tasks.qwen_induction.effect_task."
            "QwenInductionAdditiveRidgeBaseline"
        ),
        known_failure_modes=(
            "Cannot represent interactions between ablated heads.",
            "Predicts a prompt-averaged effect and does not model effect dispersion.",
        ),
        metadata={
            "feature_schema": QWEN_INDUCTION_MASK_FEATURE_SCHEMA,
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


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError:
        raise FileNotFoundError(f"required frozen artifact is missing: {path}") from None
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(
                f"unexpected CSV columns in {path.name}; expected "
                + ", ".join(expected_columns)
            )
        return [dict(row) for row in reader]


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
    missing = sorted(required_labels - set(map(str, hashes)))
    if missing:
        raise ValueError(f"artifact manifest is missing entries: {missing}")
    for raw_label, raw_digest in sorted(hashes.items(), key=lambda item: str(item[0])):
        label = str(raw_label)
        path = _safe_artifact_path(root, label)
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {label}")
        if file_sha256(path) != str(raw_digest):
            raise ValueError(f"frozen artifact hash mismatch: {label}")


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"invalid integer field {field!r}") from None
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"invalid integer field {field!r}")
    return int(value)


def _finite(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"invalid finite field {field!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"invalid finite field {field!r}")
    return value


def _load_selected_heads(
    design_dir: Path,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows = _read_csv(design_dir / "selected_heads.csv", _SELECTED_HEAD_COLUMNS)
    if len(rows) != QWEN_INDUCTION_N_HEADS:
        raise ValueError("selected_heads.csv must contain exactly 8 heads")
    ordered = sorted(rows, key=lambda row: _integer(row, "component_index"))
    if [_integer(row, "component_index") for row in ordered] != list(
        range(QWEN_INDUCTION_N_HEADS)
    ):
        raise ValueError("selected-head component_index must be contiguous from zero")
    labels = tuple(str(row["head_label"]).strip() for row in ordered)
    layers = tuple(_integer(row, "layer") for row in ordered)
    heads = tuple(_integer(row, "head") for row in ordered)
    kv_groups = tuple(_integer(row, "kv_group") for row in ordered)
    if any(not label for label in labels):
        raise ValueError("selected head labels must be non-empty")
    if len(set(labels)) != QWEN_INDUCTION_N_HEADS:
        raise ValueError("selected head labels must be unique")
    if len(set(zip(layers, heads))) != QWEN_INDUCTION_N_HEADS:
        raise ValueError("selected layer/head coordinates must be unique")
    if any(value < 0 for values in (layers, heads, kv_groups) for value in values):
        raise ValueError("selected-head integer metadata must be non-negative")
    return labels, layers, heads, kv_groups


def _mask_features(
    row: Mapping[str, str],
    *,
    head_metadata: tuple[
        tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> InductionMaskFeatures:
    mask_id = str(row.get("mask_id", "")).strip()
    bits = str(row.get("mask_bits", "")).strip()
    if not mask_id or len(bits) != QWEN_INDUCTION_N_HEADS or set(bits) - {"0", "1"}:
        raise ValueError(f"invalid Qwen induction mask record: {mask_id!r}, {bits!r}")
    head_mask = tuple(int(bit) for bit in bits)
    if _integer(row, "n_heads") != sum(head_mask):
        raise ValueError(f"mask {mask_id} has inconsistent n_heads")
    labels, layers, heads, kv_groups = head_metadata
    return InductionMaskFeatures(
        mask_id=mask_id,
        mask_bits=bits,
        head_mask=head_mask,
        head_labels=labels,
        head_layers=layers,
        head_indices=heads,
        head_kv_groups=kv_groups,
        ablated_heads=tuple(
            label for label, included in zip(labels, head_mask) if included
        ),
        n_heads=sum(head_mask),
        candidate_pool=str(row.get("pool_id", "")).strip(),
    )


def _load_masks(
    design_dir: Path,
    *,
    head_metadata: tuple[
        tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[list[tuple[int, InductionMaskFeatures]], tuple[InductionMaskFeatures, ...]]:
    calibration_rows = _read_csv(
        design_dir / "calibration_masks.csv", _CALIBRATION_MASK_COLUMNS
    )
    test_rows = _read_csv(design_dir / "test_masks.csv", _TEST_MASK_COLUMNS)
    if len(calibration_rows) != 128 or len(test_rows) != 128:
        raise ValueError("the frozen mask partition must contain 128 calibration and 128 test masks")

    calibration: list[tuple[int, InductionMaskFeatures]] = []
    for row in calibration_rows:
        if row["bank"] != "calibration" or row["pool_id"].strip():
            raise ValueError("calibration masks must use bank=calibration and no pool_id")
        calibration.append(
            (
                _integer(row, "measurement_order"),
                _mask_features(row, head_metadata=head_metadata),
            )
        )
    calibration.sort(key=lambda item: item[0])
    if [order for order, _ in calibration] != list(range(1, 129)):
        raise ValueError("calibration measurement_order must be contiguous from one")

    candidates: list[InductionMaskFeatures] = []
    for row in test_rows:
        if row["bank"] != "test" or not row["pool_id"].strip():
            raise ValueError("test masks must use bank=test and a non-empty pool_id")
        candidates.append(_mask_features(row, head_metadata=head_metadata))
    candidates.sort(key=lambda features: features.mask_id)

    all_features = [features for _, features in calibration] + candidates
    mask_ids = [features.mask_id for features in all_features]
    mask_bits = [features.mask_bits for features in all_features]
    expected_bits = {f"{value:08b}" for value in range(256)}
    if len(set(mask_ids)) != 256:
        raise ValueError("frozen Qwen induction mask IDs must be unique")
    if set(mask_bits) != expected_bits:
        raise ValueError("calibration and test banks must partition the complete 8-bit mask universe")
    if calibration[0][1].mask_bits != "00000000":
        raise ValueError("the first calibration measurement must be exact no-op")

    pool_counts: dict[str, int] = {}
    for features in candidates:
        pool_counts[features.candidate_pool] = pool_counts.get(features.candidate_pool, 0) + 1
    if len(pool_counts) != 16 or set(pool_counts.values()) != {8}:
        raise ValueError("test masks must form 16 candidate pools of 8 masks")
    return calibration, tuple(candidates)


def _parse_input_ids(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        raise ValueError("input_ids must be a non-empty space-separated integer sequence")
    try:
        result = tuple(int(item) for item in text.split())
    except ValueError:
        raise ValueError("input_ids must be a non-empty space-separated integer sequence") from None
    if any(item < 0 for item in result):
        raise ValueError("input_ids must be non-negative")
    return result


def _load_prompts(design_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    prompts: dict[str, dict[str, dict[str, str]]] = {"train": {}, "test": {}}
    seen_ids: set[str] = set()
    for row in _read_csv(design_dir / "prompts.csv", _PROMPT_COLUMNS):
        prompt_id = str(row["prompt_id"]).strip()
        if not prompt_id or prompt_id in seen_ids:
            raise ValueError("frozen prompt IDs must be non-empty and globally unique")
        seen_ids.add(prompt_id)
        split = str(row["split"]).strip()
        if split not in prompts:
            continue
        family_id = str(row["family_id"]).strip()
        cluster_id = str(row["cluster_id"]).strip()
        token_bank_id = str(row["token_bank_id"]).strip()
        if not family_id or not cluster_id or not token_bank_id:
            raise ValueError("train/test prompt family, cluster, and token bank must be non-empty")
        input_ids = _parse_input_ids(row["input_ids"])
        sequence_length = _integer(row, "sequence_length")
        if sequence_length != len(input_ids):
            raise ValueError(f"prompt {prompt_id} sequence_length disagrees with input_ids")
        target = _integer(row, "target_token_id")
        distractors = (
            _integer(row, "distractor_token_id_1"),
            _integer(row, "distractor_token_id_2"),
        )
        if len({target, *distractors}) != 3:
            raise ValueError(f"prompt {prompt_id} target and distractors must be distinct")
        positions = (
            _integer(row, "source_key_position"),
            _integer(row, "source_value_position"),
            _integer(row, "query_position"),
        )
        if any(position < 0 or position >= sequence_length for position in positions):
            raise ValueError(f"prompt {prompt_id} contains an out-of-range position")
        if _integer(row, "repeat_gap") <= 0:
            raise ValueError(f"prompt {prompt_id} repeat_gap must be positive")
        prompts[split][prompt_id] = {
            "prompt_id": prompt_id,
            "family_id": family_id,
            "cluster_id": cluster_id,
            "token_bank_id": token_bank_id,
        }
    if not prompts["train"] or not prompts["test"]:
        raise ValueError("frozen design must contain train and test prompts")
    train_banks = {row["token_bank_id"] for row in prompts["train"].values()}
    test_banks = {row["token_bank_id"] for row in prompts["test"].values()}
    if train_banks & test_banks:
        raise ValueError("train and test token banks must be disjoint")
    return prompts


def _load_effect_split(
    effects_dir: Path,
    *,
    split: str,
    prompts: Mapping[str, Mapping[str, str]],
    masks: Mapping[str, InductionMaskFeatures],
    expected_bank: str,
) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    clean_by_prompt: dict[str, float] = {}
    for row in _read_csv(effects_dir / f"{split}_effects.csv", _EFFECT_COLUMNS):
        if row["schema_version"] != QWEN_INDUCTION_EFFECT_ROW_SCHEMA:
            raise ValueError(f"unexpected effect row schema in {split}_effects.csv")
        if row["split"] != split or row["bank"] != expected_bank:
            raise ValueError(f"{split}_effects.csv contains a different split or bank")
        prompt_id = str(row["prompt_id"]).strip()
        mask_id = str(row["mask_id"]).strip()
        key = (prompt_id, mask_id)
        if prompt_id not in prompts or mask_id not in masks or key in rows:
            raise ValueError(f"invalid or duplicate prompt-mask cell: {key}")
        prompt = prompts[prompt_id]
        features = masks[mask_id]
        if row["family_id"] != prompt["family_id"] or row["cluster_id"] != prompt["cluster_id"]:
            raise ValueError(f"effect row prompt metadata disagrees for {prompt_id}")
        expected_pool = features.candidate_pool if expected_bank == "test" else ""
        if row["mask_bits"] != features.mask_bits or row["pool_id"] != expected_pool:
            raise ValueError(f"effect row mask mapping disagrees for {mask_id}")
        clean = _finite(row, "clean_margin")
        ablated = _finite(row, "ablated_margin")
        effect = _finite(row, "drop_from_clean")
        if not math.isclose(effect, clean - ablated, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"effect arithmetic disagrees for {key}")
        prior_clean = clean_by_prompt.setdefault(prompt_id, clean)
        if not math.isclose(clean, prior_clean, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"clean margin changes across masks for {prompt_id}")
        if features.mask_bits == "00000000" and not math.isclose(
            effect, 0.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("exact no-op must have zero finite effect")
        rows[key] = effect
    expected_count = len(prompts) * len(masks)
    if len(rows) != expected_count:
        raise ValueError(
            f"{split} effects are not the complete prompt-by-mask table: "
            f"expected {expected_count}, got {len(rows)}"
        )
    return rows


def _checked_manifests(
    design_dir: Path,
    effects_dir: Path,
    *,
    verify_hashes: bool,
    require_scientific_claim: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design_path = design_dir / "design_manifest.json"
    design = _read_json(design_path)
    if design.get("schema") != QWEN_INDUCTION_DESIGN_SCHEMA:
        raise ValueError("unexpected Qwen induction design manifest schema")
    if design.get("status") != "frozen_before_outcomes":
        raise ValueError("Qwen induction design is not marked frozen before outcomes")
    if design.get("data_version") not in QWEN_INDUCTION_EFFECT_DATA_VERSIONS:
        raise ValueError("Qwen induction design has an unexpected data version")
    if not design.get("all_design_gates_pass"):
        raise ValueError("Qwen induction design did not pass its frozen design gates")
    if design.get("measurement_budgets") != list(
        QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
    ):
        raise ValueError("Qwen induction design has unexpected measurement budgets")
    model = design.get("model")
    if not isinstance(model, Mapping) or (
        model.get("requested_name") != QWEN_INDUCTION_MODEL_NAME
        or model.get("requested_revision") != QWEN_INDUCTION_MODEL_REVISION
    ):
        raise ValueError("Qwen induction design did not use the pinned base model")
    design_hashes = design.get("artifact_hashes")
    if not isinstance(design_hashes, Mapping):
        raise ValueError("Qwen induction design manifest has no artifact hashes")
    required_gate_labels = {"discovery_gate", "confirmation_gate"}
    if design.get("data_version") == "copy-v2":
        required_gate_labels.update(
            {
                "eligibility_gate",
                "reference_discovery_gate",
                "reference_confirmation_gate",
            }
        )
        if (
            require_scientific_claim
            and design.get("scientific_claim_allowed") is not True
        ):
            raise ValueError("Copy-v2 design is not eligible for scientific claims")
    gate_hashes = design.get("gate_artifact_hashes")
    if not isinstance(gate_hashes, Mapping) or not required_gate_labels.issubset(
        gate_hashes
    ):
        raise ValueError("Qwen induction design has no causal-gate proofs")
    for gate_label in sorted(required_gate_labels):
        gate_record = gate_hashes[gate_label]
        if not isinstance(gate_record, Mapping):
            raise ValueError(f"invalid causal-gate proof record: {gate_label}")
        gate_path = str(gate_record.get("path", ""))
        gate_sha = str(gate_record.get("sha256", ""))
        if not gate_path or design_hashes.get(gate_path) != gate_sha:
            raise ValueError(f"causal-gate proof is not bound to the design: {gate_label}")
        proof = _read_json(_safe_artifact_path(design_dir, gate_path))
        if proof.get("passed") is not True:
            raise ValueError(f"causal-gate proof did not pass: {gate_label}")

    effect = _read_json(effects_dir / "effect_manifest.json")
    if effect.get("schema") != QWEN_INDUCTION_EFFECT_RUN_SCHEMA:
        raise ValueError("unexpected Qwen induction effect manifest schema")
    if effect.get("status") != "complete_locked_test_outcomes":
        raise ValueError("Qwen induction effect table is not complete and locked")
    effect_data_version = effect.get(
        "data_version", QWEN_INDUCTION_EFFECT_DATA_VERSION
    )
    if effect_data_version != design.get("data_version"):
        raise ValueError("Qwen induction effect and design data versions differ")
    if (
        require_scientific_claim
        and design.get("data_version") == "copy-v2"
        and effect.get("scientific_claim_allowed") is not True
    ):
        raise ValueError("Copy-v2 effects are not eligible for scientific claims")
    if effect.get("design_manifest_sha256") != file_sha256(design_path):
        raise ValueError("Qwen induction effect table does not match the frozen design")
    effect_model = effect.get("model")
    if not isinstance(effect_model, Mapping) or (
        effect_model.get("requested_name") != QWEN_INDUCTION_MODEL_NAME
        or effect_model.get("requested_revision") != QWEN_INDUCTION_MODEL_REVISION
    ):
        raise ValueError("Qwen induction effects did not use the pinned base model")
    intervention = effect.get("intervention")
    if not isinstance(intervention, Mapping) or (
        intervention.get("site") != "final_query_head_z"
        or intervention.get("replacement") != "family_conditioned_reference_mean"
        or intervention.get("n_selected_heads") != QWEN_INDUCTION_N_HEADS
    ):
        raise ValueError("Qwen induction effects used an unexpected intervention")
    effect_hashes = effect.get("artifacts")
    if not isinstance(effect_hashes, Mapping):
        raise ValueError("Qwen induction effect manifest has no artifact hashes")

    required_design = (
        "selected_heads.csv",
        "calibration_masks.csv",
        "test_masks.csv",
        "prompts.csv",
    )
    required_effect = ("train_effects.csv", "test_effects.csv")
    if verify_hashes:
        _verify_artifacts(design_dir, design_hashes, required=required_design)
        _verify_artifacts(effects_dir, effect_hashes, required=required_effect)
    else:
        for root, labels in (
            (design_dir, required_design),
            (effects_dir, required_effect),
        ):
            for label in labels:
                if not _safe_artifact_path(root, label).is_file():
                    raise FileNotFoundError(f"frozen artifact is missing: {label}")
    return design, effect


def load_qwen_induction_effect_prediction_task(
    artifacts_root: str | Path | None = None,
    *,
    measurement_budget: int = 128,
    verify_hashes: bool = True,
    expected_data_version: str | None = None,
    require_scientific_claim: bool = True,
) -> FiniteEffectPredictionTask[InductionMaskFeatures]:
    """Load one checked Qwen task from cached tables; never run model inference."""

    budget = int(measurement_budget)
    if (
        expected_data_version is not None
        and expected_data_version not in QWEN_INDUCTION_EFFECT_DATA_VERSIONS
    ):
        raise ValueError(
            f"unsupported Qwen induction data version: {expected_data_version}"
        )
    root = Path(artifacts_root) if artifacts_root is not None else (
        DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOTS[
            expected_data_version or QWEN_INDUCTION_EFFECT_DATA_VERSION
        ]
    )
    design_dir = root / "design"
    effects_dir = root / "effects"
    design_manifest, effect_manifest = _checked_manifests(
        design_dir,
        effects_dir,
        verify_hashes=verify_hashes,
        require_scientific_claim=require_scientific_claim,
    )
    data_version = str(design_manifest["data_version"])
    if expected_data_version is not None and data_version != expected_data_version:
        raise ValueError(
            "Qwen induction artifact data version differs from the requested task"
        )
    version = qwen_induction_effect_task_version(
        budget, data_version=data_version
    )
    head_metadata = _load_selected_heads(design_dir)
    calibration, candidates = _load_masks(
        design_dir,
        head_metadata=head_metadata,
    )
    prompts = _load_prompts(design_dir)
    calibration_features = {features.mask_id: features for _, features in calibration}
    candidate_features = {features.mask_id: features for features in candidates}
    train_effects = _load_effect_split(
        effects_dir,
        split="train",
        prompts=prompts["train"],
        masks=calibration_features,
        expected_bank="calibration",
    )
    test_effects = _load_effect_split(
        effects_dir,
        split="test",
        prompts=prompts["test"],
        masks=candidate_features,
        expected_bank="test",
    )

    train_prompt_ids = sorted(prompts["train"])
    measurements: list[FiniteEffectMeasurement[InductionMaskFeatures]] = []
    for order, features in calibration[:budget]:
        values = [
            train_effects[(prompt_id, features.mask_id)]
            for prompt_id in train_prompt_ids
        ]
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

    test_prompt_ids = sorted(prompts["test"])
    queries: list[FiniteEffectQuery[InductionMaskFeatures]] = []
    targets: list[FiniteEffectTarget] = []
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

    task_id = f"{QWEN_INDUCTION_EFFECT_TASK_NAME}@{version}"
    head_labels, head_layers, head_indices, head_kv_groups = head_metadata
    card = EffectTaskCard(
        task_name=QWEN_INDUCTION_EFFECT_TASK_NAME,
        task_version=version,
        summary=(
            "Predict the held-out mean finite change in an exact induction-copy "
            "candidate margin after mean replacement of a fixed subset of eight heads."
        ),
        model_or_substrate=(
            f"{QWEN_INDUCTION_MODEL_NAME} at pinned revision "
            f"{QWEN_INDUCTION_MODEL_REVISION}"
        ),
        access_regime="cached forward-only finite intervention measurements",
        estimand=(
            "mean drop from the clean correct-successor versus two-distractor "
            "logit margin over the held-out prompt distribution"
        ),
        intervention_family=(
            "binary subsets of eight frozen induction-copy heads, replaced by "
            "family-conditioned reference means at the final query head-z site"
        ),
        measurement_design=(
            f"first {budget} masks in the frozen nested calibration order; each "
            f"effect averages {len(prompts['train'])} train prompts"
        ),
        validation_target=(
            f"mean effects for {len(candidates)} disjoint test masks, each averaged "
            f"over {len(prompts['test'])} held-out prompts"
        ),
        train_split=f"{design_manifest['design_id']}:train",
        evaluation_split=f"{design_manifest['design_id']}:test",
        primary_metrics=("mae", "rmse"),
        known_scope_limits=(
            "The task uses one pinned Qwen2.5-7B base checkpoint and one synthetic copy distribution.",
            "The eight heads were selected and causally validated on separate prompts; they are not claimed to be a complete ground-truth circuit.",
            "Measurements and targets are prompt-averaged; this task evaluates effect prediction, not behavioral safety.",
        ),
        metadata={
            "task_id": task_id,
            "data_version": data_version,
            "feature_schema": QWEN_INDUCTION_MASK_FEATURE_SCHEMA,
            "measurement_budget": budget,
            "supported_measurement_budgets": QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
            "measurement_unit": "distinct intervention mask",
            "underlying_train_effect_cells": budget * len(prompts["train"]),
            "underlying_test_effect_cells": len(candidates) * len(prompts["test"]),
            "n_queries": len(queries),
            "head_labels": head_labels,
            "head_layers": head_layers,
            "head_indices": head_indices,
            "head_kv_groups": head_kv_groups,
            "design_manifest_sha256": file_sha256(design_dir / "design_manifest.json"),
            "effect_manifest_sha256": file_sha256(effects_dir / "effect_manifest.json"),
            "model_revision": QWEN_INDUCTION_MODEL_REVISION,
            "model_inference_included": False,
            "hash_verification": bool(verify_hashes),
            "effect_manifest_status": effect_manifest["status"],
        },
    )
    return FiniteEffectPredictionTask(
        name=QWEN_INDUCTION_EFFECT_TASK_NAME,
        version=version,
        measurements=measurements,
        queries=queries,
        targets=targets,
        card=card,
    )


__all__ = [
    "DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOT",
    "DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOTS",
    "InductionMaskFeatures",
    "QWEN_INDUCTION_ADDITIVE_BASELINE_NAME",
    "QWEN_INDUCTION_ADDITIVE_BASELINE_VERSION",
    "QWEN_INDUCTION_EFFECT_DATA_VERSION",
    "QWEN_INDUCTION_EFFECT_DATA_VERSIONS",
    "QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS",
    "QWEN_INDUCTION_EFFECT_TASK_NAME",
    "QWEN_INDUCTION_MASK_FEATURE_SCHEMA",
    "QWEN_INDUCTION_MODEL_NAME",
    "QWEN_INDUCTION_MODEL_REVISION",
    "QwenInductionAdditiveRidgeBaseline",
    "load_qwen_induction_effect_prediction_task",
    "qwen_induction_additive_baseline_card",
    "qwen_induction_effect_task_version",
]
