"""Model-free serialization for the frozen Qwen induction task.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    MaskDesign,
    SequenceDesign,
    validate_mask_design,
    validate_sequence_design,
)
from observerbench.tasks.qwen_induction.effect_task import (
    QWEN_INDUCTION_DESIGN_SCHEMA,
    QWEN_INDUCTION_EFFECT_DATA_VERSION,
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_EFFECT_ROW_SCHEMA,
    QWEN_INDUCTION_EFFECT_RUN_SCHEMA,
    QWEN_INDUCTION_MODEL_NAME,
    QWEN_INDUCTION_MODEL_REVISION,
    _CALIBRATION_MASK_COLUMNS,
    _EFFECT_COLUMNS,
    _PROMPT_COLUMNS,
    _SELECTED_HEAD_COLUMNS,
    _TEST_MASK_COLUMNS,
    load_qwen_induction_effect_prediction_task,
)


PHASE09_CONFIG_SCHEMA = "observerbench.qwen_induction_phase09.v1"
TOKEN_BANKS_SCHEMA = "observerbench.qwen_induction_token_banks.v1"
PRESELECTION_MANIFEST_SCHEMA = "observerbench.qwen_induction_preselection.v1"
QWEN_INDUCTION_SCIENTIFIC_CONFIG_SHA256 = (
    "ec2835be61a85c6f963cab901c1de17f512e9fa08d5983c79bd14874000bdb77"
)

_CONFIG_BANK_TO_DESIGN_BANK = {
    "reference": "reference",
    "discovery": "discovery",
    "head_fit": "head_fit",
    "head_confirmation": "head_test",
    "calibration": "calibration",
    "locked_test": "locked_test",
}
_PROMPTS_ALL_COLUMNS = (
    "prompt_id",
    "bank",
    "family_id",
    "cluster_id",
    "token_bank_id",
    "input_ids",
    "target_token_id",
    "distractor_token_id_1",
    "distractor_token_id_2",
    "key_positions",
    "source_value_positions",
    "source_key_position",
    "source_value_position",
    "query_position",
    "sequence_length",
    "repeat_gap",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _hex_revision(value: Any, label: str) -> str:
    revision = _nonempty(value, label)
    if len(revision) != 40:
        raise ValueError(f"{label} must be a 40-character commit hash")
    try:
        int(revision, 16)
    except ValueError:
        raise ValueError(f"{label} must be a hexadecimal commit hash") from None
    return revision


def _expected_family_pairs() -> set[tuple[int, int]]:
    return {
        (family.sequence_length, family.induction_gap)
        for family in SEQUENCE_FAMILIES
    }


def validate_phase09_config(
    config: Mapping[str, Any],
    *,
    require_scientific: bool = False,
) -> Mapping[str, Any]:
    """Validate the Phase-09 fields used by artifact serialization.

    Unknown fields remain untouched so the frozen protocol can carry runtime,
    analysis, or audit settings beyond this module's model-free boundary.
    """

    config = _mapping(config, "Phase-09 config")
    if config.get("schema") != PHASE09_CONFIG_SCHEMA:
        raise ValueError("unexpected Phase-09 config schema")
    status = _nonempty(config.get("status"), "status")
    scientific_status = _nonempty(
        config.get("scientific_status"), "scientific_status"
    )
    if require_scientific and (
        status != "frozen_before_qwen_outcomes"
        or scientific_status != "prospective_second_model_second_mechanism"
    ):
        raise ValueError("scientific Phase-09 status is not frozen and prospective")

    model = _mapping(config.get("model"), "model")
    model_id = _nonempty(model.get("id"), "model.id")
    model_revision = _hex_revision(model.get("revision"), "model.revision")
    _nonempty(model.get("dtype"), "model.dtype")
    if "quantization" not in model:
        raise ValueError("model.quantization must be declared, including null")

    token_pool = _mapping(config.get("token_pool"), "token_pool")
    _integer(token_pool.get("seed"), "token_pool.seed")
    per_split_size = _integer(
        token_pool.get("per_split_size"),
        "token_pool.per_split_size",
        minimum=63,
    )
    for field in ("require_printable", "require_roundtrip", "exclude_special"):
        if not isinstance(token_pool.get(field), bool):
            raise ValueError(f"token_pool.{field} must be boolean")

    sequence = _mapping(config.get("sequence_design"), "sequence_design")
    _integer(sequence.get("seed"), "sequence_design.seed")
    families = sequence.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("sequence_design.families must be a non-empty list")
    family_pairs: list[tuple[int, int]] = []
    for index, raw_family in enumerate(families):
        family = _mapping(raw_family, f"sequence_design.families[{index}]")
        family_pairs.append(
            (
                _integer(
                    family.get("sequence_length"),
                    f"sequence_design.families[{index}].sequence_length",
                    minimum=2,
                ),
                _integer(
                    family.get("repeat_gap"),
                    f"sequence_design.families[{index}].repeat_gap",
                    minimum=2,
                ),
            )
        )
    if len(set(family_pairs)) != len(family_pairs):
        raise ValueError("sequence families must be unique")
    if not set(family_pairs).issubset(_expected_family_pairs()):
        raise ValueError("sequence config contains an unsupported length/gap family")
    prompts_per_family = _mapping(
        sequence.get("prompts_per_family"),
        "sequence_design.prompts_per_family",
    )
    if set(prompts_per_family) != set(_CONFIG_BANK_TO_DESIGN_BANK):
        raise ValueError("prompts_per_family must define the six frozen banks")
    for bank, count in prompts_per_family.items():
        _integer(count, f"sequence_design.prompts_per_family.{bank}", minimum=1)
    if _integer(
        sequence.get("n_key_value_pairs"),
        "sequence_design.n_key_value_pairs",
        minimum=1,
    ) != 3:
        raise ValueError("Phase-09 sequences must contain exactly three key-value pairs")

    discovery = _mapping(config.get("head_discovery"), "head_discovery")
    mask = _mapping(config.get("mask_design"), "mask_design")
    n_components = _integer(
        mask.get("n_components"), "mask_design.n_components", minimum=1
    )
    if _integer(discovery.get("selected_heads"), "head_discovery.selected_heads") != n_components:
        raise ValueError("selected-head count and mask component count disagree")
    if _integer(mask.get("universe_size"), "mask_design.universe_size") != 2**n_components:
        raise ValueError("mask universe size must equal two to the component count")
    calibration_count = _integer(
        mask.get("calibration_masks"), "mask_design.calibration_masks", minimum=1
    )
    test_count = _integer(
        mask.get("locked_test_masks"), "mask_design.locked_test_masks", minimum=1
    )
    if calibration_count + test_count != 2**n_components:
        raise ValueError("calibration and locked-test masks must partition the universe")
    budgets = mask.get("measurement_budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("mask_design.measurement_budgets must be a non-empty list")
    budget_values = tuple(
        _integer(value, "mask measurement budget", minimum=1) for value in budgets
    )
    if tuple(sorted(set(budget_values))) != budget_values or budget_values[-1] != calibration_count:
        raise ValueError("mask measurement budgets must be increasing and end at the calibration size")
    candidate_pools = _integer(
        mask.get("candidate_pools"), "mask_design.candidate_pools", minimum=1
    )
    masks_per_pool = _integer(
        mask.get("masks_per_pool_excluding_noop"),
        "mask_design.masks_per_pool_excluding_noop",
        minimum=1,
    )
    if candidate_pools * masks_per_pool != test_count:
        raise ValueError("candidate-pool geometry does not cover the locked-test masks")
    if mask.get("include_analytic_noop_in_every_pool") is not True:
        raise ValueError("every Phase-09 action pool must include analytic no-op")
    _integer(mask.get("seed"), "mask_design.seed")

    intervention = _mapping(config.get("intervention"), "intervention")
    if (
        intervention.get("primary") != "family_conditioned_mean_ablation"
        or intervention.get("position") != "final_query"
        or intervention.get("hook_point") != "attention_z_before_output_projection"
    ):
        raise ValueError("Phase-09 config has an unexpected primary intervention")

    collateral = config.get("collateral_diagnostic")
    if require_scientific:
        models = _mapping(config.get("models"), "models")
        if (
            _integer(
                models.get("primary_prediction_budget"),
                "models.primary_prediction_budget",
                minimum=1,
            )
            != 128
            or models.get("primary_prediction_contrast")
            != "additive_mae_minus_quadratic_mae"
        ):
            raise ValueError("scientific Phase-09 primary prediction contrast changed")
        collateral = _mapping(collateral, "collateral_diagnostic")
        expected_collateral = {
            "status": "secondary_not_gate",
            "prompt_bank": "locked_test",
            "control_transform": (
                "deterministic_exact_multiset_final_key_filler_swap"
            ),
            "action_scope": (
                "unique_masks_selected_by_frozen_actions_plus_analytic_noop"
            ),
            "intervention": "same_family_conditioned_mean_ablation",
            "kl_direction": "clean_to_intervened",
            "report_total_variation": True,
            "aggregation": ["mask_overall", "mask_by_family", "frozen_action"],
            "affects_primary_gate": False,
        }
        if dict(collateral) != expected_collateral:
            raise ValueError("scientific Phase-09 collateral diagnostic changed")

    _mapping(config.get("runtime"), "runtime")
    if require_scientific:
        required = {
            "schema",
            "status",
            "scientific_status",
            "model",
            "token_pool",
            "sequence_design",
            "clean_gate",
            "head_discovery",
            "intervention",
            "mask_design",
            "models",
            "targets",
            "collateral_diagnostic",
            "uncertainty",
            "runtime",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"scientific Phase-09 config lacks fields: {missing}")
        if status != "frozen_before_qwen_outcomes" or scientific_status != (
            "prospective_second_model_second_mechanism"
        ):
            raise ValueError("scientific Phase-09 status is not frozen and prospective")
        if model_id != QWEN_INDUCTION_MODEL_NAME or model_revision != (
            QWEN_INDUCTION_MODEL_REVISION
        ):
            raise ValueError("scientific Phase-09 config did not pin the declared base model")
        if set(family_pairs) != _expected_family_pairs():
            raise ValueError("scientific Phase-09 config must contain all four families")
        if n_components != 8 or calibration_count != 128 or test_count != 128:
            raise ValueError("scientific Phase-09 config must use the exhaustive eight-head design")
        if budget_values != QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS:
            raise ValueError("scientific Phase-09 measurement budgets changed")
        if candidate_pools != 16 or masks_per_pool != 8:
            raise ValueError("scientific Phase-09 pool geometry changed")
        if per_split_size < 63:
            raise ValueError("scientific token bank cannot support the longest family")
        for field in ("clean_gate", "models", "targets", "uncertainty"):
            _mapping(config.get(field), field)
    return config


def load_phase09_config(
    path: str | Path,
    *,
    require_scientific: bool = False,
) -> dict[str, Any]:
    """Read and validate one JSON Phase-09 config without discarding extras."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Phase-09 config is missing: {source}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Phase-09 config is not valid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("Phase-09 config must be a JSON object")
    validate_phase09_config(payload, require_scientific=require_scientific)
    return payload


def validate_exact_scientific_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require the exact preregistered production config, not a scaled fixture."""

    checked = validate_phase09_config(config, require_scientific=True)
    observed = json_sha256(checked)
    if observed != QWEN_INDUCTION_SCIENTIFIC_CONFIG_SHA256:
        raise ValueError(
            "scientific Phase-09 config differs from the frozen production digest"
        )
    return checked


def _config(
    value: Mapping[str, Any] | str | Path,
    *,
    require_scientific: bool,
    require_exact: bool = False,
) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        checked = load_phase09_config(value, require_scientific=require_scientific)
    else:
        checked = validate_phase09_config(
            value, require_scientific=require_scientific
        )
    if require_exact:
        validate_exact_scientific_config(checked)
    return checked


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in columns} for row in rows
        )


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError:
        raise FileNotFoundError(f"artifact is missing: {path}") from None
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"unexpected columns in {path}")
        return [dict(row) for row in reader]


def _bank_id_by_name(design: SequenceDesign) -> dict[str, str]:
    return {
        token_bank.bank: f"qbank_{token_bank.bank}_{token_bank.token_sha256[:12]}"
        for token_bank in design.token_banks
    }


def _sequence_row(
    example: Any,
    *,
    token_bank_id: str,
    split: str | None,
) -> dict[str, Any]:
    row = {
        "prompt_id": example.example_id,
        "family_id": example.family_id,
        "cluster_id": f"qcluster_{example.bank}_{example.family_index:04d}",
        "token_bank_id": token_bank_id,
        "input_ids": " ".join(map(str, example.tokens)),
        "target_token_id": example.target_value_token,
        "distractor_token_id_1": example.distractor_value_tokens[0],
        "distractor_token_id_2": example.distractor_value_tokens[1],
        "source_key_position": example.target_key_position,
        "source_value_position": example.target_key_position + 1,
        "query_position": example.final_key_position,
        "sequence_length": example.sequence_length,
        "repeat_gap": example.induction_gap,
    }
    if split is None:
        row["bank"] = example.bank
        row["key_positions"] = " ".join(map(str, example.key_positions))
        row["source_value_positions"] = " ".join(
            str(position + 1) for position in example.key_positions
        )
    else:
        row["split"] = split
    return row


def _validate_sequence_against_config(
    design: SequenceDesign,
    config: Mapping[str, Any],
) -> None:
    validate_sequence_design(design)
    token_pool = _mapping(config["token_pool"], "token_pool")
    sequence = _mapping(config["sequence_design"], "sequence_design")
    if design.per_split_size != int(token_pool["per_split_size"]):
        raise ValueError("sequence design and config use different token-bank sizes")
    if design.seed != int(sequence["seed"]):
        raise ValueError("sequence design and config use different seeds")
    configured_families = {
        f"length_{int(row['sequence_length'])}_gap_{int(row['repeat_gap'])}"
        for row in sequence["families"]
    }
    actual_families = {example.family_id for example in design.examples}
    if actual_families != configured_families:
        raise ValueError("sequence design and config contain different families")
    counts = _mapping(sequence["prompts_per_family"], "prompts_per_family")
    for config_bank, design_bank in _CONFIG_BANK_TO_DESIGN_BANK.items():
        if getattr(design.bank_counts, design_bank) != int(counts[config_bank]):
            raise ValueError(f"sequence count differs for {config_bank}")


def write_preselection_artifacts(
    sequence_design: SequenceDesign,
    config: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    require_scientific: bool = True,
    exact_scientific_config: bool = True,
) -> Path:
    """Write the outcome-free prompt/token bridge and preselection manifest.

    Engineering smoke runs must opt out explicitly.  The public frozen design
    and effect serializers retain their unconditional scientific-config gate.
    """

    frozen = _config(
        config,
        require_scientific=require_scientific,
        require_exact=bool(require_scientific and exact_scientific_config),
    )
    _validate_sequence_against_config(sequence_design, frozen)
    root = Path(artifacts_root)
    design_dir = root / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    bank_ids = _bank_id_by_name(sequence_design)

    all_rows = [
        _sequence_row(
            example,
            token_bank_id=bank_ids[example.bank],
            split=None,
        )
        for example in sequence_design.examples
    ]
    adapter_rows = []
    for example in sequence_design.examples:
        split = {"calibration": "train", "locked_test": "test"}.get(example.bank)
        if split is not None:
            adapter_rows.append(
                _sequence_row(
                    example,
                    token_bank_id=bank_ids[example.bank],
                    split=split,
                )
            )
    _write_csv(design_dir / "prompts_all.csv", all_rows, _PROMPTS_ALL_COLUMNS)
    _write_csv(design_dir / "prompts.csv", adapter_rows, _PROMPT_COLUMNS)

    token_bank_payload = {
        "schema": TOKEN_BANKS_SCHEMA,
        "sequence_design_sha256": sequence_design.design_sha256,
        "token_pool_size": sequence_design.token_pool_size,
        "token_pool_sha256": sequence_design.token_pool_sha256,
        "per_split_size": sequence_design.per_split_size,
        "banks": [
            {
                "bank": bank.bank,
                "token_bank_id": bank_ids[bank.bank],
                "token_ids": bank.token_ids,
                "token_count": bank.token_count,
                "token_sha256": bank.token_sha256,
            }
            for bank in sequence_design.token_banks
        ],
    }
    write_json(design_dir / "token_banks.json", token_bank_payload)
    artifact_names = ("prompts_all.csv", "prompts.csv", "token_banks.json")
    manifest = {
        "schema": PRESELECTION_MANIFEST_SCHEMA,
        "status": "prepared_before_model_outcomes",
        "scientific_outcomes_included": False,
        "model": {
            "requested_name": frozen["model"]["id"],
            "requested_revision": frozen["model"]["revision"],
        },
        "config_sha256": json_sha256(frozen),
        "sequence_design_sha256": sequence_design.design_sha256,
        "mask_design_included": False,
        "selected_heads_included": False,
        "prompt_counts": {
            bank: sum(example.bank == bank for example in sequence_design.examples)
            for bank in SEQUENCE_BANKS
        },
        "artifact_hashes": {
            name: file_sha256(design_dir / name) for name in artifact_names
        },
    }
    write_json(design_dir / "preselection_manifest.json", manifest)
    return design_dir / "preselection_manifest.json"


def _record_value(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    raise ValueError(f"selected-head record lacks one of: {', '.join(names)}")


def _selected_head_rows(
    selected_heads: Iterable[Any],
    panel_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = list(selected_heads)
    if len(records) != 8:
        raise ValueError("frozen scientific panel must contain exactly eight heads")
    rows = []
    for component_index, record in enumerate(records):
        supplied_index = None
        if isinstance(record, Mapping):
            supplied_index = record.get("component_index")
        elif hasattr(record, "component_index"):
            supplied_index = getattr(record, "component_index")
        if supplied_index is not None and int(supplied_index) != component_index:
            raise ValueError("selected-head component order changed")
        rows.append(
            {
                "component_index": component_index,
                "head_label": _nonempty(
                    _record_value(record, "head_label", "label"), "head label"
                ),
                "layer": _integer(_record_value(record, "layer"), "head layer"),
                "head": _integer(_record_value(record, "head"), "head index"),
                "kv_group": _integer(
                    _record_value(record, "kv_group"), "head KV group"
                ),
            }
        )
    labels = tuple(str(row["head_label"]) for row in rows)
    if labels != panel_ids:
        raise ValueError("selected-head order differs from the frozen mask panel")
    if len(set((int(row["layer"]), int(row["head"])) for row in rows)) != 8:
        raise ValueError("selected layer/head coordinates must be unique")
    return rows


def _mask_bits(mask: Any) -> str:
    return "".join(map(str, mask.bits))


def _pool_by_mask(mask_design: MaskDesign) -> dict[str, str]:
    no_op = mask_design.no_op.mask_id
    pool_by_mask: dict[str, str] = {}
    for pool in mask_design.action_pools:
        if len(pool.mask_ids) != 9 or pool.mask_ids[0] != no_op:
            raise ValueError("action pool does not contain analytic no-op plus eight masks")
        for mask_id in pool.mask_ids[1:]:
            if mask_id in pool_by_mask:
                raise ValueError("held-out mask appears in more than one action pool")
            pool_by_mask[mask_id] = pool.pool_id
    if set(pool_by_mask) != {mask.mask_id for mask in mask_design.heldout_masks}:
        raise ValueError("action pools do not cover the held-out mask bank")
    return pool_by_mask


def write_frozen_design_artifacts(
    mask_design: MaskDesign,
    selected_heads: Iterable[Any],
    config: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    all_design_gates_pass: bool,
    gate_artifacts: Mapping[str, str | Path],
    exact_scientific_config: bool = True,
) -> Path:
    """Write adapter-facing heads, masks, pools, and frozen design manifest."""

    frozen = _config(
        config,
        require_scientific=True,
        require_exact=exact_scientific_config,
    )
    validate_mask_design(mask_design)
    mask_config = _mapping(frozen["mask_design"], "mask_design")
    if mask_design.seed != int(mask_config["seed"]):
        raise ValueError("mask design and config use different seeds")
    if mask_design.supported_budgets != tuple(mask_config["measurement_budgets"]):
        raise ValueError("mask design and config use different budgets")
    root = Path(artifacts_root)
    design_dir = root / "design"
    preselection_path = design_dir / "preselection_manifest.json"
    try:
        preselection = json.loads(preselection_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("preselection artifacts must be written first") from None
    if preselection.get("schema") != PRESELECTION_MANIFEST_SCHEMA:
        raise ValueError("unexpected preselection manifest schema")
    if preselection.get("config_sha256") != json_sha256(frozen):
        raise ValueError("preselection manifest belongs to a different config")
    if all_design_gates_pass is not True:
        raise ValueError("a frozen scientific design requires passed causal gates")
    required_gate_labels = {"discovery_gate", "confirmation_gate"}
    if not isinstance(gate_artifacts, Mapping) or not required_gate_labels.issubset(
        gate_artifacts
    ):
        raise ValueError("frozen design requires discovery and confirmation gate proofs")
    copied_gate_paths: dict[str, Path] = {}
    gate_dir = design_dir / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    for label, raw_source in sorted(gate_artifacts.items()):
        if not isinstance(label, str) or not label or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in label
        ):
            raise ValueError("gate artifact labels must be lowercase safe identifiers")
        source = Path(raw_source)
        if not source.is_file():
            raise FileNotFoundError(f"gate proof is missing: {source}")
        suffix = source.suffix.lower()
        if suffix not in {".json", ".csv"}:
            raise ValueError("gate proofs must be JSON or CSV artifacts")
        destination = gate_dir / f"{label}{suffix}"
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        copied_gate_paths[label] = destination
    for label in required_gate_labels:
        path = copied_gate_paths[label]
        if path.suffix != ".json":
            raise ValueError(f"{label} proof must be JSON")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("passed") is not True:
            raise ValueError(f"{label} proof does not record a passed gate")

    head_rows = _selected_head_rows(
        selected_heads,
        mask_design.panel.component_ids,
    )
    calibration_rows = [
        {
            "measurement_order": order,
            "mask_id": mask.mask_id,
            "mask_bits": _mask_bits(mask),
            "n_heads": mask.cardinality,
            "bank": "calibration",
            "pool_id": "",
        }
        for order, mask in enumerate(mask_design.calibration_for(128), start=1)
    ]
    pool_by_mask = _pool_by_mask(mask_design)
    test_rows = [
        {
            "mask_id": mask.mask_id,
            "mask_bits": _mask_bits(mask),
            "n_heads": mask.cardinality,
            "bank": "test",
            "pool_id": pool_by_mask[mask.mask_id],
        }
        for mask in mask_design.heldout_masks
    ]
    _write_csv(design_dir / "selected_heads.csv", head_rows, _SELECTED_HEAD_COLUMNS)
    _write_csv(
        design_dir / "calibration_masks.csv",
        calibration_rows,
        _CALIBRATION_MASK_COLUMNS,
    )
    _write_csv(design_dir / "test_masks.csv", test_rows, _TEST_MASK_COLUMNS)

    artifact_names = (
        "selected_heads.csv",
        "calibration_masks.csv",
        "test_masks.csv",
        "prompts.csv",
        "prompts_all.csv",
        "token_banks.json",
        "preselection_manifest.json",
    )
    missing = [name for name in artifact_names if not (design_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen design inputs are missing: {missing}")
    identity = {
        "config_sha256": json_sha256(frozen),
        "sequence_design_sha256": preselection["sequence_design_sha256"],
        "mask_design_sha256": mask_design.design_sha256,
        "panel_sha256": mask_design.panel.panel_sha256,
    }
    manifest = {
        "schema": QWEN_INDUCTION_DESIGN_SCHEMA,
        "status": "frozen_before_outcomes",
        "design_id": f"qwen_induction_{json_sha256(identity)[:16]}",
        "data_version": QWEN_INDUCTION_EFFECT_DATA_VERSION,
        "all_design_gates_pass": bool(all_design_gates_pass),
        "measurement_budgets": list(QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS),
        "model": {
            "requested_name": QWEN_INDUCTION_MODEL_NAME,
            "requested_revision": QWEN_INDUCTION_MODEL_REVISION,
        },
        **identity,
        "gate_artifact_hashes": {
            label: {
                "path": path.relative_to(design_dir).as_posix(),
                "sha256": file_sha256(path),
            }
            for label, path in sorted(copied_gate_paths.items())
        },
        "artifact_hashes": {
            **{name: file_sha256(design_dir / name) for name in artifact_names},
            **{
                path.relative_to(design_dir).as_posix(): file_sha256(path)
                for path in copied_gate_paths.values()
            },
        },
    }
    write_json(design_dir / "design_manifest.json", manifest)
    return design_dir / "design_manifest.json"


def _rows(value: Iterable[Any]) -> list[Any]:
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
        except TypeError:
            pass
        else:
            if isinstance(records, list):
                return records
    return list(value)


def _optional_record_value(record: Any, name: str) -> Any | None:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _measurement_value(record: Any, name: str) -> Any:
    value = _optional_record_value(record, name)
    if value is None:
        raise ValueError(f"measurement row lacks {name}")
    return value


def _normalize_effect_rows(
    measurement_rows: Iterable[Any],
    design_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    prompt_rows = _read_csv(design_dir / "prompts.csv", _PROMPT_COLUMNS)
    calibration_rows = _read_csv(
        design_dir / "calibration_masks.csv", _CALIBRATION_MASK_COLUMNS
    )
    test_rows = _read_csv(design_dir / "test_masks.csv", _TEST_MASK_COLUMNS)
    prompt_by_id = {row["prompt_id"]: row for row in prompt_rows}
    mask_by_id = {row["mask_id"]: row for row in (*calibration_rows, *test_rows)}
    if len(prompt_by_id) != len(prompt_rows) or len(mask_by_id) != 256:
        raise ValueError("frozen prompt or mask IDs are not unique")

    normalized: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    seen: set[tuple[str, str]] = set()
    for record in _rows(measurement_rows):
        prompt_id = str(_measurement_value(record, "prompt_id"))
        mask_id = str(_measurement_value(record, "mask_id"))
        key = (prompt_id, mask_id)
        if key in seen or prompt_id not in prompt_by_id or mask_id not in mask_by_id:
            raise ValueError(f"invalid or duplicate measurement cell: {key}")
        seen.add(key)
        prompt = prompt_by_id[prompt_id]
        mask = mask_by_id[mask_id]
        split = prompt["split"]
        expected_bank = "calibration" if split == "train" else "test"
        if mask["bank"] != expected_bank:
            raise ValueError("measurement crosses the frozen prompt and mask splits")
        supplied_bits = _optional_record_value(record, "mask_bits")
        if supplied_bits is not None and str(supplied_bits) != mask["mask_bits"]:
            raise ValueError("measurement mask bits differ from the frozen design")
        clean = _finite(_measurement_value(record, "clean_margin"), "clean_margin")
        raw_ablated = _optional_record_value(record, "ablated_margin")
        raw_effect = _optional_record_value(record, "drop_from_clean")
        if raw_effect is None:
            raw_effect = _optional_record_value(record, "effect")
        if raw_ablated is None and raw_effect is None:
            raise ValueError("measurement requires ablated_margin or drop_from_clean")
        if raw_ablated is None:
            effect = _finite(raw_effect, "drop_from_clean")
            ablated = clean - effect
        elif raw_effect is None:
            ablated = _finite(raw_ablated, "ablated_margin")
            effect = clean - ablated
        else:
            ablated = _finite(raw_ablated, "ablated_margin")
            effect = _finite(raw_effect, "drop_from_clean")
            if not math.isclose(effect, clean - ablated, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("measurement effect arithmetic disagrees")
        normalized[split].append(
            {
                "schema_version": QWEN_INDUCTION_EFFECT_ROW_SCHEMA,
                "prompt_id": prompt_id,
                "split": split,
                "family_id": prompt["family_id"],
                "cluster_id": prompt["cluster_id"],
                "mask_id": mask_id,
                "mask_bits": mask["mask_bits"],
                "bank": mask["bank"],
                "pool_id": mask["pool_id"],
                "clean_margin": repr(clean),
                "ablated_margin": repr(ablated),
                "drop_from_clean": repr(effect),
            }
        )
    for split in normalized:
        normalized[split].sort(key=lambda row: (row["prompt_id"], row["mask_id"]))
    expected = {
        "train": sum(row["split"] == "train" for row in prompt_rows) * 128,
        "test": sum(row["split"] == "test" for row in prompt_rows) * 128,
    }
    for split, rows in normalized.items():
        if len(rows) != expected[split]:
            raise ValueError(
                f"{split} measurements are incomplete: expected {expected[split]}, got {len(rows)}"
            )
    return normalized


def _write_shards(
    effects_dir: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    shard_size: int,
) -> list[Path]:
    paths = []
    for start in range(0, len(rows), shard_size):
        stop = min(start + shard_size, len(rows))
        path = effects_dir / "shards" / split / f"effects_{start:06d}_{stop:06d}.csv"
        _write_csv(path, rows[start:stop], _EFFECT_COLUMNS)
        paths.append(path)
    return paths


def write_effect_artifacts(
    measurement_rows: Iterable[Any],
    config: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    shard_size: int = 16_384,
    exact_scientific_config: bool = True,
) -> Path:
    """Serialize supplied measurements without loading a model or choosing outcomes."""

    frozen = _config(
        config,
        require_scientific=True,
        require_exact=exact_scientific_config,
    )
    shard_size = _integer(shard_size, "shard_size", minimum=1)
    root = Path(artifacts_root)
    design_dir = root / "design"
    design_manifest_path = design_dir / "design_manifest.json"
    try:
        design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("frozen design manifest must be written first") from None
    if design_manifest.get("schema") != QWEN_INDUCTION_DESIGN_SCHEMA:
        raise ValueError("unexpected frozen design manifest schema")
    if design_manifest.get("config_sha256") != json_sha256(frozen):
        raise ValueError("frozen design belongs to a different config")
    normalized = _normalize_effect_rows(measurement_rows, design_dir)
    effects_dir = root / "effects"
    effects_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split, rows in normalized.items():
        combined = effects_dir / f"{split}_effects.csv"
        _write_csv(combined, rows, _EFFECT_COLUMNS)
        paths.append(combined)
        paths.extend(_write_shards(effects_dir, split, rows, shard_size))
    manifest = {
        "schema": QWEN_INDUCTION_EFFECT_RUN_SCHEMA,
        "status": "complete_locked_test_outcomes",
        "design_manifest_sha256": file_sha256(design_manifest_path),
        "model": {
            "requested_name": QWEN_INDUCTION_MODEL_NAME,
            "requested_revision": QWEN_INDUCTION_MODEL_REVISION,
        },
        "intervention": {
            "site": "final_query_head_z",
            "replacement": "family_conditioned_reference_mean",
            "n_selected_heads": 8,
        },
        "measurement_rows": {split: len(rows) for split, rows in normalized.items()},
        "artifacts": {
            path.relative_to(effects_dir).as_posix(): file_sha256(path)
            for path in paths
        },
    }
    write_json(effects_dir / "effect_manifest.json", manifest)
    validate_effect_artifacts(root)
    return effects_dir / "effect_manifest.json"


def validate_effect_artifacts(
    artifacts_root: str | Path,
) -> None:
    """Verify effect hashes, shard reconstruction, and adapter compatibility."""

    root = Path(artifacts_root)
    effects_dir = root / "effects"
    try:
        manifest = json.loads(
            (effects_dir / "effect_manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FileNotFoundError("effect manifest is missing") from None
    artifacts = _mapping(manifest.get("artifacts"), "effect artifacts")
    for label, expected_hash in artifacts.items():
        relative = Path(str(label))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("effect manifest contains an unsafe path")
        path = effects_dir / relative
        if not path.is_file() or file_sha256(path) != str(expected_hash):
            raise ValueError(f"frozen effect artifact hash mismatch: {label}")
    for split in ("train", "test"):
        combined = _read_csv(effects_dir / f"{split}_effects.csv", _EFFECT_COLUMNS)
        shard_paths = sorted((effects_dir / "shards" / split).glob("effects_*.csv"))
        if not shard_paths:
            raise FileNotFoundError(f"no {split} effect shards were found")
        reconstructed = [
            row
            for path in shard_paths
            for row in _read_csv(path, _EFFECT_COLUMNS)
        ]
        if reconstructed != combined:
            raise ValueError(f"{split} effect shards do not reconstruct the adapter table")
    load_qwen_induction_effect_prediction_task(
        root,
        measurement_budget=max(QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS),
        verify_hashes=True,
    )


__all__ = [
    "PHASE09_CONFIG_SCHEMA",
    "PRESELECTION_MANIFEST_SCHEMA",
    "QWEN_INDUCTION_SCIENTIFIC_CONFIG_SHA256",
    "TOKEN_BANKS_SCHEMA",
    "load_phase09_config",
    "validate_effect_artifacts",
    "validate_exact_scientific_config",
    "validate_phase09_config",
    "write_effect_artifacts",
    "write_frozen_design_artifacts",
    "write_preselection_artifacts",
]
