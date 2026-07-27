"""Staged, resumable runner for the preregistered Qwen induction study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The runner deliberately keeps model production separate from the inference-free
task adapter.  Scientific stages have a hash-checked order, intervention cells
checkpoint one mask at a time, and locked-test outcomes cannot be opened until
predictions and actions have been sealed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import (
    file_sha256,
    json_sha256,
    package_version,
    runtime_provenance,
)
from observerbench.tasks.qwen_induction.analysis import (
    bootstrap_aggregate_action_contrasts,
    bootstrap_action_contrasts,
    bootstrap_prediction_contrasts,
    effect_dispersion_decomposition,
    evaluate_fixed_actions,
    evaluate_mean_effect_predictions,
    freeze_actions,
    freeze_mean_effect_predictions,
    intervention_outcome_diagnostics,
    prediction_error_diagnostics,
)
from observerbench.tasks.qwen_induction.artifacts import (
    load_phase09_config,
    validate_effect_artifacts,
    validate_exact_scientific_config,
    validate_phase09_config,
    write_effect_artifacts,
    write_frozen_design_artifacts,
    write_preselection_artifacts,
)
from observerbench.tasks.qwen_induction.collateral import (
    CONTROL_SCHEMA,
    iter_collateral_distribution_shifts,
    make_matched_non_induction_controls,
)
from observerbench.tasks.qwen_induction.design import (
    BinaryMask,
    MaskDesign,
    SequenceBankCounts,
    build_mask_design,
    build_sequence_design,
)
from observerbench.tasks.qwen_induction.plant import (
    HeadAblationMeans,
    HeadRef,
    Qwen2InductionPlant,
    regular_token_pool,
)
from observerbench.tasks.qwen_induction.selection import (
    confirm_selected_vs_controls,
    evaluate_clean_gate,
    match_low_induction_controls,
    select_causal_heads,
    shortlist_attention_heads,
    summarize_singleton_effects,
)


RUN_AUDIT_SCHEMA = "observerbench.qwen_induction_phase09.run_audit.v1"
OBSERVER_FREEZE_SCHEMA = "observerbench.qwen_induction_phase09.observer_freeze.v1"
SOURCE_MANIFEST_SCHEMA = "observerbench.qwen_induction_phase09.source_manifest.v1"
SCIENTIFIC_STAGES: tuple[str, ...] = (
    "prepare",
    "discover",
    "confirm",
    "freeze-design",
    "measure-calibration",
    "freeze-observers",
    "measure-locked-test",
    "evaluate",
    "measure-collateral",
)
ALL_STAGES: tuple[str, ...] = (*SCIENTIFIC_STAGES, "engineering-smoke")


@dataclass(frozen=True)
class RuntimePrompt:
    """Prompt row reconstructed from the frozen, model-free CSV."""

    prompt_id: str
    bank: str
    family_id: str
    cluster_id: str
    tokens: tuple[int, ...]
    target_value_token: int
    distractor_value_tokens: tuple[int, int]
    key_positions: tuple[int, int, int]
    final_key_position: int

    @property
    def example_id(self) -> str:
        return self.prompt_id


@dataclass(frozen=True)
class RuntimeMask:
    mask_id: str
    bits: tuple[int, ...]


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    outputs: tuple[Path, ...] = ()


def _records_frame(records: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, Mapping):
            rows.append(dict(record))
        elif is_dataclass(record):
            rows.append(asdict(record))
        elif hasattr(record, "__dict__"):
            rows.append(vars(record).copy())
        else:
            raise TypeError(f"cannot serialize record of type {type(record).__name__}")
    return pd.DataFrame(rows)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def _read_csv(path: Path, *, strings: Sequence[str] = ()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={column: str for column in strings})
    except FileNotFoundError:
        raise FileNotFoundError(f"required Phase-09 artifact is missing: {path}") from None
    return frame


def _parse_bits(value: Any, expected: int) -> tuple[int, ...]:
    text = str(value).strip()
    if len(text) != expected or set(text) - {"0", "1"}:
        raise ValueError(f"mask_bits must contain exactly {expected} binary digits")
    return tuple(map(int, text))


def _head_refs(frame: pd.DataFrame) -> tuple[HeadRef, ...]:
    required = {"layer", "head", "kv_group"}
    if required - set(frame):
        raise ValueError("head table lacks layer, head, or KV-group coordinates")
    return tuple(
        HeadRef(int(row.layer), int(row.head), int(row.kv_group))
        for row in frame.itertuples(index=False)
    )


def _save_means(path: Path, means: HeadAblationMeans) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        family_ids=np.asarray(means.family_ids),
        layers=np.asarray([head.layer for head in means.heads], dtype=np.int64),
        heads=np.asarray([head.head for head in means.heads], dtype=np.int64),
        kv_groups=np.asarray([head.kv_group for head in means.heads], dtype=np.int64),
        values=np.asarray(means.values, dtype=np.float32),
        counts=np.asarray(means.counts, dtype=np.int64),
    )
    temporary.replace(path)


def _load_means(path: Path) -> HeadAblationMeans:
    try:
        archive = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"reference means are missing: {path}") from None
    with archive:
        family_ids = tuple(map(str, archive["family_ids"].tolist()))
        heads = tuple(
            HeadRef(int(layer), int(head), int(kv))
            for layer, head, kv in zip(
                archive["layers"], archive["heads"], archive["kv_groups"]
            )
        )
        values = np.asarray(archive["values"], dtype=np.float32)
        counts = tuple(map(int, archive["counts"].tolist()))
    return HeadAblationMeans(family_ids, heads, values, counts)


class Phase09Runner:
    """Compose the existing Phase-09 primitives into sealed runtime stages."""

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        artifacts_root: str | Path,
        *,
        device: str | None = None,
        local_files_only: bool = False,
        tokenizer_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        plant_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
        collateral_iterator: Callable[..., Iterable[tuple[str, list[Any]]]] | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_phase09_config(config)
        else:
            validate_phase09_config(config)
            self.config = dict(config)
        self.root = Path(artifacts_root)
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.tokenizer_factory = tokenizer_factory
        self.plant_factory = plant_factory
        self.collateral_iterator = (
            collateral_iterator or iter_collateral_distribution_shifts
        )
        self.scientific = self.config.get("status") == "frozen_before_qwen_outcomes"
        self.injected_runtime = (
            tokenizer_factory is not None
            or plant_factory is not None
            or collateral_iterator is not None
        )
        if self.scientific and not self.injected_runtime:
            validate_exact_scientific_config(self.config)
            self._validate_frozen_source_bundle()
        self.smoke_root = self.root / "engineering_smoke"
        self.work = self.root / "work" if self.scientific else self.smoke_root / "work"
        self.audit_path = self.work / "stage_audit.json"
        self._active_stage: str | None = None
        self._stage_plant_audits: list[dict[str, Any]] = []

    def _producer_source_hashes(self) -> dict[str, str]:
        repo = Path(__file__).resolve().parents[4]
        sources = (
            repo / "pyproject.toml",
            repo / "scripts" / "run_qwen_induction_phase09.py",
            repo / "src" / "observerbench" / "__init__.py",
            repo / "src" / "observerbench" / "control.py",
            repo / "src" / "observerbench" / "core.py",
            repo / "src" / "observerbench" / "effect_prediction.py",
            repo / "src" / "observerbench" / "metrics.py",
            repo / "src" / "observerbench" / "observers.py",
            repo / "src" / "observerbench" / "provenance.py",
            repo / "src" / "observerbench" / "tasks" / "__init__.py",
            repo / "src" / "observerbench" / "tasks" / "ctl1_adapter.py",
            repo / "src" / "observerbench" / "tasks" / "ctl1_analytic.py",
            repo / "src" / "observerbench" / "tasks" / "effect_registry.py",
            repo / "src" / "observerbench" / "tasks" / "registry.py",
            repo / "src" / "observerbench" / "tasks" / "ioi" / "__init__.py",
            repo / "src" / "observerbench" / "tasks" / "ioi" / "effect_task.py",
            repo / "src" / "observerbench" / "tasks" / "ioi" / "heads.py",
            repo / "src" / "observerbench" / "tasks" / "ioi" / "stage2d.py",
            Path(__file__).with_name("__init__.py"),
            Path(__file__).resolve(),
            Path(__file__).with_name("plant.py"),
            Path(__file__).with_name("design.py"),
            Path(__file__).with_name("selection.py"),
            Path(__file__).with_name("analysis.py"),
            Path(__file__).with_name("artifacts.py"),
            Path(__file__).with_name("effect_task.py"),
            Path(__file__).with_name("collateral.py"),
        )
        return {
            path.relative_to(repo).as_posix(): file_sha256(path) for path in sources
        }

    def _validate_frozen_source_bundle(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        path = (
            repo
            / "configs"
            / "revision"
            / "phase09"
            / "qwen_phase09_source_manifest.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "the frozen Phase-09 producer-source manifest is missing"
            ) from None
        if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ValueError("unexpected Phase-09 producer-source manifest schema")
        current = self._producer_source_hashes()
        if payload.get("artifact_hashes") != current:
            raise ValueError("Phase-09 producer source differs from its frozen manifest")
        observed_bundle = json_sha256(current)
        if (
            payload.get("source_bundle_sha256") != observed_bundle
            or payload.get("config_sha256") != json_sha256(self.config)
        ):
            raise ValueError("Phase-09 producer-source seal changed")

    @staticmethod
    def _hardware_record(plant: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"device": str(getattr(plant, "device", "unknown"))}
        try:  # pragma: no cover - hardware dependent
            import torch

            device = getattr(plant, "device", None)
            device_type = getattr(device, "type", str(device))
            record["torch_version"] = str(torch.__version__)
            if device_type == "cuda" and torch.cuda.is_available():
                index = 0 if getattr(device, "index", None) is None else int(device.index)
                properties = torch.cuda.get_device_properties(index)
                record.update(
                    {
                        "cuda_device_name": str(properties.name),
                        "cuda_capability": list(torch.cuda.get_device_capability(index)),
                        "cuda_total_memory_bytes": int(properties.total_memory),
                    }
                )
            elif device_type == "mps":
                record["mps_available"] = bool(
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                )
        except Exception:
            pass
        return record

    def _checkpoint_runtime_identity(self, plant: Any) -> dict[str, Any]:
        provenance = self._qwen_runtime_provenance()
        return {
            "python_version": provenance["python_version"],
            "dependencies": provenance["dependencies"],
            "plant_audit": getattr(plant, "_observerbench_runtime_audit", {}),
            "producer_source_hashes": self._producer_source_hashes(),
        }

    @staticmethod
    def _qwen_runtime_provenance() -> dict[str, Any]:
        provenance = runtime_provenance()
        provenance["dependencies"] = {
            **provenance["dependencies"],
            **{
                name: package_version(name)
                for name in ("transformers", "accelerate", "huggingface-hub")
            },
        }
        return provenance

    def _tokenizer(self) -> Any:
        if self.tokenizer_factory is not None:
            return self.tokenizer_factory(self.config)
        try:
            from transformers import AutoTokenizer
        except Exception as error:  # pragma: no cover - optional runtime
            raise ImportError("Phase-09 production requires observerbench[qwen]") from error
        model = self.config["model"]
        return AutoTokenizer.from_pretrained(
            model["id"],
            revision=model["revision"],
            local_files_only=self.local_files_only,
        )

    def _plant(self, attention_implementation: str) -> Any:
        if self.plant_factory is not None:
            plant = self.plant_factory(attention_implementation, self.config)
        else:
            model = self.config["model"]
            runtime = self.config["runtime"]
            plant = Qwen2InductionPlant.from_pretrained(
                model["id"],
                model["revision"],
                device=self.device or runtime["device"],
                dtype=model["dtype"],
                attention_implementation=attention_implementation,
                local_files_only=self.local_files_only,
            )
        self._check_architecture(plant)
        audit_payload: dict[str, Any]
        if self.scientific:
            audit_runtime = getattr(plant, "audit_runtime", None)
            if not callable(audit_runtime):
                raise ValueError("scientific Qwen plant does not expose audit_runtime")
            model = self.config["model"]
            audited = audit_runtime(
                expected_model_id=str(model["id"]),
                expected_revision=str(model["revision"]),
                expected_layers=int(model["expected_layers"]),
                expected_query_heads=int(model["expected_query_heads"]),
                expected_kv_heads=int(model["expected_kv_heads"]),
                expected_dtype=str(model["dtype"]),
                expected_attention_implementation=str(attention_implementation),
            )
            audit_payload = dict(audited)
        else:
            architecture = plant.architecture
            audit_payload = {
                "model_id": str(self.config["model"]["id"]),
                "requested_revision": str(self.config["model"]["revision"]),
                "n_layers": int(architecture.n_layers),
                "n_query_heads": int(architecture.n_query_heads),
                "n_kv_heads": int(architecture.n_kv_heads),
                "parameter_dtype": str(self.config["model"]["dtype"]),
                "attention_implementation": str(attention_implementation),
                "quantized": False,
            }
        audit_payload.update(self._hardware_record(plant))
        audit_payload["source_hashes"] = self._producer_source_hashes()
        setattr(plant, "_observerbench_runtime_audit", audit_payload)
        self._stage_plant_audits.append(audit_payload)
        return plant

    def _check_architecture(self, plant: Any) -> None:
        expected = self.config["model"]
        architecture = plant.architecture
        fields = (
            ("expected_layers", "n_layers"),
            ("expected_query_heads", "n_query_heads"),
            ("expected_kv_heads", "n_kv_heads"),
        )
        for config_name, actual_name in fields:
            if config_name in expected and int(expected[config_name]) != int(
                getattr(architecture, actual_name)
            ):
                raise ValueError(f"loaded model failed architecture gate: {config_name}")
        if self.scientific and not bool(architecture.qkv_bias_present):
            raise ValueError("loaded scientific model lacks the registered Q/K/V biases")

    @staticmethod
    def _release_plant(plant: Any) -> None:
        # Drop the heavyweight model from the caller-owned plant shell before
        # loading a plant with a different attention implementation.
        model = getattr(plant, "model", None)
        if model is not None:
            try:
                plant.model = None
            except (AttributeError, TypeError):
                pass
            del model
        gc.collect()
        try:  # pragma: no cover - hardware dependent
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
        except Exception:
            pass

    def _new_audit(self) -> dict[str, Any]:
        return {
            "schema": RUN_AUDIT_SCHEMA,
            "mode": "scientific" if self.scientific else "engineering_smoke_only",
            "config_sha256": json_sha256(self.config),
            "completed_stages": [],
            "stages": {},
            "terminal_status": None,
        }

    def _audit(self) -> dict[str, Any]:
        if not self.audit_path.exists():
            return self._new_audit()
        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        if payload.get("schema") != RUN_AUDIT_SCHEMA:
            raise ValueError("unexpected Phase-09 run-audit schema")
        if payload.get("config_sha256") != json_sha256(self.config):
            raise ValueError("Phase-09 run root already belongs to a different config")
        expected_mode = "scientific" if self.scientific else "engineering_smoke_only"
        if payload.get("mode") != expected_mode:
            raise ValueError("Phase-09 run root mixes scientific and smoke modes")
        return payload

    def _write_audit(self, audit: Mapping[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(audit, self.audit_path)

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("stage output escaped the Phase-09 artifact root") from error

    def _verify_stage_outputs(self, stage_record: Mapping[str, Any]) -> tuple[Path, ...]:
        outputs = tuple(self.root / name for name in stage_record.get("artifact_hashes", {}))
        for path in outputs:
            expected = stage_record["artifact_hashes"][self._relative(path)]
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"completed stage artifact changed: {path}")
        return outputs

    def _begin(self, stage: str) -> StageResult | None:
        if stage not in ALL_STAGES:
            raise ValueError(f"unknown Phase-09 stage: {stage}")
        if self.scientific and stage == "engineering-smoke":
            raise ValueError("engineering-smoke requires the non-scientific smoke config")
        if not self.scientific and stage != "engineering-smoke":
            raise ValueError("smoke config may run only the engineering-smoke stage")
        audit = self._audit()
        if stage in audit["completed_stages"]:
            if self.scientific:
                index = SCIENTIFIC_STAGES.index(stage)
                for prerequisite in SCIENTIFIC_STAGES[:index]:
                    self._verify_stage_outputs(audit["stages"][prerequisite])
            outputs = self._verify_stage_outputs(audit["stages"][stage])
            return StageResult(stage, str(audit["stages"][stage]["status"]), outputs)
        if audit.get("terminal_status") is not None:
            raise RuntimeError(
                f"run stopped after a preregistered gate: {audit['terminal_status']}"
            )
        if self.scientific:
            index = SCIENTIFIC_STAGES.index(stage)
            missing = [name for name in SCIENTIFIC_STAGES[:index] if name not in audit["completed_stages"]]
            if missing:
                raise RuntimeError(
                    f"stage {stage} is out of order; complete {', '.join(missing)} first"
                )
            for prerequisite in SCIENTIFIC_STAGES[:index]:
                self._verify_stage_outputs(audit["stages"][prerequisite])
        elif audit["completed_stages"]:
            raise RuntimeError("engineering smoke has already completed")
        self._active_stage = stage
        self._stage_plant_audits = []
        self._write_audit(audit)
        return None

    def _finish(
        self,
        stage: str,
        outputs: Sequence[Path],
        *,
        status: str = "complete",
        terminal_status: str | None = None,
    ) -> StageResult:
        audit = self._audit()
        if self._active_stage != stage:
            raise RuntimeError("stage runtime audit was not initialized")
        runtime_path = self.work / "runtime" / f"{stage}.json"
        _write_json_atomic(
            {
                "schema": "observerbench.qwen_induction.stage_runtime.v1",
                "stage": stage,
                "config_sha256": json_sha256(self.config),
                "runtime": self._qwen_runtime_provenance(),
                "plant_audits": self._stage_plant_audits,
                "producer_source_hashes": self._producer_source_hashes(),
            },
            runtime_path,
        )
        unique = tuple(
            dict.fromkeys(Path(path) for path in (*outputs, runtime_path))
        )
        for path in unique:
            if not path.is_file():
                raise FileNotFoundError(f"stage output was not written: {path}")
        audit["stages"][stage] = {
            "status": status,
            "artifact_hashes": {
                self._relative(path): file_sha256(path) for path in unique
            },
        }
        audit["completed_stages"].append(stage)
        if terminal_status is not None:
            audit["terminal_status"] = terminal_status
        self._write_audit(audit)
        self._active_stage = None
        self._stage_plant_audits = []
        return StageResult(stage, status, unique)

    def _prompts_root(self) -> Path:
        return self.root if self.scientific else self.smoke_root

    def _load_prompts(self, bank: str) -> tuple[RuntimePrompt, ...]:
        path = self._prompts_root() / "design" / "prompts_all.csv"
        frame = _read_csv(
            path,
            strings=(
                "prompt_id",
                "bank",
                "family_id",
                "cluster_id",
                "input_ids",
                "key_positions",
            ),
        )
        selected = frame.loc[frame["bank"].astype(str) == bank]
        if selected.empty:
            raise ValueError(f"frozen prompt table contains no {bank} prompts")
        prompts: list[RuntimePrompt] = []
        for row in selected.itertuples(index=False):
            tokens = tuple(map(int, str(row.input_ids).split()))
            values = (
                int(row.target_token_id),
                int(row.distractor_token_id_1),
                int(row.distractor_token_id_2),
            )
            key_positions = tuple(map(int, str(row.key_positions).split()))
            if len(key_positions) != 3:
                raise ValueError("private prompt row must retain all three key positions")
            prompts.append(
                RuntimePrompt(
                    prompt_id=str(row.prompt_id),
                    bank=str(row.bank),
                    family_id=str(row.family_id),
                    cluster_id=str(row.cluster_id),
                    tokens=tokens,
                    target_value_token=values[0],
                    distractor_value_tokens=(values[1], values[2]),
                    key_positions=key_positions,
                    final_key_position=int(row.query_position),
                )
            )
        return tuple(prompts)

    def _sequence_design(self) -> Any:
        tokenizer = self._tokenizer()
        token_config = self.config["token_pool"]
        sequence_config = self.config["sequence_design"]
        per_split = int(token_config["per_split_size"])
        pool = regular_token_pool(
            tokenizer,
            seed=int(token_config["seed"]),
            limit=6 * per_split,
        )
        counts = sequence_config["prompts_per_family"]
        return build_sequence_design(
            pool,
            bank_counts=SequenceBankCounts(
                reference=int(counts["reference"]),
                discovery=int(counts["discovery"]),
                head_fit=int(counts["head_fit"]),
                head_test=int(counts["head_confirmation"]),
                calibration=int(counts["calibration"]),
                locked_test=int(counts["locked_test"]),
            ),
            per_split_size=per_split,
            seed=int(sequence_config["seed"]),
        )

    def prepare(self) -> StageResult:
        previous = self._begin("prepare")
        if previous is not None:
            return previous
        validate_phase09_config(self.config, require_scientific=True)
        design = self._sequence_design()
        manifest = write_preselection_artifacts(
            design,
            self.config,
            self.root,
            exact_scientific_config=not self.injected_runtime,
        )
        design_dir = manifest.parent
        return self._finish(
            "prepare",
            (
                manifest,
                design_dir / "prompts_all.csv",
                design_dir / "prompts.csv",
                design_dir / "token_banks.json",
            ),
        )

    def _clean_gate(
        self,
        plant: Any,
        bank: str,
        *,
        batch_size: int,
        output_dir: Path,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        rows_path = output_dir / f"clean_{bank}.csv"
        gate_path = output_dir / f"clean_{bank}_gate.json"
        records = self._load_prompts(bank)
        if rows_path.exists():
            rows = _read_csv(rows_path, strings=("prompt_id", "family_id"))
            if set(rows["prompt_id"].astype(str)) != {row.prompt_id for row in records}:
                raise ValueError(f"cached clean scores do not match {bank} prompts")
        else:
            rows = _records_frame(plant.score_clean(records, batch_size=batch_size))
            _write_csv_atomic(rows, rows_path)
        gate_config = self.config["clean_gate"]
        gate = evaluate_clean_gate(
            rows,
            minimum_overall_accuracy=float(
                gate_config["minimum_candidate_accuracy_overall"]
            ),
            minimum_family_accuracy=float(
                gate_config["minimum_candidate_accuracy_per_family"]
            ),
            minimum_median_margin=float(
                gate_config["minimum_median_candidate_margin"]
            ),
        )
        payload = asdict(gate)
        payload["bank"] = bank
        write_json(gate_path, payload)
        return payload, (rows_path, gate_path)

    def _noop_gate(
        self,
        plant: Any,
        records: Sequence[RuntimePrompt],
        heads: Sequence[HeadRef],
        means: HeadAblationMeans,
        *,
        batch_size: int,
        path: Path,
    ) -> dict[str, Any]:
        result = dict(
            plant.noop_hook_parity(
                records, heads, means, batch_size=batch_size
            )
        )
        gate = self.config["clean_gate"]
        passed = bool(
            float(result["maximum_margin_error"])
            <= float(gate["maximum_noop_hook_margin_error"])
            and (
                bool(result["prediction_parity"])
                or not bool(gate["require_exact_candidate_prediction_parity"])
            )
        )
        result["passed"] = passed
        write_json(path, result)
        return result

    def _checkpoint_effects(
        self,
        plant: Any,
        records: Sequence[RuntimePrompt],
        heads: Sequence[HeadRef],
        masks: Sequence[RuntimeMask | BinaryMask],
        means: HeadAblationMeans,
        *,
        partition: str,
        batch_size: int,
        mode: str = "mean",
        position_scope: str = "final",
    ) -> pd.DataFrame:
        checkpoint_dir = self.work / "checkpoints" / partition
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        expected = {mask.mask_id: mask for mask in masks}
        extra = {
            path.stem for path in checkpoint_dir.glob("*.csv")
        } - set(expected)
        if extra:
            raise ValueError(f"checkpoint directory contains masks from another design: {sorted(extra)}")
        expected_prompts = {record.prompt_id for record in records}
        cluster_by_prompt = {
            record.prompt_id: record.cluster_id for record in records
        }

        def boolean_values(series: pd.Series, label: str) -> np.ndarray:
            if series.isna().any():
                raise ValueError(f"{label} contains missing values")
            if pd.api.types.is_bool_dtype(series):
                return series.to_numpy(bool)
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
                if not np.isin(numeric, (0.0, 1.0)).all():
                    raise ValueError(f"{label} must contain only booleans or 0/1")
                return numeric.astype(bool)
            normalized = series.astype(str).str.strip().str.lower()
            if not normalized.isin(("true", "false", "0", "1")).all():
                raise ValueError(f"{label} must contain only booleans or 0/1")
            return normalized.isin(("true", "1")).to_numpy(bool)

        checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        checkpoint_identity = {
            "schema": "observerbench.qwen_induction.mask_checkpoints.v1",
            "partition": partition,
            "config_sha256": json_sha256(self.config),
            "prompt_ids_sha256": json_sha256(tuple(record.prompt_id for record in records)),
            "heads_sha256": json_sha256(
                tuple((head.layer, head.head, head.kv_group) for head in heads)
            ),
            "reference_means_sha256": json_sha256(
                {
                    "family_ids": means.family_ids,
                    "heads": tuple(
                        (head.layer, head.head, head.kv_group) for head in means.heads
                    ),
                    "values": np.asarray(means.values).tolist(),
                    "counts": means.counts,
                }
            ),
            "masks_sha256": json_sha256(
                tuple((mask.mask_id, tuple(mask.bits)) for mask in masks)
            ),
            "intervention_mode": mode,
            "position_scope": position_scope,
            "runtime_identity_sha256": json_sha256(
                self._checkpoint_runtime_identity(plant)
            ),
            "producer_source_hashes": self._producer_source_hashes(),
        }
        if checkpoint_manifest_path.exists():
            checkpoint_manifest = json.loads(
                checkpoint_manifest_path.read_text(encoding="utf-8")
            )
            for key, value in checkpoint_identity.items():
                if checkpoint_manifest.get(key) != value:
                    raise ValueError("checkpoint manifest belongs to another measurement design")
            if not isinstance(checkpoint_manifest.get("checkpoint_hashes"), dict):
                raise ValueError("checkpoint manifest lacks per-mask hashes")
        else:
            checkpoint_manifest = {**checkpoint_identity, "checkpoint_hashes": {}}
            _write_json_atomic(checkpoint_manifest, checkpoint_manifest_path)

        def register_checkpoint(path: Path, mask_id: str) -> None:
            observed = file_sha256(path)
            registered = checkpoint_manifest["checkpoint_hashes"].get(mask_id)
            if registered is not None and registered != observed:
                raise ValueError(f"checkpoint hash changed after atomic write: {path}")
            if registered is None:
                checkpoint_manifest["checkpoint_hashes"][mask_id] = observed
                _write_json_atomic(checkpoint_manifest, checkpoint_manifest_path)

        def read_checkpoint(mask: RuntimeMask | BinaryMask) -> pd.DataFrame | None:
            path = checkpoint_dir / f"{mask.mask_id}.csv"
            if not path.exists():
                return None
            frame = _read_csv(
                path,
                strings=("prompt_id", "family_id", "mask_id", "mask_bits"),
            )
            if "cluster_id" not in frame:
                frame.insert(
                    2,
                    "cluster_id",
                    frame["prompt_id"].astype(str).map(cluster_by_prompt),
                )
                _write_csv_atomic(frame, path)
            required = {
                "prompt_id",
                "family_id",
                "cluster_id",
                "mask_id",
                "mask_bits",
                "clean_margin",
                "ablated_margin",
                "drop_from_clean",
                "clean_candidate_correct",
                "ablated_candidate_correct",
                "clean_top1_correct",
                "ablated_top1_correct",
                "clean_target_nll",
                "ablated_target_nll",
            }
            if required - set(frame) or len(frame) != len(records):
                raise ValueError(f"incomplete checkpoint: {path}")
            if set(frame["prompt_id"].astype(str)) != expected_prompts:
                raise ValueError(f"checkpoint prompt IDs changed: {path}")
            if any(
                str(row.cluster_id) != cluster_by_prompt[str(row.prompt_id)]
                for row in frame.itertuples(index=False)
            ):
                raise ValueError(f"checkpoint cluster IDs changed: {path}")
            expected_bits = "".join(map(str, mask.bits))
            if set(frame["mask_id"].astype(str)) != {mask.mask_id} or set(
                frame["mask_bits"].astype(str)
            ) != {expected_bits}:
                raise ValueError(f"checkpoint mask identity changed: {path}")
            numeric = frame[["clean_margin", "ablated_margin", "drop_from_clean"]].apply(
                pd.to_numeric, errors="raise"
            )
            if not np.isfinite(numeric.to_numpy(float)).all() or not np.allclose(
                numeric["clean_margin"] - numeric["ablated_margin"],
                numeric["drop_from_clean"],
                rtol=1e-7,
                atol=1e-7,
            ):
                raise ValueError(f"checkpoint contains invalid effect arithmetic: {path}")
            for column in (
                "clean_candidate_correct",
                "ablated_candidate_correct",
                "clean_top1_correct",
                "ablated_top1_correct",
            ):
                boolean_values(frame[column], column)
            nll = frame[["clean_target_nll", "ablated_target_nll"]].apply(
                pd.to_numeric, errors="raise"
            )
            if (
                not np.isfinite(nll.to_numpy(float)).all()
                or (nll.to_numpy(float) < 0.0).any()
            ):
                raise ValueError(f"checkpoint contains invalid target NLL: {path}")
            register_checkpoint(path, mask.mask_id)
            return frame

        missing = [mask for mask in masks if read_checkpoint(mask) is None]
        if missing:
            for yielded_id, rows in plant.iter_mask_effects(
                records,
                heads,
                missing,
                means,
                batch_size=batch_size,
                mode=mode,
                position_scope=position_scope,
            ):
                if yielded_id not in expected or yielded_id not in {
                    mask.mask_id for mask in missing
                }:
                    raise ValueError("plant yielded an unregistered mask checkpoint")
                frame = _records_frame(rows)
                frame.insert(
                    2,
                    "cluster_id",
                    frame["prompt_id"].astype(str).map(cluster_by_prompt),
                )
                _write_csv_atomic(frame, checkpoint_dir / f"{yielded_id}.csv")
                read_checkpoint(expected[yielded_id])

        frames: list[pd.DataFrame] = []
        for mask in masks:
            frame = read_checkpoint(mask)
            if frame is None:
                raise RuntimeError(f"plant did not complete mask {mask.mask_id}")
            order = {record.prompt_id: index for index, record in enumerate(records)}
            frame["_prompt_order"] = frame["prompt_id"].astype(str).map(order)
            frames.append(frame.sort_values("_prompt_order").drop(columns="_prompt_order"))
        baseline = frames[0].set_index(frames[0]["prompt_id"].astype(str))[
            [
                "clean_margin",
                "clean_candidate_correct",
                "clean_top1_correct",
                "clean_target_nll",
            ]
        ]
        baseline_margin = baseline["clean_margin"].astype(float)
        baseline_nll = baseline["clean_target_nll"].astype(float)
        baseline_candidate = boolean_values(
            baseline["clean_candidate_correct"], "clean_candidate_correct"
        )
        baseline_top1 = boolean_values(
            baseline["clean_top1_correct"], "clean_top1_correct"
        )
        for frame in frames[1:]:
            observed = frame.set_index(frame["prompt_id"].astype(str)).loc[
                baseline.index
            ]
            if (
                not np.allclose(
                    observed["clean_margin"].to_numpy(float),
                    baseline_margin.to_numpy(float),
                    rtol=1e-7,
                    atol=1e-7,
                )
                or not np.allclose(
                    observed["clean_target_nll"].to_numpy(float),
                    baseline_nll.to_numpy(float),
                    rtol=1e-7,
                    atol=1e-7,
                )
                or not np.array_equal(
                    boolean_values(
                        observed["clean_candidate_correct"],
                        "clean_candidate_correct",
                    ),
                    baseline_candidate,
                )
                or not np.array_equal(
                    boolean_values(
                        observed["clean_top1_correct"], "clean_top1_correct"
                    ),
                    baseline_top1,
                )
            ):
                raise ValueError("clean baseline changed across intervention masks")
        return pd.concat(frames, ignore_index=True)

    def _checkpoint_collateral(
        self,
        plant: Any,
        controls: Sequence[Any],
        source_records: Sequence[RuntimePrompt],
        heads: Sequence[HeadRef],
        masks: Sequence[RuntimeMask],
        means: HeadAblationMeans,
        *,
        batch_size: int,
    ) -> pd.DataFrame:
        """Measure one frozen action mask at a time on matched controls."""

        checkpoint_dir = self.work / "checkpoints" / "collateral"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        expected = {mask.mask_id: mask for mask in masks}
        if len(expected) != len(masks):
            raise ValueError("collateral action masks must have unique IDs")
        extra = {path.stem for path in checkpoint_dir.glob("*.csv")} - set(expected)
        if extra:
            raise ValueError(
                "collateral checkpoints contain masks outside the frozen actions: "
                f"{sorted(extra)}"
            )
        expected_controls = {control.prompt_id for control in controls}
        source_by_control = {
            control.prompt_id: control.source_prompt_id for control in controls
        }
        source_records_by_id = {record.prompt_id: record for record in source_records}
        cluster_by_control = {
            control.prompt_id: source_records_by_id[control.source_prompt_id].cluster_id
            for control in controls
        }
        family_by_control = {
            control.prompt_id: control.family_id for control in controls
        }
        manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        identity = {
            "schema": "observerbench.qwen_induction.collateral_checkpoints.v1",
            "config_sha256": json_sha256(self.config),
            "control_schema": CONTROL_SCHEMA,
            "controls_sha256": json_sha256(
                tuple(
                    (
                        control.prompt_id,
                        control.source_prompt_id,
                        control.family_id,
                        tuple(control.input_ids),
                        control.swapped_position,
                    )
                    for control in controls
                )
            ),
            "heads_sha256": json_sha256(
                tuple((head.layer, head.head, head.kv_group) for head in heads)
            ),
            "reference_means_sha256": json_sha256(
                {
                    "family_ids": means.family_ids,
                    "heads": tuple(
                        (head.layer, head.head, head.kv_group) for head in means.heads
                    ),
                    "values": np.asarray(means.values).tolist(),
                    "counts": means.counts,
                }
            ),
            "masks_sha256": json_sha256(
                tuple((mask.mask_id, tuple(mask.bits)) for mask in masks)
            ),
            "metrics": ["kl_clean_to_intervened", "total_variation"],
            "runtime_identity_sha256": json_sha256(
                self._checkpoint_runtime_identity(plant)
            ),
            "producer_source_hashes": self._producer_source_hashes(),
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in identity.items():
                if manifest.get(key) != value:
                    raise ValueError(
                        "collateral checkpoint manifest belongs to another design"
                    )
            if not isinstance(manifest.get("checkpoint_hashes"), dict):
                raise ValueError("collateral checkpoint manifest lacks mask hashes")
        else:
            manifest = {**identity, "checkpoint_hashes": {}}
            _write_json_atomic(manifest, manifest_path)

        def register(path: Path, mask_id: str) -> None:
            observed = file_sha256(path)
            registered = manifest["checkpoint_hashes"].get(mask_id)
            if registered is not None and registered != observed:
                raise ValueError(f"collateral checkpoint changed: {path}")
            if registered is None:
                manifest["checkpoint_hashes"][mask_id] = observed
                _write_json_atomic(manifest, manifest_path)

        def read(mask: RuntimeMask) -> pd.DataFrame | None:
            path = checkpoint_dir / f"{mask.mask_id}.csv"
            if not path.exists():
                return None
            frame = _read_csv(
                path,
                strings=(
                    "prompt_id",
                    "source_prompt_id",
                    "family_id",
                    "source_cluster_id",
                    "mask_id",
                    "mask_bits",
                ),
            )
            required = {
                "prompt_id",
                "source_prompt_id",
                "family_id",
                "source_cluster_id",
                "mask_id",
                "mask_bits",
                "kl_clean_to_intervened",
                "total_variation",
            }
            if required - set(frame) or len(frame) != len(controls):
                raise ValueError(f"incomplete collateral checkpoint: {path}")
            if set(frame["prompt_id"].astype(str)) != expected_controls or frame[
                "prompt_id"
            ].astype(str).duplicated().any():
                raise ValueError(f"collateral prompt IDs changed: {path}")
            for row in frame.itertuples(index=False):
                prompt_id = str(row.prompt_id)
                if (
                    str(row.source_prompt_id) != source_by_control[prompt_id]
                    or str(row.family_id) != family_by_control[prompt_id]
                    or str(row.source_cluster_id) != cluster_by_control[prompt_id]
                ):
                    raise ValueError(f"collateral control identity changed: {path}")
            expected_bits = "".join(map(str, mask.bits))
            if set(frame["mask_id"].astype(str)) != {mask.mask_id} or set(
                frame["mask_bits"].astype(str)
            ) != {expected_bits}:
                raise ValueError(f"collateral mask identity changed: {path}")
            metrics = frame[
                ["kl_clean_to_intervened", "total_variation"]
            ].apply(pd.to_numeric, errors="raise")
            values = metrics.to_numpy(float)
            if (
                not np.isfinite(values).all()
                or (metrics["kl_clean_to_intervened"] < 0.0).any()
                or (metrics["total_variation"] < 0.0).any()
                or (metrics["total_variation"] > 1.0).any()
            ):
                raise ValueError(f"collateral checkpoint contains invalid metrics: {path}")
            if not any(mask.bits) and not np.array_equal(
                values, np.zeros_like(values)
            ):
                raise ValueError("analytic collateral no-op must be exactly zero")
            register(path, mask.mask_id)
            return frame

        missing = [mask for mask in masks if read(mask) is None]
        if missing:
            missing_ids = {mask.mask_id for mask in missing}
            for yielded_id, rows in self.collateral_iterator(
                plant,
                controls,
                heads,
                missing,
                means,
                batch_size=batch_size,
                include_total_variation=True,
            ):
                if yielded_id not in expected or yielded_id not in missing_ids:
                    raise ValueError(
                        "collateral producer yielded an unregistered mask checkpoint"
                    )
                frame = _records_frame(rows)
                frame.insert(
                    3,
                    "source_cluster_id",
                    frame["prompt_id"].astype(str).map(cluster_by_control),
                )
                path = checkpoint_dir / f"{yielded_id}.csv"
                _write_csv_atomic(frame, path)
                read(expected[yielded_id])

        order = {control.prompt_id: index for index, control in enumerate(controls)}
        frames: list[pd.DataFrame] = []
        for mask in masks:
            frame = read(mask)
            if frame is None:
                raise RuntimeError(
                    f"collateral producer did not complete mask {mask.mask_id}"
                )
            frame["_prompt_order"] = frame["prompt_id"].astype(str).map(order)
            frames.append(
                frame.sort_values("_prompt_order").drop(columns="_prompt_order")
            )
        return pd.concat(frames, ignore_index=True)

    def discover(self) -> StageResult:
        previous = self._begin("discover")
        if previous is not None:
            return previous
        output = self.work / "discovery"
        output.mkdir(parents=True, exist_ok=True)
        runtime = self.config["runtime"]
        discovery_batch_size = int(runtime["discovery_batch_size"])
        measurement_batch_size = int(runtime["measurement_batch_size"])
        artifacts: list[Path] = []

        # Attention screening uses eager attention, but all causal outcomes and
        # their reference means use a freshly loaded SDPA plant.
        plant = self._plant(runtime["discovery_attention_implementation"])
        try:
            gate, paths = self._clean_gate(
                plant,
                "discovery",
                batch_size=discovery_batch_size,
                output_dir=output,
            )
            artifacts.extend(paths)
            if not gate["passed"]:
                return self._finish(
                    "discover",
                    artifacts,
                    status="gate_failed",
                    terminal_status="clean_gate_failed:discovery",
                )
            discovery_path = output / "attention_scan.csv"
            if discovery_path.exists():
                discovery = _read_csv(discovery_path)
            else:
                discovery = _records_frame(
                    plant.scan_attention(
                        self._load_prompts("discovery"),
                        batch_size=discovery_batch_size,
                    )
                )
                _write_csv_atomic(discovery, discovery_path)
            artifacts.append(discovery_path)
        finally:
            self._release_plant(plant)

        selection_config = self.config["head_discovery"]
        shortlist = shortlist_attention_heads(
            discovery, count=int(selection_config["attention_shortlist"])
        )
        shortlist_path = output / "attention_shortlist.csv"
        _write_csv_atomic(shortlist, shortlist_path)
        artifacts.append(shortlist_path)
        shortlist_heads = _head_refs(shortlist)

        plant = self._plant(runtime["measurement_attention_implementation"])
        try:
            gate, paths = self._clean_gate(
                plant,
                "head_fit",
                batch_size=measurement_batch_size,
                output_dir=output,
            )
            artifacts.extend(paths)
            if not gate["passed"]:
                return self._finish(
                    "discover",
                    artifacts,
                    status="gate_failed",
                    terminal_status="clean_gate_failed:head_fit",
                )
            means_path = output / "reference_shortlist_means_sdpa.npz"
            if means_path.exists():
                means = _load_means(means_path)
                if means.heads != shortlist_heads:
                    raise ValueError("cached shortlist means use a different head order")
            else:
                means = plant.capture_reference_means(
                    self._load_prompts("reference"),
                    shortlist_heads,
                    batch_size=measurement_batch_size,
                )
                _save_means(means_path, means)
            artifacts.append(means_path)
            noop_path = output / "head_fit_noop_parity.json"
            noop = self._noop_gate(
                plant,
                self._load_prompts("head_fit"),
                shortlist_heads,
                means,
                batch_size=measurement_batch_size,
                path=noop_path,
            )
            artifacts.append(noop_path)
            if not noop["passed"]:
                return self._finish(
                    "discover",
                    artifacts,
                    status="gate_failed",
                    terminal_status="noop_hook_parity_failed:head_fit",
                )
            singleton_masks = tuple(
                RuntimeMask(
                    f"singleton_{index:03d}",
                    tuple(1 if component == index else 0 for component in range(len(shortlist_heads))),
                )
                for index in range(len(shortlist_heads))
            )
            cells = self._checkpoint_effects(
                plant,
                self._load_prompts("head_fit"),
                shortlist_heads,
                singleton_masks,
                means,
                partition="head_fit_singletons",
                batch_size=measurement_batch_size,
            )
            index_by_mask = {mask.mask_id: index for index, mask in enumerate(singleton_masks)}
            mapped: list[dict[str, Any]] = []
            for row in cells.itertuples(index=False):
                component = index_by_mask[str(row.mask_id)]
                head = shortlist_heads[component]
                mapped.append(
                    {
                        "layer": head.layer,
                        "head": head.head,
                        "prompt_id": str(row.prompt_id),
                        "family_id": str(row.family_id),
                        "effect": float(row.drop_from_clean),
                    }
                )
            singleton_cells = pd.DataFrame(mapped)
            singleton_cells_path = output / "head_fit_singleton_cells.csv"
            _write_csv_atomic(singleton_cells, singleton_cells_path)
            artifacts.append(singleton_cells_path)
            summary = summarize_singleton_effects(
                singleton_cells,
                repeats=int(selection_config.get("bootstrap_repeats", 5000)),
                seed=int(selection_config.get("bootstrap_seed", 0)),
            )
            summary_path = output / "head_fit_singleton_summary.csv"
            _write_csv_atomic(summary, summary_path)
            artifacts.append(summary_path)
            try:
                selected = select_causal_heads(
                    shortlist,
                    summary,
                    count=int(selection_config["selected_heads"]),
                    maximum_per_layer=int(selection_config["maximum_selected_per_layer"]),
                    minimum_layers=int(selection_config["minimum_selected_layers"]),
                )
                controls = match_low_induction_controls(discovery, shortlist, selected)
            except ValueError as error:
                failure_path = output / "discovery_gate.json"
                write_json(
                    failure_path,
                    {"passed": False, "reason": str(error), "scientific_result": "negative"},
                )
                artifacts.append(failure_path)
                return self._finish(
                    "discover",
                    artifacts,
                    status="gate_failed",
                    terminal_status="causal_head_selection_failed",
                )
            selected_path = output / "selected_heads.csv"
            controls_path = output / "matched_controls.csv"
            _write_csv_atomic(selected, selected_path)
            _write_csv_atomic(controls, controls_path)
            artifacts.extend((selected_path, controls_path))
            gate_path = output / "discovery_gate.json"
            write_json(
                gate_path,
                {
                    "passed": bool(noop["passed"]),
                    "selected_heads": len(selected),
                    "selected_layers": int(selected["layer"].nunique()),
                    "scientific_result": "unclassified_until_confirmation",
                },
            )
            artifacts.append(gate_path)
            return self._finish("discover", artifacts)
        finally:
            self._release_plant(plant)

    def _selected_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        directory = self.work / "discovery"
        selected = _read_csv(
            directory / "selected_heads.csv", strings=("head_label",)
        ).sort_values("component_index")
        controls = _read_csv(directory / "matched_controls.csv")
        if "matched_component_index" in controls:
            controls = controls.sort_values("matched_component_index")
        if len(selected) != 8 or len(controls) != 8:
            raise ValueError("scientific confirmation requires eight selected and eight controls")
        return selected.reset_index(drop=True), controls.reset_index(drop=True)

    def confirm(self) -> StageResult:
        previous = self._begin("confirm")
        if previous is not None:
            return previous
        output = self.work / "confirmation"
        output.mkdir(parents=True, exist_ok=True)
        selected, controls = self._selected_tables()
        selected_heads = _head_refs(selected)
        control_heads = _head_refs(controls)
        all_heads = selected_heads + control_heads
        runtime = self.config["runtime"]
        batch_size = int(runtime["measurement_batch_size"])
        plant = self._plant(runtime["measurement_attention_implementation"])
        artifacts: list[Path] = []
        try:
            clean, paths = self._clean_gate(
                plant, "head_test", batch_size=batch_size, output_dir=output
            )
            artifacts.extend(paths)
            if not clean["passed"]:
                return self._finish(
                    "confirm",
                    artifacts,
                    status="gate_failed",
                    terminal_status="clean_gate_failed:head_confirmation",
                )
            combined_means_path = output / "reference_selected_and_control_means.npz"
            if combined_means_path.exists():
                means = _load_means(combined_means_path)
                if means.heads != all_heads:
                    raise ValueError("cached confirmation means use a different head order")
            else:
                means = plant.capture_reference_means(
                    self._load_prompts("reference"), all_heads, batch_size=batch_size
                )
                _save_means(combined_means_path, means)
            artifacts.append(combined_means_path)
            selected_means = HeadAblationMeans(
                means.family_ids,
                selected_heads,
                np.asarray(means.values[:, :8], dtype=np.float32),
                means.counts,
            )
            selected_means_path = output / "reference_selected_means.npz"
            _save_means(selected_means_path, selected_means)
            artifacts.append(selected_means_path)
            noop_path = output / "head_confirmation_noop_parity.json"
            noop = self._noop_gate(
                plant,
                self._load_prompts("head_test"),
                all_heads,
                means,
                batch_size=batch_size,
                path=noop_path,
            )
            artifacts.append(noop_path)
            if not noop["passed"]:
                return self._finish(
                    "confirm",
                    artifacts,
                    status="gate_failed",
                    terminal_status="noop_hook_parity_failed:head_confirmation",
                )
            masks = (
                RuntimeMask("selected_full", (1,) * 8 + (0,) * 8),
                RuntimeMask("control_full", (0,) * 8 + (1,) * 8),
            )
            effects = self._checkpoint_effects(
                plant,
                self._load_prompts("head_test"),
                all_heads,
                masks,
                means,
                partition="head_confirmation",
                batch_size=batch_size,
            )
            effects["arm"] = effects["mask_id"].map(
                {"selected_full": "selected", "control_full": "control"}
            )
            cells = effects[
                ["prompt_id", "family_id", "arm", "drop_from_clean"]
            ].rename(columns={"drop_from_clean": "effect"})
            cells_path = output / "confirmation_cells.csv"
            _write_csv_atomic(cells, cells_path)
            artifacts.append(cells_path)
            gate = confirm_selected_vs_controls(
                cells,
                repeats=int(self.config["head_discovery"].get("bootstrap_repeats", 5000)),
                seed=int(self.config["head_discovery"].get("bootstrap_seed", 0)) + 1,
            )
            fractions = tuple(map(float, self.config["targets"]["fractions"]))
            gate["targets"] = [float(gate["selected_mean_effect"]) * value for value in fractions]
            gate["scientific_result"] = "positive" if gate["passed"] else "negative"
            gate_path = output / "confirmation_gate.json"
            write_json(gate_path, gate)
            artifacts.append(gate_path)
            if not gate["passed"]:
                return self._finish(
                    "confirm",
                    artifacts,
                    status="gate_failed",
                    terminal_status="selected_vs_control_confirmation_failed",
                )
            zero_effects = self._checkpoint_effects(
                plant,
                self._load_prompts("head_test"),
                all_heads,
                masks,
                means,
                partition="head_confirmation_zero",
                batch_size=batch_size,
                mode="zero",
            )
            zero_effects["arm"] = zero_effects["mask_id"].map(
                {"selected_full": "selected", "control_full": "control"}
            )
            zero_cells = zero_effects[
                ["prompt_id", "family_id", "arm", "drop_from_clean"]
            ].rename(columns={"drop_from_clean": "effect"})
            zero_cells_path = output / "zero_confirmation_cells.csv"
            _write_csv_atomic(zero_cells, zero_cells_path)
            zero_summary = confirm_selected_vs_controls(
                zero_cells,
                repeats=int(
                    self.config["head_discovery"].get("bootstrap_repeats", 5000)
                ),
                seed=int(self.config["head_discovery"].get("bootstrap_seed", 0))
                + 2,
            )
            zero_summary.update(
                {
                    "robustness_only": True,
                    "affects_primary_gate": False,
                    "intervention": "final_query_head_zeroing",
                }
            )
            zero_summary_path = output / "zero_confirmation_summary.json"
            _write_json_atomic(zero_summary, zero_summary_path)
            artifacts.extend((zero_cells_path, zero_summary_path))
            return self._finish("confirm", artifacts)
        finally:
            self._release_plant(plant)

    def _mask_design(self, selected: pd.DataFrame | None = None) -> MaskDesign:
        if selected is None:
            selected, _ = self._selected_tables()
        labels = tuple(selected.sort_values("component_index")["head_label"].astype(str))
        return build_mask_design(labels, seed=int(self.config["mask_design"]["seed"]))

    def freeze_design(self) -> StageResult:
        previous = self._begin("freeze-design")
        if previous is not None:
            return previous
        selected, _ = self._selected_tables()
        mask_design = self._mask_design(selected)
        manifest = write_frozen_design_artifacts(
            mask_design,
            selected.to_dict("records"),
            self.config,
            self.root,
            all_design_gates_pass=True,
            gate_artifacts={
                "discovery_gate": self.work / "discovery" / "discovery_gate.json",
                "confirmation_gate": self.work / "confirmation" / "confirmation_gate.json",
                "confirmation_cells": self.work / "confirmation" / "confirmation_cells.csv",
                "zero_confirmation_cells": self.work
                / "confirmation"
                / "zero_confirmation_cells.csv",
                "zero_confirmation_summary": self.work
                / "confirmation"
                / "zero_confirmation_summary.json",
                "head_fit_noop_parity": self.work
                / "discovery"
                / "head_fit_noop_parity.json",
                "head_confirmation_noop_parity": self.work
                / "confirmation"
                / "head_confirmation_noop_parity.json",
                "discovery_runtime": self.work / "runtime" / "discover.json",
                "confirmation_runtime": self.work / "runtime" / "confirm.json",
            },
            exact_scientific_config=not self.injected_runtime,
        )
        return self._finish(
            "freeze-design",
            (
                manifest,
                manifest.parent / "selected_heads.csv",
                manifest.parent / "calibration_masks.csv",
                manifest.parent / "test_masks.csv",
            ),
        )

    def _measurement_stage(
        self,
        *,
        stage: str,
        bank: str,
        masks: Sequence[BinaryMask],
        partition: str,
        raw_name: str,
    ) -> StageResult:
        previous = self._begin(stage)
        if previous is not None:
            return previous
        if bank == "locked_test":
            self._verify_observer_freeze()
        output = self.work / "measurements"
        output.mkdir(parents=True, exist_ok=True)
        selected, _ = self._selected_tables()
        heads = _head_refs(selected)
        means_path = self.work / "confirmation" / "reference_selected_means.npz"
        means = _load_means(means_path)
        if means.heads != heads:
            raise ValueError("frozen confirmation means do not match the head panel")
        runtime = self.config["runtime"]
        batch_size = int(runtime["measurement_batch_size"])
        plant = self._plant(runtime["measurement_attention_implementation"])
        artifacts: list[Path] = []
        try:
            clean, paths = self._clean_gate(
                plant, bank, batch_size=batch_size, output_dir=output
            )
            artifacts.extend(paths)
            if not clean["passed"]:
                return self._finish(
                    stage,
                    artifacts,
                    status="gate_failed",
                    terminal_status=f"clean_gate_failed:{bank}",
                )
            noop_path = output / f"noop_parity_{bank}.json"
            noop = self._noop_gate(
                plant,
                self._load_prompts(bank),
                heads,
                means,
                batch_size=batch_size,
                path=noop_path,
            )
            artifacts.append(noop_path)
            if not noop["passed"]:
                return self._finish(
                    stage,
                    artifacts,
                    status="gate_failed",
                    terminal_status=f"noop_hook_parity_failed:{bank}",
                )
            effects = self._checkpoint_effects(
                plant,
                self._load_prompts(bank),
                heads,
                masks,
                means,
                partition=partition,
                batch_size=batch_size,
            )
            effects.insert(2, "split", "calibration" if bank == "calibration" else "locked_test")
            raw_path = output / raw_name
            _write_csv_atomic(effects, raw_path)
            artifacts.append(raw_path)
            return self._finish(stage, artifacts)
        finally:
            self._release_plant(plant)

    def measure_calibration(self) -> StageResult:
        design = self._mask_design()
        return self._measurement_stage(
            stage="measure-calibration",
            bank="calibration",
            masks=design.calibration_for(128),
            partition="calibration",
            raw_name="calibration_effects_raw.csv",
        )

    def _locked_outcome_paths(self) -> tuple[Path, ...]:
        paths = [self.work / "measurements" / "locked_test_effects_raw.csv"]
        checkpoint_dir = self.work / "checkpoints" / "locked_test"
        if checkpoint_dir.exists():
            paths.extend(checkpoint_dir.glob("*.csv"))
        paths.extend(
            (
                self.root / "effects" / "test_effects.csv",
                self.root / "effects" / "effect_manifest.json",
            )
        )
        return tuple(path for path in paths if path.exists())

    def freeze_observers(self) -> StageResult:
        previous = self._begin("freeze-observers")
        if previous is not None:
            return previous
        leaked = self._locked_outcome_paths()
        if leaked:
            raise RuntimeError(
                "locked-test outcomes exist before observer freeze: "
                + ", ".join(map(str, leaked))
            )
        design_dir = self.root / "design"
        calibration_path = self.work / "measurements" / "calibration_effects_raw.csv"
        calibration = _read_csv(
            calibration_path, strings=("prompt_id", "family_id", "mask_id", "mask_bits", "split")
        )
        calibration_masks_path = design_dir / "calibration_masks.csv"
        locked_masks_path = design_dir / "test_masks.csv"
        calibration_masks = _read_csv(calibration_masks_path, strings=("mask_id", "mask_bits"))
        locked_masks = _read_csv(
            locked_masks_path, strings=("mask_id", "mask_bits", "pool_id")
        )
        models = self.config["models"]
        prediction = freeze_mean_effect_predictions(
            calibration,
            calibration_masks,
            locked_masks,
            budgets=tuple(map(int, self.config["mask_design"]["measurement_budgets"])),
            ridge_grid=tuple(map(float, models["ridge_grid"])),
            seed=int(self.config["uncertainty"]["bootstrap_seed"]),
        )
        confirmation_path = self.work / "confirmation" / "confirmation_gate.json"
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        targets = tuple(map(float, confirmation["targets"]))
        actions = freeze_actions(
            calibration,
            calibration_masks,
            locked_masks,
            targets,
            ridge_grid=tuple(map(float, models["ridge_grid"])),
            seed=int(self.config["uncertainty"]["bootstrap_seed"]) + 26,
        )
        freeze_dir = self.work / "observer_freeze"
        freeze_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "mean_effect_predictions.csv": prediction["predictions"],
            "mean_effect_coefficients.csv": prediction["coefficients"],
            "mean_effect_ridge_diagnostics.csv": prediction["ridge_diagnostics"],
            "action_candidate_predictions.csv": actions["candidate_predictions"],
            "fixed_actions.csv": actions["fixed_actions"],
            "action_coefficients.csv": actions["coefficients"],
            "action_ridge_diagnostics.csv": actions["ridge_diagnostics"],
        }
        paths: list[Path] = []
        for name, frame in tables.items():
            path = freeze_dir / name
            _write_csv_atomic(frame, path)
            paths.append(path)
        source_paths = (
            calibration_path,
            calibration_masks_path,
            locked_masks_path,
            confirmation_path,
            design_dir / "design_manifest.json",
            self.work / "runtime" / "discover.json",
            self.work / "runtime" / "confirm.json",
            self.work / "runtime" / "measure-calibration.json",
        )
        manifest = {
            "schema": OBSERVER_FREEZE_SCHEMA,
            "status": "sealed_before_locked_test_outcomes",
            "locked_outcomes_read": False,
            "config_sha256": json_sha256(self.config),
            "targets": list(targets),
            "source_hashes": {
                self._relative(path): file_sha256(path) for path in source_paths
            },
            "artifact_hashes": {
                self._relative(path): file_sha256(path) for path in paths
            },
            "runtime": {
                **self._qwen_runtime_provenance(),
                "producer_source_hashes": self._producer_source_hashes(),
                "plant_runtime_audit_hashes": {
                    self._relative(path): file_sha256(path)
                    for path in source_paths
                    if path.parent.name == "runtime"
                },
            },
        }
        manifest_path = freeze_dir / "observer_freeze_manifest.json"
        write_json(manifest_path, manifest)
        paths.append(manifest_path)
        return self._finish("freeze-observers", paths)

    def _verify_observer_freeze(self) -> Mapping[str, Any]:
        path = self.work / "observer_freeze" / "observer_freeze_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RuntimeError("observer freeze must exist before locked-test measurement") from None
        if payload.get("schema") != OBSERVER_FREEZE_SCHEMA or payload.get("status") != (
            "sealed_before_locked_test_outcomes"
        ):
            raise ValueError("observer-freeze manifest is invalid")
        if payload.get("config_sha256") != json_sha256(self.config):
            raise ValueError("observer freeze belongs to a different config")
        if payload.get("locked_outcomes_read") is not False:
            raise ValueError("observer freeze did not certify outcome blindness")
        for section in ("source_hashes", "artifact_hashes"):
            hashes = payload.get(section)
            if not isinstance(hashes, Mapping) or not hashes:
                raise ValueError(f"observer freeze has no {section}")
            for relative, expected in hashes.items():
                artifact = self.root / relative
                if not artifact.is_file() or file_sha256(artifact) != expected:
                    raise ValueError(f"sealed observer input changed: {artifact}")
        return payload

    def measure_locked_test(self) -> StageResult:
        design = self._mask_design()
        return self._measurement_stage(
            stage="measure-locked-test",
            bank="locked_test",
            masks=design.heldout_masks,
            partition="locked_test",
            raw_name="locked_test_effects_raw.csv",
        )

    def measure_collateral(self) -> StageResult:
        """Measure frozen actions on deterministic matched non-induction controls."""

        previous = self._begin("measure-collateral")
        if previous is not None:
            return previous
        self._verify_observer_freeze()
        freeze_dir = self.work / "observer_freeze"
        actions = _read_csv(
            freeze_dir / "fixed_actions.csv",
            strings=(
                "selector",
                "pool_id",
                "selected_mask_id",
                "selected_mask_bits",
            ),
        )
        locked_masks = _read_csv(
            self.root / "design" / "test_masks.csv",
            strings=("mask_id", "mask_bits", "pool_id"),
        )
        mask_bits_by_id = dict(
            zip(
                locked_masks["mask_id"].astype(str),
                locked_masks["mask_bits"].astype(str),
            )
        )
        selected_bits: dict[str, str] = {}
        for row in actions.itertuples(index=False):
            mask_id = str(row.selected_mask_id)
            bits = str(row.selected_mask_bits)
            if len(bits) != 8 or set(bits) - {"0", "1"}:
                raise ValueError("frozen collateral action has invalid mask bits")
            previous_bits = selected_bits.setdefault(mask_id, bits)
            if previous_bits != bits:
                raise ValueError("one frozen action mask ID has conflicting bits")
            if mask_id == "analytic_noop":
                if bits != "00000000":
                    raise ValueError("analytic no-op action has a nonempty mask")
            elif mask_bits_by_id.get(mask_id) != bits:
                raise ValueError("frozen action is outside its locked candidate design")
        if selected_bits.get("analytic_noop") != "00000000":
            raise ValueError("frozen actions must retain the exact analytic no-op")
        masks = tuple(
            RuntimeMask(mask_id, tuple(map(int, selected_bits[mask_id])))
            for mask_id in sorted(
                selected_bits,
                key=lambda value: (value != "analytic_noop", value),
            )
        )

        source_records = self._load_prompts("locked_test")
        controls = make_matched_non_induction_controls(source_records)
        output = self.work / "collateral"
        output.mkdir(parents=True, exist_ok=True)
        controls_frame = pd.DataFrame(
            [
                {
                    "prompt_id": control.prompt_id,
                    "source_prompt_id": control.source_prompt_id,
                    "family_id": control.family_id,
                    "source_cluster_id": source.cluster_id,
                    "query_position": control.query_position,
                    "swapped_position": control.swapped_position,
                    "source_input_ids": " ".join(map(str, control.source_input_ids)),
                    "input_ids": " ".join(map(str, control.input_ids)),
                    "schema_version": control.schema_version,
                }
                for control, source in zip(controls, source_records)
            ]
        )
        controls_path = output / "matched_non_induction_controls.csv"
        _write_csv_atomic(controls_frame, controls_path)

        selected, _ = self._selected_tables()
        heads = _head_refs(selected)
        means = _load_means(
            self.work / "confirmation" / "reference_selected_means.npz"
        )
        if means.heads != heads:
            raise ValueError("collateral means do not match the frozen head panel")
        runtime = self.config["runtime"]
        plant = self._plant(runtime["measurement_attention_implementation"])
        try:
            shifts = self._checkpoint_collateral(
                plant,
                controls,
                source_records,
                heads,
                masks,
                means,
                batch_size=int(runtime["measurement_batch_size"]),
            )
        finally:
            self._release_plant(plant)
        raw_path = output / "collateral_shifts_raw.csv"
        _write_csv_atomic(shifts, raw_path)

        summary_rows: list[dict[str, Any]] = []
        for mask_id, mask_group in shifts.groupby("mask_id", sort=True):
            for family_id, group in (
                [("all", mask_group)]
                + list(mask_group.groupby("family_id", sort=True))
            ):
                summary_rows.append(
                    {
                        "mask_id": str(mask_id),
                        "mask_bits": str(group.iloc[0]["mask_bits"]),
                        "family_id": str(family_id),
                        "n_controls": len(group),
                        "mean_kl_clean_to_intervened": float(
                            group["kl_clean_to_intervened"].astype(float).mean()
                        ),
                        "median_kl_clean_to_intervened": float(
                            group["kl_clean_to_intervened"].astype(float).median()
                        ),
                        "maximum_kl_clean_to_intervened": float(
                            group["kl_clean_to_intervened"].astype(float).max()
                        ),
                        "mean_total_variation": float(
                            group["total_variation"].astype(float).mean()
                        ),
                    }
                )
        summary_frame = pd.DataFrame(summary_rows)
        summary_path = output / "collateral_shift_summary.csv"
        _write_csv_atomic(summary_frame, summary_path)
        overall = summary_frame.loc[summary_frame["family_id"] == "all"].drop(
            columns="family_id"
        )
        action_summary = actions.merge(
            overall,
            left_on=["selected_mask_id", "selected_mask_bits"],
            right_on=["mask_id", "mask_bits"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["mask_id", "mask_bits"])
        if action_summary["mean_kl_clean_to_intervened"].isna().any():
            raise ValueError("a frozen action lacks its collateral measurement")
        action_path = output / "collateral_by_frozen_action.csv"
        _write_csv_atomic(action_summary, action_path)
        summary_json_path = output / "summary.json"
        _write_json_atomic(
            {
                "status": "complete_secondary_diagnostic",
                "affects_primary_gate": False,
                "control_schema": CONTROL_SCHEMA,
                "prompt_bank": "locked_test",
                "action_scope": (
                    "unique_masks_selected_by_frozen_actions_plus_analytic_noop"
                ),
                "kl_direction": "clean_to_intervened",
                "n_controls": len(controls),
                "n_unique_action_masks": len(masks),
                "analytic_noop_exact_zero": bool(
                    np.array_equal(
                        shifts.loc[
                            shifts["mask_id"] == "analytic_noop",
                            ["kl_clean_to_intervened", "total_variation"],
                        ].to_numpy(float),
                        np.zeros((len(controls), 2), dtype=float),
                    )
                ),
            },
            summary_json_path,
        )
        return self._finish(
            "measure-collateral",
            (
                controls_path,
                raw_path,
                summary_path,
                action_path,
                summary_json_path,
            ),
        )

    def evaluate(self) -> StageResult:
        previous = self._begin("evaluate")
        if previous is not None:
            return previous
        self._verify_observer_freeze()
        freeze_dir = self.work / "observer_freeze"
        measurement_dir = self.work / "measurements"
        design_dir = self.root / "design"
        prediction = _read_csv(
            freeze_dir / "mean_effect_predictions.csv", strings=("model", "mask_id", "mask_bits")
        )
        actions = _read_csv(
            freeze_dir / "fixed_actions.csv",
            strings=("selector", "pool_id", "selected_mask_id", "selected_mask_bits"),
        )
        locked = _read_csv(
            measurement_dir / "locked_test_effects_raw.csv",
            strings=("prompt_id", "family_id", "mask_id", "mask_bits", "split"),
        )
        calibration = _read_csv(
            measurement_dir / "calibration_effects_raw.csv",
            strings=("prompt_id", "family_id", "mask_id", "mask_bits", "split"),
        )
        locked_masks = _read_csv(
            design_dir / "test_masks.csv", strings=("mask_id", "mask_bits", "pool_id")
        )
        prediction_metrics = evaluate_mean_effect_predictions(prediction, locked, locked_masks)
        action_metrics, oracles = evaluate_fixed_actions(actions, locked, locked_masks)
        uncertainty = self.config["uncertainty"]
        repeats = int(uncertainty["bootstrap_repeats"])
        seed = int(uncertainty["bootstrap_seed"])
        interval = float(uncertainty["interval"])
        prediction_contrasts = bootstrap_prediction_contrasts(
            prediction,
            locked,
            locked_masks,
            repeats=repeats,
            seed=seed,
            interval=interval,
        )
        action_contrasts = bootstrap_action_contrasts(
            actions,
            locked,
            locked_masks,
            repeats=repeats,
            seed=seed + 1,
            interval=interval,
        )
        aggregate_action_contrasts = bootstrap_aggregate_action_contrasts(
            actions,
            locked,
            locked_masks,
            repeats=repeats,
            seed=seed + 2,
            interval=interval,
        )
        freeze_manifest = self._verify_observer_freeze()
        dispersion = effect_dispersion_decomposition(
            locked,
            locked_masks,
            tuple(map(float, freeze_manifest["targets"])),
        )
        prediction_diagnostics = prediction_error_diagnostics(
            prediction,
            locked,
            locked_masks,
        )
        outcome_diagnostics = intervention_outcome_diagnostics(
            locked,
            locked_masks,
        )
        evaluation_dir = self.work / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for name, frame in (
            ("mean_effect_metrics.csv", prediction_metrics),
            ("fixed_action_metrics.csv", action_metrics),
            ("pool_oracles.csv", oracles),
            ("prediction_bootstrap_contrasts.csv", prediction_contrasts),
            ("action_bootstrap_contrasts.csv", action_contrasts),
            (
                "aggregate_action_bootstrap_contrasts.csv",
                aggregate_action_contrasts,
            ),
            ("effect_dispersion_decomposition.csv", dispersion),
            ("mean_effect_mask_errors.csv", prediction_diagnostics["mask_errors"]),
            (
                "mean_effect_density_summary.csv",
                prediction_diagnostics["density_summary"],
            ),
            (
                "mean_effect_prompt_cell_residual_summary.csv",
                prediction_diagnostics["prompt_error_summary"],
            ),
            (
                "intervention_outcomes_by_mask.csv",
                outcome_diagnostics["by_mask"],
            ),
            (
                "intervention_outcomes_by_density.csv",
                outcome_diagnostics["by_density"],
            ),
            (
                "intervention_outcomes_by_family.csv",
                outcome_diagnostics["by_family"],
            ),
        ):
            path = evaluation_dir / name
            _write_csv_atomic(frame, path)
            outputs.append(path)
        full_budget = int(self.config["models"]["primary_prediction_budget"])
        full = prediction_metrics.loc[prediction_metrics["measurement_budget"] == full_budget]
        mae = dict(zip(full["model"].astype(str), full["mae"].astype(float)))
        primary_prediction = prediction_contrasts.loc[
            prediction_contrasts["measurement_budget"] == full_budget
        ]
        if len(primary_prediction) != 1:
            raise ValueError("primary prediction contrast is missing or duplicated")
        primary_prediction_row = primary_prediction.iloc[0]
        selector_loss = action_metrics.groupby("selector")["actual_target_loss"].mean().to_dict()
        summary = {
            "status": "complete_locked_evaluation",
            "primary_prediction_budget": full_budget,
            "primary_prediction_contrast": str(
                self.config["models"]["primary_prediction_contrast"]
            ),
            "primary_prediction_estimate": float(primary_prediction_row["estimate"]),
            "primary_prediction_ci_lower": float(primary_prediction_row["ci_lower"]),
            "primary_prediction_ci_upper": float(primary_prediction_row["ci_upper"]),
            "primary_quadratic_better": bool(
                primary_prediction_row["quadratic_better"]
            ),
            "additive_minus_quadratic_mae": (
                float(mae["additive"] - mae["quadratic"])
                if {"additive", "quadratic"}.issubset(mae)
                else None
            ),
            "mean_actual_target_loss_by_selector": {
                str(key): float(value) for key, value in selector_loss.items()
            },
            "causal_confirmation": "passed",
        }
        summary_path = evaluation_dir / "summary.json"
        write_json(summary_path, summary)
        outputs.append(summary_path)
        effect_manifest = write_effect_artifacts(
            pd.concat([calibration, locked], ignore_index=True),
            self.config,
            self.root,
            exact_scientific_config=not self.injected_runtime,
        )
        validate_effect_artifacts(self.root)
        effect_payload = json.loads(effect_manifest.read_text(encoding="utf-8"))
        effect_paths = tuple(
            self.root / "effects" / relative
            for relative in effect_payload["artifacts"]
        )
        outputs.extend(
            (
                effect_manifest,
                *effect_paths,
            )
        )
        return self._finish("evaluate", outputs)

    def engineering_smoke(self) -> StageResult:
        previous = self._begin("engineering-smoke")
        if previous is not None:
            return previous
        if self.scientific:
            raise ValueError("engineering smoke cannot use the scientific config")
        design = self._sequence_design()
        manifest = write_preselection_artifacts(
            design,
            self.config,
            self.smoke_root,
            require_scientific=False,
        )
        output = self.smoke_root / "smoke_outputs"
        output.mkdir(parents=True, exist_ok=True)
        runtime = self.config["runtime"]
        plant = self._plant(runtime["discovery_attention_implementation"])
        outputs: list[Path] = [
            manifest,
            manifest.parent / "prompts_all.csv",
            manifest.parent / "prompts.csv",
            manifest.parent / "token_banks.json",
        ]
        discovery_records = self._load_prompts("discovery")
        try:
            clean = _records_frame(
                plant.score_clean(
                    discovery_records, batch_size=int(runtime["discovery_batch_size"])
                )
            )
            clean_path = output / "clean_scores.csv"
            _write_csv_atomic(clean, clean_path)
            outputs.append(clean_path)
            discovery = _records_frame(
                plant.scan_attention(
                    discovery_records, batch_size=int(runtime["discovery_batch_size"])
                )
            )
            discovery_path = output / "attention_scan.csv"
            _write_csv_atomic(discovery, discovery_path)
            outputs.append(discovery_path)
        finally:
            self._release_plant(plant)

        shortlist = shortlist_attention_heads(
            discovery, count=int(self.config["head_discovery"]["attention_shortlist"])
        )
        shortlist_path = output / "attention_shortlist.csv"
        _write_csv_atomic(shortlist, shortlist_path)
        outputs.append(shortlist_path)
        n_heads = int(self.config["head_discovery"]["selected_heads"])
        heads = _head_refs(shortlist.head(n_heads))

        plant = self._plant(runtime["measurement_attention_implementation"])
        try:
            means = plant.capture_reference_means(
                self._load_prompts("reference"),
                heads,
                batch_size=int(runtime["measurement_batch_size"]),
            )
            head_fit = self._load_prompts("head_fit")
            measurement_clean = _records_frame(
                plant.score_clean(
                    head_fit, batch_size=int(runtime["measurement_batch_size"])
                )
            )
            measurement_clean_path = output / "measurement_clean_scores.csv"
            _write_csv_atomic(measurement_clean, measurement_clean_path)
            outputs.append(measurement_clean_path)
            parity = dict(
                plant.noop_hook_parity(
                    head_fit,
                    heads,
                    means,
                    batch_size=int(runtime["measurement_batch_size"]),
                )
            )
            parity["passed"] = bool(
                float(parity["maximum_margin_error"]) <= 1e-5
                and bool(parity["prediction_parity"])
            )
            parity_path = output / "noop_parity.json"
            write_json(parity_path, parity)
            outputs.append(parity_path)
            if not parity["passed"]:
                raise RuntimeError("engineering smoke failed primary zero-mask parity")
            masks = tuple(
                RuntimeMask(
                    f"smoke_mask_{encoded:0{n_heads}b}",
                    tuple(map(int, f"{encoded:0{n_heads}b}")),
                )
                for encoded in range(2**n_heads)
            )
            effects = self._checkpoint_effects(
                plant,
                head_fit,
                heads,
                masks,
                means,
                partition="engineering_smoke",
                batch_size=int(runtime["measurement_batch_size"]),
            )
            effects_path = output / "effects.csv"
            _write_csv_atomic(effects, effects_path)
            outputs.append(effects_path)
            summary_path = output / "summary.json"
            write_json(
                summary_path,
                {
                    "status": "engineering_smoke_only",
                    "scientific_claim_allowed": False,
                    "model": self.config["model"],
                    "n_discovery_heads": len(discovery),
                    "n_measured_masks": len(masks),
                    "all_effects_finite": bool(
                        np.isfinite(effects["drop_from_clean"].astype(float)).all()
                    ),
                    "primary_zero_mask_parity": parity,
                },
            )
            outputs.append(summary_path)
            return self._finish("engineering-smoke", outputs)
        finally:
            self._release_plant(plant)

    def run(self, stage: str) -> StageResult | tuple[StageResult, ...]:
        dispatch = {
            "prepare": self.prepare,
            "discover": self.discover,
            "confirm": self.confirm,
            "freeze-design": self.freeze_design,
            "measure-calibration": self.measure_calibration,
            "freeze-observers": self.freeze_observers,
            "measure-locked-test": self.measure_locked_test,
            "measure-collateral": self.measure_collateral,
            "evaluate": self.evaluate,
            "engineering-smoke": self.engineering_smoke,
        }
        if stage == "all":
            if not self.scientific:
                raise ValueError("all is available only for the scientific stage chain")
            results: list[StageResult] = []
            for name in SCIENTIFIC_STAGES:
                result = dispatch[name]()
                results.append(result)
                if result.status != "complete":
                    break
            return tuple(results)
        try:
            return dispatch[stage]()
        except KeyError:
            raise ValueError(f"unknown Phase-09 stage: {stage}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one sealed stage of the Qwen induction Phase-09 study."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--stage", choices=(*ALL_STAGES, "all"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from hash-checked stage and per-mask checkpoints. "
            "The runner never overwrites a completed sealed stage."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Phase09Runner(
        args.config,
        args.artifacts_root,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    # Resume is intrinsically safe and always enabled; the explicit switch is
    # retained for notebooks and batch scripts to document their intent.
    del args.resume
    result = runner.run(args.stage)
    if isinstance(result, tuple):
        payload = [asdict(item) for item in result]
        succeeded = all(item.status == "complete" for item in result)
    else:
        payload = asdict(result)
        succeeded = result.status == "complete"
    print(json.dumps(payload, default=str, indent=2))
    return 0 if succeeded else 2


__all__ = [
    "ALL_STAGES",
    "OBSERVER_FREEZE_SCHEMA",
    "Phase09Runner",
    "RUN_AUDIT_SCHEMA",
    "RuntimeMask",
    "RuntimePrompt",
    "SCIENTIFIC_STAGES",
    "StageResult",
    "build_parser",
    "main",
]
