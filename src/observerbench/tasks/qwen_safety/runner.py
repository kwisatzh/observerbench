"""Staged Qwen2.5-7B-Instruct safety-observer experiment.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.safety import SafetyPolicy
from observerbench.tasks.qwen_safety.analysis import (
    QwenSafetyRidgeObserver,
    evaluate_clean_gate,
    evaluate_frozen_observers,
    fit_frozen_observers,
    select_observer_configs,
)
from observerbench.tasks.qwen_safety.artifacts import (
    load_activation_cache,
    load_qwen_safety_design,
    write_activation_cache,
    write_qwen_safety_design,
)
from observerbench.tasks.qwen_safety.design import (
    QwenSafetyDesignConfig,
    build_qwen_safety_design,
)
from observerbench.tasks.qwen_safety.plant import QwenSafetyActivationProducer


QWEN_SAFETY_CONFIG_SCHEMA = "observerbench.qwen_safety.v0"
QWEN_SAFETY_FREEZE_SCHEMA = "observerbench.qwen_safety.observer_freeze.v0"
QWEN_SAFETY_RESULT_SCHEMA = "observerbench.qwen_safety.result_bundle.v0"
STAGES = (
    "prepare",
    "extract-fit",
    "freeze-observers",
    "extract-locked-test",
    "evaluate",
)


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != QWEN_SAFETY_CONFIG_SCHEMA:
        raise ValueError("unsupported Qwen safety experiment config")
    required = {
        "model",
        "design",
        "activations",
        "observers",
        "policy",
        "clean_gate",
        "runtime",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Qwen safety config is missing: {sorted(missing)}")
    return payload


def _policy(config: Mapping[str, Any]) -> SafetyPolicy:
    return SafetyPolicy(**config["policy"])


def _write_observer_states(path: Path, states: Mapping[str, Mapping[str, Any]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for index, (family, state) in enumerate(sorted(states.items())):
        prefix = f"observer_{index:02d}"
        arrays[f"{prefix}__feature_mean"] = np.asarray(state["feature_mean"], dtype=np.float64)
        arrays[f"{prefix}__feature_scale"] = np.asarray(state["feature_scale"], dtype=np.float64)
        arrays[f"{prefix}__coefficients"] = np.asarray(state["coefficients"], dtype=np.float64)
        metadata[family] = {
            "prefix": prefix,
            "name": state["name"],
            "view": state["view"],
            "target": state["target"],
            "ridge": state["ridge"],
            "multiply_by_severity": state["multiply_by_severity"],
        }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load_observer_states(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        states = {}
        for family, item in metadata.items():
            prefix = item.pop("prefix")
            states[family] = {
                **item,
                "feature_mean": archive[f"{prefix}__feature_mean"].copy(),
                "feature_scale": archive[f"{prefix}__feature_scale"].copy(),
                "coefficients": archive[f"{prefix}__coefficients"].copy(),
            }
    return states


class QwenSafetyRunner:
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        artifacts_root: str | Path,
        *,
        device: str | None = None,
        local_files_only: bool = False,
        producer_factory: Any | None = None,
    ) -> None:
        self.config = load_config(config) if isinstance(config, (str, Path)) else dict(config)
        self.root = Path(artifacts_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device or str(self.config["runtime"]["device"])
        self.local_files_only = bool(local_files_only)
        self.producer_factory = producer_factory
        self._producer: QwenSafetyActivationProducer | None = None

    @property
    def design_path(self) -> Path:
        return self.root / "design" / "qwen_safety_design.json"

    def cache_path(self, bank: str) -> Path:
        return self.root / "activations" / f"{bank}_activations.npz"

    @property
    def freeze_path(self) -> Path:
        return self.root / "freeze" / "observer_freeze.json"

    @property
    def state_path(self) -> Path:
        return self.root / "freeze" / "observer_states.npz"

    def _design(self):
        return load_qwen_safety_design(self.design_path)

    def _cache(self, bank: str):
        return load_activation_cache(self.cache_path(bank))

    def _get_producer(self) -> QwenSafetyActivationProducer:
        if self._producer is None:
            if self.producer_factory is not None:
                self._producer = self.producer_factory(self.config, self.device)
            else:
                model = self.config["model"]
                self._producer = QwenSafetyActivationProducer.from_pretrained(
                    model["id"],
                    model["revision"],
                    device=self.device,
                    dtype=model["dtype"],
                    attention_implementation=model["attention_implementation"],
                    local_files_only=self.local_files_only,
                )
            model = self.config["model"]
            audit = self._producer.plant.audit_runtime(
                expected_model_id=model["id"],
                expected_revision=model["revision"],
                expected_layers=model["expected_layers"],
                expected_query_heads=model["expected_query_heads"],
                expected_kv_heads=model["expected_kv_heads"],
                expected_dtype=model["dtype"],
                expected_attention_implementation=model["attention_implementation"],
            )
            write_json(self.root / "model_runtime.json", audit)
        return self._producer

    def prepare(self, *, resume: bool = False) -> None:
        if resume and self.design_path.exists():
            self._design()
            return
        cfg = QwenSafetyDesignConfig(**self.config["design"])
        design = build_qwen_safety_design(cfg)
        write_qwen_safety_design(design, self.design_path.parent)
        write_json(
            self.root / "experiment_manifest.json",
            {
                "schema": QWEN_SAFETY_CONFIG_SCHEMA,
                "config": self.config,
                "config_sha256": json_sha256(self.config),
                "design_sha256": design.design_sha256,
                "runtime": runtime_provenance(Path(__file__).resolve().parents[4]),
            },
        )

    def _extract_bank(self, bank: str, *, resume: bool) -> None:
        path = self.cache_path(bank)
        if resume and path.exists() and path.with_suffix(".manifest.json").exists():
            self._cache(bank)
            return
        design = self._design()
        producer = self._get_producer()
        activation_cfg = self.config["activations"]
        batch = producer.encode(
            design.prompts_for(bank),
            layer_indices=activation_cfg["layers"],
            batch_size=int(self.config["runtime"]["batch_size"]),
        )
        write_activation_cache(
            path,
            prompt_ids=batch.prompt_ids,
            layer_indices=batch.layer_indices,
            activations=batch.activations,
            candidate_margins=batch.candidate_margins,
            block_minus_allow_margins=batch.block_minus_allow_margins,
            candidate_correct=batch.candidate_correct,
            top1_correct=batch.top1_correct,
            sequence_lengths=batch.sequence_lengths,
            metadata={
                "bank": bank,
                "model": dict(self.config["model"]),
                "design_sha256": design.design_sha256,
                "allow_token_id": producer.allow_token_id,
                "block_token_id": producer.block_token_id,
            },
        )

    def extract_fit(self, *, resume: bool = False) -> None:
        if not self.design_path.exists():
            raise FileNotFoundError("prepare must run before extract-fit")
        self._extract_bank("fit", resume=resume)
        self._extract_bank("calibration", resume=resume)

    def freeze_observers(self, *, resume: bool = False) -> None:
        if resume and self.freeze_path.exists() and self.state_path.exists():
            return
        if self.cache_path("locked_test").exists():
            raise RuntimeError(
                "refusing to freeze observers after locked-test activations exist; "
                "use a fresh artifact root"
            )
        design = self._design()
        caches = {bank: self._cache(bank) for bank in ("fit", "calibration")}
        activation_cfg = self.config["activations"]
        observer_cfg = self.config["observers"]
        selections = select_observer_configs(
            design,
            caches,
            layers=activation_cfg["layers"],
            ridge_grid=observer_cfg["ridge_grid"],
            projection_dim=int(activation_cfg["projection_dim"]),
            projection_seed=int(activation_cfg["projection_seed"]),
            policy=_policy(self.config),
            selection_metric=str(
                observer_cfg.get("selection_metric", "protocol_loss_mean")
            ),
        )
        states = fit_frozen_observers(
            design,
            caches,
            selections,
            projection_dim=int(activation_cfg["projection_dim"]),
            projection_seed=int(activation_cfg["projection_seed"]),
            policy=_policy(self.config),
        )
        self.freeze_path.parent.mkdir(parents=True, exist_ok=True)
        _write_observer_states(self.state_path, states)
        write_json(
            self.freeze_path,
            {
                "schema": QWEN_SAFETY_FREEZE_SCHEMA,
                "design_sha256": design.design_sha256,
                "fit_cache_sha256": file_sha256(self.cache_path("fit")),
                "calibration_cache_sha256": file_sha256(self.cache_path("calibration")),
                "projection_dim": int(activation_cfg["projection_dim"]),
                "projection_seed": int(activation_cfg["projection_seed"]),
                "policy": asdict(_policy(self.config)),
                "selections": selections,
                "state_file": self.state_path.name,
                "state_sha256": file_sha256(self.state_path),
            },
        )

    def extract_locked_test(self, *, resume: bool = False) -> None:
        if not self.freeze_path.exists() or not self.state_path.exists():
            raise FileNotFoundError("freeze-observers must run before extract-locked-test")
        self._extract_bank("locked_test", resume=resume)

    def evaluate(self, *, resume: bool = False) -> None:
        result_path = self.root / "evaluation" / "qwen_safety_results.json"
        if resume and result_path.exists():
            return
        design = self._design()
        caches = {bank: self._cache(bank) for bank in ("fit", "calibration", "locked_test")}
        freeze = json.loads(self.freeze_path.read_text(encoding="utf-8"))
        if freeze.get("schema") != QWEN_SAFETY_FREEZE_SCHEMA:
            raise ValueError("unsupported Qwen safety observer freeze")
        if freeze["design_sha256"] != design.design_sha256:
            raise ValueError("observer freeze and design disagree")
        if freeze["state_sha256"] != file_sha256(self.state_path):
            raise ValueError("frozen observer state hash mismatch")
        clean_cfg = self.config["clean_gate"]
        gate = evaluate_clean_gate(
            design,
            caches["locked_test"],
            bank="locked_test",
            minimum_overall_candidate_accuracy=clean_cfg["minimum_overall_candidate_accuracy"],
            minimum_family_candidate_accuracy=clean_cfg["minimum_family_candidate_accuracy"],
            minimum_paired_candidate_accuracy=clean_cfg["minimum_paired_candidate_accuracy"],
            minimum_median_candidate_margin=clean_cfg["minimum_median_candidate_margin"],
        )
        evaluation_dir = result_path.parent
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        write_json(evaluation_dir / "clean_gate.json", asdict(gate))
        if not gate.passed:
            write_json(
                result_path,
                {
                    "schema": QWEN_SAFETY_RESULT_SCHEMA,
                    "status": "stopped_clean_gate_failed",
                    "clean_gate": asdict(gate),
                    "results": {},
                },
            )
            return
        states = _load_observer_states(self.state_path)
        activation_cfg = self.config["activations"]
        results = evaluate_frozen_observers(
            design,
            caches,
            freeze["selections"],
            states,
            projection_dim=int(activation_cfg["projection_dim"]),
            projection_seed=int(activation_cfg["projection_seed"]),
            policy=_policy(self.config),
        )
        payload = {
            "schema": QWEN_SAFETY_RESULT_SCHEMA,
            "status": "complete",
            "clean_gate": asdict(gate),
            "selections": freeze["selections"],
            "results": {name: result.to_dict() for name, result in results.items()},
        }
        write_json(result_path, payload)
        table_path = evaluation_dir / "qwen_safety_results.csv"
        metric_names = sorted(next(iter(results.values())).metrics)
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("observer", *metric_names))
            writer.writeheader()
            for name, result in sorted(results.items()):
                writer.writerow({"observer": name, **result.metrics})

    def run(self, stage: str, *, resume: bool = False) -> None:
        if stage == "prepare":
            self.prepare(resume=resume)
        elif stage == "extract-fit":
            self.extract_fit(resume=resume)
        elif stage == "freeze-observers":
            self.freeze_observers(resume=resume)
        elif stage == "extract-locked-test":
            self.extract_locked_test(resume=resume)
        elif stage == "evaluate":
            self.evaluate(resume=resume)
        elif stage == "all":
            self.prepare(resume=resume)
            self.extract_fit(resume=resume)
            self.freeze_observers(resume=resume)
            self.extract_locked_test(resume=resume)
            self.evaluate(resume=resume)
        else:
            raise ValueError(f"unknown Qwen safety stage {stage!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = QwenSafetyRunner(
        args.config,
        args.artifacts_root,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    runner.run(args.stage, resume=args.resume)
    print(args.artifacts_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
