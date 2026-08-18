"""Clean-only stop gate for the Phase-7 canonical-template confirmation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module has no ablation or candidate-mask scoring path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    _model_tokens,
    _score_clean,
)
from observerbench.tasks.ioi.phase7_confirmation import (
    DESIGN_SCHEMA,
    DESIGN_STATUS,
    PRETEST_PASS_STATUS,
    PRETEST_SCHEMA,
    SCIENTIFIC_STATUS,
    _verify_artifact_index,
    load_phase7_protocol,
)


PRETEST_FAIL_STATUS = "clean_pretest_failed_stop_candidate_outcomes_forbidden"


@dataclass(frozen=True)
class Phase7PretestConfig:
    """Compute-only settings for clean scoring."""

    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    batch_size: int = 64
    seed: int = 27071

    def __post_init__(self) -> None:
        if self.model_name != "gpt2-small" or self.model_revision != GPT2_SMALL_REVISION:
            raise ValueError("Phase-7 pretest requires the pinned GPT-2-small revision")
        if self.batch_size <= 0:
            raise ValueError("clean pretest batch size must be positive")


def load_clean_pretest_inputs(
    design_dir: str | Path,
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Verify the design but expose only test prompts, never candidate masks."""

    root = Path(design_dir)
    manifest = json.loads((root / "design_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DESIGN_SCHEMA or manifest.get("status") != DESIGN_STATUS:
        raise ValueError("Phase-7 design is not frozen")
    if manifest.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("Phase-7 scientific status changed")
    if manifest.get("contains_model_outcomes") is not False:
        raise ValueError("Phase-7 design contains outcomes")
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("Phase-7 protocol changed after design freeze")
    _verify_artifact_index(root, manifest.get("artifact_hashes", {}), "design")
    protocol = load_phase7_protocol(protocol_path)
    prompts = pd.read_csv(root / "prompts.csv", dtype={"prompt_id": str})
    if len(prompts) != 512 or prompts["prompt_id"].astype(str).nunique() != 512:
        raise ValueError("clean pretest requires all 512 frozen prompts")
    if set(prompts["split"].astype(str)) != {"test"}:
        raise ValueError("clean pretest exposed a non-test split")
    if prompts["prompt"].astype(str).duplicated().any():
        raise ValueError("clean pretest prompt strings must be unique")
    return manifest, protocol, prompts


def clean_pretest_validity(
    clean: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the frozen overall and per-template clean-task gates."""

    if len(clean) != 512 or clean["prompt_id"].astype(str).duplicated().any():
        raise ValueError("clean pretest requires one row per frozen prompt")
    if not np.isfinite(clean["clean_ld"].to_numpy(float)).all():
        raise ValueError("clean pretest contains a non-finite score")
    records: list[dict[str, Any]] = []
    for scope, frame in (
        ("overall", clean),
        *(
            (str(template), group)
            for template, group in clean.groupby("template_id", sort=True)
        ),
    ):
        records.append(
            {
                "scope": scope,
                "prompt_count": len(frame),
                "mean_clean_logit_difference": float(frame["clean_ld"].mean()),
                "io_vs_subject_pairwise_accuracy": float(
                    (frame["clean_ld"].to_numpy(float) > 0.0).mean()
                ),
            }
        )
    table = pd.DataFrame(records)
    overall = table.loc[table["scope"] == "overall"].iloc[0]
    templates = table.loc[table["scope"] != "overall"]
    gate = protocol["clean_pretest"]
    checks = {
        "overall_accuracy": bool(
            overall["io_vs_subject_pairwise_accuracy"]
            >= float(gate["overall_IO_vs_subject_pairwise_accuracy_min"])
        ),
        "every_template_accuracy": bool(
            (
                templates["io_vs_subject_pairwise_accuracy"]
                >= float(gate["every_template_IO_vs_subject_pairwise_accuracy_min"])
            ).all()
        ),
        "every_template_positive_mean_clean_ld": bool(
            (templates["mean_clean_logit_difference"] > 0.0).all()
        ),
        "exact_prompt_count": len(clean) == int(protocol["test_prompt_count"]),
        "eight_templates_retained": len(templates) == 8,
    }
    return table, {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "overall_accuracy_min": float(
                gate["overall_IO_vs_subject_pairwise_accuracy_min"]
            ),
            "every_template_accuracy_min": float(
                gate["every_template_IO_vs_subject_pairwise_accuracy_min"]
            ),
            "every_template_mean_clean_ld_strictly_positive": True,
        },
        "failure_rule": gate["failure_rule"],
    }


def _clean_rows(prompts: pd.DataFrame, scores: Sequence[float]) -> pd.DataFrame:
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


def run_phase7_clean_pretest(
    design_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    config: Phase7PretestConfig,
) -> Path:
    """Run only clean forward passes, write the irreversible pass/stop result."""

    design_manifest, protocol, prompts = load_clean_pretest_inputs(
        design_dir, protocol_path
    )
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-7 clean pretest is never overwritten")
    output.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("loaded model is not GPT-2-small")
    tokens, io_tokens, s_tokens = _model_tokens(model, prompts)
    if not np.array_equal(io_tokens, prompts["answer_token_id"].to_numpy(int)):
        raise ValueError("clean pretest IO tokenization differs from the sealed design")
    if not np.array_equal(
        s_tokens, prompts["counterfactual_token_id"].to_numpy(int)
    ):
        raise ValueError("clean pretest subject tokenization differs from the sealed design")
    scores = _score_clean(
        model,
        prompts,
        tokens,
        io_tokens,
        s_tokens,
        batch_size=config.batch_size,
    )
    clean = _clean_rows(prompts, scores)
    table, gate = clean_pretest_validity(clean, protocol=protocol)
    clean.to_csv(output / "clean_scores_test.csv", index=False)
    table.to_csv(output / "clean_task_validity.csv", index=False)
    passed = bool(gate["passed"])
    manifest = {
        "schema": PRETEST_SCHEMA,
        "status": PRETEST_PASS_STATUS if passed else PRETEST_FAIL_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design_manifest["design_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "config": asdict(config),
        "accessed_prompt_splits": ["test"],
        "candidate_mask_metadata_loaded": False,
        "candidate_mask_forward_passes": 0,
        "candidate_effect_cells": 0,
        "candidate_outcomes_authorized": passed,
        "gate": gate,
        "counts": {
            "clean_prompt_forward_passes": len(prompts),
            "candidate_mask_forward_passes": 0,
        },
        "model": {
            "requested_name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_name": model.cfg.model_name,
            "n_layers": int(model.cfg.n_layers),
            "n_heads": int(model.cfg.n_heads),
            "d_head": int(model.cfg.d_head),
            "dtype": str(model.cfg.dtype),
            "device": str(model.cfg.device),
        },
        "artifact_hashes": {
            "clean_scores_test.csv": file_sha256(output / "clean_scores_test.csv"),
            "clean_task_validity.csv": file_sha256(output / "clean_task_validity.csv"),
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "Fit and freeze Phase-7 actions from Phase-5 train/calibration only."
            if passed
            else "STOP. Phase-7 candidate outcomes must remain unopened."
        ),
    }
    write_json(output / "pretest_manifest.json", manifest)
    return output


def rebind_frozen_phase7_clean_pretest(
    design_dir: str | Path,
    source_pretest_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
) -> Path:
    """Rebind the unchanged clean-only v1 scores to the repaired v2 protocol."""

    design_manifest, protocol, prompts = load_clean_pretest_inputs(
        design_dir, protocol_path
    )
    if not str(protocol["schema"]).endswith(".v2"):
        raise ValueError("clean-score rebinding is reserved for the repaired v2 protocol")
    source_root = Path(source_pretest_dir)
    source_manifest_path = source_root / "pretest_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema") != PRETEST_SCHEMA
        or source_manifest.get("status") != PRETEST_PASS_STATUS
    ):
        raise ValueError("the clean-score source is not a passed Phase-7 pretest")
    _verify_artifact_index(
        source_root, source_manifest.get("artifact_hashes", {}), "source pretest"
    )
    repair = protocol["v1_aborted_attempt"]
    spec_label = next(
        label
        for label in repair["artifact_hashes"]
        if str(label).endswith("measurement_run_spec.json")
    )
    repository_root = Path(protocol_path).resolve().parents[2]
    spec = json.loads(
        (repository_root / str(spec_label)).read_text(encoding="utf-8")
    )
    if spec.get("source_hashes", {}).get("pretest_manifest") != file_sha256(
        source_manifest_path
    ):
        raise ValueError("the v1 measurement spec does not bind the source pretest")

    clean_path = source_root / "clean_scores_test.csv"
    clean = pd.read_csv(clean_path, dtype={"prompt_id": str})
    if set(clean["prompt_id"].astype(str)) != set(prompts["prompt_id"].astype(str)):
        raise ValueError("the clean-score source uses a different prompt bank")
    table, gate = clean_pretest_validity(clean, protocol=protocol)
    if not gate["passed"]:
        raise ValueError("the rebound clean scores fail the v2 clean gate")

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a rebound Phase-7 clean pretest is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clean_path, output / "clean_scores_test.csv")
    table.to_csv(output / "clean_task_validity.csv", index=False)
    manifest = {
        "schema": PRETEST_SCHEMA,
        "status": PRETEST_PASS_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design_manifest["design_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "config": source_manifest["config"],
        "accessed_prompt_splits": ["test"],
        "candidate_mask_metadata_loaded": False,
        "candidate_mask_forward_passes": 0,
        "candidate_effect_cells": 0,
        "candidate_outcomes_authorized": True,
        "gate": gate,
        "counts": {
            "clean_prompt_forward_passes": 0,
            "reused_frozen_clean_scores": len(clean),
            "candidate_mask_forward_passes": 0,
        },
        "clean_reference": {
            "status": "rebound_from_v1_clean_only_pretest_that_predates_aborted_shard",
            "source_pretest_manifest_sha256": file_sha256(source_manifest_path),
            "source_clean_scores_sha256": file_sha256(clean_path),
            "source_bound_by_v1_measurement_spec": True,
            "candidate_outcome_values_read": False,
        },
        "artifact_hashes": {
            "clean_scores_test.csv": file_sha256(output / "clean_scores_test.csv"),
            "clean_task_validity.csv": file_sha256(output / "clean_task_validity.csv"),
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": "Fit and freeze Phase-7 v2 actions from Phase-5 train/calibration only.",
    }
    write_json(output / "pretest_manifest.json", manifest)
    return output
