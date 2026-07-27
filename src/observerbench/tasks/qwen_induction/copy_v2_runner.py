"""Prospective runner for the Qwen induction Copy-v2 study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

Copy-v2 is a new conditional study after Copy-v1 stopped at its frozen clean
gate.  Candidate reservoirs are fixed without model outcomes.  A separate
clean-only stage freezes eligible prompt IDs before the inherited attention,
ablation, mask, observer, or locked-outcome stages can run.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.qwen_induction.artifacts import (
    PHASE10_CONFIG_SCHEMA,
    validate_phase09_config,
)
from observerbench.tasks.qwen_induction.copy_v2_artifacts import (
    load_copy_v2_candidate_reservoir,
    verify_copy_v2_candidate_artifacts,
    verify_copy_v2_preselection_artifacts,
    write_copy_v2_candidate_artifacts,
    write_copy_v2_preselection_artifacts,
)
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SequenceBankCounts,
    build_sequence_design,
)
from observerbench.tasks.qwen_induction.eligibility import (
    COPY_V2_CANDIDATE_MARGIN_MINIMUM,
    CopyV2EligibilityResult,
    evaluate_copy_v2_clean_eligibility,
    write_copy_v2_eligibility_artifacts,
)
from observerbench.tasks.qwen_induction.plant import regular_token_pool
from observerbench.tasks.qwen_induction.runner import (
    Phase09Runner,
    RuntimePrompt,
    StageResult,
    _read_csv,
    _records_frame,
    _write_csv_atomic,
)


COPY_V2_SOURCE_MANIFEST_SCHEMA = (
    "observerbench.qwen_induction_phase10.source_manifest.v1"
)
COPY_V2_STAGES: tuple[str, ...] = (
    "prepare",
    "eligibility",
    "discover",
    "confirm",
    "freeze-design",
    "measure-calibration",
    "freeze-observers",
    "measure-locked-test",
    "evaluate",
    "measure-collateral",
)
_CONFIG_TO_DESIGN_BANK = {
    "reference": "reference",
    "discovery": "discovery",
    "head_fit": "head_fit",
    "head_confirmation": "head_test",
    "calibration": "calibration",
    "locked_test": "locked_test",
}
_DUAL_IMPLEMENTATION_BANKS = ("discovery",)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts are invalid")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - half, centre + half


class CopyV2Runner(Phase09Runner):
    """Add a sealed clean-eligibility boundary to the Phase-09 primitives."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.get("schema") != PHASE10_CONFIG_SCHEMA:
            raise ValueError("CopyV2Runner requires the Phase-10 Copy-v2 config")

    def _producer_source_hashes(self) -> dict[str, str]:
        hashes = dict(super()._producer_source_hashes())
        repo = Path(__file__).resolve().parents[4]
        additions = (
            Path(__file__).resolve(),
            Path(__file__).with_name("eligibility.py"),
            Path(__file__).with_name("copy_v2_artifacts.py"),
            repo / "scripts" / "run_qwen_induction_phase10.py",
            repo / "scripts" / "build_qwen_phase10_source_archive.py",
            repo / "notebooks" / "qwen_induction_copy_v2_colab.ipynb",
            repo / "docs" / "QWEN_INDUCTION_COPY_V2_PREREGISTRATION.md",
            repo / "configs" / "revision" / "phase10" / "colab_constraints.txt",
        )
        hashes.update(
            {
                path.relative_to(repo).as_posix(): file_sha256(path)
                for path in additions
            }
        )
        return dict(sorted(hashes.items()))

    def _validate_frozen_source_bundle(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        path = (
            repo
            / "configs"
            / "revision"
            / "phase10"
            / "qwen_copy_v2_source_manifest.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "the frozen Copy-v2 producer-source manifest is missing"
            ) from None
        if payload.get("schema") != COPY_V2_SOURCE_MANIFEST_SCHEMA:
            raise ValueError("unexpected Copy-v2 source-manifest schema")
        current = self._producer_source_hashes()
        if payload.get("artifact_hashes") != current:
            raise ValueError("Copy-v2 producer source differs from its frozen manifest")
        if (
            payload.get("source_bundle_sha256") != json_sha256(current)
            or payload.get("config_sha256") != json_sha256(self.config)
        ):
            raise ValueError("Copy-v2 producer-source seal changed")

    def _begin(self, stage: str) -> StageResult | None:
        if stage not in COPY_V2_STAGES:
            raise ValueError(f"unknown Copy-v2 stage: {stage}")
        if not self.scientific:
            raise ValueError("Copy-v2 stages require the frozen prospective config")
        audit = self._audit()
        prepare_record = audit.get("stages", {}).get("prepare")
        if isinstance(prepare_record, Mapping) and prepare_record.get("status") == "complete":
            verify_copy_v2_candidate_artifacts(self.root, self.config)
        eligibility_record = audit.get("stages", {}).get("eligibility")
        if (
            isinstance(eligibility_record, Mapping)
            and eligibility_record.get("status") == "complete"
        ):
            verify_copy_v2_preselection_artifacts(self.root, self.config)
        return self._begin_ordered_stage(stage, COPY_V2_STAGES)

    def _copy_v1_token_banks_path(self) -> Path:
        repo = Path(__file__).resolve().parents[4]
        exclusion = self.config["token_pool"]["exclude_copy_v1_allocated_tokens"]
        path = repo / str(exclusion["source_artifact"])
        if (
            not path.is_file()
            or file_sha256(path) != str(exclusion["source_artifact_sha256"])
        ):
            raise ValueError("the frozen Copy-v1 token-bank input changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload.get("token_pool_size", -1))
            != int(exclusion["source_token_pool_size"])
            or str(payload.get("token_pool_sha256"))
            != str(exclusion["source_token_pool_sha256"])
        ):
            raise ValueError("the Copy-v1 token-pool identity changed")
        return path

    @staticmethod
    def _token_ids_from_banks(path: Path) -> set[int]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(token)
            for bank in payload["banks"]
            for token in bank["token_ids"]
        }

    def _candidate_sequence_design(self) -> Any:
        tokenizer = self._tokenizer()
        token_config = self.config["token_pool"]
        sequence = self.config["sequence_design"]
        per_split = int(token_config["per_split_size"])
        copy_v1_path = self._copy_v1_token_banks_path()
        excluded = self._token_ids_from_banks(copy_v1_path)
        vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
        candidate_ids = tuple(
            token_id for token_id in range(vocab_size) if token_id not in excluded
        )
        pool = regular_token_pool(
            tokenizer,
            seed=int(token_config["seed"]),
            limit=6 * per_split,
            candidate_ids=candidate_ids,
        )
        raw_counts = sequence["reservoir_prompts_per_family"]
        counts = {
            design_bank: int(raw_counts[config_bank])
            for config_bank, design_bank in _CONFIG_TO_DESIGN_BANK.items()
        }
        return build_sequence_design(
            pool,
            bank_counts=SequenceBankCounts(**counts),
            per_split_size=per_split,
            seed=int(sequence["seed"]),
        )

    def prepare(self) -> StageResult:
        previous = self._begin("prepare")
        if previous is not None:
            return previous
        validate_phase09_config(self.config, require_scientific=True)
        design = self._candidate_sequence_design()
        manifest = write_copy_v2_candidate_artifacts(
            design,
            self.config,
            self.root,
            copy_v1_token_banks_path=self._copy_v1_token_banks_path(),
            scientific_claim_allowed=self.scientific_claim_allowed,
        )
        verify_copy_v2_candidate_artifacts(self.root, self.config)
        output = manifest.parent
        return self._finish(
            "prepare",
            (
                manifest,
                output / "prompts_all.csv",
                output / "token_banks.json",
                output / "copy_v1_token_banks.json",
                output / "copy_v1_token_exclusion.json",
            ),
        )

    @staticmethod
    def _runtime_prompts(frame: pd.DataFrame) -> tuple[RuntimePrompt, ...]:
        prompts: list[RuntimePrompt] = []
        for row in frame.itertuples(index=False):
            prompts.append(
                RuntimePrompt(
                    prompt_id=str(row.prompt_id),
                    bank=str(row.bank),
                    family_id=str(row.family_id),
                    cluster_id=str(row.cluster_id),
                    tokens=tuple(map(int, str(row.input_ids).split())),
                    target_value_token=int(row.target_token_id),
                    distractor_value_tokens=(
                        int(row.distractor_token_id_1),
                        int(row.distractor_token_id_2),
                    ),
                    key_positions=tuple(map(int, str(row.key_positions).split())),
                    final_key_position=int(row.query_position),
                )
            )
        return tuple(prompts)

    def _final_counts(self) -> dict[str, int]:
        raw = self.config["sequence_design"]["final_prompts_per_family"]
        return {
            design_bank: int(raw[config_bank])
            for config_bank, design_bank in _CONFIG_TO_DESIGN_BANK.items()
        }

    @staticmethod
    def _decoded_token_properties(tokenizer: Any, token_ids: Sequence[int]) -> dict[str, Any]:
        texts = [
            str(
                tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            for token_id in token_ids
        ]
        lengths = np.asarray([len(text) for text in texts], dtype=float)
        return {
            "decoded_length": {
                "minimum": int(lengths.min()),
                "median": float(np.median(lengths)),
                "mean": float(lengths.mean()),
                "maximum": int(lengths.max()),
            },
            "character_class_counts": {
                "ascii_only": sum(text.isascii() for text in texts),
                "contains_non_ascii": sum(not text.isascii() for text in texts),
                "contains_alpha": sum(any(character.isalpha() for character in text) for text in texts),
                "contains_digit": sum(any(character.isdigit() for character in text) for text in texts),
                "other_only": sum(
                    not any(character.isalnum() for character in text) for text in texts
                ),
            },
        }

    def _coverage_diagnostics(
        self,
        result: CopyV2EligibilityResult,
        tokenizer: Any,
    ) -> dict[str, Any]:
        coverage_rows = []
        for row in result.coverage.to_dict("records"):
            lower, upper = _wilson_interval(
                int(row["eligible_count"]), int(row["candidate_count"])
            )
            coverage_rows.append(
                {**row, "wilson_95_lower": lower, "wilson_95_upper": upper}
            )

        def summary(frame: pd.DataFrame) -> dict[str, Any]:
            if frame.empty:
                return {"count": 0}
            margins = pd.to_numeric(frame["candidate_margin"], errors="raise").to_numpy(float)
            nll = pd.to_numeric(frame["target_nll"], errors="raise").to_numpy(float)
            token_ids = pd.to_numeric(frame["target_token_id"], errors="raise").to_numpy(int)
            quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            finite_margins = margins[np.isfinite(margins)]
            finite_nll = nll[np.isfinite(nll)]
            return {
                "count": len(frame),
                "candidate_accuracy": float(frame["candidate_correct"].astype(bool).mean()),
                "full_vocabulary_top1_accuracy": float(frame["top1_correct"].astype(bool).mean()),
                "candidate_margin_quantiles": {
                    str(value): (
                        float(np.quantile(finite_margins, value))
                        if finite_margins.size
                        else None
                    )
                    for value in quantiles
                },
                "finite_candidate_margin_count": int(finite_margins.size),
                "nonfinite_candidate_margin_count": int(
                    margins.size - finite_margins.size
                ),
                "target_nll": {
                    "finite_count": int(finite_nll.size),
                    "nonfinite_count": int(nll.size - finite_nll.size),
                    "median": (
                        float(np.median(finite_nll)) if finite_nll.size else None
                    ),
                    "mean": float(finite_nll.mean()) if finite_nll.size else None,
                },
                "target_token_id": {
                    "median": float(np.median(token_ids)),
                    "minimum": int(token_ids.min()),
                    "maximum": int(token_ids.max()),
                },
                **self._decoded_token_properties(tokenizer, token_ids),
            }

        decisions = result.decisions
        slices = {
            "all_candidates": decisions,
            "eligible": decisions.loc[decisions["eligible"]],
            "selected": decisions.loc[decisions["selected"]],
            "excluded": decisions.loc[~decisions["eligible"]],
        }
        return {
            "schema": "observerbench.qwen_induction.copy_v2.coverage_diagnostics.v1",
            "passed": result.passed,
            "coverage_with_descriptive_wilson_95": coverage_rows,
            "slices": {name: summary(frame) for name, frame in slices.items()},
            "gating_note": "Wilson intervals and token diagnostics are descriptive only.",
        }

    def eligibility(self) -> StageResult:
        previous = self._begin("eligibility")
        if previous is not None:
            return previous
        candidates = load_copy_v2_candidate_reservoir(self.root, self.config)
        records = self._runtime_prompts(candidates)
        runtime = self.config["runtime"]
        batch_size = int(runtime["measurement_batch_size"])
        scoring = self.work / "clean_scoring"
        scoring.mkdir(parents=True, exist_ok=True)
        sdpa_path = scoring / "candidate_scores_sdpa.csv"
        eager_path = scoring / "candidate_scores_eager_discovery.csv"
        tokenizer: Any

        plant = self._plant(runtime["eligibility_attention_implementation"])
        tokenizer = self._tokenizer()
        try:
            sdpa = _records_frame(
                plant.score_clean(records, batch_size=batch_size)
            )
            _write_csv_atomic(sdpa, sdpa_path)
        finally:
            self._release_plant(plant)
        if set(sdpa["prompt_id"].astype(str)) != set(candidates["prompt_id"].astype(str)):
            raise ValueError("SDPA clean scores do not cover the frozen reservoir")

        eager_records = tuple(
            record for record in records if record.bank in _DUAL_IMPLEMENTATION_BANKS
        )
        plant = self._plant(runtime["discovery_attention_implementation"])
        try:
            eager = _records_frame(
                plant.score_clean(
                    eager_records,
                    batch_size=int(runtime["discovery_batch_size"]),
                )
            )
            _write_csv_atomic(eager, eager_path)
        finally:
            self._release_plant(plant)
        if set(eager["prompt_id"].astype(str)) != {
            record.prompt_id for record in eager_records
        }:
            raise ValueError("eager clean scores do not cover the eager-stage reservoirs")

        bank_by_prompt = candidates.set_index("prompt_id")["bank"].astype(str)
        scores = sdpa.copy()
        scores["bank"] = scores["prompt_id"].astype(str).map(bank_by_prompt)
        scores["eager_sdpa_candidate_prediction_agreement"] = True
        scores["sdpa_candidate_margin"] = pd.to_numeric(
            scores["candidate_margin"], errors="raise"
        )
        scores["sdpa_candidate_predicted_token_id"] = pd.to_numeric(
            scores["candidate_predicted_token_id"], errors="raise"
        ).astype(int)
        scores["sdpa_candidate_logits_finite"] = scores[
            "candidate_logits_finite"
        ].astype(bool)
        scores["eager_candidate_margin"] = np.nan
        scores["eager_candidate_predicted_token_id"] = -1
        scores["eager_candidate_logits_finite"] = True
        scores["finite_candidate_logits"] = scores[
            "candidate_logits_finite"
        ].astype(bool)
        scores["stage_candidate_margin"] = pd.to_numeric(
            scores["candidate_margin"], errors="raise"
        )
        scores["stage_candidate_predicted_token_id"] = pd.to_numeric(
            scores["candidate_predicted_token_id"], errors="raise"
        ).astype(int)
        eager_by_id = eager.set_index(eager["prompt_id"].astype(str))
        sdpa_by_id = scores.set_index(scores["prompt_id"].astype(str))
        for prompt_id in eager_by_id.index:
            agreement = int(eager_by_id.loc[prompt_id, "candidate_predicted_token_id"]) == int(
                sdpa_by_id.loc[prompt_id, "candidate_predicted_token_id"]
            )
            selected = scores["prompt_id"].astype(str) == prompt_id
            scores.loc[selected, "eager_sdpa_candidate_prediction_agreement"] = agreement
            eager_margin = float(eager_by_id.loc[prompt_id, "candidate_margin"])
            scores.loc[selected, "eager_candidate_margin"] = eager_margin
            scores.loc[selected, "eager_candidate_predicted_token_id"] = int(
                eager_by_id.loc[prompt_id, "candidate_predicted_token_id"]
            )
            eager_finite = bool(eager_by_id.loc[prompt_id, "candidate_logits_finite"])
            scores.loc[selected, "eager_candidate_logits_finite"] = eager_finite
            scores.loc[selected, "finite_candidate_logits"] = bool(
                scores.loc[selected, "finite_candidate_logits"].iloc[0]
            ) and eager_finite
            scores.loc[selected, "candidate_margin"] = min(
                float(sdpa_by_id.loc[prompt_id, "candidate_margin"]), eager_margin
            )
            # Discovery is the only downstream stage that uses eager attention.
            scores.loc[selected, "stage_candidate_margin"] = float(
                eager_by_id.loc[prompt_id, "candidate_margin"]
            )
            scores.loc[selected, "stage_candidate_predicted_token_id"] = int(
                eager_by_id.loc[prompt_id, "candidate_predicted_token_id"]
            )
        scores["finite_target_nll"] = np.isfinite(
            pd.to_numeric(scores["target_nll"], errors="coerce").to_numpy(float)
        )
        scores = scores.drop(columns=["target_token_id"])
        result = evaluate_copy_v2_clean_eligibility(
            candidates,
            scores,
            required_counts=self._final_counts(),
        )
        eligibility_dir = self.work / "eligibility"
        gate_manifest = write_copy_v2_eligibility_artifacts(
            result, eligibility_dir
        )
        diagnostics_path = eligibility_dir / "coverage_diagnostics.json"
        write_json(diagnostics_path, self._coverage_diagnostics(result, tokenizer))
        outputs: list[Path] = [
            sdpa_path,
            eager_path,
            gate_manifest,
            eligibility_dir / "candidate_decisions.csv",
            eligibility_dir / "selected_prompt_ids.csv",
            eligibility_dir / "coverage.csv",
            diagnostics_path,
        ]
        if not result.passed:
            return self._finish(
                "eligibility",
                outputs,
                status="gate_failed",
                terminal_status="copy_v2_clean_eligibility_failed",
            )
        preselection = write_copy_v2_preselection_artifacts(
            result,
            self.config,
            self.root,
            runtime_record={
                "runtime": self._qwen_runtime_provenance(),
                "plant_audits": list(self._stage_plant_audits),
                "scientific_claim_allowed": self.scientific_claim_allowed,
            },
            source_hashes=self._producer_source_hashes(),
        )
        verify_copy_v2_preselection_artifacts(self.root, self.config)
        outputs.extend(
            (
                preselection,
                preselection.parent / "prompts_all.csv",
                preselection.parent / "prompts.csv",
                preselection.parent / "token_banks.json",
                preselection.parent / "eligibility" / "eligibility_manifest.json",
                preselection.parent / "eligibility" / "candidate_decisions.csv",
                preselection.parent / "eligibility" / "selected_prompt_ids.csv",
                preselection.parent / "eligibility" / "coverage.csv",
            )
        )
        return self._finish("eligibility", outputs)

    def _clean_gate(
        self,
        plant: Any,
        bank: str,
        *,
        batch_size: int,
        output_dir: Path,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        gate, paths = super()._clean_gate(
            plant,
            bank,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        rows = _read_csv(
            paths[0],
            strings=("prompt_id", "family_id"),
        )
        decisions = _read_csv(
            self.root
            / "design"
            / "eligibility"
            / "candidate_decisions.csv",
            strings=("prompt_id", "bank", "family_id"),
        )
        selected = decisions.loc[
            (decisions["bank"].astype(str) == bank)
            & decisions["selected"].astype(str).str.lower().isin(("true", "1"))
        ].copy()
        current = rows.set_index(rows["prompt_id"].astype(str))
        frozen = selected.set_index(selected["prompt_id"].astype(str))
        if set(current.index) != set(frozen.index):
            raise ValueError("clean rescore prompt IDs differ from frozen eligibility")
        aligned = frozen.loc[current.index]
        current_margin = pd.to_numeric(current["candidate_margin"], errors="raise").to_numpy(float)
        frozen_margin = pd.to_numeric(aligned["stage_candidate_margin"], errors="raise").to_numpy(float)
        margin_error = np.abs(current_margin - frozen_margin)
        current_prediction = pd.to_numeric(
            current["candidate_predicted_token_id"], errors="raise"
        ).to_numpy(int)
        frozen_prediction = pd.to_numeric(
            aligned["stage_candidate_predicted_token_id"], errors="raise"
        ).to_numpy(int)
        current_correct = current["candidate_correct"].astype(str).str.lower().isin(("true", "1")).to_numpy(bool)
        rescore_config = self.config["clean_eligibility"]["rescore_gate"]
        rescore_passed = bool(
            np.array_equal(current_prediction, frozen_prediction)
            and current_correct.all()
            and (current_margin >= float(rescore_config["minimum_candidate_margin"])).all()
        )
        gate["eligibility_rescore"] = {
            "passed": rescore_passed,
            "exact_candidate_prediction_agreement": bool(
                np.array_equal(current_prediction, frozen_prediction)
            ),
            "all_candidate_correct": bool(current_correct.all()),
            "all_margins_remain_eligible": bool(
                (current_margin >= COPY_V2_CANDIDATE_MARGIN_MINIMUM).all()
            ),
            "maximum_margin_difference": float(margin_error.max(initial=0.0)),
            "margin_difference_is_gate": False,
        }
        if not rescore_passed:
            gate["passed"] = False
            gate["failures"] = [*gate.get("failures", ()), "eligibility_rescore"]
        write_json(paths[1], gate)
        return gate, paths

    def run(self, stage: str) -> StageResult | tuple[StageResult, ...]:
        dispatch = {
            "prepare": self.prepare,
            "eligibility": self.eligibility,
            "discover": self.discover,
            "confirm": self.confirm,
            "freeze-design": self.freeze_design,
            "measure-calibration": self.measure_calibration,
            "freeze-observers": self.freeze_observers,
            "measure-locked-test": self.measure_locked_test,
            "evaluate": self.evaluate,
            "measure-collateral": self.measure_collateral,
        }
        if stage == "all":
            results: list[StageResult] = []
            for name in COPY_V2_STAGES:
                result = dispatch[name]()
                results.append(result)
                if result.status != "complete":
                    break
            return tuple(results)
        try:
            return dispatch[stage]()
        except KeyError:
            raise ValueError(f"unknown Copy-v2 stage: {stage}") from None


__all__ = ["COPY_V2_SOURCE_MANIFEST_SCHEMA", "COPY_V2_STAGES", "CopyV2Runner"]
