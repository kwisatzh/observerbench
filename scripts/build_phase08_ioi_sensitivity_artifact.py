#!/usr/bin/env python3
"""Build the checked Phase-8 IOI target-sensitivity paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from observerbench.core import write_json
from observerbench.provenance import file_sha256, source_hashes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results/revision/phase08/ioi_target_sensitivity"
DEFAULT_OUTDIR = REPO_ROOT / "paper/generated_phase08/ioi_sensitivity"
PROTOCOL = REPO_ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json"

INPUTS = {
    "protocol": PROTOCOL,
    "aborted_measurement_v1": DEFAULT_ROOT / "aborted_measurement_v1.json",
    "freeze_manifest": DEFAULT_ROOT / "prediction_freeze/prediction_action_manifest.json",
    "all_selected_masks": DEFAULT_ROOT / "prediction_freeze/all_selected_masks.csv",
    "new_measurement_masks": DEFAULT_ROOT / "prediction_freeze/new_measurement_masks.csv",
    "reused_phase7_masks": DEFAULT_ROOT / "prediction_freeze/reused_phase7_masks.csv",
    "fixed_actions": DEFAULT_ROOT / "prediction_freeze/fixed_actions.csv",
    "preoutcome_audit": DEFAULT_ROOT / "preoutcome_audit_v2/preoutcome_audit.json",
    "measurement_spec": DEFAULT_ROOT / "new_measurement_v2/measurement_run_spec.json",
    "measurement_manifest": DEFAULT_ROOT / "new_measurement_v2/measurement_manifest.json",
    "evaluation_manifest": DEFAULT_ROOT / "evaluation_v2/evaluation_manifest.json",
    "result_digest": DEFAULT_ROOT / "evaluation_v2/result_digest.json",
    "secondary_contrasts": DEFAULT_ROOT / "evaluation_v2/secondary_contrasts.csv",
    "observer_summary": DEFAULT_ROOT / "evaluation_v2/observer_summary.csv",
    "prediction_diagnostics": DEFAULT_ROOT / "evaluation_v2/prediction_diagnostics.csv",
    "fixed_action_prompt_losses": (
        DEFAULT_ROOT / "evaluation_v2/fixed_action_prompt_losses.csv.gz"
    ),
}
EXPECTED_HASHES = {
    "protocol": "d22614d2deecf3818e0765a2851a8bd38151ed7c9ecedef7e619810fea537508",
    "aborted_measurement_v1": "24b84c6334c0b5ae3833538ea32c0e83f4ae8c6ec2fbaac0797075a228389cda",
    "freeze_manifest": "3792abb1d1e674d27d5d9742244a87efd54a59eb24c21879b7b380c609bd4bad",
    "all_selected_masks": "c0e27f97100c288958125b8c76b4dad2b6f5d6276defbc3dfd5b2a97554a79ce",
    "new_measurement_masks": "667bb263d4e25f3d4bdcf3b3f877797bee0bfc0270a673987f9069422dfd2032",
    "reused_phase7_masks": "a3e93f8811f6ffacb9a8c064f148420f4847b18c696a05457c856c2832e8bc60",
    "fixed_actions": "b4287522fc4691210982b99a6fec3b7a3890b4bf4ddcb5b19f7e1eac427831b1",
    "preoutcome_audit": "a95c8b2b065704e4dc654f17799acb062170b101c7590142d9744155e1ad0f13",
    "measurement_spec": "d475855cbb71fc7342775e9858fdd9abbc68c1357acd4bee8d07e2978d34dcc9",
    "measurement_manifest": "c14cb5e1143f36ed300e61516b26a6de3aa7c43bb76844ebe6ec5c8d2b3e1b66",
    "evaluation_manifest": "bde96228a884b4f45f60a95394c2b552fabae3dd07d046afecc439dc188e7ccc",
    "result_digest": "849932cca2a57ff116f00a6d967d093f6dcde00f4f0e6d53372f8d72e4f4ea38",
    "secondary_contrasts": "347319c8928ef9d80a4faba7660ac6a7d771f0bad36b830621572e7089af33c0",
    "observer_summary": "abcd52a570b5780490476b39d9a1e60a6d34d8168c58d3f5762b5f7f12870fe1",
    "prediction_diagnostics": "12b6c71c38c202db7ce2acc62e6a6af9acf3226a669c5711596dddd020a871d5",
    "fixed_action_prompt_losses": "4498f46d759555e15ba684641355dbe2e552f33fe0658c53cf761a219dbbf9e6",
}

SCIENTIFIC_STATUS = "post_review_post_confirmatory_target_sensitivity"
DIRECT = "direct_risk_head_pair_quadratic"
MEAN = "natural_mean_effect_head_pair_quadratic"
TRANSFORMED = "transformed_mean_head_pair_quadratic"
NOOP = "exact_noop"
REFERENCES = (MEAN, TRANSFORMED, NOOP)
TARGET_SCOPES = ("target_0.5", "target_1", "target_1.5", "all_three_equal_weight")
TARGET_LABELS = {
    "target_0.5": "0.5",
    "target_1": "1.0",
    "target_1.5": "1.5",
    "all_three_equal_weight": "Equal-target mean",
}
EXPECTED_COUNTS = {
    "candidate_prediction_rows": 13_392,
    "fixed_actions": 576,
    "exact_noop_baseline_action_rows": 144,
    "fitted_selector_noop_selections": 12,
    "new_effect_cells_to_measure": 75_776,
    "new_selected_masks_to_measure": 148,
    "reused_phase7_measured_masks": 89,
    "selected_unique_nonnoop_masks": 237,
    "total_selected_noop_action_rows": 156,
}
EXPECTED_DOWNSTREAM_SOURCES = {
    "scripts/audit_ioi_phase08_preoutcome.py",
    "scripts/evaluate_ioi_phase08_sensitivity.py",
    "scripts/freeze_ioi_phase08_sensitivity.py",
    "scripts/run_ioi_phase08_selected_measurement.py",
    "src/observerbench/core.py",
    "src/observerbench/provenance.py",
    "src/observerbench/tasks/ioi/heads.py",
    "src/observerbench/tasks/ioi/phase2_capacity.py",
    "src/observerbench/tasks/ioi/phase5_analysis.py",
    "src/observerbench/tasks/ioi/phase5_design.py",
    "src/observerbench/tasks/ioi/phase5_effects.py",
    "src/observerbench/tasks/ioi/phase6_risk.py",
    "src/observerbench/tasks/ioi/phase7_confirmation.py",
    "src/observerbench/tasks/ioi/phase7_evaluation.py",
    "src/observerbench/tasks/ioi/phase7_freeze_audit.py",
    "src/observerbench/tasks/ioi/phase7_measurement.py",
    "src/observerbench/tasks/ioi/phase8_evaluation.py",
    "src/observerbench/tasks/ioi/phase8_measurement.py",
    "src/observerbench/tasks/ioi/phase8_sensitivity.py",
    "src/observerbench/tasks/ioi/stage2d.py",
}
EXPECTED_CONTRASTS = {
    ("target_0.5", MEAN): (
        "1.0352206105211128",
        "0.731124513122874",
        "0.3040960973982389",
        "0.2937500415927403",
        "0.2092748425032672",
        "0.40502803805332704",
    ),
    ("target_1", MEAN): (
        "1.092759578100716",
        "0.8906227769330144",
        "0.2021368011677017",
        "0.18497829277234798",
        "0.11527971806838953",
        "0.29309062029011934",
    ),
    ("target_1.5", MEAN): (
        "1.1983789706913133",
        "1.0704367071739398",
        "0.12794226351737356",
        "0.10676277425292856",
        "0.020483544047844287",
        "0.24346541372312144",
    ),
    ("all_three_equal_weight", MEAN): (
        "1.108786386437714",
        "0.897394665743276",
        "0.211391720694438",
        "0.19065143951991775",
        "0.14147918892429312",
        "0.28706179157274564",
    ),
    ("target_0.5", TRANSFORMED): (
        "0.9232284441047037",
        "0.731124513122874",
        "0.19210393098182976",
        "0.2080784362835805",
        "0.1022460644482635",
        "0.288206098274289",
    ),
    ("target_1", TRANSFORMED): (
        "1.0588377003829617",
        "0.8906227769330144",
        "0.16821492344994718",
        "0.15886752369046456",
        "0.0894079872314857",
        "0.2531921749973358",
    ),
    ("target_1.5", TRANSFORMED): (
        "1.180473820170543",
        "1.0704367071739398",
        "0.11003711299660306",
        "0.09321436114585413",
        "0.011334839001453174",
        "0.2169369199332625",
    ),
    ("all_three_equal_weight", TRANSFORMED): (
        "1.0541799882194027",
        "0.897394665743276",
        "0.15678532247612667",
        "0.14872728018765569",
        "0.09121647906512306",
        "0.2254439552715565",
    ),
    ("target_0.5", NOOP): (
        "0.5",
        "0.731124513122874",
        "-0.231124513122874",
        "-0.462249026245748",
        "-0.31389352317395",
        "-0.1614723555438104",
    ),
    ("target_1", NOOP): (
        "1.0",
        "0.8906227769330144",
        "0.10937722306698561",
        "0.10937722306698561",
        "0.014741812775901053",
        "0.19645887116494112",
    ),
    ("target_1.5", NOOP): (
        "1.5",
        "1.0704367071739398",
        "0.4295632928260602",
        "0.2863755285507068",
        "0.3205612675922263",
        "0.52473849109471",
    ),
    ("all_three_equal_weight", NOOP): (
        "1.0",
        "0.897394665743276",
        "0.10260533425672395",
        "0.10260533425672395",
        "0.020589340312290014",
        "0.17694636049062762",
    ),
}


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not a finite decimal: {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"{label} is not a finite decimal")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected={expected!r}, actual={actual!r}")


def _decimal_equal(actual: Any, expected: str, label: str) -> None:
    _equal(_decimal(actual, label), Decimal(expected), label)


def _close(actual: Decimal, expected: Decimal, label: str) -> None:
    if abs(actual - expected) > Decimal("1e-13"):
        raise ValueError(f"{label} changed: expected={expected}, actual={actual}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(
    rows: Iterable[Mapping[str, str]], label: str, **criteria: str
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(column) == expected for column, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {label} row for {criteria}, found {len(matches)}")
    return matches[0]


def _verify_pins(paths: Mapping[str, Path]) -> None:
    _equal(set(paths), set(EXPECTED_HASHES), "pinned input labels")
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _equal(file_sha256(path), EXPECTED_HASHES[label], f"SHA-256 for {label}")


def _verify_artifact_index(root: Path, index: Mapping[str, Any], label: str) -> None:
    if not index:
        raise ValueError(f"{label} has no artifact hash index")
    for relative, expected in index.items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        _equal(file_sha256(path), str(expected), f"{label} artifact {relative}")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    _equal(protocol["schema"], "observerbench.ioi_phase08_target_sensitivity.v1", "protocol schema")
    _equal(protocol["status"], SCIENTIFIC_STATUS, "protocol status")
    _equal(protocol["targets"], [Decimal("0.5"), Decimal("1.0"), Decimal("1.5")], "targets")
    _equal(protocol["selectors"], [DIRECT, MEAN, TRANSFORMED], "selectors")
    _equal(protocol["analytic_reference"], NOOP, "analytic reference")
    _equal(protocol["candidate_pool_count"], 48, "candidate pools")
    _equal(protocol["candidate_pool_size_including_noop"], 31, "candidate pool size")
    _equal(protocol["test_prompt_count"], 512, "test prompts")
    _equal(protocol["test_pair_clusters"], 32, "pair clusters")
    _equal(protocol["templates"], 8, "templates")
    _equal(protocol["bootstrap"]["repeats"], 5000, "bootstrap repeats")
    _equal(protocol["bootstrap"]["seed"], 2_808_101, "bootstrap seed")
    _equal(
        protocol["bootstrap"]["axes"],
        ["32 unordered name-pair clusters", "48 action pools"],
        "bootstrap axes",
    )
    if "No Phase-8 threshold licenses a primary claim." not in protocol["bootstrap"]["reporting"]:
        raise ValueError("Phase-8 no-success-gate disclosure changed")


def _validate_aborted_attempt(root: Path, record: Mapping[str, Any]) -> None:
    _equal(
        record["status"],
        "aborted_after_complete_measurement_before_value_inspection_or_evaluation",
        "aborted v1 status",
    )
    _equal(record["candidate_outcome_values_inspected"], False, "aborted v1 inspection")
    _equal(record["evaluation_run"], False, "aborted v1 evaluation")
    _equal(
        record["actions_or_measurement_union_changed_for_replacement"],
        False,
        "replacement action stability",
    )
    _verify_artifact_index(root, record["artifact_hashes"], "aborted v1")


def _validate_freeze(root: Path, freeze: Mapping[str, Any]) -> None:
    _equal(freeze["schema"], "observerbench.ioi_phase08_prediction_action_freeze.v1", "freeze schema")
    _equal(freeze["status"], "all_sensitivity_actions_frozen_before_new_outcomes", "freeze status")
    _equal(freeze["scientific_status"], SCIENTIFIC_STATUS, "freeze scientific status")
    _equal(freeze["protocol_sha256"], EXPECTED_HASHES["protocol"], "freeze protocol")
    _equal(freeze["new_candidate_outcomes_loaded"], False, "freeze new outcomes")
    _equal(
        freeze["phase7_outcome_values_loaded_during_fit"],
        False,
        "freeze inherited outcome reads",
    )
    _equal(freeze["counts"], EXPECTED_COUNTS, "freeze counts")
    _equal(freeze["basis_columns"], 92, "basis columns")
    _equal(freeze["targets"], [Decimal("0.5"), Decimal("1.0"), Decimal("1.5")], "freeze targets")
    _equal(freeze["selectors"], [DIRECT, MEAN, TRANSFORMED], "freeze selectors")
    _verify_artifact_index(root / "prediction_freeze", freeze["artifact_hashes"], "freeze")


def _validate_audit(repo_root: Path, audit: Mapping[str, Any]) -> None:
    _equal(audit["schema"], "observerbench.ioi_phase08_preoutcome_audit.v1", "audit schema")
    _equal(
        audit["status"],
        "deterministic_recomputation_passed_new_outcomes_unopened",
        "audit status",
    )
    _equal(audit["scientific_status"], SCIENTIFIC_STATUS, "audit scientific status")
    _equal(audit["protocol_sha256"], EXPECTED_HASHES["protocol"], "audit protocol")
    _equal(audit["freeze_manifest_sha256"], EXPECTED_HASHES["freeze_manifest"], "audit freeze")
    _equal(audit["new_outcome_values_loaded"], False, "audit outcome gate")
    _equal(audit["recomputed_all_predictions_actions_and_unions"], True, "audit recomputation")
    _equal(audit["frozen_counts"], EXPECTED_COUNTS, "audit counts")
    if _decimal(audit["maximum_numeric_recomputation_difference"], "audit maximum difference") > Decimal("5e-15"):
        raise ValueError("Phase-8 deterministic recomputation tolerance changed")
    hashes = audit["downstream_source_hashes"]
    _equal(set(hashes), EXPECTED_DOWNSTREAM_SOURCES, "downstream source closure")
    for relative, expected in hashes.items():
        _equal(file_sha256(repo_root / relative), str(expected), f"audited source {relative}")


def _validate_measurement(root: Path, spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected_counts = {
        "mask_shards": 10,
        "new_effect_cells": 75_776,
        "new_selected_unique_nonnoop_masks": 148,
        "noop_ablation_forward_passes": 0,
        "reused_phase7_masks_remeasured": 0,
        "test_prompts": 512,
        "unselected_candidate_masks_measured": 0,
    }
    _equal(spec["schema"], "observerbench.ioi_phase08_selected_measurement_spec.v1", "measurement spec schema")
    _equal(spec["status"], "sealed_before_new_outcomes", "measurement spec status")
    _equal(spec["scientific_status"], SCIENTIFIC_STATUS, "measurement spec scientific status")
    _equal(spec["counts"], expected_counts, "measurement spec counts")
    for field, expected in (
        ("protocol", EXPECTED_HASHES["protocol"]),
        ("phase8_freeze_manifest", EXPECTED_HASHES["freeze_manifest"]),
        ("phase8_preoutcome_audit", EXPECTED_HASHES["preoutcome_audit"]),
        ("phase8_new_measurement_masks", EXPECTED_HASHES["new_measurement_masks"]),
    ):
        _equal(spec["source_hashes"][field], expected, f"measurement spec {field}")

    _equal(manifest["schema"], "observerbench.ioi_phase08_selected_measurement.v1", "measurement schema")
    _equal(
        manifest["status"],
        "all_frozen_new_selected_mask_outcomes_measured",
        "measurement status",
    )
    _equal(manifest["scientific_status"], SCIENTIFIC_STATUS, "measurement scientific status")
    _equal(manifest["measurement_spec_sha256"], EXPECTED_HASHES["measurement_spec"], "measurement spec binding")
    _equal(manifest["counts"], expected_counts, "measurement counts")
    _equal(manifest["fixed_actions"], 576, "measurement fixed actions")
    _equal(manifest["noop_ablation_forward_passes"], 0, "measurement no-op passes")
    _equal(manifest["reused_phase7_masks_remeasured"], 0, "measurement reused masks")
    _equal(manifest["unselected_candidate_masks_measured"], 0, "measurement unselected masks")
    _equal(manifest["source_hashes"], spec["source_hashes"], "measurement source bindings")
    _verify_artifact_index(root / "new_measurement_v2", manifest["artifact_hashes"], "measurement")


def _validate_mask_scope(paths: Mapping[str, Path]) -> dict[str, Any]:
    all_rows = _read_csv(paths["all_selected_masks"])
    new_rows = _read_csv(paths["new_measurement_masks"])
    reused_rows = _read_csv(paths["reused_phase7_masks"])
    _equal(len(all_rows), 237, "combined selected-mask count")
    _equal(len(new_rows), 148, "new selected-mask count")
    _equal(len(reused_rows), 89, "reused selected-mask count")
    all_ids = {row["mask_id"] for row in all_rows}
    new_ids = {row["mask_id"] for row in new_rows}
    reused_ids = {row["mask_id"] for row in reused_rows}
    _equal(len(all_ids), 237, "combined mask-id uniqueness")
    _equal(len(new_ids), 148, "new mask-id uniqueness")
    _equal(len(reused_ids), 89, "reused mask-id uniqueness")
    _equal(new_ids & reused_ids, set(), "new/reused mask disjointness")
    _equal(new_ids | reused_ids, all_ids, "new/reused mask union")
    for label, rows in (
        ("all", all_rows),
        ("new", new_rows),
        ("reused", reused_rows),
    ):
        if any(row["mask_bits"].zfill(13) == "0" * 13 for row in rows):
            raise ValueError(f"{label} measured union contains exact no action")
        if any(row["is_noop"].lower() != "false" for row in rows):
            raise ValueError(f"{label} measured union contains a no-op flag")
    return {
        "selected_unique_nonnoop_masks": 237,
        "newly_measured_masks": 148,
        "hash_verified_reused_phase7_masks": 89,
        "noop_ablation_forward_passes": 0,
    }


def _validate_fixed_actions(rows: list[dict[str, str]]) -> None:
    _equal(len(rows), 576, "fixed-action row count")
    _equal({row["scientific_status"] for row in rows}, {SCIENTIFIC_STATUS}, "fixed-action status")
    _equal({row["selector"] for row in rows}, {DIRECT, MEAN, TRANSFORMED, NOOP}, "fixed-action selectors")
    _equal({row["target"] for row in rows}, {"0.5", "1.0", "1.5"}, "fixed-action targets")
    if len({(row["selector"], row["target"], row["pool_id"]) for row in rows}) != 576:
        raise ValueError("fixed-action table contains duplicate selector-target-pool rows")
    _equal(
        sum(row["selected_is_noop"].lower() == "true" for row in rows),
        156,
        "fixed-action no-op selections",
    )


def _validate_evaluation(
    repo_root: Path,
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    _equal(manifest["schema"], "observerbench.ioi_phase08_evaluation.v1", "evaluation schema")
    _equal(
        manifest["status"],
        "post_confirmatory_secondary_sensitivity_complete",
        "evaluation status",
    )
    _equal(manifest["scientific_status"], SCIENTIFIC_STATUS, "evaluation scientific status")
    _equal(manifest["secondary_no_success_gate"], True, "evaluation success gate")
    for field, expected in (
        ("protocol_sha256", EXPECTED_HASHES["protocol"]),
        ("freeze_manifest_sha256", EXPECTED_HASHES["freeze_manifest"]),
        ("preoutcome_audit_sha256", EXPECTED_HASHES["preoutcome_audit"]),
        ("phase8_measurement_manifest_sha256", EXPECTED_HASHES["measurement_manifest"]),
        (
            "phase7_measurement_manifest_sha256",
            "6681b04b114b4aba1129edf9d73e5197426213130ae44b21333c859a2bed7a9e",
        ),
    ):
        _equal(manifest[field], expected, f"evaluation {field}")
    _verify_artifact_index(root / "evaluation_v2", manifest["artifact_hashes"], "evaluation")

    phase8_measurement = _read_json(root / "new_measurement_v2/measurement_manifest.json")
    phase8_shards = {
        relative: str(value)
        for relative, value in phase8_measurement["artifact_hashes"].items()
        if str(relative).startswith("shards/")
    }
    _equal(manifest["phase8_shard_hashes"], phase8_shards, "evaluation Phase-8 shards")
    phase7_root = repo_root / "results/revision/phase07/ioi_canonical_noop_confirmation_v2/selected_measurement"
    for relative, expected in manifest["phase7_shard_hashes"].items():
        _equal(file_sha256(phase7_root / relative), str(expected), f"evaluation Phase-7 shard {relative}")


def _validate_contrasts(
    contrast_rows: list[dict[str, str]],
    digest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _equal(len(contrast_rows), 12, "secondary contrast count")
    output: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], Mapping[str, str]] = {}
    for key, expected in EXPECTED_CONTRASTS.items():
        scope, reference = key
        row = _one(
            contrast_rows,
            "secondary contrast",
            target_scope=scope,
            reference=reference,
        )
        _equal(row["analysis_status"], SCIENTIFIC_STATUS, f"{key} status")
        _equal(row["candidate"], DIRECT, f"{key} candidate")
        _equal(row["bootstrap_repeats"], "5000", f"{key} bootstrap repeats")
        _equal(row["secondary_no_success_gate"], "True", f"{key} success gate")
        for field, value in zip(
            (
                "reference_mean_loss",
                "candidate_mean_loss",
                "absolute_loss_reduction",
                "relative_loss_reduction",
                "q025",
                "q975",
            ),
            expected,
        ):
            _decimal_equal(row[field], value, f"{key} {field}")
        reference_loss, candidate_loss, difference, relative, _low, _high = (
            _decimal(value, f"{key} expected value") for value in expected
        )
        _close(reference_loss - candidate_loss, difference, f"{key} absolute identity")
        _close(difference / reference_loss, relative, f"{key} relative identity")
        by_key[key] = row

    for scope in TARGET_SCOPES:
        direct_losses = {
            _decimal(by_key[(scope, reference)]["candidate_mean_loss"], f"{scope} direct loss")
            for reference in REFERENCES
        }
        _equal(len(direct_losses), 1, f"{scope} direct-loss consistency")
        output.append(
            {
                "target_scope": scope,
                "target_label": TARGET_LABELS[scope],
                "direct_risk_mean_loss": float(next(iter(direct_losses))),
                **{
                    f"{prefix}_{field}": float(
                        _decimal(
                            by_key[(scope, reference)][column],
                            f"{scope} {reference} {column}",
                        )
                    )
                    for prefix, reference in (
                        ("natural_mean", MEAN),
                        ("transformed_mean", TRANSFORMED),
                        ("noop", NOOP),
                    )
                    for field, column in (
                        ("reference_mean_loss", "reference_mean_loss"),
                        ("absolute_gain", "absolute_loss_reduction"),
                        ("relative_gain_fraction", "relative_loss_reduction"),
                        ("ci95_low", "q025"),
                        ("ci95_high", "q975"),
                    )
                },
            }
        )

    for scope in TARGET_SCOPES:
        for reference in (MEAN, TRANSFORMED):
            if _decimal(by_key[(scope, reference)]["q025"], "positive estimand interval") <= 0:
                raise ValueError(f"{scope} versus {reference} no longer excludes zero")
    if _decimal(by_key[("target_0.5", NOOP)]["q975"], "negative no-op interval") >= 0:
        raise ValueError("the retained t=0.5 negative no-op result changed")
    for scope in ("target_1", "target_1.5", "all_three_equal_weight"):
        if _decimal(by_key[(scope, NOOP)]["q025"], "positive no-op interval") <= 0:
            raise ValueError(f"{scope} versus no action no longer excludes zero")

    _equal(digest["scientific_status"], SCIENTIFIC_STATUS, "digest scientific status")
    _equal(digest["secondary_no_success_gate"], True, "digest success gate")
    _equal(digest["targets"], [Decimal("0.5"), Decimal("1.0"), Decimal("1.5")], "digest targets")
    _equal(len(digest["contrasts"]), 12, "digest contrast count")
    digest_rows = {
        (str(row["target_scope"]), str(row["reference"])): row
        for row in digest["contrasts"]
    }
    _equal(set(digest_rows), set(EXPECTED_CONTRASTS), "digest contrast keys")
    for key, row in digest_rows.items():
        csv_row = by_key[key]
        for field in (
            "reference_mean_loss",
            "candidate_mean_loss",
            "absolute_loss_reduction",
            "relative_loss_reduction",
            "q025",
            "q975",
        ):
            _decimal_equal(row[field], csv_row[field], f"digest {key} {field}")
    return output


def _validate_summary(
    rows: list[dict[str, str]], normalized: list[dict[str, Any]]
) -> None:
    _equal(len(rows), 12, "observer-summary row count")
    for entry in normalized[:3]:
        target = entry["target_label"]
        for selector, field in (
            (DIRECT, "direct_risk_mean_loss"),
            (MEAN, "natural_mean_reference_mean_loss"),
            (TRANSFORMED, "transformed_mean_reference_mean_loss"),
            (NOOP, "noop_reference_mean_loss"),
        ):
            row = _one(rows, "observer summary", target=target, selector=selector)
            _decimal_equal(
                row["mean_target_loss"],
                str(entry[field]),
                f"observer summary {target} {selector}",
            )


def _validate_prompt_loss_means(
    path: Path, normalized: list[dict[str, Any]]
) -> None:
    totals: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["target"], row["selector"])
            if key[0] not in {"0.5", "1.0", "1.5"} or key[1] not in {
                DIRECT,
                MEAN,
                TRANSFORMED,
                NOOP,
            }:
                raise ValueError(f"unexpected prompt-loss cell: {key}")
            totals[key] += _decimal(row["actual_target_loss"], "prompt target loss")
            counts[key] += 1
    _equal(len(totals), 12, "prompt-loss selector-target cells")
    _equal(set(counts.values()), {48 * 512}, "prompt-loss rows per cell")
    by_target = {row["target_label"]: row for row in normalized[:3]}
    for target, entry in by_target.items():
        for selector, field in (
            (DIRECT, "direct_risk_mean_loss"),
            (MEAN, "natural_mean_reference_mean_loss"),
            (TRANSFORMED, "transformed_mean_reference_mean_loss"),
            (NOOP, "noop_reference_mean_loss"),
        ):
            key = (target, selector)
            observed = totals[key] / Decimal(counts[key])
            _close(observed, _decimal(entry[field], f"{key} expected mean"), f"{key} prompt mean")


def _validate_prediction_diagnostics(
    rows: list[dict[str, str]], digest: Mapping[str, Any]
) -> dict[str, Any]:
    _equal(len(rows), 9, "prediction-diagnostic row count")
    expected_negative = {"0.5": 44, "1.0": 44, "1.5": 16}
    for row in rows:
        _equal(row["design_rank"], "92", "diagnostic rank")
        _equal(row["n_columns"], "92", "diagnostic columns")
        _equal(row["selected_action_count"], "48", "diagnostic selected actions")
    transformed = {
        row["target"]: row for row in rows if row["selector"] == TRANSFORMED
    }
    _equal(set(transformed), set(expected_negative), "transformed diagnostic targets")
    for target, count in expected_negative.items():
        _equal(
            transformed[target]["selected_raw_negative_count"],
            str(count),
            f"transformed raw-negative selections at {target}",
        )
    recorded = {
        str(_decimal(row["target"], "digest diagnostic target")): int(
            row["selected_raw_negative_count"]
        )
        for row in digest["transformed_mean_prediction_diagnostics"]
    }
    _equal(recorded, expected_negative, "digest transformed diagnostics")
    return {
        "raw_unclipped_rule": True,
        "transformed_mean_selected_raw_negative_counts": {
            "target_0.5": 44,
            "target_1.0": 44,
            "target_1.5": 16,
        },
        "selected_action_count_per_target": 48,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gain_cell(row: Mapping[str, Any], prefix: str, *, emphasize: bool = False) -> str:
    relative = 100 * float(row[f"{prefix}_relative_gain_fraction"])
    low = float(row[f"{prefix}_ci95_low"])
    high = float(row[f"{prefix}_ci95_high"])
    sign = "+" if relative >= 0 else "$-$"
    value = f"{sign}{abs(relative):.1f}\\% [{low:.3f}, {high:.3f}]"
    return rf"\textbf{{{value}}}" if emphasize else value


def _write_tex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Target & Direct loss & Gain vs. mean & Gain vs. transformed mean & Gain vs. no action \\",
        r"\midrule",
    ]
    for row in rows:
        scope = row["target_scope"]
        target = (
            rf"${row['target_label']}$"
            if scope != "all_three_equal_weight"
            else "Equal-target mean"
        )
        lines.append(
            f"{target} & {row['direct_risk_mean_loss']:.3f} & "
            f"{_gain_cell(row, 'natural_mean')} & "
            f"{_gain_cell(row, 'transformed_mean')} & "
            f"{_gain_cell(row, 'noop', emphasize=scope == 'target_0.5')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Post-confirmatory Study-4 target sensitivity. This secondary "
                r"analysis was designed after the $t=1$ confirmation and has no success "
                r"gate. Each gain cell reports the relative loss change and, in brackets, "
                r"a 95\% paired bootstrap interval for the absolute reference-minus-direct "
                r"loss difference. The three targets share 512 prompts and 48 frozen action "
                r"pools; the last row weights the targets equally. The negative no-action "
                r"result at $t=0.5$ is retained.}"
            ),
            r"\label{tab:phase8-ioi-sensitivity}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _contains_private_path(paths: Iterable[Path]) -> bool:
    banned = ("/Users/", "/Downloads/", "observerbench-review-phase01")
    return any(
        marker in path.read_text(encoding="utf-8", errors="replace")
        for path in paths
        for marker in banned
    )


def build(*, repo_root: Path, result_root: Path, outdir: Path) -> dict[str, Any]:
    paths = {
        label: (
            repo_root / path.relative_to(REPO_ROOT)
            if path.is_relative_to(REPO_ROOT)
            else path
        )
        for label, path in INPUTS.items()
    }
    if result_root != DEFAULT_ROOT:
        for label, path in list(paths.items()):
            default_path = (
                repo_root / path.relative_to(REPO_ROOT)
                if path.is_relative_to(REPO_ROOT)
                else path
            )
            default_root = repo_root / DEFAULT_ROOT.relative_to(REPO_ROOT)
            if default_path.is_relative_to(default_root):
                paths[label] = result_root / default_path.relative_to(default_root)

    _verify_pins(paths)
    protocol = _read_json(paths["protocol"])
    freeze = _read_json(paths["freeze_manifest"])
    audit = _read_json(paths["preoutcome_audit"])
    spec = _read_json(paths["measurement_spec"])
    measurement = _read_json(paths["measurement_manifest"])
    evaluation = _read_json(paths["evaluation_manifest"])
    digest = _read_json(paths["result_digest"])

    _validate_protocol(protocol)
    _validate_aborted_attempt(result_root, _read_json(paths["aborted_measurement_v1"]))
    _validate_freeze(result_root, freeze)
    _validate_audit(repo_root, audit)
    _validate_measurement(result_root, spec, measurement)
    scope = _validate_mask_scope(paths)
    _validate_fixed_actions(_read_csv(paths["fixed_actions"]))
    _validate_evaluation(repo_root, result_root, evaluation)
    normalized = _validate_contrasts(_read_csv(paths["secondary_contrasts"]), digest)
    _validate_summary(_read_csv(paths["observer_summary"]), normalized)
    _validate_prompt_loss_means(paths["fixed_action_prompt_losses"], normalized)
    diagnostics = _validate_prediction_diagnostics(
        _read_csv(paths["prediction_diagnostics"]), digest
    )
    _equal(digest["reused_phase7_masks"], 89, "digest reused masks")
    _equal(digest["newly_measured_masks"], 148, "digest new masks")
    _equal(digest["combined_selected_mask_union"], 237, "digest mask union")
    _equal(digest["noop_ablation_forward_passes"], 0, "digest no-op passes")

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ioi_target_sensitivity_results.csv"
    tex_path = outdir / "ioi_target_sensitivity_table.tex"
    _write_csv(normalized, csv_path)
    _write_tex(normalized, tex_path)
    output_paths = [csv_path, tex_path]
    checks = {
        "all_pinned_input_hashes_match": True,
        "aborted_v1_was_uninspected_and_unevaluated": True,
        "expanded_preoutcome_source_closure_matches": True,
        "predictions_actions_and_mask_union_were_frozen_before_new_outcomes": True,
        "only_148_new_masks_were_measured_and_89_hash_verified_masks_reused": True,
        "evaluation_is_bound_to_the_expanded_audit_and_replacement_measurement": True,
        "all_three_targets_and_all_three_references_are_reported": True,
        "prompt_level_means_recompute_to_the_published_contrasts": True,
        "mean_and_transformed_mean_intervals_are_positive_at_every_target": True,
        "target_0.5_noop_interval_is_negative_and_retained": True,
        "secondary_analysis_has_no_success_gate": True,
    }
    manifest = {
        "schema": "observerbench.phase08.ioi_target_sensitivity_artifact.v1",
        "scientific_status": SCIENTIFIC_STATUS,
        "result_calibration": "good_positive_estimand_result_with_target_dependent_intervention_value",
        "claim_status": {
            "direct_vs_natural_mean_all_three_targets": "positive_intervals_exclude_zero",
            "direct_vs_transformed_mean_all_three_targets": "positive_intervals_exclude_zero",
            "direct_vs_noop_target_0.5": "negative_interval_excludes_zero",
            "direct_vs_noop_target_1.0": "positive_interval_excludes_zero",
            "direct_vs_noop_target_1.5": "positive_interval_excludes_zero",
            "equal_target_aggregate_vs_all_references": "positive_intervals_exclude_zero",
            "success_gate": "none_post_confirmatory_secondary",
        },
        "results": normalized,
        "measurement_scope": scope,
        "prediction_diagnostics": diagnostics,
        "provenance": {
            "v1_measurement": "aborted_before_value_inspection_or_evaluation",
            "v2_measurement": "used_after_expanded_source_closure_audit",
            "targets_and_references_suppressed": 0,
            "success_gate": None,
        },
        "interpretation_boundary": {
            "supported": [
                "On this frozen IOI action family and prompt surface, direct-risk selection has lower loss than natural-mean and transformed-mean selection at each of the three reported targets.",
                "Intervention value relative to exact no action depends on the target: it is negative at t=0.5 and positive at t=1.0 and t=1.5.",
                "The equal-target aggregate is positive against all three references.",
            ],
            "not_supported": [
                "A new primary or preregistered claim.",
                "Uniform intervention value over targets, action families, circuits, models, or tasks.",
                "Held-out-template, behavioral-utility, deployment, or safety claims.",
            ],
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "input_hashes": source_hashes(paths.values(), repo_root),
        "code_hashes": source_hashes(
            [repo_root / "scripts/build_phase08_ioi_sensitivity_artifact.py"],
            repo_root,
        ),
        "output_hashes": source_hashes(output_paths, repo_root),
        "contains_private_local_path": _contains_private_path(output_paths),
        "commands_to_reproduce": [
            "python scripts/build_phase08_ioi_sensitivity_artifact.py"
        ],
    }
    write_json(outdir / "ioi_target_sensitivity_artifact_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build(
        repo_root=args.repo_root.resolve(),
        result_root=args.result_root.resolve(),
        outdir=args.outdir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
