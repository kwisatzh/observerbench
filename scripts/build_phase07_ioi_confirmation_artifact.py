#!/usr/bin/env python3
"""Build the checked Phase-7 IOI confirmation paper artifact.

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
DEFAULT_ROOT = (
    REPO_ROOT
    / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"
)
DEFAULT_OUTDIR = REPO_ROOT / "paper/generated_phase07/ioi_confirmation"
PROTOCOL = REPO_ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json"

INPUTS = {
    "protocol": PROTOCOL,
    "design_manifest": DEFAULT_ROOT / "design/design_manifest.json",
    "pretest_manifest": DEFAULT_ROOT / "clean_pretest/pretest_manifest.json",
    "clean_task_validity": DEFAULT_ROOT / "clean_pretest/clean_task_validity.csv",
    "freeze_manifest": DEFAULT_ROOT / "prediction_freeze/prediction_action_manifest.json",
    "selected_measurement_masks": DEFAULT_ROOT / "prediction_freeze/selected_measurement_masks.csv",
    "preoutcome_audit": DEFAULT_ROOT / "preoutcome_audit/preoutcome_audit.json",
    "measurement_spec": DEFAULT_ROOT / "selected_measurement/measurement_run_spec.json",
    "measurement_manifest": DEFAULT_ROOT / "selected_measurement/measurement_manifest.json",
    "evaluation_manifest": DEFAULT_ROOT / "evaluation/evaluation_manifest.json",
    "hypothesis_audit": DEFAULT_ROOT / "evaluation/hypothesis_audit.json",
    "result_digest": DEFAULT_ROOT / "evaluation/result_digest.json",
    "primary_contrasts": DEFAULT_ROOT / "evaluation/primary_contrasts.csv",
    "observer_summary": DEFAULT_ROOT / "evaluation/observer_summary.csv",
    "pool_signs": DEFAULT_ROOT / "evaluation/pool_signs.csv",
    "template_sensitivity": DEFAULT_ROOT / "evaluation/template_sensitivity.csv",
    "fixed_action_prompt_losses": DEFAULT_ROOT / "evaluation/fixed_action_prompt_losses.csv.gz",
}
EXPECTED_HASHES = {
    "protocol": "198a1d30a61e9463863a7597becc7973c7f7a2ee8a915962d12b6612e194befb",
    "design_manifest": "3dfa6352524c66b6b57794ec9ed2c89a900afd7df3d9879cd7f21cb0686feee5",
    "pretest_manifest": "7c148b1137460fa7d4674f97f3984ce8a39068cd1dbe394852fed2f9141ae2a2",
    "clean_task_validity": "b4bbd616a93503183d1c2f59b753207c154342f222ffc96eed87f33fd0ad7d58",
    "freeze_manifest": "fce0b027f232841653d5b7f24f3254243a71d3fbd282cc9c468c4781eab334e5",
    "selected_measurement_masks": "7f843bc5c2649ca2e84a52dccb6a7fc12330fbeab586186e4ad766517b9869eb",
    "preoutcome_audit": "738063c000b2b38ea77ab9135bdd1e2f0fe9c04165c57e9637bf8a518b0865cf",
    "measurement_spec": "ddf84f9cb0b41cc06cf148d7e9f74387add7220436ea6339ecc877c1c4f92549",
    "measurement_manifest": "6681b04b114b4aba1129edf9d73e5197426213130ae44b21333c859a2bed7a9e",
    "evaluation_manifest": "41b541d8eef690bdbb7ded24c991bf1ab713e857db4ef2843109aec8198a3535",
    "hypothesis_audit": "0eff4ab684557f6c6b6ae591708e32cf038408bd1ddfc524f64751db14e683ed",
    "result_digest": "384c52005128490e9d05835969d8d30f6c5d76c3f2972b68e3957f3a24b22322",
    "primary_contrasts": "e46d31d6f85d2acd2b66f256ba8f7bea22c89c828252722180e406082485b45f",
    "observer_summary": "902e019f4a782df724c26010ec339479f8f158c0885629003f56edf989ecbc78",
    "pool_signs": "e8c7001f093c99b8407684681312a38b0d01491e40a56131e233d48471eb4c70",
    "template_sensitivity": "a0018a60054dd16a67512942fef15eb3020afbdeaf5ff14328048c74355c6a12",
    "fixed_action_prompt_losses": "7633ca90d531812d129a13598bb667b71794aa8d64a26230d914f4eb63a09c0e",
}

DIRECT = "direct_risk_head_pair_quadratic"
MEAN = "natural_mean_effect_head_pair_quadratic"
NOOP = "exact_noop"
CONTRASTS = {
    "H1a_estimand": {
        "reference": MEAN,
        "reference_mean_loss": "1.092759578100716",
        "candidate_mean_loss": "0.8906227769330144",
        "absolute_loss_reduction": "0.2021368011677017",
        "relative_loss_reduction": "0.18497829277234798",
        "q025": "0.11496571685759895",
        "q975": "0.29256013182554524",
        "positive_pool_count": "30",
        "zero_pool_count": "7",
        "negative_pool_count": "11",
        "template_direction_count": 8,
    },
    "H1b_intervention_value": {
        "reference": NOOP,
        "reference_mean_loss": "1.0",
        "candidate_mean_loss": "0.8906227769330144",
        "absolute_loss_reduction": "0.10937722306698561",
        "relative_loss_reduction": "0.10937722306698561",
        "q025": "0.01246399262114816",
        "q975": "0.1955910766065547",
        "positive_pool_count": "30",
        "zero_pool_count": "0",
        "negative_pool_count": "18",
        "template_direction_count": 6,
    },
}
DECOMPOSITION = {
    DIRECT: {
        "mean_effect": "0.6839233261222640694417317708333333333333",
        "total_loss": "0.8906227769330143935094742838541666666667",
        "mean_to_target": "0.3969548799796029905517578125",
        "dispersion": "0.4936678969534114029577164713541666666667",
    },
    MEAN: {
        "mean_effect": "0.7708828156270707681688944498697916666667",
        "total_loss": "1.092759578100716074430281575520833333333",
        "mean_to_target": "0.2859615176372850918001810709635416666667",
        "dispersion": "0.8067980604634309826301005045572916666667",
    },
    NOOP: {
        "mean_effect": "0",
        "total_loss": "1",
        "mean_to_target": "1",
        "dispersion": "0",
    },
}
CLEAN = {
    "overall": ("512", "3.606913924217224", "0.9765625"),
    "abba_meeting": ("64", "2.4733858555555344", "0.953125"),
    "abba_office": ("64", "2.4009827971458435", "0.953125"),
    "abba_park": ("64", "3.507099986076355", "0.953125"),
    "abba_store": ("64", "4.167588159441948", "0.953125"),
    "baba_meeting": ("64", "3.0559157580137253", "1.0"),
    "baba_office": ("64", "3.795596659183502", "1.0"),
    "baba_park": ("64", "4.519814997911453", "1.0"),
    "baba_store": ("64", "4.9349271804094315", "1.0"),
}


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not a decimal: {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"{label} is not finite")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected={expected!r}, actual={actual!r}")


def _decimal_equal(actual: Any, expected: str, label: str) -> None:
    _equal(_decimal(actual, label), Decimal(expected), label)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Iterable[Mapping[str, str]], label: str, **criteria: str) -> Mapping[str, str]:
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
        _equal(file_sha256(path), str(expected), f"{label} artifact {relative}")


def _validate_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    protocol = _read_json(paths["protocol"])
    design = _read_json(paths["design_manifest"])
    pretest = _read_json(paths["pretest_manifest"])
    freeze = _read_json(paths["freeze_manifest"])
    preaudit = _read_json(paths["preoutcome_audit"])
    spec = _read_json(paths["measurement_spec"])
    measurement = _read_json(paths["measurement_manifest"])
    evaluation = _read_json(paths["evaluation_manifest"])

    _equal(protocol["schema"], "observerbench.ioi_phase07_canonical_noop_confirmation.v2", "protocol schema")
    _decimal_equal(protocol["target"], "1", "target")
    if "only screened target" not in protocol["target_choice"]["rule"]:
        raise ValueError("v2 target rationale changed")
    _equal(protocol["v1_aborted_attempt"]["candidate_outcome_values_inspected"], False, "v1 inspection disclosure")
    _equal(design["status"], "outcome_free_design_frozen", "design status")
    _equal(design["contains_model_outcomes"], False, "design outcome gate")
    _verify_artifact_index(paths["design_manifest"].parent, design["artifact_hashes"], "design")

    _equal(pretest["status"], "clean_pretest_passed_candidate_outcomes_unopened", "pretest status")
    _equal(pretest["gate"]["passed"], True, "clean gate")
    _equal(
        pretest["gate"]["checks"],
        {
            "eight_templates_retained": True,
            "every_template_accuracy": True,
            "every_template_positive_mean_clean_ld": True,
            "exact_prompt_count": True,
            "overall_accuracy": True,
        },
        "clean checks",
    )
    _verify_artifact_index(paths["pretest_manifest"].parent, pretest["artifact_hashes"], "pretest")

    _equal(freeze["status"], "actions_frozen_candidate_outcomes_unopened", "freeze status")
    _equal(freeze["phase7_candidate_outcomes_loaded"], False, "freeze candidate-outcome gate")
    _equal(freeze["phase7_candidate_mask_forward_passes"], 0, "freeze candidate passes")
    _equal(
        freeze["counts"],
        {
            "analytic_noop_action_rows": 48,
            "candidate_action_rows": 1488,
            "fixed_actions": 144,
            "phase5_train_calibration_cells": 30720,
            "selected_unique_nonnoop_masks": 89,
        },
        "freeze counts",
    )
    _verify_artifact_index(paths["freeze_manifest"].parent, freeze["artifact_hashes"], "freeze")

    _equal(preaudit["status"], "independent_recomputation_passed_v2_outcomes_unopened", "preoutcome audit status")
    _equal(preaudit["all_checks_pass"], True, "preoutcome audit gate")
    _equal(preaudit["counts"]["phase7_candidate_outcome_rows_read"], 0, "preoutcome outcome reads")
    _equal(preaudit["counts"]["selected_unique_nonnoop_masks"], 89, "preoutcome selected masks")

    upstream = {
        "protocol": EXPECTED_HASHES["protocol"],
        "design_manifest": EXPECTED_HASHES["design_manifest"],
        "pretest_manifest": EXPECTED_HASHES["pretest_manifest"],
        "prediction_action_manifest": EXPECTED_HASHES["freeze_manifest"],
        "selected_measurement_masks": EXPECTED_HASHES["selected_measurement_masks"],
        "preoutcome_audit": EXPECTED_HASHES["preoutcome_audit"],
    }
    for field, expected in upstream.items():
        _equal(str(spec["source_hashes"][field]), expected, f"measurement spec {field}")
        _equal(str(measurement["source_hashes"][field]), expected, f"measurement {field}")
    _equal(spec["status"], "sealed_before_candidate_outcomes", "measurement spec status")
    _equal(measurement["status"], "all_frozen_selected_nonnoop_outcomes_measured", "measurement status")
    _equal(measurement["measurement_spec_sha256"], EXPECTED_HASHES["measurement_spec"], "measurement spec binding")
    _equal(
        measurement["counts"],
        {
            "effect_cells": 45568,
            "mask_shards": 6,
            "noop_ablation_forward_passes": 0,
            "selected_unique_nonnoop_masks": 89,
            "test_prompts": 512,
            "unselected_candidate_masks_measured": 0,
        },
        "measurement counts",
    )
    _equal(measurement["accessed_mask_bank"], "frozen selected non-noop union only", "measurement mask scope")
    _equal(measurement["unselected_candidate_masks_measured"], 0, "unselected measurements")
    _equal(measurement["noop_ablation_forward_passes"], 0, "no-op forward passes")
    _verify_artifact_index(paths["measurement_manifest"].parent, measurement["artifact_hashes"], "measurement")

    _equal(evaluation["status"], "frozen_joint_primary_evaluation_complete", "evaluation status")
    _equal(evaluation["joint_primary_passed"], True, "joint primary gate")
    for field, expected in (
        ("protocol_sha256", EXPECTED_HASHES["protocol"]),
        ("design_manifest_sha256", EXPECTED_HASHES["design_manifest"]),
        ("pretest_manifest_sha256", EXPECTED_HASHES["pretest_manifest"]),
        ("prediction_action_manifest_sha256", EXPECTED_HASHES["freeze_manifest"]),
        ("preoutcome_audit_sha256", EXPECTED_HASHES["preoutcome_audit"]),
        ("measurement_manifest_sha256", EXPECTED_HASHES["measurement_manifest"]),
    ):
        _equal(str(evaluation[field]), expected, f"evaluation {field}")
    _verify_artifact_index(paths["evaluation_manifest"].parent, evaluation["artifact_hashes"], "evaluation")
    measured_shards = {
        key: str(value)
        for key, value in measurement["artifact_hashes"].items()
        if str(key).startswith("shards/")
    }
    _equal(
        {key: str(value) for key, value in evaluation["measured_shard_hashes"].items()},
        measured_shards,
        "evaluation measured-shard binding",
    )
    return {
        "protocol": protocol,
        "pretest": pretest,
        "freeze": freeze,
        "measurement": measurement,
        "evaluation": evaluation,
    }


def _validate_clean(rows: list[dict[str, str]]) -> dict[str, Any]:
    _equal(len(rows), 9, "clean row count")
    for scope, expected in CLEAN.items():
        row = _one(rows, "clean", scope=scope)
        for field, value in zip(
            ("prompt_count", "mean_clean_logit_difference", "io_vs_subject_pairwise_accuracy"),
            expected,
        ):
            _decimal_equal(row[field], value, f"clean {scope} {field}")
    template_accuracies = [Decimal(values[2]) for key, values in CLEAN.items() if key != "overall"]
    return {
        "status": "pass",
        "overall_pairwise_accuracy": float(Decimal(CLEAN["overall"][2])),
        "worst_template_pairwise_accuracy": float(min(template_accuracies)),
        "overall_threshold": 0.95,
        "per_template_threshold": 0.90,
        "template_count": 8,
        "prompt_count": 512,
    }


def _validate_selected_scope(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    _equal(len(rows), 89, "selected measurement row count")
    _equal(len({row["mask_id"] for row in rows}), 89, "selected mask-id uniqueness")
    _equal(len({row["mask_bits"].zfill(13) for row in rows}), 89, "selected mask-bit uniqueness")
    if any(row["mask_bits"].zfill(13) == "0" * 13 for row in rows):
        raise ValueError("analytic no-op entered the measured union")
    if any(row["is_noop"].lower() != "false" for row in rows):
        raise ValueError("selected union contains a no-op flag")
    return {"selected_unique_nonnoop_masks": 89, "analytic_noop_measured": False}


def _validate_primary(
    contrast_rows: list[dict[str, str]],
    pool_rows: list[dict[str, str]],
    audit: Mapping[str, Any],
    digest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _equal(len(contrast_rows), 2, "primary contrast count")
    _equal(audit["joint_primary_passed"], True, "hypothesis joint gate")
    _equal(digest["joint_primary"]["joint_primary_passed"], True, "digest joint gate")
    output: list[dict[str, Any]] = []
    for comparison, expected in CONTRASTS.items():
        row = _one(contrast_rows, "primary contrast", comparison_id=comparison)
        _equal(row["candidate"], DIRECT, f"{comparison} candidate")
        _equal(row["reference"], expected["reference"], f"{comparison} reference")
        for field in (
            "reference_mean_loss",
            "candidate_mean_loss",
            "absolute_loss_reduction",
            "relative_loss_reduction",
            "q025",
            "q975",
        ):
            _decimal_equal(row[field], expected[field], f"{comparison} {field}")
        for field, value in (
            ("target", "1.0"),
            ("pair_clusters", "32"),
            ("candidate_pools", "48"),
            ("bootstrap_repeats", "5000"),
        ):
            _equal(row[field], value, f"{comparison} {field}")
        entry = audit["comparisons"][comparison]
        _equal(entry["passed"], True, f"{comparison} pass")
        _equal(
            entry["checks"],
            {
                "paired_cluster_pool_interval_lower_strictly_positive": True,
                "relative_reduction_at_least_five_percent": True,
            },
            f"{comparison} checks",
        )
        for field in ("absolute_loss_reduction", "relative_loss_reduction", "q025", "q975"):
            _decimal_equal(entry[field], expected[field], f"{comparison} audit {field}")
        pool = _one(pool_rows, "pool signs", comparison_id=comparison)
        for field in ("positive_pool_count", "zero_pool_count", "negative_pool_count"):
            _equal(pool[field], expected[field], f"{comparison} {field}")
        output.append(
            {
                "comparison_id": comparison,
                "status": "pass",
                "reference": expected["reference"],
                "reference_mean_loss": float(Decimal(expected["reference_mean_loss"])),
                "direct_risk_mean_loss": float(Decimal(expected["candidate_mean_loss"])),
                "absolute_loss_reduction": float(Decimal(expected["absolute_loss_reduction"])),
                "relative_loss_reduction_fraction": float(Decimal(expected["relative_loss_reduction"])),
                "ci95_low": float(Decimal(expected["q025"])),
                "ci95_high": float(Decimal(expected["q975"])),
                "positive_pool_count": int(expected["positive_pool_count"]),
                "template_direction_count": expected["template_direction_count"],
                "template_count": 8,
            }
        )
    return output


def _read_prompt_losses(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _decomposition_and_template_directions(
    rows: list[dict[str, str]], template_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, int]]:
    with localcontext() as context:
        context.prec = 40
        return _decomposition_and_template_directions_with_context(
            rows, template_rows
        )


def _decomposition_and_template_directions_with_context(
    rows: list[dict[str, str]], template_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, int]]:
    _equal(len(rows), 512 * 48 * 3, "fixed-action prompt-loss row count")
    selectors = {DIRECT, MEAN, NOOP}
    _equal({row["selector"] for row in rows}, selectors, "prompt-loss selectors")
    if len({(row["selector"], row["pool_id"], row["prompt_id"]) for row in rows}) != len(rows):
        raise ValueError("fixed-action prompt losses contain duplicate cells")
    by_pool: dict[tuple[str, str], list[Any]] = defaultdict(
        lambda: [Decimal(0), Decimal(0), 0]
    )
    by_template: dict[tuple[str, str], list[Any]] = defaultdict(
        lambda: [Decimal(0), 0]
    )
    for row in rows:
        selector = row["selector"]
        key = (selector, row["pool_id"])
        by_pool[key][0] += _decimal(row["finite_effect"], "finite effect")
        by_pool[key][1] += _decimal(row["actual_target_loss"], "target loss")
        by_pool[key][2] += 1
        template_key = (row["template_id"], selector)
        by_template[template_key][0] += _decimal(row["actual_target_loss"], "template loss")
        by_template[template_key][1] += 1
    _equal(len(by_pool), 144, "selector-pool decomposition count")

    totals: dict[str, dict[str, Decimal]] = {}
    for selector in selectors:
        accum = {name: Decimal(0) for name in ("mean_effect", "total_loss", "mean_to_target", "dispersion")}
        pools = [(key, value) for key, value in by_pool.items() if key[0] == selector]
        _equal(len(pools), 48, f"{selector} pool count")
        for _key, (sum_effect, sum_loss, count) in pools:
            _equal(count, 512, f"{selector} prompt count per pool")
            mu = sum_effect / Decimal(count)
            risk = sum_loss / Decimal(count)
            mean_term = abs(mu - Decimal(1))
            dispersion = risk - mean_term
            _equal(risk, mean_term + dispersion, f"{selector} Jensen identity")
            accum["mean_effect"] += mu
            accum["total_loss"] += risk
            accum["mean_to_target"] += mean_term
            accum["dispersion"] += dispersion
        totals[selector] = {name: value / Decimal(48) for name, value in accum.items()}
        for field, expected in DECOMPOSITION[selector].items():
            _decimal_equal(totals[selector][field], expected, f"{selector} {field}")

    if not totals[DIRECT]["mean_to_target"] > totals[MEAN]["mean_to_target"]:
        raise ValueError("Phase-7 direct risk no longer accepts a worse mean term")
    if not totals[DIRECT]["dispersion"] < totals[MEAN]["dispersion"]:
        raise ValueError("Phase-7 direct risk no longer reduces dispersion")
    if not totals[DIRECT]["total_loss"] < totals[MEAN]["total_loss"]:
        raise ValueError("Phase-7 direct risk no longer lowers total loss")

    template_means = {
        key: value[0] / Decimal(value[1]) for key, value in by_template.items()
    }
    templates = sorted({key[0] for key in template_means})
    _equal(len(templates), 8, "template count")
    directions = {
        "H1a_estimand": sum(template_means[(name, DIRECT)] < template_means[(name, MEAN)] for name in templates),
        "H1b_intervention_value": sum(template_means[(name, DIRECT)] < template_means[(name, NOOP)] for name in templates),
    }
    _equal(directions, {"H1a_estimand": 8, "H1b_intervention_value": 6}, "template directions")
    _equal(len(template_rows), 16, "template sensitivity row count")
    for comparison, count in directions.items():
        recorded = [row for row in template_rows if row["comparison_id"] == comparison]
        _equal(len(recorded), 8, f"{comparison} template rows")
        _equal(
            sum(_decimal(row["reference_minus_direct_loss"], "template direction") > 0 for row in recorded),
            count,
            f"{comparison} recorded template directions",
        )
    output = {
        selector: {field: float(value) for field, value in fields.items()}
        for selector, fields in totals.items()
    }
    output["interpretation"] = {
        "status": "descriptive",
        "phase6_pattern_repeated": True,
        "direct_risk_accepts_larger_mean_to_target_term": True,
        "direct_risk_reduces_dispersion_more_than_the_mean_term_worsens": True,
    }
    return output, directions


def _validate_observer_summary(
    rows: list[dict[str, str]], decomposition: Mapping[str, Any]
) -> None:
    _equal(len(rows), 3, "observer summary row count")
    for selector in (DIRECT, MEAN, NOOP):
        row = _one(rows, "observer summary", selector=selector)
        expected = decomposition[selector]
        _decimal_equal(row["mean_target_loss"], str(expected["total_loss"]), f"{selector} summary loss")
        _decimal_equal(row["mean_finite_effect"], str(expected["mean_effect"]), f"{selector} summary effect")


def _write_results(
    rows: list[dict[str, Any]], decomposition: Mapping[str, Any], path: Path
) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    decomposition_path = path.with_name("ioi_confirmation_decomposition.json")
    write_json(decomposition_path, decomposition)


def _write_tex(rows: list[dict[str, Any]], decomposition: Mapping[str, Any], path: Path) -> None:
    labels = {
        "H1a_estimand": "Same-basis mean-effect plug-in",
        "H1b_intervention_value": "Exact no action",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrl@{}}",
        r"\toprule",
        r"Reference & Ref. loss & Risk loss & Reduction & 95\% interval & Templates \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['comparison_id']]} & {row['reference_mean_loss']:.3f} & "
            f"{row['direct_risk_mean_loss']:.3f} & "
            f"{100 * row['relative_loss_reduction_fraction']:.1f}\\% & "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] & "
            f"{row['template_direction_count']}/8 \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Study-4 fixed-action target tracking at the pilot-informed target "
                r"$t=1.0$. On 512 new prompt strings from eight fixed canonical templates, "
                r"intervals resample 32 name-pair clusters and 48 frozen action pools (5,000 "
                r"draws). Intervals are absolute loss reductions; the Reduction column is "
                r"relative. The last column is descriptive and shows templates with the same "
                r"direction. The clean IO-versus-subject gate passed before measurement "
                r"(97.7\% overall; 95.3\% worst template).}"
            ),
            r"\label{tab:phase7-ioi-confirmation}",
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
            if path.is_relative_to(DEFAULT_ROOT):
                paths[label] = result_root / path.relative_to(DEFAULT_ROOT)
    _verify_pins(paths)
    chain = _validate_chain(paths)
    clean = _validate_clean(_read_csv(paths["clean_task_validity"]))
    scope = _validate_selected_scope(paths["selected_measurement_masks"])
    audit = _read_json(paths["hypothesis_audit"])
    digest = _read_json(paths["result_digest"])
    headline = _validate_primary(
        _read_csv(paths["primary_contrasts"]),
        _read_csv(paths["pool_signs"]),
        audit,
        digest,
    )
    decomposition, directions = _decomposition_and_template_directions(
        _read_prompt_losses(paths["fixed_action_prompt_losses"]),
        _read_csv(paths["template_sensitivity"]),
    )
    _validate_observer_summary(_read_csv(paths["observer_summary"]), decomposition)
    _equal(
        digest["selected_unique_nonnoop_masks_measured"],
        scope["selected_unique_nonnoop_masks"],
        "digest selected-mask scope",
    )
    _equal(digest["unselected_candidate_masks_measured"], 0, "digest unselected measurements")
    _equal(digest["noop_ablation_forward_passes"], 0, "digest no-op passes")

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ioi_confirmation_results.csv"
    decomposition_path = outdir / "ioi_confirmation_decomposition.json"
    tex_path = outdir / "ioi_confirmation_table.tex"
    _write_results(headline, decomposition, csv_path)
    _write_tex(headline, decomposition, tex_path)
    output_paths = [csv_path, decomposition_path, tex_path]
    checks = {
        "all_pinned_input_hashes_match": True,
        "protocol_design_pretest_freeze_audit_chain_matches": True,
        "clean_gate_passes_with_exact_values": True,
        "measurement_is_exact_selected_nonnoop_scope": True,
        "evaluation_is_bound_to_measurement_and_preoutcome_seals": True,
        "joint_H1a_H1b_gate_passes_with_exact_values": True,
        "per_pool_jensen_identity_and_exact_decomposition_match": True,
        "template_directions_are_descriptive_8_of_8_and_6_of_8": True,
        "bounded_claim_language_only": True,
    }
    manifest = {
        "schema": "observerbench.phase07.ioi_confirmation_artifact.v1",
        "scientific_status": "pilot_informed_outcome_sealed_canonical_template_confirmation",
        "result_calibration": "good_positive_bounded",
        "claim_status": {
            "joint_primary": "pass",
            "H1a_estimand": "pass",
            "H1b_target_tracking_vs_noop": "pass",
            "clean_task_validity": "pass",
        },
        "headline_contrasts": headline,
        "clean_task_gate": clean,
        "measured_scope": scope,
        "jensen_decomposition": decomposition,
        "template_directions": {
            "status": "descriptive",
            "H1a_estimand": f"{directions['H1a_estimand']}/8",
            "H1b_target_tracking_vs_noop": f"{directions['H1b_intervention_value']}/8",
        },
        "interpretation_boundary": {
            "supported": [
                "Lower pooled fixed-action finite-effect target loss than the same-basis mean-effect plug-in.",
                "Lower pooled fixed-action finite-effect target loss than exact no action.",
                "The Study-3 mean-term-versus-dispersion pattern repeats descriptively.",
            ],
            "not_supported": [
                "Uniform improvement over no action across all eight templates.",
                "Held-out-template generalization; the eight canonical templates are fixed.",
                "Broad IOI, model-scale, behavioral-utility, deployment, or safety claims.",
            ],
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "input_hashes": source_hashes(paths.values(), repo_root),
        "code_hashes": source_hashes(
            [repo_root / "scripts/build_phase07_ioi_confirmation_artifact.py"],
            repo_root,
        ),
        "output_hashes": source_hashes(output_paths, repo_root),
        "contains_private_local_path": _contains_private_path(output_paths),
        "commands_to_reproduce": [
            "python scripts/build_phase07_ioi_confirmation_artifact.py"
        ],
        "chain_summary": {
            "design_id": chain["evaluation"]["design_id"],
            "selected_unique_nonnoop_masks": chain["measurement"]["counts"]["selected_unique_nonnoop_masks"],
            "effect_cells": chain["measurement"]["counts"]["effect_cells"],
        },
    }
    manifest_path = outdir / "ioi_confirmation_artifact_manifest.json"
    write_json(manifest_path, manifest)
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
