#!/usr/bin/env python3
"""Build the checked Phase-6 IOI confirmation paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from observerbench.core import write_json
from observerbench.provenance import file_sha256, portable_artifact_path, source_hashes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT / "results/revision/phase06/ioi_fresh_confirmation/evaluation"
)
DEFAULT_OUTDIR = REPO_ROOT / "paper/generated_phase06/ioi_confirmation"
DEFAULT_PROTOCOL = REPO_ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json"
EXPECTED_PROTOCOL_HASH = "b6d44ac16c88a48c738416a18abb205771598ae713f7d44bf0dfdc73f3dff099"

INPUT_NAMES = (
    "hypothesis_audit.json",
    "prespecified_contrasts.csv",
    "natural_mean_estimand_metrics.csv",
    "clean_task_validity.csv",
    "candidate_actual_risk.csv",
    "decision_quality.csv",
)
EXPECTED_INPUT_HASHES = {
    "hypothesis_audit.json": (
        "0c3844c1fde73e21d4da5a49385494e00744836c3c43b8a46538776f47de382e"
    ),
    "prespecified_contrasts.csv": (
        "d5d932ada991cc2abdfb1107fff4daf8fe4ec49512e4c52e876d1f95f57917af"
    ),
    "natural_mean_estimand_metrics.csv": (
        "6e731217b89b8497c2a0f2ff707a692f42791b81fb19af0484e126c4b3a1379f"
    ),
    "clean_task_validity.csv": (
        "3aed5fff1dcc71709e45a28628430e831794b5ae90b616c92e7b9127a36825a8"
    ),
    "candidate_actual_risk.csv": (
        "06d5a1fc6a831d175f4c3b119bcc50e8e53b258bfda73d4010f9bb47778a8150"
    ),
    "decision_quality.csv": (
        "b7c77f4f757b5dba754dea00b14e749e6fe3f817ac132261497118354c99da7f"
    ),
}

CONTRAST_EXPECTATIONS: tuple[Mapping[str, str], ...] = (
    {
        "comparison_id": "H1_primary_estimand",
        "result": "pass",
        "reference_label": "Natural mean, quadratic basis",
        "reference_selector_family": "natural_mean_effect",
        "reference_model": "head_pair_quadratic_screen",
        "reference_mean": "1.0757814324557937",
        "candidate_mean": "0.8325709196312042",
        "mean": "0.2432105128245894",
        "relative_reduction_fraction": "0.22607799826902428",
        "q025": "0.11385469850793016",
        "q975": "0.40237179996935696",
    },
    {
        "comparison_id": "H2_secondary_vs_additive",
        "result": "fail",
        "reference_label": "Additive direct risk",
        "reference_selector_family": "direct_risk",
        "reference_model": "additive_head",
        "reference_mean": "0.858623386816665",
        "candidate_mean": "0.8325709196312042",
        "mean": "0.026052467185460653",
        "relative_reduction_fraction": "0.030342135545655048",
        "q025": "-0.025486295092559876",
        "q975": "0.07640456896770047",
    },
    {
        "comparison_id": "H2_secondary_vs_count",
        "result": "fail",
        "reference_label": "Count-additive direct risk",
        "reference_selector_family": "direct_risk",
        "reference_model": "count_additive",
        "reference_mean": "0.863129189965548",
        "candidate_mean": "0.8325709196312042",
        "mean": "0.030558270334343735",
        "relative_reduction_fraction": "0.03540405154825488",
        "q025": "-0.021157178251451116",
        "q975": "0.078829896710522",
    },
    {
        "comparison_id": "Jensen_parameter_count_sensitivity",
        "result": "pass",
        "reference_label": "Target-specific transformed-mean score",
        "reference_selector_family": "target_specific_jensen_score",
        "reference_model": "head_pair_quadratic_screen",
        "reference_mean": "1.0766507619409822",
        "candidate_mean": "0.8325709196312042",
        "mean": "0.24407984230977794",
        "relative_reduction_fraction": "0.22670289283941217",
        "q025": "0.11912255681357542",
        "q975": "0.39854511026981826",
    },
)

H1_TARGET_EXPECTATIONS = {
    "target_0.5": {
        "mean": "0.3001229533110745",
        "relative_reduction_fraction": "0.29801405769819317",
    },
    "target_1": {
        "mean": "0.18629807233810425",
        "relative_reduction_fraction": "0.16277876047850276",
    },
}

NATURAL_MEAN_EXPECTATIONS = {
    "count_plus_all_bin4": {
        "measurement_budget": "160",
        "candidate_count": "1536",
        "heldout_mean_effect_mae": "0.17686858671830388",
        "heldout_mean_effect_rmse": "0.22620523608100127",
        "heldout_mean_effect_r2": "0.6913228501700867",
        "heldout_mean_effect_rank_correlation": "0.8443785875388623",
        "descriptive_non_gating": "True",
    },
    "head_pair_quadratic_screen": {
        "measurement_budget": "160",
        "candidate_count": "1536",
        "heldout_mean_effect_mae": "0.12849992265111734",
        "heldout_mean_effect_rmse": "0.1607022884733953",
        "heldout_mean_effect_r2": "0.8442086655989596",
        "heldout_mean_effect_rank_correlation": "0.9378843706488591",
        "descriptive_non_gating": "True",
    },
}

CLEAN_EXPECTATIONS = {
    "overall": ("512", "512", "192", "1.8722557500004768", "0.8671875"),
    "p6_abba_kitchen": ("64", "64", "24", "3.799344450235367", "0.984375"),
    "p6_abba_library": ("64", "64", "24", "1.9367087185382843", "0.890625"),
    "p6_abba_museum": ("64", "64", "24", "1.000433325767517", "0.796875"),
    "p6_abba_train": ("64", "64", "24", "1.2292743027210236", "0.78125"),
    "p6_baba_kitchen": ("64", "64", "24", "2.5092716366052628", "0.984375"),
    "p6_baba_library": ("64", "64", "24", "2.4432227462530136", "0.984375"),
    "p6_baba_museum": ("64", "64", "24", "0.32617056369781494", "0.578125"),
    "p6_baba_train": ("64", "64", "24", "1.7336202561855316", "0.9375"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-6 input: {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-6 input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected={expected!r}, recorded={actual!r}")


def _decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not a finite decimal: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} is not a finite decimal: {value!r}")
    return parsed


def _expect_decimal(actual: Any, expected: str, label: str) -> None:
    _expect_equal(_decimal(actual, label), Decimal(expected), label)


def _one_row(
    rows: Iterable[Mapping[str, str]],
    *,
    label: str,
    **criteria: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(column) == expected for column, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {label} row for {criteria}, found {len(matches)}"
        )
    return matches[0]


def _validate_frozen_hashes(input_paths: Mapping[str, Path]) -> None:
    for name, expected in EXPECTED_INPUT_HASHES.items():
        path = input_paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase-6 input: {path}")
        _expect_equal(file_sha256(path), expected, f"SHA-256 for {name}")


def _primary_contrast_row(
    rows: Iterable[Mapping[str, str]], comparison_id: str, target_scope: str
) -> Mapping[str, str]:
    return _one_row(
        rows,
        label="prespecified target-loss contrast",
        comparison_id=comparison_id,
        policy="target_loss",
        metric="absolute_target_loss_reduction",
        target_scope=target_scope,
    )


def _validate_contrasts(
    audit: Mapping[str, Any], rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    _expect_equal(audit.get("result_calibration"), "mixed", "result calibration")
    _expect_equal(audit.get("ioi_language_allowed"), False, "IOI language gate")
    _expect_equal(
        audit.get("clean_failure_never_changes_H1_or_H2"),
        True,
        "clean failure independence",
    )

    headline: list[dict[str, Any]] = []
    for expected in CONTRAST_EXPECTATIONS:
        comparison_id = expected["comparison_id"]
        row = _primary_contrast_row(rows, comparison_id, "primary_pooled")
        for field, categorical in (
            ("candidate_selector_family", "direct_risk"),
            ("candidate_model", "head_pair_quadratic_screen"),
            ("reference_selector_family", expected["reference_selector_family"]),
            ("reference_model", expected["reference_model"]),
            ("targets", "0.5,1"),
            ("row_count", "49152"),
            ("bootstrap_repeats", "5000"),
        ):
            _expect_equal(row.get(field), categorical, f"{comparison_id} {field}")
        for field in (
            "reference_mean",
            "candidate_mean",
            "mean",
            "relative_reduction_fraction",
            "q025",
            "q975",
        ):
            _expect_decimal(row.get(field), expected[field], f"{comparison_id} {field}")

        if comparison_id == "H1_primary_estimand":
            audit_entry = audit[comparison_id]
            expected_checks = {
                "paired_cluster_interval_lower_strictly_positive": True,
                "relative_loss_reduction_at_least_frozen_threshold": True,
                "target_0.5_nonnegative": True,
                "target_1_nonnegative": True,
            }
        elif comparison_id.startswith("H2_"):
            audit_entry = audit["H2_secondary_structure"]["comparisons"][comparison_id]
            expected_checks = {
                "paired_cluster_interval_lower_strictly_positive": False,
                "relative_loss_reduction_at_least_frozen_threshold": False,
                "target_0.5_nonnegative": True,
                "target_1_nonnegative": True,
            }
        else:
            audit_entry = audit[comparison_id]
            expected_checks = {
                "paired_cluster_interval_lower_strictly_positive": True,
                "pooled_point_direction_nonnegative": True,
            }
            _expect_equal(
                audit_entry["per_primary_target_directions_reported_not_gated"],
                {"target_0.5_nonnegative": True, "target_1_nonnegative": True},
                "Jensen sensitivity target directions",
            )
        _expect_equal(
            audit_entry["passed"], expected["result"] == "pass", f"{comparison_id} pass"
        )
        _expect_equal(audit_entry["checks"], expected_checks, f"{comparison_id} checks")
        for field in ("q025", "q975"):
            _expect_decimal(
                audit_entry[field], expected[field], f"{comparison_id} audit {field}"
            )
        if "relative_reduction_fraction" in audit_entry:
            _expect_decimal(
                audit_entry["relative_reduction_fraction"],
                expected["relative_reduction_fraction"],
                f"{comparison_id} audit relative reduction",
            )

        headline.append(
            {
                "comparison_id": comparison_id,
                "claim_role": row["claim_role"],
                "status": expected["result"],
                "candidate": "Direct risk, quadratic basis",
                "reference": expected["reference_label"],
                "reference_mean_loss": float(Decimal(expected["reference_mean"])),
                "candidate_mean_loss": float(Decimal(expected["candidate_mean"])),
                "absolute_loss_reduction": float(Decimal(expected["mean"])),
                "relative_loss_reduction_fraction": float(
                    Decimal(expected["relative_reduction_fraction"])
                ),
                "ci95_low": float(Decimal(expected["q025"])),
                "ci95_high": float(Decimal(expected["q975"])),
                "bootstrap_repeats": 5000,
            }
        )

    _expect_equal(audit["H1_primary_estimand"]["passed"], True, "H1 status")
    _expect_equal(audit["H2_secondary_structure"]["passed"], False, "H2 status")
    _expect_equal(
        audit["Jensen_parameter_count_sensitivity"]["passed"],
        True,
        "Jensen sensitivity status",
    )
    for scope, values in H1_TARGET_EXPECTATIONS.items():
        row = _primary_contrast_row(rows, "H1_primary_estimand", scope)
        for field, expected in values.items():
            _expect_decimal(row.get(field), expected, f"H1 {scope} {field}")
    return headline


def _validate_natural_mean(rows: list[dict[str, str]]) -> dict[str, Any]:
    _expect_equal(len(rows), 2, "natural-mean diagnostic row count")
    for model, expected in NATURAL_MEAN_EXPECTATIONS.items():
        row = _one_row(
            rows,
            label="natural-mean diagnostic",
            selector_family="natural_mean_effect",
            model=model,
        )
        for field, value in expected.items():
            if field == "descriptive_non_gating":
                _expect_equal(row.get(field), value, f"{model} {field}")
            else:
                _expect_decimal(row.get(field), value, f"{model} {field}")
    quadratic = NATURAL_MEAN_EXPECTATIONS["head_pair_quadratic_screen"]
    return {
        "status": "descriptive_non_gating",
        "model": "Natural mean, quadratic basis",
        "heldout_mean_effect_mae": float(
            Decimal(quadratic["heldout_mean_effect_mae"])
        ),
        "heldout_mean_effect_rmse": float(
            Decimal(quadratic["heldout_mean_effect_rmse"])
        ),
        "heldout_mean_effect_r2": float(
            Decimal(quadratic["heldout_mean_effect_r2"])
        ),
        "heldout_mean_effect_rank_correlation": float(
            Decimal(quadratic["heldout_mean_effect_rank_correlation"])
        ),
    }


def _validate_clean_gate(
    audit: Mapping[str, Any], rows: list[dict[str, str]]
) -> dict[str, Any]:
    _expect_equal(len(rows), len(CLEAN_EXPECTATIONS), "clean validity row count")
    fields = (
        "test_prompt_count",
        "reference_prompt_count",
        "train_prompt_count",
        "mean_clean_logit_difference",
        "io_vs_subject_pairwise_accuracy",
    )
    for scope, expected_values in CLEAN_EXPECTATIONS.items():
        row = _one_row(rows, label="clean validity", scope=scope)
        for field, expected in zip(fields, expected_values):
            _expect_decimal(row.get(field), expected, f"clean {scope} {field}")

    clean_audit = audit["clean_task_validity"]
    _expect_equal(clean_audit["passed"], False, "clean-task gate status")
    _expect_equal(
        clean_audit["checks"],
        {
            "every_template_accuracy": False,
            "every_template_positive_mean_clean_ld": True,
            "no_rows_filtered": True,
            "overall_accuracy": False,
        },
        "clean-task gate checks",
    )
    _expect_equal(
        clean_audit["ioi_mechanism_or_fresh_template_generalization_language_allowed"],
        False,
        "clean-task interpretation boundary",
    )
    accuracies = {
        scope: Decimal(values[-1])
        for scope, values in CLEAN_EXPECTATIONS.items()
        if scope != "overall"
    }
    worst_scope = min(accuracies, key=accuracies.__getitem__)
    _expect_equal(worst_scope, "p6_baba_museum", "worst clean template")
    return {
        "status": "fail",
        "overall_accuracy": float(Decimal(CLEAN_EXPECTATIONS["overall"][-1])),
        "overall_accuracy_threshold": 0.90,
        "worst_template": worst_scope,
        "worst_template_accuracy": float(accuracies[worst_scope]),
        "every_template_accuracy_threshold": 0.75,
        "every_template_mean_clean_logit_difference_positive": True,
        "rows_filtered": False,
    }


def _validate_selected_action_decomposition(
    actual_risk_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    actual_means = {
        (row["mask_id"], row["target"]): _decimal(
            row["actual_mean_effect"], "candidate actual mean effect"
        )
        for row in actual_risk_rows
    }
    expected = {
        "natural_mean_effect": (
            "0.4332697935266575451041666667",
            "0.6425116389291360996875000003",
            "1.075781432455793644791666667",
        ),
        "direct_risk": (
            "0.5499065573094412672604166667",
            "0.2826643623217629896145833333",
            "0.832570919631204256875",
        ),
    }
    result: dict[str, dict[str, float]] = {}
    for family, expected_values in expected.items():
        rows = [
            row
            for row in decision_rows
            if row["selector_family"] == family
            and row["model"] == "head_pair_quadratic_screen"
            and row["measurement_budget"] == "160"
            and row["policy"] == "target_loss"
            and row["target"] in {"0.5", "1.0"}
        ]
        _expect_equal(len(rows), 96, f"{family} primary selected-action rows")
        mean_terms = [
            abs(actual_means[(row["selected_mask_id"], row["target"])] - Decimal(row["target"]))
            for row in rows
        ]
        total_losses = [Decimal(row["selected_mean_target_loss"]) for row in rows]
        mean_term = sum(mean_terms, Decimal(0)) / Decimal(len(rows))
        total = sum(total_losses, Decimal(0)) / Decimal(len(rows))
        dispersion = total - mean_term
        for actual, expected_value, label in zip(
            (mean_term, dispersion, total),
            expected_values,
            ("mean-to-target term", "dispersion term", "total loss"),
        ):
            _expect_decimal(actual, expected_value, f"{family} {label}")
        result[family] = {
            "mean_to_target": float(mean_term),
            "dispersion": float(dispersion),
            "total_loss": float(total),
        }
    return result


def _summary_rows(
    contrasts: list[dict[str, Any]],
    natural_mean: Mapping[str, Any],
    clean_gate: Mapping[str, Any],
    no_op: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in contrasts]
    rows.append(
        {
            "comparison_id": "postoutcome_no_op_audit",
            "claim_role": "postoutcome_limitation",
            "status": "limitation",
            "candidate": "No action",
            "reference": "Direct risk, quadratic basis",
            "reference_mean_loss": no_op["direct_risk_mean_loss"],
            "candidate_mean_loss": no_op["mean_loss"],
            "absolute_loss_reduction": no_op["direct_risk_minus_no_op"],
            "relative_loss_reduction_fraction": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_repeats": 0,
            "heldout_mean_effect_mae": None,
            "heldout_mean_effect_r2": None,
            "heldout_mean_effect_rank_correlation": None,
            "overall_clean_accuracy": None,
            "worst_template_accuracy": None,
        }
    )
    rows.append(
        {
            "comparison_id": "natural_mean_estimand_diagnostic",
            "claim_role": "descriptive_non_gating",
            "status": "descriptive",
            "candidate": natural_mean["model"],
            "reference": "Held-out candidate mean effect",
            "reference_mean_loss": None,
            "candidate_mean_loss": None,
            "absolute_loss_reduction": None,
            "relative_loss_reduction_fraction": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_repeats": 0,
            "heldout_mean_effect_mae": natural_mean["heldout_mean_effect_mae"],
            "heldout_mean_effect_r2": natural_mean["heldout_mean_effect_r2"],
            "heldout_mean_effect_rank_correlation": natural_mean[
                "heldout_mean_effect_rank_correlation"
            ],
            "overall_clean_accuracy": None,
            "worst_template_accuracy": None,
        }
    )
    rows.append(
        {
            "comparison_id": "clean_task_validity",
            "claim_role": "interpretation_gate",
            "status": clean_gate["status"],
            "candidate": "Frozen Phase-6 test prompts",
            "reference": "Prespecified clean-task thresholds",
            "reference_mean_loss": None,
            "candidate_mean_loss": None,
            "absolute_loss_reduction": None,
            "relative_loss_reduction_fraction": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_repeats": 0,
            "heldout_mean_effect_mae": None,
            "heldout_mean_effect_r2": None,
            "heldout_mean_effect_rank_correlation": None,
            "overall_clean_accuracy": clean_gate["overall_accuracy"],
            "worst_template_accuracy": clean_gate["worst_template_accuracy"],
        }
    )
    for row in rows[: len(contrasts)]:
        row.update(
            {
                "heldout_mean_effect_mae": None,
                "heldout_mean_effect_r2": None,
                "heldout_mean_effect_rank_correlation": None,
                "overall_clean_accuracy": None,
                "worst_template_accuracy": None,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(
    contrasts: list[dict[str, Any]],
    natural_mean: Mapping[str, Any],
    clean_gate: Mapping[str, Any],
    no_op: Mapping[str, Any],
    path: Path,
) -> None:
    labels = {
        "H1_primary_estimand": "H1: risk vs. natural mean",
        "H2_secondary_vs_additive": "H2: risk basis vs. additive",
        "H2_secondary_vs_count": "H2: risk basis vs. count-additive",
        "Jensen_parameter_count_sensitivity": "Target-specific transformed-mean sensitivity",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrl@{}}",
        r"\toprule",
        r"Check & Status & Point statistic & 95\% interval \\",
        r"\midrule",
    ]
    for row in contrasts:
        status = "Pass" if row["status"] == "pass" else "Fail"
        point = f"{100.0 * row['relative_loss_reduction_fraction']:.2f}\\% reduction"
        interval = f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        lines.append(
            f"{labels[row['comparison_id']]} & {status} & {point} & {interval} \\\\"
        )
    lines.extend(
        [
            (
                "Natural-mean own-estimand diagnostic & Descriptive & "
                f"MAE {natural_mean['heldout_mean_effect_mae']:.3f}; "
                f"$R^2$ {natural_mean['heldout_mean_effect_r2']:.3f} & -- \\\\"
            ),
            (
                "No-action audit & Limitation & "
                f"loss {no_op['mean_loss']:.3f} vs. risk {no_op['direct_risk_mean_loss']:.3f} "
                "& not prespecified \\\\"
            ),
            (
                "Clean-task validity & Fail & "
                f"overall {clean_gate['overall_accuracy']:.3f} $<$ 0.900 & "
                f"worst template {clean_gate['worst_template_accuracy']:.3f} "
                "$<$ 0.750 \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Pilot-informed, outcome-sealed Study-3 result. H1 passes; H2 fails; the "
                r"target-specific transformed-mean sensitivity passes; and the clean-task "
                r"gate fails. Contrast point statistics are relative fixed-action loss "
                r"reductions, while intervals are paired absolute loss reductions from "
                r"the frozen pair-cluster-by-pool bootstrap. The nonempty candidate family "
                r"excluded no action; its exact pooled loss is lower than the selected "
                r"direct-risk loss. The clean-gate failure bars IOI-mechanism and "
                r"fresh-template-generalization language.}"
            ),
            r"\label{tab:phase6-ioi-confirmation}",
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


def build(*, repo_root: Path, results_dir: Path, outdir: Path) -> dict[str, Any]:
    input_paths = {name: results_dir / name for name in INPUT_NAMES}
    _validate_frozen_hashes(input_paths)
    protocol_path = repo_root / "configs/revision/ioi_phase06_fresh_confirmation_v1.json"
    _expect_equal(file_sha256(protocol_path), EXPECTED_PROTOCOL_HASH, "Phase-6 protocol SHA-256")
    protocol = _read_json(protocol_path)
    audit = _read_json(input_paths["hypothesis_audit.json"])
    contrast_rows = _read_csv(input_paths["prespecified_contrasts.csv"])
    natural_rows = _read_csv(input_paths["natural_mean_estimand_metrics.csv"])
    clean_rows = _read_csv(input_paths["clean_task_validity.csv"])
    actual_risk_rows = _read_csv(input_paths["candidate_actual_risk.csv"])
    decision_rows = _read_csv(input_paths["decision_quality.csv"])

    contrasts = _validate_contrasts(audit, contrast_rows)
    natural_mean = _validate_natural_mean(natural_rows)
    clean_gate = _validate_clean_gate(audit, clean_rows)
    decomposition = _validate_selected_action_decomposition(
        actual_risk_rows, decision_rows
    )
    _expect_equal(protocol.get("primary_targets"), [Decimal("0.5"), Decimal("1.0")], "primary targets")
    if any(int(value) == 0 for value in protocol.get("candidate_head_counts", [])):
        raise ValueError("the frozen Phase-6 candidate family unexpectedly contains no action")
    no_op_loss = sum(protocol["primary_targets"], Decimal(0)) / Decimal(
        len(protocol["primary_targets"])
    )
    direct_risk_loss = Decimal(str(contrasts[0]["candidate_mean_loss"]))
    no_op = {
        "status": "postoutcome_limitation_not_a_frozen_comparison",
        "included_in_frozen_candidate_family": False,
        "primary_targets": [float(value) for value in protocol["primary_targets"]],
        "mean_loss": float(no_op_loss),
        "direct_risk_mean_loss": float(direct_risk_loss),
        "direct_risk_minus_no_op": float(direct_risk_loss - no_op_loss),
    }
    checks = {
        "frozen_input_hashes_match": True,
        "H1_exact_values_match_audit_and_contrasts": True,
        "H2_exact_values_match_audit_and_contrasts": True,
        "Jensen_sensitivity_exact_values_match_audit_and_contrasts": True,
        "natural_mean_estimand_diagnostic_matches": True,
        "clean_task_gate_exact_values_match_audit_and_table": True,
        "no_op_loss_is_exact_and_candidate_family_excluded_it": True,
        "selected_action_decomposition_recomputes_exactly": True,
        "bounded_claim_language_only": True,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ioi_confirmation_results.csv"
    tex_path = outdir / "ioi_confirmation_table.tex"
    summary = _summary_rows(contrasts, natural_mean, clean_gate, no_op)
    _write_csv(summary, csv_path)
    _write_latex(contrasts, natural_mean, clean_gate, no_op, tex_path)

    output_paths = [csv_path, tex_path]
    manifest = {
        "schema": "observerbench.phase06.ioi_confirmation_artifact.v1",
        "scientific_status": (
            "pilot_informed_prospective_fresh_template_name_mask_confirmation"
        ),
        "result_calibration": "mixed",
        "claim_status": {
            "H1_primary_estimand": "pass",
            "H2_secondary_structure": "fail",
            "Jensen_parameter_count_sensitivity": "pass",
            "clean_task_validity": "fail",
        },
        "interpretation_boundary": {
            "supported": [
                (
                    "H1 passes for fixed-action selection among the frozen nonempty "
                    "candidate masks."
                ),
                (
                    "The H1 result passes the target-specific transformed-mean "
                    "parameter-count sensitivity."
                ),
            ],
            "not_supported": [
                "H2's interaction-aware risk-basis claim.",
                "IOI-mechanism or fresh-template-generalization claims.",
                "An absolute control-efficacy claim against a no-op action.",
            ],
        },
        "headline_contrasts": contrasts,
        "natural_mean_estimand_diagnostic": natural_mean,
        "clean_task_gate": clean_gate,
        "no_op_audit": no_op,
        "selected_action_decomposition": decomposition,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "input_hashes": source_hashes([*input_paths.values(), protocol_path], repo_root),
        "code_hashes": source_hashes(
            [repo_root / "scripts/build_phase06_ioi_confirmation_artifact.py"],
            repo_root,
        ),
        "output_hashes": source_hashes(output_paths, repo_root),
        "contains_private_local_path": _contains_private_path(output_paths),
        "commands_to_reproduce": [
            "python scripts/build_phase06_ioi_confirmation_artifact.py",
        ],
    }
    manifest_path = outdir / "ioi_confirmation_artifact_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    manifest = build(
        repo_root=repo_root,
        results_dir=args.results_dir.resolve(),
        outdir=args.outdir.resolve(),
    )
    print(portable_artifact_path(args.outdir.resolve(), repo_root))
    return 0 if manifest["all_checks_pass"] and not manifest["contains_private_local_path"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
