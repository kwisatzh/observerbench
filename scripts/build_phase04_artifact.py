#!/usr/bin/env python3
"""Build the checked Phase-4 integration audit and ObserverCards.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from observerbench.core import write_json
from observerbench.paper_cards import build_phase4_cards, write_phase4_observer_cards
from observerbench.provenance import file_sha256, portable_artifact_path, source_hashes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "results/revision/phase04"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase04_audit(root: Path, cards: list[dict[str, Any]]) -> dict[str, Any]:
    ctl2 = _json(root / "results/revision/phase04/ctl2_phase04_audit.json")
    ioi = _json(root / "results/revision/phase02/ioi_phase02_claim_audit.json")
    ctl2_rows = ctl2["multiseed"]["fixed_direction_ise_effects"]
    nonzero_unprojected = [
        row
        for row in ctl2_rows
        if row["gamma"] > 0 and row["support_mode"] == "unprojected"
    ]
    gamma_zero = [row for row in ctl2_rows if row["gamma"] == 0]

    rank_checks: dict[str, bool] = {}
    model_checks: dict[str, bool] = {}
    for stage in ("ioi_stage2b", "ioi_stage2c"):
        base = root / f"results/revision/phase02/{stage}_capacity"
        capacity = pd.read_csv(base / "capacity_audit.csv").set_index("model")
        comparison = pd.read_csv(base / "model_comparison.csv").set_index("model")
        single = capacity.loc[
            [
                "count_plus_PB_bin4",
                "count_plus_PE_bin4",
                "count_plus_BE_bin4",
            ],
            "rank_added_vs_count_additive",
        ]
        rank_checks[f"{stage}_single_pairs_add_four_ranks"] = bool((single == 4).all())
        rank_checks[f"{stage}_all_pairs_add_twelve_ranks"] = bool(
            capacity.loc["count_plus_all_bin4", "rank_added_vs_count_additive"] == 12
        )
        model_checks[f"{stage}_all_pairs_beats_both_baselines"] = bool(
            comparison.loc["count_plus_all_bin4", "mae"]
            < min(
                comparison.loc["additive_head", "mae"],
                comparison.loc["count_additive", "mae"],
            )
        )

    ctl2_card = next(card for card in cards if card["task_name"] == "trained_ctl2")
    stage2b_card = next(card for card in cards if card["task_name"] == "ioi_stage2b")
    stage2c_card = next(card for card in cards if card["task_name"] == "ioi_stage2c")
    checks = {
        "ctl2_phase01_audit_passes": bool(ctl2["all_required_checks_pass"]),
        "ctl2_nonzero_fixed_direction_effects_are_6_of_6": bool(
            len(nonzero_unprojected) == 8
            and all(row["n_positive_fo_minus_lifted"] == 6 for row in nonzero_unprojected)
        ),
        "ctl2_gamma_zero_factorial_null_within_1e_5": bool(
            max(abs(row["mean"]) for row in gamma_zero) < 1e-5
        ),
        "ctl2_gain_match_is_unclipped_but_target_worsens": bool(
            ctl2["gain_match_small_setpoint"]["control_clip_fraction"] == 0
            and ctl2["gain_match_small_setpoint"]["final_target_mse"]
            > ctl2["gain_match_small_setpoint"]["initial_target_mse"]
        ),
        "ioi_phase02_audit_passes": bool(ioi["all_checks_pass"]),
        **rank_checks,
        **model_checks,
        "checked_cards_cover_ctl2_and_both_ioi_designs": bool(
            {card["task_name"] for card in cards}
            >= {"trained_ctl2", "ioi_stage2b", "ioi_stage2c"}
        ),
        "checked_cards_have_no_unknown_task": bool(
            all(card["task_name"] != "unknown_task" for card in cards)
        ),
        "ctl2_card_pins_48_of_48_seed_level_contrasts": bool(
            ctl2_card["primary_metrics"]["positive_seed_level_fixed_direction_contrasts"] == 48
            and ctl2_card["primary_metrics"]["total_seed_level_fixed_direction_contrasts"] == 48
        ),
        "stage2b_card_retracts_additive_sufficiency": bool(
            stage2b_card["primary_metrics"]["all_pairs_delta_mae_vs_additive_q05"] > 0
            and "strong baseline" in stage2b_card["recommendation"]
        ),
        "stage2c_card_records_conditional_pb": bool(
            stage2c_card["primary_metrics"]["PB_add_one_delta_mae_q05"] < 0
            and stage2c_card["primary_metrics"]["PB_leave_one_out_delta_mae_mean"] > 0
        ),
    }
    return {
        "schema": "observerbench.phase04.integration_audit.v1",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "claim": (
            "Checked Ctl-2 factorial, null, gain, rank, and IOI model-comparison "
            "results agree with the Phase-4 cards and current manuscript claims."
        ),
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# ObserverBench Phase-4 integration audit",
        "",
        f"All checks pass: **{audit['all_checks_pass']}**.",
        "",
        audit["claim"],
        "",
        "## Checked gates",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in audit["checks"].items()
    )
    lines.extend([
        "",
        "No model is trained and no GPT-2 data are downloaded by this integration step.",
        "",
    ])
    return "\n".join(lines)


def _contains_private_path(paths: list[Path]) -> bool:
    banned = ("/Users/", "/Downloads/", "observerbench-review-phase01")
    for path in paths:
        if path.suffix.lower() not in {".json", ".md", ".csv", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(value in text for value in banned):
            return True
    return False


def build(root: Path, outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    cards = build_phase4_cards(root)
    card_json, card_md = write_phase4_observer_cards(root, outdir / "observer_cards")
    audit = _phase04_audit(root, cards)
    audit_path = outdir / "phase04_claim_audit.json"
    audit_md_path = outdir / "PHASE04_INTEGRATION_AUDIT.md"
    write_json(audit_path, audit)
    audit_md_path.write_text(_audit_markdown(audit), encoding="utf-8")

    input_paths = [
        root / "results/revision/phase04/ctl2_phase04_audit.json",
        root / "results/revision/phase04/ctl2_multiseed_sweep_current/phase01_sweep_results.csv",
        root / "results/revision/phase02/ioi_phase02_claim_audit.json",
    ]
    for stage in ("ioi_stage2b", "ioi_stage2c"):
        base = root / f"results/revision/phase02/{stage}_capacity"
        input_paths.extend([
            base / "model_comparison.csv",
            base / "capacity_audit.csv",
            base / "add_one_vs_additive_head.csv",
            base / "add_one_vs_count_additive.csv",
            base / "leave_one_out_contrasts.csv",
        ])
    output_paths = [card_json, card_md, audit_path, audit_md_path]
    manifest = {
        "schema": "observerbench.phase04.artifact.v1",
        "result_status": "checked_revision",
        "all_checks_pass": audit["all_checks_pass"],
        "input_hashes": source_hashes(input_paths, root),
        "code_hashes": source_hashes(
            [
                root / "src/observerbench/cards.py",
                root / "src/observerbench/paper_cards.py",
                root / "scripts/build_phase04_artifact.py",
                root / "scripts/reproduce_paper_fast.py",
            ],
            root,
        ),
        "output_hashes": source_hashes(output_paths, root),
        "contains_private_local_path": _contains_private_path(output_paths),
        "commands_to_reproduce": [
            "python scripts/build_phase04_artifact.py",
            "python scripts/reproduce_paper_fast.py",
        ],
    }
    manifest_path = outdir / "phase04_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repo_root.resolve()
    outdir = args.outdir.resolve()
    manifest = build(root, outdir)
    print(portable_artifact_path(outdir, root))
    return 0 if manifest["all_checks_pass"] and not manifest["contains_private_local_path"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
