from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from observerbench.cards import validate_observer_card_bundle, write_observer_card_bundle


def write_ctl2_results(path: Path, rows: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "trained_transformer_ctl2_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def ctl2_row(observer: str, **overrides):
    row = {
        "observer": observer,
        "integrated_squared_error": 1.0,
        "cumulative_collateral_abs": 1.0,
        "divergence_rate": 0.0,
        "divergence_rate_mse_growth": 0.0,
        "target_error_worsened_rate": 0.0,
        "observer_bias_mae_path": 0.1,
        "ise_ratio_vs_lifted": 0.8,
        "cumulative_collateral_ratio_vs_lifted": 0.8,
    }
    row.update(overrides)
    return row


def load_bundle(path: Path) -> dict:
    return json.loads((path / "observer_card.json").read_text(encoding="utf-8"))


def test_metric_derived_recommendations_ignore_observer_name_stamps(tmp_path: Path) -> None:
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    write_ctl2_results(good_dir, [ctl2_row("same_observer")])
    write_ctl2_results(
        bad_dir,
        [
            ctl2_row(
                "same_observer",
                divergence_rate=0.5,
                divergence_rate_mse_growth=0.4,
                target_error_worsened_rate=0.7,
                observer_bias_mae_path=2.0,
                ise_ratio_vs_lifted=2.0,
            )
        ],
    )

    write_observer_card_bundle(good_dir, good_dir / "cards")
    write_observer_card_bundle(bad_dir, bad_dir / "cards")
    good_rec = load_bundle(good_dir / "cards")["cards"][0]["recommendation"]
    bad_rec = load_bundle(bad_dir / "cards")["cards"][0]["recommendation"]

    assert good_rec != bad_rec

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    identical_metrics = ctl2_row("observer_a")
    write_ctl2_results(first_dir, [identical_metrics])
    identical_metrics["observer"] = "observer_b"
    write_ctl2_results(second_dir, [identical_metrics])

    write_observer_card_bundle(first_dir, first_dir / "cards")
    write_observer_card_bundle(second_dir, second_dir / "cards")
    first_rec = load_bundle(first_dir / "cards")["cards"][0]["recommendation"]
    second_rec = load_bundle(second_dir / "cards")["cards"][0]["recommendation"]

    assert first_rec == second_rec


def test_threshold_changes_update_failure_modes(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    divergent_dir = tmp_path / "divergent"
    write_ctl2_results(clean_dir, [ctl2_row("ctl2_observer")])
    write_ctl2_results(
        divergent_dir,
        [
            ctl2_row(
                "ctl2_observer",
                divergence_rate=0.2,
                observer_bias_mae_path=0.9,
            )
        ],
    )

    write_observer_card_bundle(clean_dir, clean_dir / "cards")
    write_observer_card_bundle(divergent_dir, divergent_dir / "cards")
    clean_failures = load_bundle(clean_dir / "cards")["cards"][0]["failure_modes_detected"]
    divergent_failures = load_bundle(divergent_dir / "cards")["cards"][0]["failure_modes_detected"]

    assert clean_failures != divergent_failures
    assert any("divergence_rate" in failure for failure in divergent_failures)
    assert any("observer_bias_mae_path" in failure for failure in divergent_failures)


def test_observer_card_json_validates_against_schema(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    cards_dir = tmp_path / "cards"
    write_ctl2_results(results_dir, [ctl2_row("schema_observer")])

    write_observer_card_bundle(results_dir, cards_dir)
    bundle = load_bundle(cards_dir)

    validate_observer_card_bundle(bundle)
    assert bundle["cards"][0]["thresholds"]["divergence_rate_max"] == 0.05


def test_observer_card_markdown_is_human_readable(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    cards_dir = tmp_path / "cards"
    write_ctl2_results(results_dir, [ctl2_row("markdown_observer")])

    write_observer_card_bundle(results_dir, cards_dir)
    text = (cards_dir / "observer_card.md").read_text(encoding="utf-8")

    assert "trained_ctl2" in text
    assert "markdown_observer" in text
    assert "Recommendation" in text
    assert "Scope limits" in text
    assert "observerbench make-card" in text
