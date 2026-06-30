from __future__ import annotations

from pathlib import Path

import pandas as pd

from observerbench.tasks.ioi.heads import BACKUP_NAME_MOVERS, NAME_MOVERS, NEGATIVE_NAME_MOVERS
from observerbench.tasks.ioi.stage2d import STAGE2D_MODELS, available_design_columns


ROOT = Path(__file__).resolve().parents[1]


def test_ioi_canonical_head_groups() -> None:
    assert NAME_MOVERS == ((9, 9), (9, 6), (10, 0))
    assert BACKUP_NAME_MOVERS == (
        (9, 0),
        (9, 7),
        (10, 1),
        (10, 2),
        (10, 6),
        (10, 10),
        (11, 2),
        (11, 9),
    )
    assert NEGATIVE_NAME_MOVERS == ((10, 7), (11, 10))


def test_stage2d_design_matrix_models_and_pair_columns() -> None:
    cols = available_design_columns()

    assert set(STAGE2D_MODELS) == {
        "additive_head",
        "count_additive",
        "count_plus_PB_count",
        "count_plus_PE_count",
        "count_plus_BE_count",
        "count_plus_all_pairs",
    }
    assert "P_B_count" in cols["count_plus_PB_count"]
    assert "P_E_count" in cols["count_plus_PE_count"]
    assert "B_E_count" in cols["count_plus_BE_count"]
    assert {"P_B_count", "P_E_count", "B_E_count"}.issubset(cols["count_plus_all_pairs"])


def test_stage2d_scientific_fixture_pe_dominates_single_pair() -> None:
    comparison = pd.read_csv(ROOT / "tests/fixtures/ioi_stage2d/ioi_stage2d_model_comparison.csv").set_index("model")

    pb = comparison.loc["count_plus_PB_count", "delta_mae_vs_additive_mean"]
    pe = comparison.loc["count_plus_PE_count", "delta_mae_vs_additive_mean"]
    be = comparison.loc["count_plus_BE_count", "delta_mae_vs_additive_mean"]

    assert pe > pb
    assert pe > be
    assert comparison.loc["count_plus_PE_count", "delta_mae_vs_additive_mean"] > comparison.loc["count_plus_PB_count", "delta_mae_vs_additive_mean"]
    assert pb < 0.10 * pe
    assert not bool(comparison.loc["count_plus_BE_count", "main_success"])
    assert comparison.loc["count_plus_BE_count", "p_delta_vs_count_additive_gt_0"] == 1.0
    assert comparison.loc["count_plus_BE_count", "p_delta_vs_additive_gt_0"] < 0.95


def test_stage2d_direct_group_mask_fixture_pe_exceeds_pb() -> None:
    interactions = pd.read_csv(ROOT / "tests/fixtures/ioi_stage2d/ioi_stage2d_group_interactions.csv").set_index("pair")

    assert interactions.loc["PE", "interaction"] > interactions.loc["PB", "interaction"]
