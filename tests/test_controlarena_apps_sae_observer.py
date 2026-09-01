"""Focused tests for the published SAE observer composition.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from observerbench.ai_control_public import (
    PublicAIControlScore,
    read_public_ai_control_scores,
    write_public_ai_control_scores,
)
from observerbench.tasks.controlarena_apps.sae_probe import (
    BatchTopKSAE,
    SparseFeatureMatrix,
    encode_batch_topk_sae,
    load_sae_encoding,
    select_dense_logistic_observer,
    select_sparse_logistic_observer,
    write_sae_encoding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sparse_from_dense(values: np.ndarray) -> SparseFeatureMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in matrix:
        active = np.flatnonzero(row)
        indices.extend(active.tolist())
        data.extend(row[active].tolist())
        indptr.append(len(data))
    return SparseFeatureMatrix(
        data=np.asarray(data, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.int64),
        shape=matrix.shape,
    )


def _difference_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    c: float,
    penalty: str,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, object]]:
    del seed, max_iter
    labels = np.asarray(train_y, dtype=bool)
    weight = np.mean(train_x[labels], axis=0) - np.mean(
        train_x[~labels], axis=0
    )
    logits = np.asarray(test_x) @ weight * float(c)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(scores, dtype=float), {
        "solver": "deterministic-test-double",
        "penalty": penalty,
        "converged": True,
    }


def test_sparse_probe_ranks_features_and_uses_frozen_splits() -> None:
    dense = np.asarray(
        [
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.1],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    splits = np.asarray(
        ["fit"] * 4 + ["calibration"] * 2 + ["locked_test"] * 2
    )
    features = _sparse_from_dense(dense)

    order, magnitude = features.rank_by_class_mean_difference(
        labels, splits == "fit"
    )
    assert order[0] == 2
    assert magnitude[2] > magnitude[1]
    np.testing.assert_allclose(
        features.dense_columns(np.ones(8, dtype=bool), [2, 1]),
        dense[:, [2, 1]],
    )

    selection, scores = select_sparse_logistic_observer(
        features,
        labels,
        splits,
        feature_count_grid=[1, 2],
        c_grid=[0.5, 1.0],
        selection_budget=0.5,
        seed=42,
        max_iter=50,
        fit_predict=_difference_classifier,
    )
    assert selection["refit_rank_top_feature_ids"][0] == 2
    assert len(selection["candidate_grid"]) == 4
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_dense_controls_share_selection_and_locked_test_boundary() -> None:
    features = np.asarray(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.1],
            [1.0, 0.1],
            [-1.5, 0.0],
            [1.5, 0.0],
            [-3.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    splits = np.asarray(
        ["fit"] * 4 + ["calibration"] * 2 + ["locked_test"] * 2
    )
    selection, scores = select_dense_logistic_observer(
        features,
        labels,
        splits,
        c_grid=[0.1, 1.0],
        selection_budget=0.5,
        seed=42,
        max_iter=50,
        fit_predict=_difference_classifier,
    )
    assert selection["selection_metric"] == "realized_violation_rate"
    assert len(selection["candidate_grid"]) == 2
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_minimal_batch_topk_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "tiny-ae.pt"
    torch.save(
        {
            "encoder.weight": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
            ),
            "encoder.bias": torch.zeros(4),
            "decoder.weight": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            "b_dec": torch.zeros(3),
            "threshold": torch.tensor(0.25),
            "k": torch.tensor(2),
        },
        checkpoint,
    )
    sae = BatchTopKSAE.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_activation_dim=3,
        expected_dictionary_size=4,
        expected_k=2,
    )
    source = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [-1.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    encoding = encode_batch_topk_sae(
        sae,
        source,
        batch_size=2,
        metadata={"checkpoint": "tiny"},
    )
    assert encoding.features.shape == (3, 4)
    assert encoding.reconstruction.shape == source.shape
    assert encoding.features.l0_per_row.tolist() == [2, 3, 1]

    cache = tmp_path / "encoded.npz"
    write_sae_encoding(cache, encoding)
    loaded = load_sae_encoding(
        cache,
        expected_metadata={"checkpoint": "tiny"},
    )
    np.testing.assert_array_equal(loaded.features.indptr, encoding.features.indptr)
    np.testing.assert_array_equal(loaded.features.indices, encoding.features.indices)
    np.testing.assert_allclose(loaded.features.data, encoding.features.data)
    np.testing.assert_allclose(loaded.reconstruction, encoding.reconstruction)


def test_public_score_writer_round_trips_without_targets(tmp_path: Path) -> None:
    rows = (
        PublicAIControlScore("p1::honest", "sae", 0.125),
        PublicAIControlScore("p1::attack", "sae", 0.875),
    )
    path = tmp_path / "predictions.csv"
    write_public_ai_control_scores(path, rows)
    assert read_public_ai_control_scores(path) == rows
    text = path.read_text(encoding="utf-8")
    assert "backdoor_works" not in text
    assert "eval_mode" not in text


def test_frozen_sae_config_pins_method_and_checkpoint() -> None:
    path = (
        REPO_ROOT
        / "configs/revision/ai_control/controlarena_apps_qwen_sae_probe_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["status"] == "post_outcome_method_integration_frozen_before_sae_results"
    assert config["sae"]["revision"] == "9e5ce2d10931473918813eaf48e885b77b703cae"
    assert config["sae"]["checkpoint_sha256"] == (
        "a651eeda01072b4407e4b955ee28c04b2345419f301e8c518b2bd73f9655d271"
    )
    assert config["sae"]["activation_dim"] == 3584
    assert config["probe"]["feature_count_grid"] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert config["probe"]["method_paper"].endswith("kantamneni25a.html")
