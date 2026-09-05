# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
spec = importlib.util.spec_from_file_location(
    "monitor_transfer", Path(__file__).resolve().parents[1] / "scripts/run_qwen_monitor_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def test_primal_ridge_matches_existing_kernel_fit_and_ignores_test_labels():
    from observerbench.tasks.qwen_safety.followup import _CenteredLinearKernel
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 10)).astype(np.float32)
    y = np.tile([0., 1.], 12)
    fit = np.arange(24) < 12
    calibration = (np.arange(24) >= 12) & (np.arange(24) < 20)
    model, alpha = transfer.fit_ridge_with_calibration(
        x, y, fit, calibration, ridge_grid=[.01, 1., 100.], budget=.25)
    reference = _CenteredLinearKernel(x[:20]).predict(y[:20], x[20:], ridge=alpha)
    np.testing.assert_allclose(model.predict(x[20:]), reference, atol=1e-7)
    y[20:] = 1 - y[20:]
    other, other_alpha = transfer.fit_ridge_with_calibration(
        x, y, fit, calibration, ridge_grid=[.01, 1., 100.], budget=.25)
    assert alpha == other_alpha
    np.testing.assert_array_equal(model.coef_, other.coef_)


def test_adaptation_splits_keep_problems_together_and_budgets_nested():
    rows = [{"problem_id": i, "split": "fit" if i < 20 else "locked_test"}
            for i in range(24) for _ in range(2)]
    valid = np.ones(len(rows), dtype=bool)
    previous = set()
    for pairs in [4, 8, 16]:
        f, c, ids = transfer.adaptation_split(rows, pairs, 9120, valid)
        assert not (f & c).any()
        assert not (f[40:].any() or c[40:].any())
        assert previous <= set(ids)
        assert np.array_equal(f[::2], f[1::2])
        assert np.array_equal(c[::2], c[1::2])
        previous = set(ids)


def test_positive_affine_calibration_cannot_repair_top_budget_order():
    from observerbench.ai_control_followup import budget_metrics
    labels = np.array([True, False, False, True])
    scores = np.array([.3, .4, .2, .5])
    assert budget_metrics(labels, scores, budget=.25) == budget_metrics(
        labels, 3 * scores + 7, budget=.25)
