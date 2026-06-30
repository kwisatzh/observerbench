"""Closed-loop trained-transformer Ctl-2 task for ObserverBench.

Ctl-1 measured one-shot target/collateral geometry: fit an observer, form an
observer-derived residual direction, apply one proportional intervention, and
score target movement and collateral. Ctl-2 closes the loop. The controller
repeatedly reads the observer estimate on the edited residual state, applies the
same proportional law, and accumulates target error, actuation energy, and
collateral over time.

This task is intentionally small. It is not a new controller. It is a falsifier
for the observer-control claim: with the same plant, same controller, and same
normalization of initial target authority, does swapping the observer change the
closed-loop trajectory?

v7 additions:
  * an oracle_target arm, used as a gain/controller sanity check;
  * divergence criteria based on final-vs-initial error growth, not a stale
    absolute threshold alone;
  * per-example, per-step trajectories and fan plots, so the closed-loop result
    is visible as a trajectory rather than only as unbounded ratios.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

from observerbench.core import ObserverResult, write_json
from observerbench.cards import results_to_dataframe, write_cards
from observerbench.metrics import mse, mae, r2_score
from observerbench.tasks.trained_ctl1 import (
    TrainedTransformerCtl1Config,
    generate_data,
    train_model,
    extract_hidden,
    _fit_ridge,
    _predict_ridge,
    normalize_direction,
)


@dataclass
class TrainedTransformerCtl2Config(TrainedTransformerCtl1Config):
    # Closed-loop-specific settings.
    loop_steps: int = 15
    controller_gain: float = 0.35
    max_strength: float = 1.0
    target_tolerance: float = 0.10

    # Old absolute threshold kept for backward compatibility, but the primary
    # divergence flag below uses a bounded, growth-based definition.
    divergence_threshold: float = 4.0
    divergence_abs_error_threshold: float = 2.0
    divergence_initial_error_multiplier: float = 2.0
    divergence_final_mse_multiplier: float = 4.0

    # Include oracle arm by default. It uses the true target readout as the
    # observer estimate and the true target-head direction as the actuation
    # direction. It answers the control-reviewer question: at this gain, does an
    # accurate observer/controller stay stable?
    include_oracle: bool = True

    # If True, use target_ref from config. If False, set a per-example target
    # offset above the model's initial output. The default fixed target mirrors
    # Ctl-1 and is easier to interpret.
    use_relative_target: bool = False
    relative_target_offset: float = 0.75


def _linear_from_probe(h: np.ndarray, probe: np.ndarray) -> np.ndarray:
    return probe[0] + h @ probe[1:]


def _observer_estimate(
    h: np.ndarray,
    observer_name: str,
    coef: np.ndarray | None,
    probe_x1: np.ndarray,
    probe_x2: np.ndarray,
    probe_int: np.ndarray,
    target_vec: np.ndarray,
    target_bias: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x1h = _linear_from_probe(h, probe_x1)
    x2h = _linear_from_probe(h, probe_x2)
    ih = _linear_from_probe(h, probe_int)
    if observer_name == "first_order":
        assert coef is not None
        zhat = coef[0] + coef[1] * x1h + coef[2] * x2h
    elif observer_name == "lifted_interaction":
        assert coef is not None
        zhat = coef[0] + coef[1] * x1h + coef[2] * x2h + coef[3] * ih
    elif observer_name == "oracle_target":
        zhat = target_bias + h @ target_vec
    else:
        raise ValueError(f"unknown observer {observer_name}")
    return zhat, {"x1_hat": x1h, "x2_hat": x2h, "interaction_hat": ih}


def _settling_steps(abs_err: np.ndarray, tol: float) -> np.ndarray:
    """First step from which all remaining errors are below tol; nan if never."""
    n, _t = abs_err.shape
    out = np.full(n, np.nan)
    ok = abs_err <= tol
    suffix_ok = np.flip(np.cumprod(np.flip(ok.astype(int), axis=1), axis=1), axis=1).astype(bool)
    for i in range(n):
        hits = np.where(suffix_ok[i])[0]
        if len(hits):
            out[i] = float(hits[0])
    return out


def _prepare_observers(cfg: TrainedTransformerCtl2Config):
    train, test = generate_data(cfg)
    model, train_info = train_model(cfg, train)
    train_pred, train_feats, h_train = extract_hidden(model, train, cfg)
    test_pred, test_feats, h_test = extract_hidden(model, test, cfg)

    # Residual probes for latent variables and nuisance.
    probe_x1 = _fit_ridge(h_train, train["x1"], cfg.ridge)
    probe_x2 = _fit_ridge(h_train, train["x2"], cfg.ridge)
    probe_int = _fit_ridge(h_train, train["interaction"], cfg.ridge)
    probe_nuis = _fit_ridge(h_train, train["nuisance"], cfg.ridge)

    x1h_tr = _predict_ridge(h_train, probe_x1)
    x2h_tr = _predict_ridge(h_train, probe_x2)
    ih_tr = _predict_ridge(h_train, probe_int)
    x1h_te = _predict_ridge(h_test, probe_x1)
    x2h_te = _predict_ridge(h_test, probe_x2)
    ih_te = _predict_ridge(h_test, probe_int)

    fo_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr], train["target_clean"], cfg.ridge)
    li_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr, ih_tr], train["target_clean"], cfg.ridge)

    target_vec = model.target_head.weight.detach().cpu().numpy().reshape(-1)
    target_bias = float(model.target_head.bias.detach().cpu().numpy().reshape(-1)[0])
    nuisance_vec = probe_nuis[1:]
    nuisance_bias = float(probe_nuis[0])

    raw_fo_direction = fo_coef[1] * probe_x1[1:] + fo_coef[2] * probe_x2[1:]
    raw_li_direction = li_coef[1] * probe_x1[1:] + li_coef[2] * probe_x2[1:] + li_coef[3] * probe_int[1:]
    raw_oracle_direction = target_vec.copy()

    observers = {
        "first_order": {"coef": fo_coef, "raw_direction": raw_fo_direction},
        "lifted_interaction": {"coef": li_coef, "raw_direction": raw_li_direction},
    }
    if cfg.include_oracle:
        observers["oracle_target"] = {"coef": None, "raw_direction": raw_oracle_direction}

    for obs in observers.values():
        direction, raw_target_gain, direction_scale, norm_diag = normalize_direction(obs["raw_direction"], target_vec, cfg)
        obs.update({
            "direction": direction,
            "raw_target_gain": raw_target_gain,
            "direction_scale": direction_scale,
            "norm_diag": norm_diag,
        })

    env = {
        "train": train,
        "test": test,
        "model": model,
        "train_info": train_info,
        "h_train": h_train,
        "h_test": h_test,
        "test_pred": test_pred,
        "target_vec": target_vec,
        "target_bias": target_bias,
        "nuisance_vec": nuisance_vec,
        "nuisance_bias": nuisance_bias,
        "probe_x1": probe_x1,
        "probe_x2": probe_x2,
        "probe_int": probe_int,
        "probe_nuis": probe_nuis,
        "x1h_te": x1h_te,
        "x2h_te": x2h_te,
        "ih_te": ih_te,
        "observers": observers,
    }
    return env


def _rollout_observer(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    observer_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = env["observers"][observer_name]
    h0 = env["h_test"].copy()
    h = h0.copy()
    n = h.shape[0]
    direction = obs["direction"]
    coef = obs["coef"]
    target_vec = env["target_vec"]
    target_bias = env["target_bias"]
    nuisance_vec = env["nuisance_vec"]
    nuisance_bias = env["nuisance_bias"]
    probe_x1 = env["probe_x1"]
    probe_x2 = env["probe_x2"]
    probe_int = env["probe_int"]

    target_initial = target_bias + h @ target_vec
    nuisance_initial = nuisance_bias + h @ nuisance_vec
    if cfg.use_relative_target:
        target_ref = target_initial + cfg.relative_target_offset
    else:
        target_ref = np.full(n, cfg.target_ref, dtype=float)

    rows = []
    trajectory_example_rows = []
    strength_history = []
    h_history_targets = []
    h_history_collateral = []
    h_history_obs = []
    previous_recorded_h = None

    for t in range(cfg.loop_steps + 1):
        target_t = target_bias + h @ target_vec
        nuisance_t = nuisance_bias + h @ nuisance_vec
        zhat_t, _feat_dict = _observer_estimate(
            h,
            observer_name,
            coef,
            probe_x1,
            probe_x2,
            probe_int,
            target_vec,
            target_bias,
        )
        err_true = target_ref - target_t
        err_obs = target_ref - zhat_t
        collateral_delta = nuisance_t - nuisance_initial
        if t < cfg.loop_steps:
            next_strength = np.clip(cfg.controller_gain * err_obs, -cfg.max_strength, cfg.max_strength)
        else:
            next_strength = np.zeros(n, dtype=float)
        if previous_recorded_h is None:
            state_step_delta = np.zeros(n, dtype=float)
        else:
            state_step_delta = np.linalg.norm(h - previous_recorded_h, axis=1)
        state_delta_from_initial = np.linalg.norm(h - h0, axis=1)
        previous_recorded_h = h.copy()
        h_history_targets.append(target_t.copy())
        h_history_collateral.append(collateral_delta.copy())
        h_history_obs.append(zhat_t.copy())
        rows.append({
            "observer": observer_name,
            "step": t,
            "target_mse": float(np.mean(err_true ** 2)),
            "target_mae": float(np.mean(np.abs(err_true))),
            "observer_error_mae": float(np.mean(np.abs(zhat_t - target_t))),
            "collateral_abs_delta": float(np.mean(np.abs(collateral_delta))),
            "collateral_rms_delta": float(np.sqrt(np.mean(collateral_delta ** 2))),
            "mean_observer_estimate": float(np.mean(zhat_t)),
            "mean_true_target": float(np.mean(target_t)),
            "mean_target_ref": float(np.mean(target_ref)),
        })
        trajectory_example_rows.extend({
            "observer": observer_name,
            "example_idx": int(i),
            "step": int(t),
            "target": float(target_t[i]),
            "target_ref": float(target_ref[i]),
            "target_error": float(err_true[i]),
            "target_abs_error": float(abs(err_true[i])),
            "target_squared_error": float(err_true[i] ** 2),
            "observer_estimate": float(zhat_t[i]),
            "observer_bias": float(zhat_t[i] - target_t[i]),
            "observer_abs_bias": float(abs(zhat_t[i] - target_t[i])),
            "collateral_delta": float(collateral_delta[i]),
            "collateral_abs_delta": float(abs(collateral_delta[i])),
            "residual_l2_delta_from_initial": float(state_delta_from_initial[i]),
            "residual_l2_delta_from_prev_step": float(state_step_delta[i]),
            "next_control_strength": float(next_strength[i]),
        } for i in range(n))
        if t == cfg.loop_steps:
            break
        strength = next_strength
        strength_history.append(strength.copy())
        h = h + strength[:, None] * direction[None, :]

    target_mat = np.stack(h_history_targets, axis=1)
    collateral_mat = np.stack(h_history_collateral, axis=1)
    obs_mat = np.stack(h_history_obs, axis=1)
    ref_mat = target_ref[:, None]
    err_mat = ref_mat - target_mat
    abs_err = np.abs(err_mat)
    sq_err = err_mat ** 2
    settling = _settling_steps(abs_err, cfg.target_tolerance)
    strengths = np.stack(strength_history, axis=1) if strength_history else np.zeros((n, 0))

    # Integrated metrics over the trajectory. We include the initial step in ISE
    # because a controller with a better observer should reduce this integral.
    ise = np.sum(sq_err, axis=1)
    iae = np.sum(abs_err, axis=1)
    cumulative_collateral = np.sum(np.abs(collateral_mat), axis=1)
    final_collateral = np.abs(collateral_mat[:, -1])
    total_energy = np.sum(np.abs(strengths), axis=1) * float(np.sum(np.abs(direction))) if strengths.shape[1] else np.zeros(n)
    overshoot = np.maximum(0.0, target_mat - ref_mat)
    undershoot = np.maximum(0.0, ref_mat - target_mat)

    initial_abs_error = abs_err[:, 0]
    final_abs_error = abs_err[:, -1]
    initial_sq_error = sq_err[:, 0]
    final_sq_error = sq_err[:, -1]
    old_abs_diverged = np.max(abs_err, axis=1) > cfg.divergence_threshold
    growth_diverged = (
        (final_abs_error > cfg.divergence_abs_error_threshold) &
        (final_abs_error > cfg.divergence_initial_error_multiplier * np.maximum(initial_abs_error, 1e-12))
    )
    mse_growth_diverged = final_sq_error > cfg.divergence_final_mse_multiplier * np.maximum(initial_sq_error, 1e-12)
    worsened = final_sq_error > initial_sq_error

    initial_target_mse = float(np.mean(initial_sq_error))
    final_target_mse = float(np.mean(final_sq_error))
    target_improvement_mse = initial_target_mse - final_target_mse
    metrics = {
        "loop_steps": float(cfg.loop_steps),
        "model_test_mse": mse(env["test_pred"], env["test"]["target"]),
        "model_test_r2": r2_score(env["test"]["target_clean"], env["test_pred"]),
        "observer_initial_r2_vs_true_target": r2_score(env["test"]["target_clean"], obs_mat[:, 0]),
        "observer_initial_mae_vs_true_target": mae(env["test"]["target_clean"], obs_mat[:, 0]),
        "initial_target_mse": initial_target_mse,
        "final_target_mse": final_target_mse,
        "final_over_initial_target_mse": float(final_target_mse / max(initial_target_mse, 1e-12)),
        "target_improvement_mse": target_improvement_mse,
        "target_improvement_fraction": target_improvement_mse / max(initial_target_mse, 1e-12),
        "integrated_squared_error": float(np.mean(ise)),
        "integrated_absolute_error": float(np.mean(iae)),
        "final_target_mae": float(np.mean(final_abs_error)),
        "cumulative_collateral_abs": float(np.mean(cumulative_collateral)),
        "final_collateral_abs": float(np.mean(final_collateral)),
        "max_collateral_abs": float(np.mean(np.max(np.abs(collateral_mat), axis=1))),
        "actuation_energy_l1": float(np.mean(total_energy)),
        "mean_abs_strength": float(np.mean(np.abs(strengths))) if strengths.size else 0.0,
        "mean_strength": float(np.mean(strengths)) if strengths.size else 0.0,
        "max_abs_strength": float(np.max(np.abs(strengths))) if strengths.size else 0.0,
        "overshoot_mean": float(np.mean(np.max(overshoot, axis=1))),
        "undershoot_mean": float(np.mean(np.max(undershoot, axis=1))),
        "settled_fraction": float(np.mean(~np.isnan(settling))),
        "settling_step_mean": float(np.nanmean(settling)) if np.any(~np.isnan(settling)) else float("nan"),
        # Primary divergence metrics: bounded rates rather than unbounded ratios.
        "divergence_rate": float(np.mean(growth_diverged)),
        "divergence_rate_mse_growth": float(np.mean(mse_growth_diverged)),
        "divergence_rate_old_abs_threshold": float(np.mean(old_abs_diverged)),
        "target_error_worsened_rate": float(np.mean(worsened)),
        "mean_observer_bias_initial": float(np.mean(obs_mat[:, 0] - target_mat[:, 0])),
        "mean_observer_bias_final": float(np.mean(obs_mat[:, -1] - target_mat[:, -1])),
        "observer_bias_mae_path": float(np.mean(np.abs(obs_mat - target_mat))),
        "effective_target_gain": float(target_vec @ direction),
        "effective_collateral_gain": float(nuisance_vec @ direction),
        "raw_target_gain": float(obs["raw_target_gain"]),
        "direction_norm": float(np.linalg.norm(direction)),
        "raw_direction_norm": float(np.linalg.norm(obs["raw_direction"])),
        "normalization_mode": obs["norm_diag"].get("normalization_mode"),
        "post_normalization_target_gain_error": obs["norm_diag"].get("post_normalization_target_gain_error", float("nan")),
        "gamma": float(cfg.gamma),
        "nuisance_interaction_weight": float(cfg.nuisance_interaction_weight),
        "controller_gain": float(cfg.controller_gain),
        "max_strength": float(cfg.max_strength),
        "target_ref_mean": float(np.mean(target_ref)),
        "divergence_abs_error_threshold": float(cfg.divergence_abs_error_threshold),
        "divergence_initial_error_multiplier": float(cfg.divergence_initial_error_multiplier),
        "divergence_final_mse_multiplier": float(cfg.divergence_final_mse_multiplier),
    }

    # Per-example metrics for paired comparisons.
    per_example = pd.DataFrame({
        "observer": observer_name,
        "example_idx": np.arange(n),
        "initial_target": target_mat[:, 0],
        "final_target": target_mat[:, -1],
        "target_ref": target_ref,
        "integrated_squared_error": ise,
        "integrated_absolute_error": iae,
        "initial_target_abs_error": initial_abs_error,
        "final_target_abs_error": final_abs_error,
        "initial_target_squared_error": initial_sq_error,
        "final_target_squared_error": final_sq_error,
        "final_over_initial_squared_error": final_sq_error / np.maximum(initial_sq_error, 1e-12),
        "cumulative_collateral_abs": cumulative_collateral,
        "final_collateral_abs": final_collateral,
        "actuation_energy_l1": total_energy,
        "settling_step": settling,
        "max_abs_error": np.max(abs_err, axis=1),
        "old_abs_threshold_diverged": old_abs_diverged,
        "growth_diverged": growth_diverged,
        "mse_growth_diverged": mse_growth_diverged,
        "target_error_worsened": worsened,
        "observer_bias_abs_mean": np.mean(np.abs(obs_mat - target_mat), axis=1),
    })

    traj = pd.DataFrame(rows)
    traj_examples = pd.DataFrame(trajectory_example_rows)
    return metrics, traj, per_example, traj_examples


def _plot_fan(traj_examples: pd.DataFrame, value_col: str, outpath: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for observer_name, g in traj_examples.groupby("observer"):
        q = g.groupby("step")[value_col].quantile([0.1, 0.5, 0.9]).unstack()
        mean = g.groupby("step")[value_col].mean()
        steps = q.index.values
        ax.plot(steps, mean.values, marker="o", label=f"{observer_name} mean")
        ax.fill_between(steps, q[0.1].values, q[0.9].values, alpha=0.15)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_trained_transformer_ctl2(cfg: TrainedTransformerCtl2Config, outdir: str | Path) -> List[ObserverResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    env = _prepare_observers(cfg)

    observer_order = ["first_order", "lifted_interaction"] + (["oracle_target"] if cfg.include_oracle else [])
    results: List[ObserverResult] = []
    trajectories = []
    per_examples = []
    trajectory_examples = []
    for observer_name in observer_order:
        metrics, traj, per_ex, traj_ex = _rollout_observer(cfg, env, observer_name)
        trajectories.append(traj)
        per_examples.append(per_ex)
        trajectory_examples.append(traj_ex)
        results.append(ObserverResult(
            task="trained_transformer_ctl2_closed_loop",
            observer=observer_name,
            access_regime="white-box residual representation; fixed residual-space actuator and iterative controller",
            observer_family="linear/oracle observers over learned residual features, rerun each feedback step",
            metrics=metrics,
            metadata={
                "estimand": "closed-loop control-relevant target state represented in a trained transformer residual stream",
                "measurement_design": "fit residual probes and observer-derived actuation directions; run an iterative proportional controller with observer re-measurement at each edited residual state",
                "validation_target": "integrated target tracking error, cumulative collateral, actuation energy, observer bias, and divergence rate along trajectory",
                "notes": "Ctl-2 is the closed-loop version of trained Ctl-1. Same plant, same controller, same target-gain normalization; only the observer map is swapped. The oracle_target arm is a gain/controller sanity check.",
            },
        ))

    # Pairwise ratios/deltas for easy reading; lifted is the main reference.
    by_name = {r.observer: r for r in results}
    li = by_name["lifted_interaction"].metrics
    for r in results:
        r.metrics["ise_ratio_vs_lifted"] = r.metrics["integrated_squared_error"] / max(li["integrated_squared_error"], 1e-12)
        r.metrics["cumulative_collateral_ratio_vs_lifted"] = r.metrics["cumulative_collateral_abs"] / max(li["cumulative_collateral_abs"], 1e-12)
        r.metrics["final_mse_ratio_vs_lifted"] = r.metrics["final_target_mse"] / max(li["final_target_mse"], 1e-12)
        r.metrics["energy_ratio_vs_lifted"] = r.metrics["actuation_energy_l1"] / max(li["actuation_energy_l1"], 1e-12)

    df = results_to_dataframe(results)
    df.to_csv(out / "observerbench_results.csv", index=False)
    pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results]).to_csv(out / "trained_transformer_ctl2_results.csv", index=False)
    traj_df = pd.concat(trajectories, ignore_index=True)
    traj_df.to_csv(out / "trained_transformer_ctl2_trajectory_summary.csv", index=False)
    per_ex_df = pd.concat(per_examples, ignore_index=True)
    per_ex_df.to_csv(out / "trained_transformer_ctl2_per_example.csv", index=False)
    traj_ex_df = pd.concat(trajectory_examples, ignore_index=True)
    traj_ex_df.to_csv(out / "trained_transformer_ctl2_per_step_examples.csv", index=False)

    # Paired per-example comparison: first_order minus lifted. Positive means FO is worse for errors/collateral.
    fo_ex = per_ex_df[per_ex_df["observer"] == "first_order"].set_index("example_idx")
    li_ex = per_ex_df[per_ex_df["observer"] == "lifted_interaction"].set_index("example_idx")
    paired = pd.DataFrame({"example_idx": fo_ex.index})
    for col in [
        "integrated_squared_error", "integrated_absolute_error", "final_target_abs_error",
        "final_target_squared_error", "cumulative_collateral_abs", "final_collateral_abs",
        "actuation_energy_l1", "observer_bias_abs_mean", "max_abs_error",
        "final_over_initial_squared_error",
    ]:
        paired[f"delta_{col}_fo_minus_lifted"] = fo_ex[col].values - li_ex[col].values
    for col in ["growth_diverged", "mse_growth_diverged", "target_error_worsened"]:
        paired[f"delta_{col}_fo_minus_lifted"] = fo_ex[col].astype(float).values - li_ex[col].astype(float).values
    paired.to_csv(out / "trained_transformer_ctl2_paired_deltas.csv", index=False)

    write_cards(results, out / "observer_cards")
    write_json(out / "trained_transformer_ctl2_config.json", asdict(cfg))
    write_json(out / "trained_transformer_ctl2_train_info.json", env["train_info"])

    fo = by_name["first_order"].metrics
    summary = {
        "delta_ise_fo_minus_lifted": float(fo["integrated_squared_error"] - li["integrated_squared_error"]),
        "delta_cumulative_collateral_fo_minus_lifted": float(fo["cumulative_collateral_abs"] - li["cumulative_collateral_abs"]),
        "delta_final_target_mse_fo_minus_lifted": float(fo["final_target_mse"] - li["final_target_mse"]),
        "delta_divergence_rate_fo_minus_lifted": float(fo["divergence_rate"] - li["divergence_rate"]),
        "delta_worsened_rate_fo_minus_lifted": float(fo["target_error_worsened_rate"] - li["target_error_worsened_rate"]),
        "fo_ise_ratio_vs_lifted": float(fo["integrated_squared_error"] / max(li["integrated_squared_error"], 1e-12)),
        "fo_collateral_ratio_vs_lifted": float(fo["cumulative_collateral_abs"] / max(li["cumulative_collateral_abs"], 1e-12)),
        "fo_energy_ratio_vs_lifted": float(fo["actuation_energy_l1"] / max(li["actuation_energy_l1"], 1e-12)),
        "strong_success_candidate": bool(
            (fo["integrated_squared_error"] <= 1.25 * li["integrated_squared_error"]) and
            (fo["cumulative_collateral_abs"] >= 2.0 * li["cumulative_collateral_abs"])
        ),
        "alternative_success_candidate": bool(fo["integrated_squared_error"] > li["integrated_squared_error"]),
    }
    if "oracle_target" in by_name:
        oracle = by_name["oracle_target"].metrics
        summary.update({
            "oracle_integrated_squared_error": float(oracle["integrated_squared_error"]),
            "oracle_final_target_mse": float(oracle["final_target_mse"]),
            "oracle_divergence_rate": float(oracle["divergence_rate"]),
            "oracle_target_error_worsened_rate": float(oracle["target_error_worsened_rate"]),
            "oracle_sanity_converges": bool((oracle["divergence_rate"] == 0.0) and (oracle["target_error_worsened_rate"] < 0.5)),
        })
    write_json(out / "trained_transformer_ctl2_pairwise_summary.json", summary)

    # Figures: summary and fan plots.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for observer_name, g in traj_df.groupby("observer"):
        ax.plot(g["step"], g["target_mse"], marker="o", label=observer_name)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel("mean target MSE")
    ax.set_title("Ctl-2: closed-loop target tracking")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "ctl2_target_mse_trajectory.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for observer_name, g in traj_df.groupby("observer"):
        ax.plot(g["step"], g["collateral_abs_delta"], marker="o", label=observer_name)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel("mean |collateral displacement|")
    ax.set_title("Ctl-2: collateral accumulation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "ctl2_collateral_trajectory.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for observer_name, g in traj_df.groupby("observer"):
        ax.plot(g["step"], g["observer_error_mae"], marker="o", label=observer_name)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel("observer MAE vs plant target")
    ax.set_title("Ctl-2: observer bias along trajectory")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "ctl2_observer_bias_trajectory.png", dpi=180); plt.close(fig)

    _plot_fan(traj_ex_df, "target_squared_error", out / "ctl2_target_mse_fan.png", "Ctl-2: target error trajectories", "per-example target squared error")
    _plot_fan(traj_ex_df, "collateral_abs_delta", out / "ctl2_collateral_fan.png", "Ctl-2: collateral trajectories", "per-example |collateral displacement|")
    _plot_fan(traj_ex_df, "observer_abs_bias", out / "ctl2_observer_bias_fan.png", "Ctl-2: observer-bias trajectories", "per-example |observer-target| bias")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(results)); width = 0.35
    ax.bar(x - width/2, [r.metrics["integrated_squared_error"] for r in results], width, label="integrated target squared error")
    ax.bar(x + width/2, [r.metrics["cumulative_collateral_abs"] for r in results], width, label="cumulative collateral")
    ax.set_xticks(x); ax.set_xticklabels([r.observer for r in results], rotation=20, ha="right")
    ax.set_title("Ctl-2 summary")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "ctl2_summary_bar.png", dpi=180); plt.close(fig)

    return results
