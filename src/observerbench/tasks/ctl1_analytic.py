from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from observerbench.core import ObserverResult, write_json
from observerbench.metrics import mse, mae, r2_score
from observerbench.observers import LinearObserver, first_order_features, lifted_pair_features, interaction_only_features


@dataclass
class CollateralTaskConfig:
    """Configuration for the ObserverBench collateral-control task.

    The task uses one fixed actuator geometry over latent activation coordinates
    [x1, x2, x1*x2]. Observers differ only in the basis they use to estimate the
    target and therefore in the actuation direction they ask the fixed actuator to
    apply. Collateral is induced by the overlap between the observer-derived
    direction and a fixed nuisance readout.

    The important control condition is ``normalize_direction_to_target_gain``:
    every observer-derived direction is scaled to have the same linearized target
    gain. A unit control command therefore has the same first-order target effect
    across observers. Any collateral difference is due to geometry, not due to
    giving one observer more control authority.
    """

    n_train: int = 4000
    n_test: int = 4000
    p_feature: float = 0.35
    seed: int = 0
    target_noise: float = 0.03
    collateral_noise: float = 0.03
    target_ref: float = 1.0
    controller_gain: float = 0.8
    max_strength: float = 2.0

    # Target state: beta1*x1 + beta2*x2 + gamma*(x1*x2).
    beta1: float = 0.35
    beta2: float = 0.25
    gamma: float = 1.15

    # Base nuisance/off-target readout in the main-effect subspace.
    nuisance_x1: float = 0.90
    nuisance_x2: float = -0.25

    # Nuisance placement sweep. 0.0 means nuisance is purely in the main-effect
    # subspace [nuisance_x1, nuisance_x2, 0]. 1.0 means nuisance is purely the
    # interaction coordinate [0, 0, nuisance_interaction_unit]. Intermediate
    # values linearly interpolate the two. This makes explicit that collateral is
    # governed by subspace overlap, not by an observer's name.
    nuisance_interaction_weight: float = 0.0
    nuisance_interaction_unit: float = 1.0

    include_interaction_only: bool = False

    # Normalize every observer-derived actuation direction to have this target
    # gain. This keeps the actuator/controller fair.
    normalize_direction_to_target_gain: bool = True
    target_actuation_gain: float = 0.85
    ridge: float = 1e-6


def target_readout(cfg: CollateralTaskConfig) -> np.ndarray:
    return np.array([cfg.beta1, cfg.beta2, cfg.gamma], dtype=float)


def nuisance_readout(cfg: CollateralTaskConfig) -> np.ndarray:
    w = float(np.clip(cfg.nuisance_interaction_weight, 0.0, 1.0))
    main = np.array([cfg.nuisance_x1, cfg.nuisance_x2, 0.0], dtype=float)
    interaction = np.array([0.0, 0.0, cfg.nuisance_interaction_unit], dtype=float)
    return (1.0 - w) * main + w * interaction


def make_split(n: int, cfg: CollateralTaskConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x1 = rng.binomial(1, cfg.p_feature, size=n).astype(float)
    x2 = rng.binomial(1, cfg.p_feature, size=n).astype(float)
    interaction = x1 * x2
    H = np.c_[x1, x2, interaction]
    tr = target_readout(cfg)
    nr = nuisance_readout(cfg)
    target_clean = H @ tr
    nuisance_clean = H @ nr
    target = target_clean + rng.normal(0, cfg.target_noise, size=n)
    nuisance = nuisance_clean + rng.normal(0, cfg.collateral_noise, size=n)
    return {
        "x1": x1,
        "x2": x2,
        "interaction": interaction,
        "activation": H,
        "target_clean": target_clean,
        "target": target,
        "nuisance_clean": nuisance_clean,
        "nuisance": nuisance,
    }


def generate_data(cfg: CollateralTaskConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    return make_split(cfg.n_train, cfg, rng), make_split(cfg.n_test, cfg, rng)


def observer_direction_in_activation_space(name: str, observer: LinearObserver) -> np.ndarray:
    """Map observer coefficients into fixed activation coordinates.

    Coefficients exclude the intercept. The fixed actuator coordinates are
    [x1, x2, interaction].
    """
    if observer.coef_ is None:
        raise RuntimeError("Observer must be fit before extracting direction.")
    c = np.asarray(observer.coef_[1:], dtype=float)
    if name == "first_order":
        if len(c) != 2:
            raise ValueError(f"first_order expected 2 coefficients, got {len(c)}")
        return np.array([c[0], c[1], 0.0], dtype=float)
    if name == "lifted_interaction":
        if len(c) != 3:
            raise ValueError(f"lifted_interaction expected 3 coefficients, got {len(c)}")
        return c.astype(float)
    if name == "interaction_only":
        if len(c) != 1:
            raise ValueError(f"interaction_only expected 1 coefficient, got {len(c)}")
        return np.array([0.0, 0.0, c[0]], dtype=float)
    raise ValueError(f"Unknown observer name: {name}")


def normalize_direction(direction: np.ndarray, tr: np.ndarray, cfg: CollateralTaskConfig) -> Tuple[np.ndarray, float, float]:
    raw_target_gain = float(tr @ direction)
    if not cfg.normalize_direction_to_target_gain:
        return direction.copy(), raw_target_gain, 1.0
    if abs(raw_target_gain) < 1e-12:
        return np.zeros_like(direction), raw_target_gain, 0.0
    scale = cfg.target_actuation_gain / raw_target_gain
    return direction * scale, raw_target_gain, scale


def evaluate_observer(name: str, observer: LinearObserver, train: dict, test: dict, cfg: CollateralTaskConfig) -> ObserverResult:
    tr = target_readout(cfg)
    nr = nuisance_readout(cfg)

    observer.fit(train, train["target"])
    zhat = observer.predict(test)
    observer_r2 = r2_score(test["target"], zhat)
    observer_mae = mae(test["target"], zhat)

    raw_direction = observer_direction_in_activation_space(name, observer)
    direction, raw_target_gain, direction_scale = normalize_direction(raw_direction, tr, cfg)
    effective_target_gain = float(tr @ direction)
    effective_collateral_gain = float(nr @ direction)
    collateral_per_target_gain = effective_collateral_gain / effective_target_gain if abs(effective_target_gain) > 1e-12 else float("nan")

    baseline_target_mse = mse(np.full_like(test["target"], cfg.target_ref), test["target"])
    strength = np.clip(cfg.controller_gain * (cfg.target_ref - zhat), -cfg.max_strength, cfg.max_strength)
    delta = strength[:, None] * direction[None, :]
    target_delta = delta @ tr
    nuisance_delta = delta @ nr
    target_after = test["target"] + target_delta
    nuisance_after = test["nuisance"] + nuisance_delta
    control_target_mse = mse(np.full_like(target_after, cfg.target_ref), target_after)

    metrics = {
        "observer_r2": observer_r2,
        "observer_mae": observer_mae,
        "baseline_target_mse": baseline_target_mse,
        "control_target_mse": control_target_mse,
        "target_improvement_mse": baseline_target_mse - control_target_mse,
        "collateral_abs_delta": float(np.mean(np.abs(nuisance_after - test["nuisance"]))),
        "actuation_energy_l1": float(np.mean(np.sum(np.abs(delta), axis=1))),
        "mean_abs_strength": float(np.mean(np.abs(strength))),
        "mean_strength": float(np.mean(strength)),
        "raw_direction_x1": float(raw_direction[0]),
        "raw_direction_x2": float(raw_direction[1]),
        "raw_direction_interaction": float(raw_direction[2]),
        "direction_x1": float(direction[0]),
        "direction_x2": float(direction[1]),
        "direction_interaction": float(direction[2]),
        "target_readout_x1": float(tr[0]),
        "target_readout_x2": float(tr[1]),
        "target_readout_interaction": float(tr[2]),
        "nuisance_readout_x1": float(nr[0]),
        "nuisance_readout_x2": float(nr[1]),
        "nuisance_readout_interaction": float(nr[2]),
        "nuisance_interaction_weight": float(cfg.nuisance_interaction_weight),
        "gamma": float(cfg.gamma),
        "raw_target_gain": raw_target_gain,
        "direction_scale": direction_scale,
        "effective_target_gain": effective_target_gain,
        "effective_collateral_gain": effective_collateral_gain,
        "collateral_per_target_gain": collateral_per_target_gain,
    }
    return ObserverResult(
        task="collateral_interaction_control",
        observer=name,
        access_regime="white-box or supervised observer; fixed vector actuator and fixed controller",
        observer_family="linear observer over first-order or lifted basis",
        metrics=metrics,
        metadata={
            "estimand": "finite-control target state with a configurable interactional component",
            "measurement_design": "fit observer on target labels; convert observer coefficients to a direction in a fixed activation space; validate target and collateral under the same actuator geometry",
            "validation_target": "target tracking under a fixed controller with low movement of a fixed nuisance readout",
            "notes": "Collateral emerges from observer coefficient geometry. Directions are normalized to equal linearized target gain; there are no observer-specific collateral gains.",
        },
    )


def _derive_result_level_metrics(results: List[ObserverResult]) -> None:
    best_collateral = min(r.metrics["collateral_abs_delta"] for r in results)
    best_target_mse = min(r.metrics["control_target_mse"] for r in results)
    best_improvement = max(r.metrics["target_improvement_mse"] for r in results)
    for r in results:
        r.metrics["collateral_ratio_vs_best"] = r.metrics["collateral_abs_delta"] / max(best_collateral, 1e-12)
        r.metrics["target_mse_ratio_vs_best"] = r.metrics["control_target_mse"] / max(best_target_mse, 1e-12)
        r.metrics["target_improvement_fraction_vs_best"] = r.metrics["target_improvement_mse"] / max(best_improvement, 1e-12)


def run_task(cfg: CollateralTaskConfig, outdir: str | Path) -> List[ObserverResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = generate_data(cfg)
    observers = [
        ("first_order", LinearObserver("first_order", first_order_features, ridge=cfg.ridge)),
        ("lifted_interaction", LinearObserver("lifted_interaction", lifted_pair_features, ridge=cfg.ridge)),
    ]
    if cfg.include_interaction_only:
        observers.append(("interaction_only", LinearObserver("interaction_only", interaction_only_features, ridge=cfg.ridge)))
    results = [evaluate_observer(name, obs, train, test, cfg) for name, obs in observers]
    _derive_result_level_metrics(results)

    # Outputs.
    df = pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results])
    df.to_csv(out / "collateral_task_results.csv", index=False)
    write_json(out / "collateral_task_config.json", asdict(cfg))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, [r.metrics["control_target_mse"] for r in results], width, label="target MSE after control")
    ax.bar(x + width/2, [r.metrics["collateral_abs_delta"] for r in results], width, label="collateral |Δ|")
    ax.axhline(results[0].metrics["baseline_target_mse"], linestyle="--", linewidth=1, label="baseline target MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([r.observer for r in results], rotation=20, ha="right")
    ax.set_title("ObserverBench Ctl-1: fixed actuator, observer-induced collateral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "collateral_task_target_vs_collateral.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for r in results:
        ax.scatter(r.metrics["effective_target_gain"], r.metrics["effective_collateral_gain"], label=r.observer, s=80)
        ax.annotate(r.observer, (r.metrics["effective_target_gain"], r.metrics["effective_collateral_gain"]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("target gain of observer-derived actuation direction")
    ax.set_ylabel("collateral gain of same direction")
    ax.set_title("Fixed actuator geometry: collateral emerges from direction overlap")
    fig.tight_layout()
    fig.savefig(out / "collateral_task_direction_geometry.png", dpi=180)
    plt.close(fig)
    return results
