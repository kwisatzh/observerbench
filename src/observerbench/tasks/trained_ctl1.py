"""Trained-transformer Ctl-1 task for ObserverBench.

This is the model-based version of the analytic collateral-control task. A tiny
Transformer is trained on token sequences whose target contains a first-order
part and an interactional part. We then fit observers from the learned final
residual representation and ask whether the same fixed controller/actuator can
control the target with low collateral.

The task is intentionally small: it is a bridge from the analytic Ctl-1 geometry
to learned residual directions, not a language-model benchmark.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from observerbench.core import ObserverResult, write_json
from observerbench.cards import results_to_dataframe, write_cards
from observerbench.metrics import mse, mae, r2_score


@dataclass
class TrainedTransformerCtl1Config:
    seed: int = 0
    n_train: int = 3000
    n_test: int = 1500
    seq_len: int = 4
    p_feature: float = 0.35
    beta1: float = 0.35
    beta2: float = 0.25
    gamma: float = 1.15
    target_noise: float = 0.02
    nuisance_x1: float = 0.90
    nuisance_x2: float = -0.25
    nuisance_interaction_weight: float = 0.0
    nuisance_interaction_unit: float = 1.0
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 96
    train_steps: int = 600
    batch_size: int = 256
    lr: float = 1e-3
    aux_feature_weight: float = 0.05
    ridge: float = 1e-4
    target_ref: float = 1.0
    controller_gain: float = 0.8
    max_strength: float = 2.0
    target_actuation_gain: float = 0.85
    # Direction normalization for residual-space actuation.
    # "target_gain_scale" is the original normalization: scale the entire
    # observer direction so target_head @ direction == target_actuation_gain.
    # This can be ill-conditioned when the observer direction is nearly
    # orthogonal to the target head.
    # "parallel_orthogonal" decomposes the observer direction into the
    # component parallel to the model's target head and its orthogonal residual.
    # The parallel component is replaced by a fixed target-gain component, while
    # the orthogonal component is kept at its measured scale. This holds the
    # linearized target effect fixed without amplifying the orthogonal part by
    # a near-zero denominator.
    direction_normalization: str = "parallel_orthogonal"
    orthogonal_scale: float = 1.0
    device: str = "auto"


def _device(cfg: TrainedTransformerCtl1Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax((q @ k.transpose(-2, -1)) / (self.d_head ** 0.5), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerRegressor(nn.Module):
    def __init__(self, vocab_size: int, cfg: TrainedTransformerCtl1Config):
        super().__init__()
        self.token = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([TinyBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.target_head = nn.Linear(cfg.d_model, 1)
        self.feature_head = nn.Linear(cfg.d_model, 3)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token(tokens) + self.pos[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.ln(x[:, -1, :])

    def forward(self, tokens: torch.Tensor, return_h: bool = False):
        h = self.encode(tokens)
        y = self.target_head(h).squeeze(-1)
        feats = self.feature_head(h)
        if return_h:
            return y, feats, h
        return y, feats


def target_readout(cfg: TrainedTransformerCtl1Config) -> np.ndarray:
    return np.array([cfg.beta1, cfg.beta2, cfg.gamma], dtype=float)


def nuisance_readout(cfg: TrainedTransformerCtl1Config) -> np.ndarray:
    w = float(np.clip(cfg.nuisance_interaction_weight, 0.0, 1.0))
    main = np.array([cfg.nuisance_x1, cfg.nuisance_x2, 0.0], dtype=float)
    inter = np.array([0.0, 0.0, cfg.nuisance_interaction_unit], dtype=float)
    return (1.0 - w) * main + w * inter


def make_data(n: int, cfg: TrainedTransformerCtl1Config, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x1 = rng.binomial(1, cfg.p_feature, size=n).astype(float)
    x2 = rng.binomial(1, cfg.p_feature, size=n).astype(float)
    inter = x1 * x2
    H = np.c_[x1, x2, inter]
    y_clean = H @ target_readout(cfg)
    y = y_clean + rng.normal(0, cfg.target_noise, size=n)
    nuisance = H @ nuisance_readout(cfg)
    # [BOS, x1 token, x2 token, READOUT]
    tokens = np.zeros((n, cfg.seq_len), dtype=np.int64)
    tokens[:, 0] = 0
    tokens[:, 1] = np.where(x1 > 0.5, 2, 1)
    tokens[:, 2] = np.where(x2 > 0.5, 4, 3)
    tokens[:, 3] = 5
    return {"tokens": tokens, "x1": x1, "x2": x2, "interaction": inter, "activation": H, "target": y, "target_clean": y_clean, "nuisance": nuisance}


def generate_data(cfg: TrainedTransformerCtl1Config) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    return make_data(cfg.n_train, cfg, rng), make_data(cfg.n_test, cfg, rng)


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]
    reg = np.eye(X1.shape[1]) * ridge
    reg[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)


def _predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ coef


def train_model(cfg: TrainedTransformerCtl1Config, train: Dict[str, np.ndarray]) -> Tuple[TinyTransformerRegressor, Dict[str, float]]:
    if torch is not None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    if torch is None:
        raise RuntimeError("PyTorch is required for the trained-transformer task.")
    dev = _device(cfg)
    torch.manual_seed(cfg.seed)
    model = TinyTransformerRegressor(vocab_size=6, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    tokens = torch.tensor(train["tokens"], dtype=torch.long, device=dev)
    y = torch.tensor(train["target"], dtype=torch.float32, device=dev)
    feats = torch.tensor(np.c_[train["x1"], train["x2"], train["interaction"]], dtype=torch.float32, device=dev)
    n = len(tokens)
    losses: List[float] = []
    for step in range(cfg.train_steps):
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), device=dev)
        pred, feat_logits = model(tokens[idx])
        loss = torch.mean((pred - y[idx]) ** 2) + cfg.aux_feature_weight * F.binary_cross_entropy_with_logits(feat_logits, feats[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, cfg.train_steps // 20) == 0 or step == cfg.train_steps - 1:
            losses.append(float(loss.detach().cpu()))
    return model, {"final_train_loss": losses[-1], "loss_trace": losses, "device": dev}


def extract_hidden(model: TinyTransformerRegressor, data: Dict[str, np.ndarray], cfg: TrainedTransformerCtl1Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev = _device(cfg)
    model.eval()
    ys, fs, hs = [], [], []
    bs = 2048
    with torch.no_grad():
        tokens = torch.tensor(data["tokens"], dtype=torch.long, device=dev)
        for i in range(0, len(tokens), bs):
            y, f, h = model(tokens[i:i+bs], return_h=True)
            ys.append(y.detach().cpu().numpy())
            fs.append(torch.sigmoid(f).detach().cpu().numpy())
            hs.append(h.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(fs), np.concatenate(hs)


def normalize_direction(direction: np.ndarray, target_vec: np.ndarray, cfg: TrainedTransformerCtl1Config):
    """Normalize an observer-derived residual direction for fair control.

    The original trained-transformer prototype scaled the *entire* observer
    direction by ``target_actuation_gain / (target_vec @ direction)``. That is
    correct in the analytic task but can be ill-conditioned in learned residual
    space: if the observer direction is nearly orthogonal to the model's target
    head, the denominator is small and the orthogonal part is amplified.

    The default mode used here separates the two effects. We decompose the raw
    direction into a component parallel to the model target head and an
    orthogonal component. The parallel component is replaced by the unique unit
    that gives the requested linearized target gain; the orthogonal component is
    carried over without target-gain amplification. Collateral then comes from
    (i) incidental nuisance overlap of the shared parallel target move and
    (ii) the observer-specific orthogonal geometry.
    """
    target_vec = np.asarray(target_vec, dtype=float)
    direction = np.asarray(direction, dtype=float)
    raw_gain = float(target_vec @ direction)

    mode = getattr(cfg, "direction_normalization", "parallel_orthogonal")
    if mode == "target_gain_scale":
        if abs(raw_gain) < 1e-12:
            return np.zeros_like(direction), raw_gain, 0.0, {
                "normalization_mode": mode,
                "orientation_sign": 0.0,
                "raw_parallel_norm": 0.0,
                "raw_orthogonal_norm": float(np.linalg.norm(direction)),
                "target_head_norm": float(np.linalg.norm(target_vec)),
            }
        scale = cfg.target_actuation_gain / raw_gain
        return direction * scale, raw_gain, scale, {
            "normalization_mode": mode,
            "orientation_sign": float(np.sign(raw_gain)),
            "raw_parallel_norm": float("nan"),
            "raw_orthogonal_norm": float("nan"),
            "target_head_norm": float(np.linalg.norm(target_vec)),
        }

    if mode != "parallel_orthogonal":
        raise ValueError(f"Unknown direction_normalization mode: {mode}")

    target_norm = float(np.linalg.norm(target_vec))
    if target_norm < 1e-12:
        return np.zeros_like(direction), raw_gain, 0.0, {
            "normalization_mode": mode,
            "orientation_sign": 0.0,
            "raw_parallel_norm": 0.0,
            "raw_orthogonal_norm": float(np.linalg.norm(direction)),
            "target_head_norm": target_norm,
        }

    # Orient the observer's orthogonal geometry so positive controller commands
    # correspond to increasing the target. If raw_gain is numerically zero, use
    # +1; the target component below sets the actual target direction.
    orientation_sign = 1.0 if raw_gain >= 0 else -1.0
    oriented = orientation_sign * direction
    t_hat = target_vec / target_norm
    raw_parallel = float(oriented @ t_hat) * t_hat
    raw_orthogonal = oriented - raw_parallel

    fixed_parallel = (cfg.target_actuation_gain / target_norm) * t_hat
    final_direction = fixed_parallel + cfg.orthogonal_scale * raw_orthogonal
    # By construction, target_vec @ raw_orthogonal == 0 up to numerical error.
    scale_equivalent = float("nan")
    diagnostics = {
        "normalization_mode": mode,
        "orientation_sign": orientation_sign,
        "raw_parallel_norm": float(np.linalg.norm(raw_parallel)),
        "raw_orthogonal_norm": float(np.linalg.norm(raw_orthogonal)),
        "fixed_parallel_norm": float(np.linalg.norm(fixed_parallel)),
        "target_head_norm": target_norm,
        "post_normalization_target_gain_error": float(target_vec @ final_direction - cfg.target_actuation_gain),
    }
    return final_direction, raw_gain, scale_equivalent, diagnostics


def run_trained_transformer_ctl1(cfg: TrainedTransformerCtl1Config, outdir: str | Path) -> List[ObserverResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = generate_data(cfg)
    model, train_info = train_model(cfg, train)
    train_pred, train_feats, h_train = extract_hidden(model, train, cfg)
    test_pred, test_feats, h_test = extract_hidden(model, test, cfg)

    # Probes from learned final residual to latent variables and nuisance.
    probe_x1 = _fit_ridge(h_train, train["x1"], cfg.ridge)
    probe_x2 = _fit_ridge(h_train, train["x2"], cfg.ridge)
    probe_int = _fit_ridge(h_train, train["interaction"], cfg.ridge)
    probe_nuis = _fit_ridge(h_train, train["nuisance"], cfg.ridge)
    x1h_tr = _predict_ridge(h_train, probe_x1); x2h_tr = _predict_ridge(h_train, probe_x2); ih_tr = _predict_ridge(h_train, probe_int)
    x1h_te = _predict_ridge(h_test, probe_x1); x2h_te = _predict_ridge(h_test, probe_x2); ih_te = _predict_ridge(h_test, probe_int)
    nuisance_before = _predict_ridge(h_test, probe_nuis)

    fo_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr], train["target_clean"], cfg.ridge)
    li_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr, ih_tr], train["target_clean"], cfg.ridge)
    observers = {
        "first_order": (fo_coef, np.c_[x1h_te, x2h_te], fo_coef[1] * probe_x1[1:] + fo_coef[2] * probe_x2[1:]),
        "lifted_interaction": (li_coef, np.c_[x1h_te, x2h_te, ih_te], li_coef[1] * probe_x1[1:] + li_coef[2] * probe_x2[1:] + li_coef[3] * probe_int[1:]),
    }
    target_vec = model.target_head.weight.detach().cpu().numpy().reshape(-1)
    nuisance_vec = probe_nuis[1:]
    baseline_target_mse = mse(np.full_like(test_pred, cfg.target_ref), test_pred)
    results: List[ObserverResult] = []
    for name, (coef, features_test, raw_direction) in observers.items():
        zhat = _predict_ridge(features_test, coef)
        direction, raw_target_gain, direction_scale, norm_diag = normalize_direction(raw_direction, target_vec, cfg)
        eff_target_gain = float(target_vec @ direction)
        eff_collateral_gain = float(nuisance_vec @ direction)
        strength = np.clip(cfg.controller_gain * (cfg.target_ref - zhat), -cfg.max_strength, cfg.max_strength)
        target_after = test_pred + strength * eff_target_gain
        nuisance_after = nuisance_before + strength * eff_collateral_gain
        control_target_mse = mse(np.full_like(target_after, cfg.target_ref), target_after)
        metrics = {
            "model_test_mse": mse(test_pred, test["target"]),
            "model_test_r2": r2_score(test["target_clean"], test_pred),
            "probe_x1_r2": r2_score(test["x1"], x1h_te),
            "probe_x2_r2": r2_score(test["x2"], x2h_te),
            "probe_interaction_r2": r2_score(test["interaction"], ih_te),
            "probe_nuisance_r2": r2_score(test["nuisance"], nuisance_before),
            "observer_r2": r2_score(test["target_clean"], zhat),
            "observer_mae": mae(test["target_clean"], zhat),
            "baseline_target_mse": baseline_target_mse,
            "control_target_mse": control_target_mse,
            "target_improvement_mse": baseline_target_mse - control_target_mse,
            "collateral_abs_delta": float(np.mean(np.abs(nuisance_after - nuisance_before))),
            "actuation_energy_l1": float(np.mean(np.abs(strength) * np.sum(np.abs(direction)))),
            "mean_abs_strength": float(np.mean(np.abs(strength))),
            "mean_strength": float(np.mean(strength)),
            "raw_direction_norm": float(np.linalg.norm(raw_direction)),
            "direction_norm": float(np.linalg.norm(direction)),
            "raw_target_gain": raw_target_gain,
            "direction_scale": direction_scale,
            "effective_target_gain": eff_target_gain,
            "effective_collateral_gain": eff_collateral_gain,
            "collateral_per_target_gain": eff_collateral_gain / eff_target_gain if abs(eff_target_gain) > 1e-12 else float("nan"),
            "normalization_mode": norm_diag.get("normalization_mode"),
            "orientation_sign": norm_diag.get("orientation_sign"),
            "raw_parallel_norm": norm_diag.get("raw_parallel_norm"),
            "raw_orthogonal_norm": norm_diag.get("raw_orthogonal_norm"),
            "fixed_parallel_norm": norm_diag.get("fixed_parallel_norm", float("nan")),
            "target_head_norm": norm_diag.get("target_head_norm"),
            "post_normalization_target_gain_error": norm_diag.get("post_normalization_target_gain_error", float("nan")),
            "gamma": float(cfg.gamma),
            "nuisance_interaction_weight": float(cfg.nuisance_interaction_weight),
        }
        results.append(ObserverResult(
            task="trained_transformer_ctl1",
            observer=name,
            access_regime="white-box residual representation; fixed residual-space actuator and controller",
            observer_family="linear probes over learned residual features",
            metrics=metrics,
            metadata={
                "estimand": "control-relevant target state represented in a trained transformer residual stream",
                "measurement_design": "train latent probes on residual state; form observer-derived residual actuation directions; validate target and collateral under a fixed controller",
                "validation_target": "target tracking with low movement of a fixed nuisance probe",
                "notes": "Prototype trained-transformer Ctl-1. Directions are normalized with parallel/orthogonal target-head normalization by default: the target-parallel component gives equal linearized target gain, while observer-specific collateral comes from the unamplified orthogonal geometry.",
            },
        ))

    best_collateral = min(r.metrics["collateral_abs_delta"] for r in results)
    best_target_mse = min(r.metrics["control_target_mse"] for r in results)
    best_improvement = max(r.metrics["target_improvement_mse"] for r in results)
    for r in results:
        r.metrics["collateral_ratio_vs_best"] = r.metrics["collateral_abs_delta"] / max(best_collateral, 1e-12)
        r.metrics["target_mse_ratio_vs_best"] = r.metrics["control_target_mse"] / max(best_target_mse, 1e-12)
        r.metrics["target_improvement_fraction_vs_best"] = r.metrics["target_improvement_mse"] / max(best_improvement, 1e-12)

    df = results_to_dataframe(results)
    df.to_csv(out / "observerbench_results.csv", index=False)
    pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results]).to_csv(out / "trained_transformer_ctl1_results.csv", index=False)
    write_cards(results, out / "observer_cards")
    write_json(out / "trained_transformer_ctl1_config.json", asdict(cfg))
    write_json(out / "trained_transformer_ctl1_train_info.json", train_info)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(results)); width = 0.35
    ax.bar(x - width/2, [r.metrics["control_target_mse"] for r in results], width, label="target MSE after control")
    ax.bar(x + width/2, [r.metrics["collateral_abs_delta"] for r in results], width, label="collateral |delta|")
    ax.axhline(baseline_target_mse, linestyle="--", linewidth=1, label="baseline target MSE")
    ax.set_xticks(x); ax.set_xticklabels([r.observer for r in results], rotation=20, ha="right")
    ax.set_title("Trained-transformer Ctl-1: first-order vs interaction-aware observer")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "trained_transformer_ctl1_target_vs_collateral.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for r in results:
        ax.scatter(r.metrics["effective_target_gain"], r.metrics["effective_collateral_gain"], s=80, label=r.observer)
        ax.annotate(r.observer, (r.metrics["effective_target_gain"], r.metrics["effective_collateral_gain"]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("target gain of observer-derived residual direction")
    ax.set_ylabel("nuisance gain of same direction")
    ax.set_title("Learned residual geometry")
    fig.tight_layout(); fig.savefig(out / "trained_transformer_ctl1_direction_geometry.png", dpi=180); plt.close(fig)
    return results
