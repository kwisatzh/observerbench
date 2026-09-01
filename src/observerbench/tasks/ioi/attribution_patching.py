"""Attribution-patching observers for the frozen GPT-2-small IOI effect task."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from observerbench.effect_prediction import (
    EffectObserverCard,
    FiniteEffectMeasurement,
    FiniteEffectQuery,
)
from observerbench.tasks.ioi.effect_task import IOIMaskFeatures
from observerbench.tasks.ioi.heads import head_records
from observerbench.tasks.ioi.phase5_effects import (
    _batched,
    _hook_layout,
    _length_buckets,
    _logit_difference,
    _model_tokens,
    load_locked_ioi_design,
)


IOI_RAW_ATP_BASELINE_NAME = "ioi-attribution-patching-raw"
IOI_CALIBRATED_ATP_BASELINE_NAME = "ioi-attribution-patching-scalar-calibrated"
IOI_ATP_BASELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class IOIAttributionMap:
    """A prompt-averaged attribution-patching map in canonical head order."""

    head_effects: tuple[float, ...]
    n_prompts: int
    prompt_split: str
    model_revision: str
    intervention: str

    def __post_init__(self) -> None:
        if len(self.head_effects) != len(head_records()):
            raise ValueError("IOI attribution map must contain exactly 13 head effects")
        if not self.head_effects or not all(math.isfinite(value) for value in self.head_effects):
            raise ValueError("IOI attribution effects must be finite")
        if self.n_prompts <= 0:
            raise ValueError("IOI attribution map requires at least one prompt")
        if not self.prompt_split or not self.model_revision or not self.intervention:
            raise ValueError("IOI attribution-map provenance fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        records = head_records()
        return {
            "schema_version": "observerbench.ioi_attribution_map.v1",
            "head_effects": [
                {
                    "component_index": index,
                    "head_label": str(record["label"]),
                    "layer": int(record["layer"]),
                    "head": int(record["head"]),
                    "predicted_effect": float(effect),
                }
                for index, (record, effect) in enumerate(zip(records, self.head_effects))
            ],
            "n_prompts": self.n_prompts,
            "prompt_split": self.prompt_split,
            "model_revision": self.model_revision,
            "intervention": self.intervention,
        }


class IOIAttributionPatchingBaseline:
    """Add head-wise attribution-patching effects over each requested mask."""

    def __init__(
        self,
        attribution_map: IOIAttributionMap,
        *,
        calibrate_scalar: bool,
    ) -> None:
        self.attribution_map = attribution_map
        self.calibrate_scalar = bool(calibrate_scalar)
        self.name = (
            IOI_CALIBRATED_ATP_BASELINE_NAME
            if self.calibrate_scalar
            else IOI_RAW_ATP_BASELINE_NAME
        )
        self.gain_: float | None = None

    def _raw_effect(self, features: IOIMaskFeatures) -> float:
        if not isinstance(features, IOIMaskFeatures):
            raise TypeError("IOIAttributionPatchingBaseline requires IOIMaskFeatures")
        return float(
            sum(
                effect * included
                for effect, included in zip(
                    self.attribution_map.head_effects,
                    features.head_mask,
                )
            )
        )

    def fit(
        self,
        measurements: Sequence[FiniteEffectMeasurement[IOIMaskFeatures]],
    ) -> None:
        if not measurements:
            raise ValueError("at least one IOI finite-effect measurement is required")
        if not self.calibrate_scalar:
            self.gain_ = 1.0
            return
        raw = np.asarray(
            [self._raw_effect(row.features) for row in measurements],
            dtype=float,
        )
        observed = np.asarray(
            [row.observed_effect for row in measurements],
            dtype=float,
        )
        denominator = float(raw @ raw)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("attribution predictions cannot identify a scalar gain")
        gain = float((raw @ observed) / denominator)
        if not math.isfinite(gain):
            raise ValueError("fitted attribution gain is not finite")
        self.gain_ = gain

    def predict(
        self,
        queries: Sequence[FiniteEffectQuery[IOIMaskFeatures]],
    ) -> Sequence[float]:
        if self.gain_ is None:
            raise RuntimeError("fit must be called before predict")
        return tuple(self.gain_ * self._raw_effect(row.features) for row in queries)


def ioi_attribution_patching_card(
    *,
    calibrated: bool,
    attribution_map: IOIAttributionMap,
    fitted_gain: float | None = None,
) -> EffectObserverCard:
    """Describe one raw or scalar-calibrated attribution-patching observer."""

    if fitted_gain is not None and not math.isfinite(fitted_gain):
        raise ValueError("fitted_gain must be finite when supplied")
    if not calibrated and fitted_gain not in (None, 1.0):
        raise ValueError("the raw attribution-patching observer has unit gain")

    name = (
        IOI_CALIBRATED_ATP_BASELINE_NAME
        if calibrated
        else IOI_RAW_ATP_BASELINE_NAME
    )
    fit_procedure = (
        "fit one no-intercept scalar gain to the task-supplied finite calibration effects"
        if calibrated
        else "no finite-effect fitting; use the prompt-averaged local attribution map directly"
    )
    access = (
        "white-box gradients on the public train prompts plus frozen finite calibration effects"
        if calibrated
        else "white-box gradients on the public train prompts"
    )
    return EffectObserverCard(
        observer_name=name,
        observer_version=IOI_ATP_BASELINE_VERSION,
        observer_family=(
            "scalar-calibrated attribution patching"
            if calibrated
            else "attribution patching"
        ),
        access_regime=access,
        measurement_basis=(
            "clean-state gradient dot (head-z minus template-conditioned reference mean), "
            "averaged over train prompts in the canonical 13-head IOI basis"
        ),
        fit_procedure=fit_procedure,
        implementation=(
            "observerbench.tasks.ioi.attribution_patching."
            "IOIAttributionPatchingBaseline"
        ),
        known_failure_modes=(
            "First-order additivity cannot represent interactions between ablated heads.",
            "The local gradient may not predict a finite mean-replacement intervention.",
            "The prompt-averaged map does not model prompt-level effect dispersion.",
        ),
        metadata={
            "effective_gradient_prompts": attribution_map.n_prompts,
            "gradient_prompt_split": attribution_map.prompt_split,
            "model_revision": attribution_map.model_revision,
            "scalar_calibrated": bool(calibrated),
            "calibration_intercept": 0.0 if calibrated else None,
            "fitted_scalar_gain": fitted_gain,
        },
    )


def measure_ioi_attribution_map(
    model: Any,
    design_dir: str | Path,
    template_head_means: np.ndarray,
    templates: Sequence[str],
    *,
    model_revision: str,
    prompt_split: str = "train",
    batch_size: int = 32,
    max_prompts: int | None = None,
) -> IOIAttributionMap:
    """Measure a clean-state AtP map using the Phase-5 mean-ablation direction."""

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive when supplied")
    prompts, _masks, _manifest = load_locked_ioi_design(design_dir)
    prompts = prompts[prompts["split"] == prompt_split].copy().reset_index(drop=True)
    if max_prompts is not None:
        prompts = prompts.head(max_prompts).reset_index(drop=True)
    if prompts.empty:
        raise ValueError(f"no prompts are available for split {prompt_split!r}")

    records = head_records()
    means = np.asarray(template_head_means)
    expected_shape = (len(templates), len(records), int(model.cfg.d_head))
    if means.shape != expected_shape or not np.isfinite(means).all():
        raise ValueError(
            f"template head means must have shape {expected_shape}, got {means.shape}"
        )
    template_to_index = {str(template): index for index, template in enumerate(templates)}
    mapped_templates = prompts["template_id"].astype(str).map(template_to_index)
    if mapped_templates.isna().any():
        raise ValueError("an attribution prompt has no template-conditioned reference mean")
    template_rows = mapped_templates.astype(int).to_numpy()

    token_rows, io_tokens, subject_tokens = _model_tokens(model, prompts)
    layout = _hook_layout()
    hook_names = tuple(f"blocks.{layer}.attn.hook_z" for layer in layout)
    totals = np.zeros(len(records), dtype=np.float64)
    measured = 0

    for bucket in _length_buckets(token_rows):
        for batch_indices in _batched(bucket, batch_size):
            tokens = torch.stack([token_rows[index] for index in batch_indices]).to(
                model.cfg.device
            )
            io = torch.as_tensor(io_tokens[batch_indices], device=model.cfg.device)
            subject = torch.as_tensor(
                subject_tokens[batch_indices], device=model.cfg.device
            )
            references = torch.as_tensor(
                means[template_rows[batch_indices]],
                dtype=model.W_E.dtype,
                device=model.cfg.device,
            )
            captured: dict[str, Any] = {}

            def retain(z: Any, hook: Any) -> Any:
                z.retain_grad()
                captured[str(hook.name)] = z
                return z

            model.zero_grad(set_to_none=True)
            with torch.enable_grad(), model.hooks(
                fwd_hooks=[(name, retain) for name in hook_names]
            ):
                logits = model(tokens, return_type="logits")
                score = _logit_difference(logits, io, subject).sum()
                score.backward()

            batch_effects = torch.zeros(
                (len(batch_indices), len(records)),
                dtype=torch.float64,
                device=model.cfg.device,
            )
            for layer, (canonical_np, heads_np) in layout.items():
                name = f"blocks.{layer}.attn.hook_z"
                activation = captured[name]
                if activation.grad is None:
                    raise RuntimeError(f"no gradient was retained for {name}")
                canonical = torch.as_tensor(
                    canonical_np, dtype=torch.long, device=model.cfg.device
                )
                heads = torch.as_tensor(
                    heads_np, dtype=torch.long, device=model.cfg.device
                )
                clean = activation[:, -1].index_select(1, heads)
                gradient = activation.grad[:, -1].index_select(1, heads)
                reference = references.index_select(1, canonical)
                effects = ((clean - reference) * gradient).sum(dim=-1).double()
                batch_effects.index_copy_(1, canonical, effects)
            totals += batch_effects.detach().cpu().numpy().sum(axis=0)
            measured += len(batch_indices)
            captured.clear()

    if measured != len(prompts):
        raise RuntimeError("attribution measurement did not cover every selected prompt")
    return IOIAttributionMap(
        head_effects=tuple(float(value) for value in totals / measured),
        n_prompts=measured,
        prompt_split=prompt_split,
        model_revision=model_revision,
        intervention="final-position template-conditioned mean replacement at head-z",
    )


def load_template_head_means(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load and validate the Phase-5 template-conditioned head means."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != {"means", "templates"}:
            raise ValueError("template-head-mean cache has unexpected arrays")
        means = np.asarray(payload["means"])
        templates = tuple(str(value) for value in payload["templates"].tolist())
    if not templates or len(set(templates)) != len(templates):
        raise ValueError("template-head-mean cache has invalid template labels")
    return means, templates


__all__ = [
    "IOI_ATP_BASELINE_VERSION",
    "IOI_CALIBRATED_ATP_BASELINE_NAME",
    "IOI_RAW_ATP_BASELINE_NAME",
    "IOIAttributionMap",
    "IOIAttributionPatchingBaseline",
    "ioi_attribution_patching_card",
    "load_template_head_means",
    "measure_ioi_attribution_map",
]
