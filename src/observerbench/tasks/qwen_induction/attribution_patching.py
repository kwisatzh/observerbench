"""Attribution-patching observers for the frozen Qwen Copy-v2 effect task."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.effect_prediction import (
    EffectObserverCard,
    FiniteEffectMeasurement,
    FiniteEffectQuery,
)
from observerbench.tasks.qwen_induction.effect_task import (
    InductionMaskFeatures,
    QWEN_INDUCTION_MODEL_REVISION,
    QWEN_INDUCTION_N_HEADS,
)
from observerbench.tasks.qwen_induction.plant import (
    HeadAblationMeans,
    HeadRef,
    _distractor_tokens,
    _family_id,
    _query_position,
    _target_token,
)


QWEN_RAW_ATP_BASELINE_NAME = "qwen-induction-attribution-patching-raw"
QWEN_CALIBRATED_ATP_BASELINE_NAME = (
    "qwen-induction-attribution-patching-scalar-calibrated"
)
QWEN_ATP_BASELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class QwenInductionAttributionMap:
    """A prompt-averaged attribution map in the frozen eight-head order."""

    head_effects: tuple[float, ...]
    head_labels: tuple[str, ...]
    n_prompts: int
    prompt_split: str
    model_revision: str
    intervention: str

    def __post_init__(self) -> None:
        if len(self.head_effects) != QWEN_INDUCTION_N_HEADS:
            raise ValueError("Qwen attribution map must contain exactly 8 head effects")
        if len(self.head_labels) != QWEN_INDUCTION_N_HEADS:
            raise ValueError("Qwen attribution map must contain exactly 8 head labels")
        if len(set(self.head_labels)) != QWEN_INDUCTION_N_HEADS:
            raise ValueError("Qwen attribution-map head labels must be unique")
        if not all(math.isfinite(value) for value in self.head_effects):
            raise ValueError("Qwen attribution effects must be finite")
        if self.n_prompts <= 0:
            raise ValueError("Qwen attribution map requires at least one prompt")
        if not self.prompt_split or not self.model_revision or not self.intervention:
            raise ValueError("Qwen attribution-map provenance fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "observerbench.qwen_induction_attribution_map.v1",
            "head_effects": [
                {
                    "component_index": index,
                    "head_label": label,
                    "predicted_effect": float(effect),
                }
                for index, (label, effect) in enumerate(
                    zip(self.head_labels, self.head_effects)
                )
            ],
            "n_prompts": self.n_prompts,
            "prompt_split": self.prompt_split,
            "model_revision": self.model_revision,
            "intervention": self.intervention,
        }


class QwenInductionAttributionPatchingBaseline:
    """Sum frozen head-wise AtP effects over each requested intervention mask."""

    def __init__(
        self,
        attribution_map: QwenInductionAttributionMap,
        *,
        calibrate_scalar: bool,
    ) -> None:
        self.attribution_map = attribution_map
        self.calibrate_scalar = bool(calibrate_scalar)
        self.name = (
            QWEN_CALIBRATED_ATP_BASELINE_NAME
            if self.calibrate_scalar
            else QWEN_RAW_ATP_BASELINE_NAME
        )
        self.gain_: float | None = None

    def _raw_effect(self, features: InductionMaskFeatures) -> float:
        if not isinstance(features, InductionMaskFeatures):
            raise TypeError(
                "QwenInductionAttributionPatchingBaseline requires "
                "InductionMaskFeatures"
            )
        if features.head_labels != self.attribution_map.head_labels:
            raise ValueError("query head order differs from the attribution map")
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
        measurements: Sequence[FiniteEffectMeasurement[InductionMaskFeatures]],
    ) -> None:
        if not measurements:
            raise ValueError("at least one Qwen finite-effect measurement is required")
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
        queries: Sequence[FiniteEffectQuery[InductionMaskFeatures]],
    ) -> Sequence[float]:
        if self.gain_ is None:
            raise RuntimeError("fit must be called before predict")
        return tuple(self.gain_ * self._raw_effect(row.features) for row in queries)


def qwen_induction_attribution_patching_card(
    *,
    calibrated: bool,
    attribution_map: QwenInductionAttributionMap,
    fitted_gain: float | None = None,
) -> EffectObserverCard:
    """Describe one raw or scalar-calibrated Qwen AtP observer."""

    if fitted_gain is not None and not math.isfinite(fitted_gain):
        raise ValueError("fitted_gain must be finite when supplied")
    if not calibrated and fitted_gain not in (None, 1.0):
        raise ValueError("the raw attribution-patching observer has unit gain")
    name = (
        QWEN_CALIBRATED_ATP_BASELINE_NAME
        if calibrated
        else QWEN_RAW_ATP_BASELINE_NAME
    )
    return EffectObserverCard(
        observer_name=name,
        observer_version=QWEN_ATP_BASELINE_VERSION,
        observer_family=(
            "scalar-calibrated attribution patching"
            if calibrated
            else "attribution patching"
        ),
        access_regime=(
            "white-box gradients on the public train prompts plus frozen finite "
            "calibration effects"
            if calibrated
            else "white-box gradients on the public train prompts"
        ),
        measurement_basis=(
            "clean-state gradient of the exact candidate margin dotted with "
            "(head-z minus its family-conditioned reference mean), averaged over "
            "train prompts in the frozen eight-head basis"
        ),
        fit_procedure=(
            "fit one no-intercept scalar gain to the task-supplied finite "
            "calibration effects"
            if calibrated
            else "no finite-effect fitting; use the local attribution map directly"
        ),
        implementation=(
            "observerbench.tasks.qwen_induction.attribution_patching."
            "QwenInductionAttributionPatchingBaseline"
        ),
        known_failure_modes=(
            "First-order additivity cannot represent interactions between ablated heads.",
            "The clean local gradient may not predict a finite mean replacement.",
            "The prompt-averaged map does not model prompt-level effect dispersion.",
        ),
        metadata={
            "effective_gradient_prompts": attribution_map.n_prompts,
            "gradient_prompt_split": attribution_map.prompt_split,
            "model_revision": attribution_map.model_revision,
            "scalar_calibrated": bool(calibrated),
            "calibration_intercept": 0.0 if calibrated else None,
            "fitted_scalar_gain": fitted_gain,
            "result_status": "post-outcome published-method baseline",
        },
    )


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = int(str(row[field]))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"invalid integer field {field!r}") from None
    if value < 0:
        raise ValueError(f"invalid integer field {field!r}")
    return value


def load_qwen_train_prompts(
    path: str | Path,
    *,
    max_prompts: int | None = None,
) -> tuple[dict[str, str], ...]:
    """Load the public Copy-v2 train prompts without opening test outcomes."""

    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive when supplied")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("split") == "train"]
    if not rows:
        raise ValueError("no public train prompts were found")
    if max_prompts is not None:
        rows = rows[:max_prompts]
    required = {
        "prompt_id",
        "family_id",
        "input_ids",
        "target_token_id",
        "distractor_token_id_1",
        "distractor_token_id_2",
        "source_key_position",
        "source_value_position",
        "query_position",
        "sequence_length",
    }
    seen: set[str] = set()
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Qwen train prompt table is missing required columns")
        prompt_id = str(row["prompt_id"]).strip()
        if not prompt_id or prompt_id in seen:
            raise ValueError("Qwen train prompt IDs must be non-empty and unique")
        seen.add(prompt_id)
        tokens = tuple(int(item) for item in str(row["input_ids"]).split())
        if len(tokens) != _integer(row, "sequence_length"):
            raise ValueError(f"prompt {prompt_id} has inconsistent sequence length")
    return tuple(rows)


def load_qwen_reference_means(path: str | Path) -> HeadAblationMeans:
    """Load the frozen selected-head family means in their stored order."""

    with np.load(Path(path), allow_pickle=False) as payload:
        expected = {"family_ids", "layers", "heads", "kv_groups", "values", "counts"}
        if set(payload.files) != expected:
            raise ValueError("Qwen reference-mean cache has unexpected arrays")
        family_ids = tuple(str(value) for value in payload["family_ids"].tolist())
        layers = tuple(int(value) for value in payload["layers"].tolist())
        heads = tuple(int(value) for value in payload["heads"].tolist())
        kv_groups = tuple(int(value) for value in payload["kv_groups"].tolist())
        values = np.asarray(payload["values"], dtype=np.float32)
        counts = tuple(int(value) for value in payload["counts"].tolist())
    if not (
        len(layers)
        == len(heads)
        == len(kv_groups)
        == QWEN_INDUCTION_N_HEADS
    ):
        raise ValueError("Qwen reference means must contain exactly 8 heads")
    return HeadAblationMeans(
        family_ids=family_ids,
        heads=tuple(
            HeadRef(layer=layer, head=head, kv_group=kv_group)
            for layer, head, kv_group in zip(layers, heads, kv_groups)
        ),
        values=values,
        counts=counts,
    )


def measure_qwen_induction_attribution_map(
    plant: Any,
    prompts: Sequence[Mapping[str, str]],
    means: HeadAblationMeans,
    *,
    model_revision: str = QWEN_INDUCTION_MODEL_REVISION,
    prompt_split: str = "train",
    batch_size: int = 4,
) -> QwenInductionAttributionMap:
    """Measure clean-state AtP for the exact Copy-v2 finite intervention."""

    torch = plant.torch
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = tuple(prompts)
    if not records:
        raise ValueError("at least one Qwen attribution prompt is required")
    heads = tuple(means.heads)
    if len(heads) != QWEN_INDUCTION_N_HEADS:
        raise ValueError("Qwen attribution measurement requires exactly 8 heads")
    checked = tuple(plant._checked_heads(heads))
    if checked != heads:
        raise ValueError("Qwen selected-head order changed during validation")
    if means.head_dim != int(plant.architecture.head_dim):
        raise ValueError("Qwen reference means and model head dimension differ")
    unknown_families = sorted(
        {_family_id(row) for row in records} - set(means.family_ids)
    )
    if unknown_families:
        raise ValueError(f"Qwen prompts have no reference mean: {unknown_families}")

    for parameter in plant.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    by_layer: dict[int, list[tuple[int, HeadRef]]] = {}
    for component, head in enumerate(heads):
        by_layer.setdefault(head.layer, []).append((component, head))
    selected_layers = tuple(sorted(by_layer))
    earliest_layer = selected_layers[0]
    totals = np.zeros(QWEN_INDUCTION_N_HEADS, dtype=np.float64)
    measured = 0

    for batch in plant._batches(records, batch_size):
        captured: dict[int, Any] = {}
        handles = []
        for layer in selected_layers:
            def capture(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                layer: int = layer,
            ) -> tuple[Any, ...] | None:
                hidden = inputs[0]
                if layer == earliest_layer:
                    hidden = hidden.detach().requires_grad_(True)
                    captured[layer] = hidden
                    return (hidden, *inputs[1:])
                if not hidden.requires_grad:
                    raise RuntimeError("Qwen suffix graph was not connected to the first hook")
                captured[layer] = hidden
                return None

            handles.append(
                plant.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
                    capture
                )
            )
        try:
            with torch.enable_grad():
                output = plant.model(
                    input_ids=plant._inputs(batch),
                    use_cache=False,
                    logits_to_keep=1,
                )
                logits = output.logits
                if logits.ndim != 3 or logits.shape[0] != len(batch):
                    raise ValueError("Qwen attribution run returned unexpected logits")
                if logits.shape[1] != 1:
                    rows = torch.arange(len(batch), device=logits.device)
                    positions = torch.as_tensor(
                        [_query_position(row) for row in batch],
                        dtype=torch.long,
                        device=logits.device,
                    )
                    final_logits = logits[rows, positions].float()
                else:
                    final_logits = logits[:, 0].float()
                targets = torch.as_tensor(
                    [_target_token(row) for row in batch],
                    dtype=torch.long,
                    device=logits.device,
                )
                distractors = torch.as_tensor(
                    [_distractor_tokens(row) for row in batch],
                    dtype=torch.long,
                    device=logits.device,
                )
                target_logits = final_logits.gather(1, targets[:, None]).squeeze(1)
                distractor_logits = final_logits.gather(1, distractors)
                margins = target_logits - (
                    torch.logsumexp(distractor_logits, dim=1) - math.log(2.0)
                )
                if set(captured) != set(selected_layers):
                    raise RuntimeError("not every selected Qwen layer was captured")
                gradients = torch.autograd.grad(
                    margins.sum(),
                    tuple(captured[layer] for layer in selected_layers),
                    allow_unused=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        gradient_by_layer = dict(zip(selected_layers, gradients))
        positions = torch.as_tensor(
            [_query_position(row) for row in batch],
            dtype=torch.long,
            device=plant.device,
        )
        rows = torch.arange(len(batch), device=plant.device)
        family_rows = torch.as_tensor(
            [means.family_index(_family_id(row)) for row in batch],
            dtype=torch.long,
            device=plant.device,
        )
        reference_values = torch.as_tensor(
            means.values,
            dtype=torch.float32,
            device=plant.device,
        )
        batch_effects = torch.zeros(
            (len(batch), QWEN_INDUCTION_N_HEADS),
            dtype=torch.float64,
            device=plant.device,
        )
        for layer, entries in by_layer.items():
            hidden = captured[layer]
            gradient = gradient_by_layer[layer]
            clean_z = hidden[rows, positions].reshape(
                len(batch),
                plant.architecture.n_query_heads,
                plant.architecture.head_dim,
            )
            gradient_z = gradient[rows, positions].reshape_as(clean_z)
            for component, head in entries:
                reference = reference_values[family_rows, component].to(
                    dtype=clean_z.dtype
                )
                displacement = (clean_z[:, head.head] - reference).float()
                contribution = (
                    gradient_z[:, head.head].float() * displacement
                ).sum(dim=-1)
                batch_effects[:, component] = contribution.double()
        totals += batch_effects.detach().cpu().numpy().sum(axis=0)
        measured += len(batch)
        del output, logits, gradients, captured, gradient_by_layer, batch_effects

    labels = tuple(head.label for head in heads)
    return QwenInductionAttributionMap(
        head_effects=tuple(float(value) for value in totals / measured),
        head_labels=labels,
        n_prompts=measured,
        prompt_split=prompt_split,
        model_revision=model_revision,
        intervention=(
            "final-query family-conditioned mean replacement at query-head z "
            "before the attention output projection"
        ),
    )


__all__ = [
    "QWEN_ATP_BASELINE_VERSION",
    "QWEN_CALIBRATED_ATP_BASELINE_NAME",
    "QWEN_RAW_ATP_BASELINE_NAME",
    "QwenInductionAttributionMap",
    "QwenInductionAttributionPatchingBaseline",
    "load_qwen_reference_means",
    "load_qwen_train_prompts",
    "measure_qwen_induction_attribution_map",
    "qwen_induction_attribution_patching_card",
]
