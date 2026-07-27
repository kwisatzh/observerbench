"""Native-Hugging-Face producer for exact Qwen induction interventions.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The public benchmark loader never imports this module. This optional producer
owns model loading, hook validation, clean scoring, head discovery, reference
means, and finite mean-ablation measurements.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class HeadRef:
    layer: int
    head: int
    kv_group: int

    @property
    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True)
class Qwen2Architecture:
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    qkv_bias_present: bool
    output_bias_present: bool


@dataclass(frozen=True)
class CleanScore:
    prompt_id: str
    family_id: str
    candidate_margin: float
    candidate_correct: bool
    top1_correct: bool
    target_nll: float
    target_token_id: int
    candidate_predicted_token_id: int
    predicted_token_id: int
    candidate_logits_finite: bool


@dataclass(frozen=True)
class EffectRow:
    prompt_id: str
    family_id: str
    mask_id: str
    mask_bits: str
    clean_margin: float
    ablated_margin: float
    drop_from_clean: float
    clean_candidate_correct: bool
    ablated_candidate_correct: bool
    clean_top1_correct: bool
    ablated_top1_correct: bool
    clean_target_nll: float
    ablated_target_nll: float


@dataclass(frozen=True)
class HeadDiscoveryScore:
    layer: int
    head: int
    kv_group: int
    attention_to_target: float
    attention_to_distractors: float
    attention_specificity: float
    direct_logit_attribution: float
    output_norm: float
    n_prompts: int


@dataclass(frozen=True)
class HeadAblationMeans:
    """Family-conditioned final-query z means in supplied head order."""

    family_ids: tuple[str, ...]
    heads: tuple[HeadRef, ...]
    values: np.ndarray
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 3 or values.shape[:2] != (
            len(self.family_ids),
            len(self.heads),
        ):
            raise ValueError("reference means must have family-by-head-by-head_dim shape")
        if len(self.counts) != len(self.family_ids) or any(count <= 0 for count in self.counts):
            raise ValueError("reference means require a positive count per family")
        if len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("reference-mean family IDs must be unique")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("reference-mean heads must be unique")
        if not np.isfinite(values).all():
            raise ValueError("reference means must be finite")

    @property
    def head_dim(self) -> int:
        return int(self.values.shape[2])

    def family_index(self, family_id: str) -> int:
        try:
            return self.family_ids.index(str(family_id))
        except ValueError as error:
            raise KeyError(f"unknown reference-mean family {family_id!r}") from error


def _record_value(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    raise ValueError(f"record lacks one of: {', '.join(names)}")


def _prompt_id(record: Any) -> str:
    return str(_record_value(record, "prompt_id", "example_id"))


def _family_id(record: Any) -> str:
    return str(_record_value(record, "family_id"))


def _tokens(record: Any) -> tuple[int, ...]:
    value = _record_value(record, "tokens", "input_ids")
    if isinstance(value, str):
        result = tuple(int(item) for item in value.split())
    else:
        result = tuple(int(item) for item in value)
    if not result or any(item < 0 for item in result):
        raise ValueError("input token IDs must be a nonempty nonnegative sequence")
    return result


def _query_position(record: Any) -> int:
    return int(_record_value(record, "final_key_position", "query_position"))


def _target_token(record: Any) -> int:
    return int(_record_value(record, "target_value_token", "target_token_id"))


def _distractor_tokens(record: Any) -> tuple[int, int]:
    try:
        value = _record_value(record, "distractor_value_tokens")
        result = tuple(int(item) for item in value)
    except ValueError:
        result = (
            int(_record_value(record, "distractor_token_id_1")),
            int(_record_value(record, "distractor_token_id_2")),
        )
    if len(result) != 2 or len(set(result)) != 2:
        raise ValueError("each induction prompt requires two distinct distractors")
    return result[0], result[1]


def _source_value_positions(record: Any) -> tuple[int, int, int]:
    try:
        raw_positions = _record_value(record, "key_positions")
        if isinstance(raw_positions, str):
            raw_positions = raw_positions.split()
        positions = tuple(int(item) + 1 for item in raw_positions)
    except ValueError:
        positions = (int(_record_value(record, "source_value_position")),)
    if len(positions) == 1:
        # Adapter-facing prompts retain only the target source position. The
        # producer uses full design records for attention discovery.
        raise ValueError("attention discovery requires all three key_positions")
    if len(positions) != 3:
        raise ValueError("attention discovery requires three source-value positions")
    return positions


def _validate_fixture_record(record: Any) -> None:
    """Reject any producer input that changes the exact next-token estimand."""

    tokens = _tokens(record)
    query = _query_position(record)
    if query != len(tokens) - 1:
        raise ValueError("induction query must be the final supplied token")
    try:
        source_key = int(
            _record_value(record, "source_key_position", "target_key_position")
        )
    except ValueError:
        raw_key_positions = _record_value(record, "key_positions")
        if isinstance(raw_key_positions, str):
            raw_key_positions = raw_key_positions.split()
        key_positions = tuple(map(int, raw_key_positions))
        if len(key_positions) != 3:
            raise ValueError("fixture must declare its target source key") from None
        source_key = key_positions[0]
    if source_key < 0 or source_key + 1 >= query:
        raise ValueError("target source bigram is outside the causal prefix")
    target = _target_token(record)
    if tokens[source_key + 1] != target or tokens[query] != tokens[source_key]:
        raise ValueError("target source bigram or repeated query key is inconsistent")
    distractors = _distractor_tokens(record)
    if len({target, *distractors}) != 3:
        raise ValueError("target and distractor continuation tokens must be distinct")
    try:
        source_values = _source_value_positions(record)
    except ValueError as error:
        if "requires all three key_positions" in str(error):
            return
        raise
    if any(position <= 0 or position >= query for position in source_values):
        raise ValueError("source value positions must lie in the causal prefix")
    observed = tuple(tokens[position] for position in source_values)
    if observed[0] != target or observed[1:] != distractors:
        raise ValueError("all-three source positions disagree with declared candidates")


def regular_token_pool(
    tokenizer: Any,
    *,
    seed: int,
    limit: int,
    candidate_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return printable, regular, non-special, single-token round trips."""

    if limit <= 0:
        raise ValueError("regular-token limit must be positive")
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    candidates = np.asarray(
        tuple(range(vocab_size)) if candidate_ids is None else tuple(map(int, candidate_ids)),
        dtype=int,
    )
    if len(candidates) != len(set(candidates.tolist())):
        raise ValueError("candidate token IDs must be unique")
    rng = np.random.default_rng(int(seed))
    candidates = candidates[rng.permutation(len(candidates))]
    specials = set(map(int, getattr(tokenizer, "all_special_ids", ())))
    accepted: list[int] = []
    for raw_token_id in candidates:
        token_id = int(raw_token_id)
        if token_id < 0 or token_id >= vocab_size or token_id in specials:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or not text or not text.isprintable() or "\ufffd" in text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if tuple(map(int, encoded)) != (token_id,):
            continue
        accepted.append(token_id)
        if len(accepted) == limit:
            return tuple(sorted(accepted))
    raise ValueError(
        f"only {len(accepted)} printable round-tripping regular tokens were found; "
        f"{limit} are required"
    )


def validate_qwen2_architecture(model: Any) -> Qwen2Architecture:
    """Audit the exact Qwen2 head geometry and Q/K/V bias tensors."""

    config = model.config
    n_layers = int(config.num_hidden_layers)
    n_query_heads = int(config.num_attention_heads)
    n_kv_heads = int(config.num_key_value_heads)
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // n_query_heads))
    if n_layers <= 0 or n_query_heads <= 0 or n_kv_heads <= 0:
        raise ValueError("invalid Qwen2 architecture dimensions")
    if n_query_heads % n_kv_heads != 0 or head_dim * n_query_heads != hidden_size:
        raise ValueError("Qwen2 query/KV grouping or head dimension is inconsistent")
    layers = tuple(model.model.layers)
    if len(layers) != n_layers:
        raise ValueError("Qwen2 layer module count differs from config")
    qkv_bias = all(
        layer.self_attn.q_proj.bias is not None
        and layer.self_attn.k_proj.bias is not None
        and layer.self_attn.v_proj.bias is not None
        for layer in layers
    )
    output_bias = any(layer.self_attn.o_proj.bias is not None for layer in layers)
    if not qkv_bias:
        raise ValueError("pinned Qwen2 producer requires Q/K/V biases")
    return Qwen2Architecture(
        n_layers=n_layers,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        qkv_bias_present=qkv_bias,
        output_bias_present=output_bias,
    )


def _torch_dtype(torch: Any, name: str) -> Any:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported Qwen producer dtype: {name}") from error


def _resolve_device(torch: Any, requested: str) -> Any:
    device = str(requested)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def _mask_record(mask: Any, n_heads: int) -> tuple[str, tuple[int, ...]]:
    bits_value = getattr(mask, "bits", mask)
    if isinstance(bits_value, str):
        bits = tuple(int(bit) for bit in bits_value)
    else:
        bits = tuple(int(bit) for bit in bits_value)
    if len(bits) != n_heads or set(bits) - {0, 1}:
        raise ValueError(f"intervention masks must contain {n_heads} binary entries")
    mask_id = str(getattr(mask, "mask_id", f"mask_{''.join(map(str, bits))}"))
    return mask_id, bits


class Qwen2InductionPlant:
    """Thin raw-HF plant with query-head z interventions."""

    def __init__(self, model: Any, tokenizer: Any, *, device: Any) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - base project depends on torch
            raise ImportError("Qwen induction production requires torch") from error
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        self.architecture = validate_qwen2_architecture(model)
        self._batch_context: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        local_files_only: bool = False,
    ) -> "Qwen2InductionPlant":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Qwen production requires the optional qwen dependencies"
            ) from error
        resolved = _resolve_device(torch, device)
        resolved_dtype = _torch_dtype(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=resolved_dtype,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        plant = cls(model, tokenizer, device=resolved)
        plant.model_id = str(model_id)
        plant.model_revision = str(revision)
        plant.resolved_model_revision = getattr(model.config, "_commit_hash", None)
        plant.attention_implementation = str(attention_implementation)
        return plant

    def audit_runtime(
        self,
        *,
        expected_model_id: str,
        expected_revision: str,
        expected_layers: int,
        expected_query_heads: int,
        expected_kv_heads: int,
        expected_dtype: str,
        expected_attention_implementation: str,
    ) -> dict[str, Any]:
        """Fail closed on the pinned scientific checkpoint and execution mode."""

        requested_model = getattr(self, "model_id", None)
        requested_revision = getattr(self, "model_revision", None)
        resolved_revision = getattr(self, "resolved_model_revision", None)
        attention = getattr(
            self,
            "attention_implementation",
            getattr(self.model.config, "_attn_implementation", None),
        )
        if requested_model != str(expected_model_id):
            raise ValueError("loaded Qwen model ID differs from the frozen config")
        if requested_revision != str(expected_revision):
            raise ValueError("requested Qwen revision differs from the frozen config")
        if resolved_revision != str(expected_revision):
            raise ValueError("resolved Qwen checkpoint hash differs from the frozen revision")
        if (
            self.architecture.n_layers != int(expected_layers)
            or self.architecture.n_query_heads != int(expected_query_heads)
            or self.architecture.n_kv_heads != int(expected_kv_heads)
        ):
            raise ValueError("loaded Qwen architecture differs from the frozen config")
        if str(attention) != str(expected_attention_implementation):
            raise ValueError("loaded attention implementation differs from the stage config")
        parameter_dtypes = {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in self.model.parameters()
        }
        if parameter_dtypes != {str(expected_dtype)}:
            raise ValueError(
                "loaded Qwen parameter dtype differs from the unquantized frozen config"
            )
        quantization_config = getattr(self.model.config, "quantization_config", None)
        if quantization_config is not None or getattr(self.model, "hf_quantizer", None) is not None:
            raise ValueError("the scientific Qwen producer forbids quantized weights")
        return {
            "model_id": requested_model,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "n_layers": self.architecture.n_layers,
            "n_query_heads": self.architecture.n_query_heads,
            "n_kv_heads": self.architecture.n_kv_heads,
            "parameter_dtype": next(iter(parameter_dtypes)),
            "attention_implementation": str(attention),
            "quantized": False,
        }

    def _batches(
        self,
        records: Sequence[Any],
        batch_size: int,
    ) -> Iterator[list[Any]]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        buckets: dict[int, list[Any]] = {}
        for record in records:
            buckets.setdefault(len(_tokens(record)), []).append(record)
        for length in sorted(buckets):
            rows = buckets[length]
            for start in range(0, len(rows), batch_size):
                yield rows[start : start + batch_size]

    def _inputs(self, records: Sequence[Any]) -> Any:
        torch = self.torch
        for record in records:
            _validate_fixture_record(record)
        ids = torch.as_tensor(
            [_tokens(record) for record in records],
            dtype=torch.long,
            device=self.device,
        )
        return ids

    def _scores_from_logits(self, records: Sequence[Any], logits: Any) -> list[CleanScore]:
        torch = self.torch
        rows = torch.arange(len(records), device=logits.device)
        positions = torch.as_tensor(
            [_query_position(record) for record in records],
            dtype=torch.long,
            device=logits.device,
        )
        if logits.shape[1] == 1:
            if any(
                _query_position(record) != len(_tokens(record)) - 1
                for record in records
            ):
                raise ValueError("last-logit scoring requires a final-position query")
            final_logits = logits[:, 0].float()
        else:
            final_logits = logits[rows, positions].float()
        targets = torch.as_tensor(
            [_target_token(record) for record in records],
            dtype=torch.long,
            device=logits.device,
        )
        distractors = torch.as_tensor(
            [_distractor_tokens(record) for record in records],
            dtype=torch.long,
            device=logits.device,
        )
        candidate_tokens = torch.column_stack([targets, distractors])
        candidate_logits = final_logits.gather(1, candidate_tokens)
        candidate_logits_finite = torch.isfinite(candidate_logits).all(dim=1)
        target_logits = candidate_logits[:, 0]
        distractor_logits = candidate_logits[:, 1:]
        margins = target_logits - (
            torch.logsumexp(distractor_logits, dim=1) - math.log(2.0)
        )
        candidate_prediction = candidate_tokens.gather(
            1, candidate_logits.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        candidate_correct = candidate_prediction == targets
        predicted = final_logits.argmax(dim=1)
        top1 = predicted == targets
        nll = -torch.log_softmax(final_logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return [
            CleanScore(
                prompt_id=_prompt_id(record),
                family_id=_family_id(record),
                candidate_margin=float(margin.detach().cpu()),
                candidate_correct=bool(candidate.detach().cpu()),
                top1_correct=bool(correct.detach().cpu()),
                target_nll=float(loss.detach().cpu()),
                target_token_id=int(target.detach().cpu()),
                candidate_predicted_token_id=int(candidate_prediction_token.detach().cpu()),
                predicted_token_id=int(prediction.detach().cpu()),
                candidate_logits_finite=bool(finite.detach().cpu()),
            )
            for record, margin, candidate, correct, loss, target, candidate_prediction_token, prediction, finite in zip(
                records,
                margins,
                candidate_correct,
                top1,
                nll,
                targets,
                candidate_prediction,
                predicted,
                candidate_logits_finite,
            )
        ]

    def score_clean(self, records: Sequence[Any], *, batch_size: int = 16) -> list[CleanScore]:
        scores: list[CleanScore] = []
        with self.torch.inference_mode():
            for batch in self._batches(tuple(records), batch_size):
                output = self.model(
                    input_ids=self._inputs(batch),
                    use_cache=False,
                    logits_to_keep=1,
                )
                scores.extend(self._scores_from_logits(batch, output.logits))
        return scores

    def _checked_heads(self, heads: Sequence[HeadRef]) -> tuple[HeadRef, ...]:
        result = tuple(heads)
        if not result or len(set(result)) != len(result):
            raise ValueError("head panel must be nonempty and unique")
        groups = self.architecture.n_query_heads // self.architecture.n_kv_heads
        for head in result:
            if not 0 <= head.layer < self.architecture.n_layers:
                raise ValueError(f"head layer is out of range: {head}")
            if not 0 <= head.head < self.architecture.n_query_heads:
                raise ValueError(f"query-head index is out of range: {head}")
            if head.kv_group != head.head // groups:
                raise ValueError(f"query-head KV-group label is wrong: {head}")
        return result

    def all_heads(self) -> tuple[HeadRef, ...]:
        group_size = self.architecture.n_query_heads // self.architecture.n_kv_heads
        return tuple(
            HeadRef(layer, head, head // group_size)
            for layer in range(self.architecture.n_layers)
            for head in range(self.architecture.n_query_heads)
        )

    def capture_reference_means(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        *,
        batch_size: int = 16,
    ) -> HeadAblationMeans:
        heads = self._checked_heads(heads)
        family_ids = tuple(sorted({_family_id(record) for record in records}))
        family_index = {family: index for index, family in enumerate(family_ids)}
        head_index = {head: index for index, head in enumerate(heads)}
        values = self.torch.zeros(
            (len(family_ids), len(heads), self.architecture.head_dim),
            dtype=self.torch.float32,
            device=self.device,
        )
        counts = self.torch.zeros(
            len(family_ids), dtype=self.torch.long, device=self.device
        )
        handles = []
        heads_by_layer: dict[int, list[HeadRef]] = {}
        for head in heads:
            heads_by_layer.setdefault(head.layer, []).append(head)

        for layer, layer_heads in heads_by_layer.items():
            def hook(_module: Any, inputs: tuple[Any, ...], layer_heads=tuple(layer_heads)) -> None:
                hidden = inputs[0]
                context = self._batch_context
                if context is None:
                    raise RuntimeError("reference hook fired outside an active batch")
                rows = self.torch.arange(hidden.shape[0], device=hidden.device)
                positions = context["query_positions"]
                query_z = hidden[rows, positions].reshape(
                    hidden.shape[0],
                    self.architecture.n_query_heads,
                    self.architecture.head_dim,
                )
                family_rows = context["family_rows"]
                for head in layer_heads:
                    values[:, head_index[head]].index_add_(
                        0, family_rows, query_z[:, head.head].detach().float()
                    )

            handles.append(
                self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
            )
        try:
            with self.torch.inference_mode():
                for batch in self._batches(tuple(records), batch_size):
                    family_rows = self.torch.as_tensor(
                        [family_index[_family_id(record)] for record in batch],
                        dtype=self.torch.long,
                        device=self.device,
                    )
                    self._batch_context = {
                        "family_rows": family_rows,
                        "query_positions": self.torch.as_tensor(
                            [_query_position(record) for record in batch],
                            dtype=self.torch.long,
                            device=self.device,
                        ),
                    }
                    self.model.model(input_ids=self._inputs(batch), use_cache=False)
                    counts += self.torch.bincount(
                        family_rows, minlength=len(family_ids)
                    )
        finally:
            self._batch_context = None
            for handle in handles:
                handle.remove()
        values /= counts[:, None, None]
        return HeadAblationMeans(
            family_ids=family_ids,
            heads=heads,
            values=values.detach().cpu().numpy(),
            counts=tuple(map(int, counts.detach().cpu().tolist())),
        )

    @contextmanager
    def head_intervention(
        self,
        heads: Sequence[HeadRef],
        *,
        mask_rows: np.ndarray,
        query_positions: Sequence[int],
        family_ids: Sequence[str],
        mode: str = "mean",
        position_scope: str = "final",
        means: HeadAblationMeans | None = None,
    ) -> Iterator[None]:
        heads = self._checked_heads(heads)
        masks = np.asarray(mask_rows, dtype=np.uint8)
        if masks.ndim != 2 or masks.shape[1] != len(heads) or not np.isin(masks, (0, 1)).all():
            raise ValueError("batch intervention masks have the wrong binary shape")
        if len(query_positions) != len(masks) or len(family_ids) != len(masks):
            raise ValueError("intervention batch metadata lengths differ")
        if mode not in {"mean", "zero", "noop"}:
            raise ValueError("head intervention mode must be mean, zero, or noop")
        if position_scope not in {"final", "all"}:
            raise ValueError("head intervention position scope must be final or all")
        if mode == "mean":
            if means is None or means.heads != heads:
                raise ValueError("mean ablation requires aligned family-conditioned means")
            if means.head_dim != self.architecture.head_dim:
                raise ValueError("mean-ablation head dimension differs from model")
        family_rows = None
        mean_values = None
        if mode == "mean":
            family_rows = self.torch.as_tensor(
                [means.family_index(str(family)) for family in family_ids],
                dtype=self.torch.long,
                device=self.device,
            )
            mean_values = self.torch.as_tensor(
                means.values, dtype=self.torch.float32, device=self.device
            )
        self._batch_context = {
            "mask_rows": self.torch.as_tensor(
                masks, dtype=self.torch.bool, device=self.device
            ),
            "query_positions": self.torch.as_tensor(
                query_positions, dtype=self.torch.long, device=self.device
            ),
            "family_rows": family_rows,
            "mean_values": mean_values,
        }
        handles = []
        by_layer: dict[int, list[tuple[int, HeadRef]]] = {}
        for component, head in enumerate(heads):
            by_layer.setdefault(head.layer, []).append((component, head))
        for layer, entries in by_layer.items():
            def hook(
                _module: Any,
                inputs: tuple[Any, ...],
                entries=tuple(entries),
            ) -> tuple[Any, ...] | None:
                if mode == "noop":
                    return None
                hidden = inputs[0].clone()
                context = self._batch_context
                if context is None:
                    raise RuntimeError("head intervention hook has no active batch")
                for component, head in entries:
                    active = self.torch.nonzero(
                        context["mask_rows"][:, component], as_tuple=False
                    ).squeeze(1)
                    if active.numel() == 0:
                        continue
                    start = head.head * self.architecture.head_dim
                    stop = start + self.architecture.head_dim
                    if position_scope == "all":
                        if mode != "zero":
                            raise ValueError("all-position replacement is defined only for zero robustness")
                        hidden[active, :, start:stop] = 0
                        continue
                    position_tensor = context["query_positions"][active]
                    if mode == "zero":
                        hidden[active, position_tensor, start:stop] = 0
                    else:
                        replacement = context["mean_values"][
                            context["family_rows"][active], component
                        ]
                        hidden[active, position_tensor, start:stop] = replacement.to(
                            dtype=hidden.dtype
                        )
                return (hidden, *inputs[1:])

            handles.append(
                self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
            )
        try:
            yield
        finally:
            self._batch_context = None
            for handle in handles:
                handle.remove()

    def _score_with_mask(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        mask: Sequence[int],
        means: HeadAblationMeans | None,
        *,
        batch_size: int,
        mode: str,
        position_scope: str,
    ) -> list[CleanScore]:
        scores: list[CleanScore] = []
        for batch in self._batches(tuple(records), batch_size):
            batch_masks = np.tile(np.asarray(mask, dtype=np.uint8), (len(batch), 1))
            with self.head_intervention(
                heads,
                mask_rows=batch_masks,
                query_positions=[_query_position(record) for record in batch],
                family_ids=[_family_id(record) for record in batch],
                mode=mode,
                position_scope=position_scope,
                means=means,
            ), self.torch.inference_mode():
                output = self.model(
                    input_ids=self._inputs(batch),
                    use_cache=False,
                    logits_to_keep=1,
                )
            scores.extend(self._scores_from_logits(batch, output.logits))
        return scores

    def noop_hook_parity(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        means: HeadAblationMeans,
        *,
        batch_size: int = 16,
    ) -> dict[str, float | bool]:
        """Compare hook-free scoring with the primary mean hook at zero mask."""

        clean = self.score_clean(records, batch_size=batch_size)
        hooked = self._score_with_mask(
            records,
            heads,
            (0,) * len(heads),
            means,
            batch_size=batch_size,
            mode="mean",
            position_scope="final",
        )
        if [row.prompt_id for row in clean] != [row.prompt_id for row in hooked]:
            raise AssertionError("no-op parity changed prompt order")
        maximum = max(
            abs(left.candidate_margin - right.candidate_margin)
            for left, right in zip(clean, hooked)
        )
        exact = all(
            left.candidate_correct == right.candidate_correct
            and left.candidate_predicted_token_id
            == right.candidate_predicted_token_id
            and left.predicted_token_id == right.predicted_token_id
            for left, right in zip(clean, hooked)
        )
        return {"maximum_margin_error": float(maximum), "prediction_parity": bool(exact)}

    def measure_mask_effects(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        masks: Sequence[Any],
        means: HeadAblationMeans,
        *,
        batch_size: int = 16,
        mode: str = "mean",
        position_scope: str = "final",
    ) -> list[EffectRow]:
        return [
            row
            for _mask_id, rows in self.iter_mask_effects(
                records,
                heads,
                masks,
                means,
                batch_size=batch_size,
                mode=mode,
                position_scope=position_scope,
            )
            for row in rows
        ]

    def iter_mask_effects(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        masks: Sequence[Any],
        means: HeadAblationMeans,
        *,
        batch_size: int = 16,
        mode: str = "mean",
        position_scope: str = "final",
    ) -> Iterator[tuple[str, list[EffectRow]]]:
        """Yield one completed mask at a time for resumable serialization."""

        heads = self._checked_heads(heads)
        clean_rows = self.score_clean(records, batch_size=batch_size)
        clean = {row.prompt_id: row for row in clean_rows}
        for raw_mask in masks:
            mask_id, bits = _mask_record(raw_mask, len(heads))
            if not any(bits):
                ablated_rows = clean_rows
            else:
                ablated_rows = self._score_with_mask(
                    records,
                    heads,
                    bits,
                    means,
                    batch_size=batch_size,
                    mode=mode,
                    position_scope=position_scope,
                )
            mask_output: list[EffectRow] = []
            for ablated in ablated_rows:
                baseline = clean[ablated.prompt_id]
                mask_output.append(
                    EffectRow(
                        prompt_id=ablated.prompt_id,
                        family_id=ablated.family_id,
                        mask_id=mask_id,
                        mask_bits="".join(map(str, bits)),
                        clean_margin=baseline.candidate_margin,
                        ablated_margin=ablated.candidate_margin,
                        drop_from_clean=baseline.candidate_margin - ablated.candidate_margin,
                        clean_candidate_correct=baseline.candidate_correct,
                        ablated_candidate_correct=ablated.candidate_correct,
                        clean_top1_correct=baseline.top1_correct,
                        ablated_top1_correct=ablated.top1_correct,
                        clean_target_nll=baseline.target_nll,
                        ablated_target_nll=ablated.target_nll,
                    )
                )
            yield mask_id, mask_output

    def measure_single_head_effects(
        self,
        records: Sequence[Any],
        heads: Sequence[HeadRef],
        means: HeadAblationMeans,
        *,
        batch_size: int = 16,
    ) -> list[EffectRow]:
        identity = np.eye(len(heads), dtype=np.uint8)
        return self.measure_mask_effects(
            records,
            heads,
            [tuple(map(int, row)) for row in identity],
            means,
            batch_size=batch_size,
        )

    def scan_attention(
        self,
        records: Sequence[Any],
        *,
        batch_size: int = 4,
    ) -> list[HeadDiscoveryScore]:
        """Scan every query head using attention specificity and DLA diagnostics."""

        if getattr(self, "attention_implementation", "eager") != "eager":
            raise ValueError("attention discovery requires an eager-attention plant")
        n_layers = self.architecture.n_layers
        n_heads = self.architecture.n_query_heads
        target_sum = self.torch.zeros(
            (n_layers, n_heads), dtype=self.torch.float32, device=self.device
        )
        distractor_sum = self.torch.zeros_like(target_sum)
        dla_sum = self.torch.zeros_like(target_sum)
        norm_sum = self.torch.zeros_like(target_sum)
        count = 0
        handles = []

        for layer_index, layer in enumerate(self.model.model.layers):
            def attention_hook(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: tuple[Any, Any],
                layer_index=layer_index,
            ) -> None:
                weights = output[1]
                if weights is None:
                    raise RuntimeError("eager attention did not return attention weights")
                context = self._batch_context
                rows = self.torch.arange(weights.shape[0], device=weights.device)
                query = context["query_positions"]
                target_positions = context["source_positions"][:, 0]
                distractor_positions = context["source_positions"][:, 1:]
                target_values = weights[rows, :, query, target_positions]
                first = weights[rows, :, query, distractor_positions[:, 0]]
                second = weights[rows, :, query, distractor_positions[:, 1]]
                target_sum[layer_index].add_(target_values.detach().float().sum(0))
                distractor_sum[layer_index].add_(
                    (0.5 * (first + second)).detach().float().sum(0)
                )

            handles.append(layer.self_attn.register_forward_hook(attention_hook))

            def z_hook(
                module: Any,
                inputs: tuple[Any, ...],
                layer_index=layer_index,
            ) -> None:
                hidden = inputs[0]
                context = self._batch_context
                rows = self.torch.arange(hidden.shape[0], device=hidden.device)
                query = context["query_positions"]
                z = hidden[rows, query].reshape(
                    hidden.shape[0], n_heads, self.architecture.head_dim
                )
                contrast = context["unembed_contrast"].to(dtype=hidden.dtype)
                weight = module.weight.T.reshape(
                    n_heads, self.architecture.head_dim, self.architecture.hidden_size
                )
                projected = self.torch.einsum("bo,hdo->bhd", contrast, weight)
                dla = (z * projected).sum(-1)
                residual = self.torch.einsum("bhd,hdo->bho", z, weight)
                norms = residual.float().norm(dim=-1)
                dla_sum[layer_index].add_(dla.detach().float().sum(0))
                norm_sum[layer_index].add_(norms.detach().float().sum(0))

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(z_hook))

        try:
            with self.torch.inference_mode():
                for batch in self._batches(tuple(records), batch_size):
                    targets = self.torch.as_tensor(
                        [_target_token(record) for record in batch],
                        dtype=self.torch.long,
                        device=self.model.lm_head.weight.device,
                    )
                    distractors = self.torch.as_tensor(
                        [_distractor_tokens(record) for record in batch],
                        dtype=self.torch.long,
                        device=self.model.lm_head.weight.device,
                    )
                    unembed = self.model.lm_head.weight.detach()
                    contrast = unembed[targets] - unembed[distractors].mean(dim=1)
                    self._batch_context = {
                        "query_positions": self.torch.as_tensor(
                            [_query_position(record) for record in batch],
                            dtype=self.torch.long,
                            device=self.device,
                        ),
                        "source_positions": self.torch.as_tensor(
                            [_source_value_positions(record) for record in batch],
                            dtype=self.torch.long,
                            device=self.device,
                        ),
                        "unembed_contrast": contrast,
                    }
                    self.model.model(input_ids=self._inputs(batch), use_cache=False)
                    count += len(batch)
        finally:
            self._batch_context = None
            for handle in handles:
                handle.remove()
        if count == 0:
            raise ValueError("attention discovery requires at least one prompt")
        target_values = (target_sum / count).detach().cpu().numpy()
        distractor_values = (distractor_sum / count).detach().cpu().numpy()
        dla_values = (dla_sum / count).detach().cpu().numpy()
        norm_values = (norm_sum / count).detach().cpu().numpy()
        group_size = n_heads // self.architecture.n_kv_heads
        rows: list[HeadDiscoveryScore] = []
        for layer in range(n_layers):
            for head in range(n_heads):
                target = target_values[layer, head]
                distractor = distractor_values[layer, head]
                rows.append(
                    HeadDiscoveryScore(
                        layer=layer,
                        head=head,
                        kv_group=head // group_size,
                        attention_to_target=float(target),
                        attention_to_distractors=float(distractor),
                        attention_specificity=float(target - distractor),
                        direct_logit_attribution=float(dla_values[layer, head]),
                        output_norm=float(norm_values[layer, head]),
                        n_prompts=count,
                    )
                )
        return rows


__all__ = [
    "CleanScore",
    "EffectRow",
    "HeadAblationMeans",
    "HeadDiscoveryScore",
    "HeadRef",
    "Qwen2Architecture",
    "Qwen2InductionPlant",
    "regular_token_pool",
    "validate_qwen2_architecture",
]
