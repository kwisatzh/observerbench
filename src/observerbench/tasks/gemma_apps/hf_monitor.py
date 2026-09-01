"""Small Hugging Face monitor path for the matched Gemma APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module captures only the final-token residuals requested by the frozen
task. It avoids retaining every layer and every token in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Iterable, Sequence

import numpy as np


def integer_score_token_ids(
    tokenizer: Any, bins: Iterable[int] = range(10)
) -> tuple[int, ...]:
    """Resolve the frozen integer bins and fail if any is not one unique token."""

    token_ids: list[int] = []
    for value in bins:
        encoded = tokenizer.encode(str(int(value)), add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if len(encoded) != 1:
            raise ValueError(f"score bin {value} is not one token in the pinned tokenizer")
        token_ids.append(int(encoded[0]))
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("integer score-token IDs must be distinct")
    return tuple(token_ids)


def validate_contextual_score_token_ids(
    tokenizer: Any,
    rendered_prefixes: Sequence[str],
    *,
    bins: Sequence[int],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    """Verify that each frozen bin remains the declared token after each prefix."""

    if not rendered_prefixes:
        raise ValueError("at least one rendered score prefix is required")
    values = tuple(map(int, bins))
    expected = tuple(map(int, token_ids))
    if len(values) != len(expected):
        raise ValueError("score bins and token IDs must have equal length")
    prefix_lengths = []
    for prefix_index, prefix in enumerate(rendered_prefixes):
        prefix_ids = tokenizer.encode(str(prefix), add_special_tokens=False)
        if hasattr(prefix_ids, "tolist"):
            prefix_ids = prefix_ids.tolist()
        prefix_tuple = tuple(map(int, prefix_ids))
        prefix_lengths.append(len(prefix_tuple))
        for value, token_id in zip(values, expected, strict=True):
            combined = tokenizer.encode(str(prefix) + str(value), add_special_tokens=False)
            if hasattr(combined, "tolist"):
                combined = combined.tolist()
            if tuple(map(int, combined)) != (*prefix_tuple, token_id):
                raise ValueError(
                    f"score bin {value} is not the declared one-token suffix "
                    f"after rendered prefix {prefix_index}"
                )
    return {
        "prefixes_checked": len(rendered_prefixes),
        "bins_checked_per_prefix": len(values),
        "prefix_token_lengths": prefix_lengths,
        "exact_suffix_match": True,
    }


def _dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype {name!r}") from error


def _last_hidden(layer_output: Any) -> Any:
    hidden = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output
    if not hasattr(hidden, "ndim") or hidden.ndim != 3:
        raise ValueError("decoder layer hook did not return batch-by-token-by-hidden state")
    return hidden[:, -1]


@dataclass(frozen=True)
class EncodedMonitorView:
    """Final-token measurements for one frozen prompt view."""

    activations: np.ndarray
    lengths: np.ndarray
    score_log_probabilities: np.ndarray | None = None
    expected_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        activations = np.asarray(self.activations)
        lengths = np.asarray(self.lengths)
        if activations.ndim != 3 or lengths.shape != (len(activations),):
            raise ValueError("encoded activations and token lengths have incompatible rows")
        if not np.isfinite(activations).all():
            raise ValueError("encoded activations must be finite")
        if self.score_log_probabilities is None:
            if self.expected_scores is not None:
                raise ValueError("expected scores require score log probabilities")
            return
        log_probs = np.asarray(self.score_log_probabilities)
        expected = np.asarray(self.expected_scores)
        if log_probs.ndim != 2 or log_probs.shape[0] != len(activations):
            raise ValueError("score log probabilities have incompatible rows")
        if expected.shape != (len(activations),):
            raise ValueError("expected scores have incompatible rows")
        if not np.isfinite(log_probs).all() or not np.isfinite(expected).all():
            raise ValueError("score readouts must be finite")


@dataclass
class HFCausalMonitor:
    """A pinned causal LM plus the narrow measurements used by this study."""

    torch: Any
    tokenizer: Any
    model: Any
    device: str
    model_id: str
    revision: str
    dtype_name: str
    attention_implementation: str

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        *,
        device: str,
        dtype: str,
        attention_implementation: str,
        local_files_only: bool = False,
    ) -> "HFCausalMonitor":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("install ObserverBench with the ai-control extra") from error
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=_dtype(torch, dtype),
            attn_implementation=attention_implementation,
            device_map={"": device},
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        model.eval()
        return cls(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=str(device),
            model_id=str(model_id),
            revision=str(revision),
            dtype_name=str(dtype),
            attention_implementation=str(attention_implementation),
        )

    def _decoder_layers(self) -> Sequence[Any]:
        backbone = getattr(self.model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError("model does not expose model.layers decoder blocks")
        return layers

    def audit_runtime(
        self,
        *,
        expected_model_type: str,
        expected_layers: int,
        expected_hidden_size: int,
    ) -> dict[str, Any]:
        config = self.model.config
        try:
            observed_dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except StopIteration:
            observed_dtype = self.dtype_name
        observed = {
            "requested_model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_model_revision": getattr(config, "_commit_hash", None),
            "model_type": str(getattr(config, "model_type", "")),
            "layers": int(getattr(config, "num_hidden_layers")),
            "hidden_size": int(getattr(config, "hidden_size")),
            "dtype": observed_dtype,
            "attention_implementation": str(
                getattr(config, "_attn_implementation", self.attention_implementation)
            ),
            "logits_to_keep_supported": "logits_to_keep"
            in inspect.signature(self.model.forward).parameters,
        }
        expected = {
            "model_type": str(expected_model_type),
            "layers": int(expected_layers),
            "hidden_size": int(expected_hidden_size),
        }
        for key, value in expected.items():
            if observed[key] != value:
                raise ValueError(f"model {key} is {observed[key]!r}, expected {value!r}")
        if len(self._decoder_layers()) != int(expected_layers):
            raise ValueError("exposed decoder-layer count differs from model configuration")
        resolved_revision = observed["resolved_model_revision"]
        if resolved_revision is not None and resolved_revision != self.revision:
            raise ValueError("resolved model revision differs from the frozen revision")
        if observed["dtype"] != self.dtype_name:
            raise ValueError("loaded model dtype differs from the frozen dtype")
        if observed["attention_implementation"] != self.attention_implementation:
            raise ValueError("loaded attention implementation differs from the frozen setting")
        if not observed["logits_to_keep_supported"]:
            raise ValueError("pinned model runtime lacks memory-bounded logits_to_keep support")
        return observed

    def render_chat(self, content: str, *, add_generation_prompt: bool) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(content)}],
                tokenize=False,
                add_generation_prompt=bool(add_generation_prompt),
            )
        )

    def token_lengths(
        self, rendered: Sequence[str], *, batch_size: int = 64
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("token-length batch size must be positive")
        lengths: list[int] = []
        for start in range(0, len(rendered), batch_size):
            encoded = self.tokenizer(
                list(rendered[start : start + batch_size]),
                add_special_tokens=False,
                truncation=False,
                return_length=True,
            )
            raw = encoded.get("length")
            if raw is None:
                raw = [len(ids) for ids in encoded["input_ids"]]
            lengths.extend(map(int, raw))
        return np.asarray(lengths, dtype=np.int32)

    def encode_rendered(
        self,
        rendered: Sequence[str],
        *,
        layers: Sequence[int],
        max_length: int,
        batch_size: int,
        score_token_ids: Sequence[int] | None = None,
        score_values: Sequence[float] | None = None,
    ) -> EncodedMonitorView:
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("max length and batch size must be positive")
        requested_layers = tuple(map(int, layers))
        decoder_layers = self._decoder_layers()
        if len(requested_layers) != len(set(requested_layers)):
            raise ValueError("requested residual layers must be unique")
        if any(layer < 0 or layer >= len(decoder_layers) for layer in requested_layers):
            raise ValueError("requested residual layer lies outside the model")
        if (score_token_ids is None) != (score_values is None):
            raise ValueError("score token IDs and values must be supplied together")
        if (
            score_token_ids is not None
            and score_values is not None
            and len(score_token_ids) != len(score_values)
        ):
            raise ValueError("score token IDs and values must have equal length")

        torch = self.torch
        score_ids = (
            torch.as_tensor(tuple(map(int, score_token_ids)), device=self.device)
            if score_token_ids is not None
            else None
        )
        values = (
            torch.as_tensor(tuple(map(float, score_values)), device=self.device)
            if score_values is not None
            else None
        )
        activation_chunks: list[np.ndarray] = []
        length_chunks: list[np.ndarray] = []
        log_probability_chunks: list[np.ndarray] = []
        expected_chunks: list[np.ndarray] = []
        for start in range(0, len(rendered), batch_size):
            batch_rendered = list(rendered[start : start + batch_size])
            tokens = self.tokenizer(
                batch_rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
                add_special_tokens=False,
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            captured: dict[int, Any] = {}
            handles = []
            for layer in requested_layers:
                def capture(_module: Any, _inputs: Any, output: Any, *, key: int = layer) -> None:
                    captured[key] = _last_hidden(output)

                handles.append(decoder_layers[layer].register_forward_hook(capture))
            try:
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
            finally:
                for handle in handles:
                    handle.remove()
            if tuple(sorted(captured)) != tuple(sorted(requested_layers)):
                raise ValueError("one or more requested residual hooks did not fire")
            if requested_layers:
                activations = torch.stack(
                    [captured[layer] for layer in requested_layers], dim=1
                )
                activation_chunks.append(
                    activations.to(dtype=torch.float16).cpu().numpy()
                )
            else:
                hidden_size = int(getattr(self.model.config, "hidden_size"))
                activation_chunks.append(
                    np.empty((len(batch_rendered), 0, hidden_size), dtype=np.float16)
                )
            length_chunks.append(attention_mask.sum(dim=1).cpu().numpy())
            if score_ids is not None and values is not None:
                candidate_logits = outputs.logits[:, -1].float().index_select(1, score_ids)
                log_probs = torch.log_softmax(candidate_logits, dim=1)
                log_probability_chunks.append(log_probs.cpu().numpy().astype(np.float32))
                expected_chunks.append(
                    (torch.exp(log_probs) * values).sum(dim=1).cpu().numpy().astype(np.float32)
                )
            del outputs, input_ids, attention_mask, captured
        log_probabilities = (
            np.concatenate(log_probability_chunks)
            if log_probability_chunks
            else None
        )
        expected_scores = np.concatenate(expected_chunks) if expected_chunks else None
        return EncodedMonitorView(
            activations=np.concatenate(activation_chunks),
            lengths=np.concatenate(length_chunks).astype(np.int32),
            score_log_probabilities=log_probabilities,
            expected_scores=expected_scores,
        )


__all__ = [
    "EncodedMonitorView",
    "HFCausalMonitor",
    "integer_score_token_ids",
    "validate_contextual_score_token_ids",
]
