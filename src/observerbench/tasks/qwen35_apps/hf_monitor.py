"""Narrow Hugging Face monitor adapter for the Qwen3.5 APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

Qwen3.5 stores its language-model configuration below ``text_config`` and may
nest its decoder blocks below a language-model wrapper. The generic measurement
loop used by the Gemma APPS study is otherwise sufficient, so this module only
adapts those two boundaries and makes non-thinking chat rendering explicit.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

from observerbench.tasks.gemma_apps.hf_monitor import (
    EncodedMonitorView,
    HFCausalMonitor,
    integer_score_token_ids,
    validate_contextual_score_token_ids,
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _at_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for name in path:
        current = _field(current, name)
        if current is None:
            return None
    return current


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class Qwen35CausalMonitor(HFCausalMonitor):
    """Qwen3.5 specialization of the checked final-token monitor path."""

    _DECODER_PATHS = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
    )

    def _decoder_layers_with_path(self) -> tuple[Sequence[Any], str]:
        matches: list[tuple[Sequence[Any], str]] = []
        for path in self._DECODER_PATHS:
            layers = _at_path(self.model, path)
            if layers is None or not hasattr(layers, "__len__") or not hasattr(
                layers, "__getitem__"
            ):
                continue
            matches.append((layers, ".".join(path)))
        if not matches:
            tried = ", ".join(".".join(path) for path in self._DECODER_PATHS)
            raise ValueError(f"Qwen3.5 decoder blocks were not found; tried: {tried}")
        lengths = {len(layers) for layers, _ in matches}
        if len(lengths) != 1:
            raise ValueError("Qwen3.5 decoder paths expose inconsistent layer counts")
        return matches[0]

    def _decoder_layers(self) -> Sequence[Any]:
        layers, _ = self._decoder_layers_with_path()
        return layers

    def audit_runtime(
        self,
        *,
        expected_model_type: str,
        expected_text_model_type: str,
        expected_layers: int,
        expected_hidden_size: int,
        selected_layers: Sequence[int] = (),
        expected_selected_layer_types: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        """Audit the pinned nested text configuration and residual-hook path."""

        config = self.model.config
        nested_text_config = _field(config, "text_config")
        text_config = nested_text_config if nested_text_config is not None else config
        config_layout = (
            "nested_text_config"
            if nested_text_config is not None
            else "extracted_text_config"
        )
        layer_types = tuple(map(str, _field(text_config, "layer_types", ())))
        layers, decoder_path = self._decoder_layers_with_path()
        try:
            observed_dtype = str(next(self.model.parameters()).dtype).removeprefix(
                "torch."
            )
        except StopIteration:
            observed_dtype = self.dtype_name
        forward_parameters = inspect.signature(self.model.forward).parameters
        observed = {
            "requested_model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_model_revision": _field(config, "_commit_hash"),
            "model_type": str(_field(config, "model_type", "")),
            "text_model_type": str(_field(text_config, "model_type", "")),
            "config_layout": config_layout,
            "layers": int(_field(text_config, "num_hidden_layers", -1)),
            "hidden_size": int(_field(text_config, "hidden_size", -1)),
            "layer_types": list(layer_types),
            "decoder_path": decoder_path,
            "dtype": observed_dtype,
            "attention_implementation": str(
                _first_non_none(
                    _field(text_config, "_attn_implementation"),
                    _field(config, "_attn_implementation"),
                    self.attention_implementation,
                )
            ),
            "logits_to_keep_supported": "logits_to_keep" in forward_parameters,
            "thinking_enabled": False,
        }
        expected = {
            "text_model_type": str(expected_text_model_type),
            "layers": int(expected_layers),
            "hidden_size": int(expected_hidden_size),
        }
        for key, value in expected.items():
            if observed[key] != value:
                raise ValueError(
                    f"Qwen3.5 model {key} is {observed[key]!r}, expected {value!r}"
                )
        if nested_text_config is not None and observed["model_type"] != str(
            expected_model_type
        ):
            raise ValueError(
                f"Qwen3.5 model model_type is {observed['model_type']!r}, "
                f"expected {str(expected_model_type)!r}"
            )
        if len(layers) != int(expected_layers):
            raise ValueError("Qwen3.5 decoder-layer count differs from text_config")
        if layer_types and len(layer_types) != int(expected_layers):
            raise ValueError("Qwen3.5 layer_types length differs from the layer count")
        if observed["dtype"] != self.dtype_name:
            raise ValueError("loaded Qwen3.5 dtype differs from the frozen dtype")
        if observed["attention_implementation"] != self.attention_implementation:
            raise ValueError(
                "loaded Qwen3.5 attention implementation differs from the frozen setting"
            )
        revision = observed["resolved_model_revision"]
        if revision is not None and revision != self.revision:
            raise ValueError("resolved Qwen3.5 revision differs from the frozen revision")
        if not observed["logits_to_keep_supported"]:
            raise ValueError("pinned Qwen3.5 runtime lacks logits_to_keep support")

        selected = tuple(map(int, selected_layers))
        declared_types = dict(expected_selected_layer_types or {})
        selected = tuple(dict.fromkeys((*selected, *map(int, declared_types))))
        if any(index < 0 or index >= int(expected_layers) for index in selected):
            raise ValueError("selected Qwen3.5 residual layer lies outside the model")
        if selected and not layer_types:
            raise ValueError("selected Qwen3.5 layers require declared layer_types")
        selected_types = {str(index): layer_types[index] for index in selected}
        for raw_index, expected_type in declared_types.items():
            index = int(raw_index)
            if layer_types[index] != str(expected_type):
                raise ValueError(
                    f"Qwen3.5 layer {index} is {layer_types[index]!r}, "
                    f"expected {expected_type!r}"
                )
        observed["selected_layer_types"] = selected_types
        return observed

    def render_chat(self, content: str, *, add_generation_prompt: bool) -> str:
        """Render one text-only user turn with Qwen3.5 thinking disabled."""

        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(content)}],
                tokenize=False,
                add_generation_prompt=bool(add_generation_prompt),
                enable_thinking=False,
            )
        except TypeError as error:
            raise ValueError(
                "the pinned Qwen3.5 tokenizer cannot enforce enable_thinking=False"
            ) from error
        return str(rendered)


__all__ = [
    "EncodedMonitorView",
    "Qwen35CausalMonitor",
    "integer_score_token_ids",
    "validate_contextual_score_token_ids",
]
