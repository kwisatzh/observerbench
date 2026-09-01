"""Focused tests for the narrow Qwen3.5 Hugging Face monitor adapter.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from observerbench.tasks.qwen35_apps.hf_monitor import Qwen35CausalMonitor


def _layer_types() -> list[str]:
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(32)
    ]


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return "rendered-without-thinking"


class _Model:
    def __init__(
        self,
        *,
        nested_language_model: bool = False,
        extracted_text_config: bool = False,
    ) -> None:
        layers = [object() for _ in range(32)]
        if nested_language_model:
            self.model = SimpleNamespace(
                language_model=SimpleNamespace(layers=layers)
            )
        else:
            self.model = SimpleNamespace(layers=layers)
        text_config = SimpleNamespace(
            model_type="qwen3_5_text",
            num_hidden_layers=32,
            hidden_size=4096,
            layer_types=_layer_types(),
            _attn_implementation="sdpa",
            _commit_hash="pinned-revision",
        )
        self.config = (
            text_config
            if extracted_text_config
            else SimpleNamespace(
                model_type="qwen3_5",
                _commit_hash="pinned-revision",
                text_config=text_config,
            )
        )
        self._parameter = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
        )

    def parameters(self):
        return iter((self._parameter,))

    def forward(self, input_ids=None, logits_to_keep=None):
        return input_ids, logits_to_keep


def _monitor(
    *,
    nested_language_model: bool = False,
    extracted_text_config: bool = False,
) -> Qwen35CausalMonitor:
    return Qwen35CausalMonitor(
        torch=torch,
        tokenizer=_Tokenizer(),
        model=_Model(
            nested_language_model=nested_language_model,
            extracted_text_config=extracted_text_config,
        ),
        device="cpu",
        model_id="Qwen/Qwen3.5-9B",
        revision="pinned-revision",
        dtype_name="bfloat16",
        attention_implementation="sdpa",
    )


def test_runtime_audit_reads_nested_text_config_and_selected_layer_types() -> None:
    monitor = _monitor()

    audit = monitor.audit_runtime(
        expected_model_type="qwen3_5",
        expected_text_model_type="qwen3_5_text",
        expected_layers=32,
        expected_hidden_size=4096,
        selected_layers=(12, 20, 28),
        expected_selected_layer_types={
            12: "linear_attention",
            20: "linear_attention",
            28: "linear_attention",
        },
    )

    assert audit["decoder_path"] == "model.layers"
    assert audit["config_layout"] == "nested_text_config"
    assert audit["selected_layer_types"] == {
        "12": "linear_attention",
        "20": "linear_attention",
        "28": "linear_attention",
    }
    assert audit["thinking_enabled"] is False
    assert audit["logits_to_keep_supported"] is True


def test_decoder_path_accepts_a_language_model_wrapper() -> None:
    monitor = _monitor(nested_language_model=True)

    audit = monitor.audit_runtime(
        expected_model_type="qwen3_5",
        expected_text_model_type="qwen3_5_text",
        expected_layers=32,
        expected_hidden_size=4096,
    )

    assert audit["decoder_path"] == "model.language_model.layers"
    assert len(monitor._decoder_layers()) == 32


def test_runtime_audit_accepts_transformers_extracted_text_config() -> None:
    monitor = _monitor(extracted_text_config=True)

    audit = monitor.audit_runtime(
        expected_model_type="qwen3_5",
        expected_text_model_type="qwen3_5_text",
        expected_layers=32,
        expected_hidden_size=4096,
        selected_layers=(12, 20, 28),
    )

    assert audit["config_layout"] == "extracted_text_config"
    assert audit["model_type"] == "qwen3_5_text"
    assert audit["text_model_type"] == "qwen3_5_text"


def test_runtime_audit_rejects_selected_layer_type_drift() -> None:
    monitor = _monitor()

    with pytest.raises(ValueError, match="layer 12"):
        monitor.audit_runtime(
            expected_model_type="qwen3_5",
            expected_text_model_type="qwen3_5_text",
            expected_layers=32,
            expected_hidden_size=4096,
            expected_selected_layer_types={12: "full_attention"},
        )


def test_chat_rendering_explicitly_disables_thinking() -> None:
    monitor = _monitor()

    rendered = monitor.render_chat("inspect this code", add_generation_prompt=True)

    assert rendered == "rendered-without-thinking"
    call = monitor.tokenizer.calls[0]
    assert call["enable_thinking"] is False
    assert call["add_generation_prompt"] is True
    assert call["messages"] == [
        {"role": "user", "content": "inspect this code"}
    ]


def test_chat_rendering_fails_if_thinking_cannot_be_disabled() -> None:
    class _OldTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize: bool, add_generation_prompt: bool
        ):
            return str(messages), tokenize, add_generation_prompt

    monitor = _monitor()
    monitor.tokenizer = _OldTokenizer()

    with pytest.raises(ValueError, match="enable_thinking=False"):
        monitor.render_chat("inspect this code", add_generation_prompt=True)
