"""Model-free tests for the generic Hugging Face monitor path.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from observerbench.tasks.gemma_apps.hf_monitor import (
    HFCausalMonitor,
    integer_score_token_ids,
    validate_contextual_score_token_ids,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"
    truncation_side = "left"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text.isdigit():
            return [100 + int(text)]
        if text and text[-1].isdigit():
            return [1 + (ord(character) % 31) for character in text[:-1]] + [
                100 + int(text[-1])
            ]
        return [1 + (ord(character) % 31) for character in text]

    def apply_chat_template(self, messages, **_kwargs):
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(
        self,
        texts,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        return_length=False,
    ):
        del add_special_tokens
        rows = [self.encode(text) for text in texts]
        if truncation:
            rows = [row[-int(max_length) :] for row in rows]
        if return_tensors is None:
            payload = {"input_ids": rows}
            if return_length:
                payload["length"] = [len(row) for row in rows]
            return payload
        width = max(map(len, rows)) if padding else len(rows[0])
        ids = []
        masks = []
        for row in rows:
            pad = width - len(row)
            ids.append([0] * pad + row)
            masks.append([0] * pad + [1] * len(row))
        return {
            "input_ids": torch.as_tensor(ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(masks, dtype=torch.long),
        }


class FakeLayer(torch.nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = float(increment)

    def forward(self, hidden):
        return (hidden + self.increment,)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gemma2",
            num_hidden_layers=3,
            hidden_size=4,
            _attn_implementation="sdpa",
        )
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([FakeLayer(1), FakeLayer(2), FakeLayer(3)])
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    ):
        del attention_mask, output_hidden_states, use_cache, return_dict
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = torch.zeros((len(input_ids), int(logits_to_keep), 128))
        logits[:, -1, 100:110] = torch.arange(10, dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def _monitor() -> HFCausalMonitor:
    return HFCausalMonitor(
        torch=torch,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_id="fixture/gemma",
        revision="1" * 40,
        dtype_name="float32",
        attention_implementation="sdpa",
    )


def test_score_token_contract_is_single_and_unique() -> None:
    assert integer_score_token_ids(FakeTokenizer()) == tuple(range(100, 110))

    class Broken(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "4" else super().encode(text, add_special_tokens)

    with pytest.raises(ValueError, match="not one token"):
        integer_score_token_ids(Broken())


def test_contextual_score_tokens_must_remain_exact_suffixes() -> None:
    tokenizer = FakeTokenizer()
    token_ids = integer_score_token_ids(tokenizer)
    audit = validate_contextual_score_token_ids(
        tokenizer,
        ("<score>\n", "detailed<score>\n"),
        bins=tuple(range(10)),
        token_ids=token_ids,
    )
    assert audit["exact_suffix_match"] is True

    class ContextMerge(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith("\n4"):
                return super().encode(text[:-2], add_special_tokens) + [77]
            return super().encode(text, add_special_tokens)

    with pytest.raises(ValueError, match="one-token suffix"):
        validate_contextual_score_token_ids(
            ContextMerge(),
            ("<score>\n",),
            bins=tuple(range(10)),
            token_ids=token_ids,
        )


def test_monitor_captures_only_requested_post_layer_final_states() -> None:
    monitor = _monitor()
    audit = monitor.audit_runtime(
        expected_model_type="gemma2", expected_layers=3, expected_hidden_size=4
    )
    assert audit["logits_to_keep_supported"] is True
    rendered = ["abc", "longer"]
    view = monitor.encode_rendered(
        rendered,
        layers=(0, 2),
        max_length=32,
        batch_size=2,
        score_token_ids=tuple(range(100, 110)),
        score_values=np.linspace(0.0, 10.0, 10),
    )

    assert view.activations.shape == (2, 2, 4)
    assert view.score_log_probabilities.shape == (2, 10)
    assert view.expected_scores.shape == (2,)
    assert np.all(view.activations[:, 1] > view.activations[:, 0])
    np.testing.assert_array_equal(monitor.token_lengths(rendered), [3, 6])


def test_monitor_rejects_out_of_range_or_duplicate_layers() -> None:
    monitor = _monitor()
    with pytest.raises(ValueError, match="unique"):
        monitor.encode_rendered(["x"], layers=(1, 1), max_length=8, batch_size=1)
    with pytest.raises(ValueError, match="outside"):
        monitor.encode_rendered(["x"], layers=(3,), max_length=8, batch_size=1)
