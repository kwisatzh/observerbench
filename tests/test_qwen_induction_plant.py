"""Tests for the optional native-HF Qwen induction producer.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from observerbench.tasks.qwen_induction.plant import (
    HeadAblationMeans,
    HeadRef,
    Qwen2InductionPlant,
    regular_token_pool,
    validate_qwen2_architecture,
)


class FakeTokenizer:
    vocab_size = 80
    all_special_ids = [0, 79]

    def __len__(self):
        return self.vocab_size

    def decode(self, values, **_kwargs):
        value = int(values[0])
        if value == 2:
            return "\ufffd"
        return chr(0x100 + value)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        value = ord(text) - 0x100
        if value == 3:
            return [3, 4]
        return [value]


class FakeAttention:
    def __init__(self, hidden: int):
        self.q_proj = SimpleNamespace(bias=np.zeros(1))
        self.k_proj = SimpleNamespace(bias=np.zeros(1))
        self.v_proj = SimpleNamespace(bias=np.zeros(1))
        self.o_proj = SimpleNamespace(bias=None, weight=np.zeros((hidden, hidden)))


def test_regular_token_pool_excludes_special_invalid_and_nonroundtrip_ids() -> None:
    pool = regular_token_pool(
        FakeTokenizer(), seed=4, limit=5, candidate_ids=tuple(range(10))
    )
    assert len(pool) == 5
    assert 0 not in pool
    assert 2 not in pool
    assert 3 not in pool


def test_architecture_audit_records_gqa_and_biases() -> None:
    hidden = 16
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=hidden,
            head_dim=4,
        ),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=FakeAttention(hidden)),
                SimpleNamespace(self_attn=FakeAttention(hidden)),
            ]
        ),
    )
    architecture = validate_qwen2_architecture(model)
    assert architecture.n_layers == 2
    assert architecture.n_query_heads == 4
    assert architecture.n_kv_heads == 2
    assert architecture.qkv_bias_present
    assert not architecture.output_bias_present


def test_reference_mean_shape_and_family_lookup() -> None:
    heads = (HeadRef(0, 0, 0), HeadRef(1, 2, 1))
    means = HeadAblationMeans(
        family_ids=("a", "b"),
        heads=heads,
        values=np.ones((2, 2, 4), dtype=np.float32),
        counts=(3, 5),
    )
    assert means.head_dim == 4
    assert means.family_index("b") == 1
    with pytest.raises(KeyError):
        means.family_index("missing")


def test_tiny_random_qwen_hooks_measure_exact_noop_and_finite_effect() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=79,
        eos_token_id=79,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).eval()
    plant = Qwen2InductionPlant(model, tokenizer=FakeTokenizer(), device="cpu")
    plant.model_id = "tiny-qwen"
    plant.model_revision = "0" * 40
    plant.resolved_model_revision = "0" * 40
    plant.attention_implementation = "eager"
    audit = plant.audit_runtime(
        expected_model_id="tiny-qwen",
        expected_revision="0" * 40,
        expected_layers=2,
        expected_query_heads=4,
        expected_kv_heads=2,
        expected_dtype="float32",
        expected_attention_implementation="eager",
    )
    assert audit["quantized"] is False
    records = [
        {
            "prompt_id": "p0",
            "family_id": "f0",
            "input_ids": (10, 11, 20, 21, 30, 31, 10),
            "query_position": 6,
            "target_token_id": 11,
            "distractor_token_id_1": 21,
            "distractor_token_id_2": 31,
            "key_positions": (0, 2, 4),
        },
        {
            "prompt_id": "p1",
            "family_id": "f0",
            "input_ids": (12, 13, 22, 23, 32, 33, 12),
            "query_position": 6,
            "target_token_id": 13,
            "distractor_token_id_1": 23,
            "distractor_token_id_2": 33,
            "key_positions": (0, 2, 4),
        },
    ]
    heads = (HeadRef(0, 0, 0), HeadRef(1, 3, 1))
    means = plant.capture_reference_means(records, heads, batch_size=2)
    effects = plant.measure_mask_effects(
        records,
        heads,
        [(0, 0), (1, 0)],
        means,
        batch_size=2,
    )
    noop = [row for row in effects if row.mask_bits == "00"]
    ablated = [row for row in effects if row.mask_bits == "10"]
    assert len(noop) == len(ablated) == 2
    assert all(row.drop_from_clean == pytest.approx(0.0) for row in noop)
    assert all(np.isfinite(row.drop_from_clean) for row in ablated)
    parity = plant.noop_hook_parity(records, heads, means, batch_size=2)
    assert parity["maximum_margin_error"] == pytest.approx(0.0)
    assert parity["prediction_parity"]
    clean = plant.score_clean(records, batch_size=2)
    assert all(
        row.candidate_predicted_token_id
        in {row.target_token_id, 21, 23, 31, 33}
        for row in clean
    )
    scan = plant.scan_attention(records, batch_size=2)
    assert len(scan) == 8
    assert all(np.isfinite(row.attention_specificity) for row in scan)


def test_tiny_qwen_rejects_nonfinal_query_fixture() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    plant = Qwen2InductionPlant(model, tokenizer=FakeTokenizer(), device="cpu")
    malformed = {
        "prompt_id": "bad",
        "family_id": "f0",
        "input_ids": (10, 11, 20, 21, 10, 30),
        "query_position": 4,
        "source_key_position": 0,
        "target_token_id": 11,
        "distractor_token_id_1": 21,
        "distractor_token_id_2": 31,
    }
    with pytest.raises(ValueError, match="final supplied token"):
        plant.score_clean([malformed])


def test_mean_hook_changes_only_selected_query_head_slice() -> None:
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    config = transformers.Qwen2Config(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen2ForCausalLM(config).eval()
    plant = Qwen2InductionPlant(model, tokenizer=FakeTokenizer(), device="cpu")
    records = [
        {
            "prompt_id": "f0",
            "family_id": "f0",
            "input_ids": (10, 11, 20, 21, 30, 31, 10),
            "query_position": 6,
            "target_token_id": 11,
            "distractor_token_id_1": 21,
            "distractor_token_id_2": 31,
            "key_positions": (0, 2, 4),
        },
        {
            "prompt_id": "f1",
            "family_id": "f1",
            "input_ids": (12, 13, 22, 23, 32, 33, 12),
            "query_position": 6,
            "target_token_id": 13,
            "distractor_token_id_1": 23,
            "distractor_token_id_2": 33,
            "key_positions": (0, 2, 4),
        },
    ]
    heads = (HeadRef(0, 0, 0), HeadRef(0, 1, 0))
    replacements = np.zeros((2, 2, 8), dtype=np.float32)
    replacements[0, 0] = 3.0
    replacements[1, 0] = 5.0
    means = HeadAblationMeans(
        family_ids=("f0", "f1"),
        heads=heads,
        values=replacements,
        counts=(1, 1),
    )

    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().clone())

    baseline_handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(capture)
    with torch.inference_mode():
        model.model(input_ids=plant._inputs(records), use_cache=False)
    baseline_handle.remove()
    baseline = captured.pop()

    with plant.head_intervention(
        heads,
        mask_rows=np.asarray([[1, 0], [1, 0]], dtype=np.uint8),
        query_positions=(6, 6),
        family_ids=("f0", "f1"),
        means=means,
    ):
        intervention_handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(capture)
        with torch.inference_mode():
            model.model(input_ids=plant._inputs(records), use_cache=False)
        intervention_handle.remove()
    intervened = captured.pop()

    assert torch.equal(intervened[:, :6], baseline[:, :6])
    assert torch.equal(intervened[:, 6, 8:], baseline[:, 6, 8:])
    assert torch.equal(
        intervened[:, 6, :8], torch.tensor([[3.0] * 8, [5.0] * 8])
    )
