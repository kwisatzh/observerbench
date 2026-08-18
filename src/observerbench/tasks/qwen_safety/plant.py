"""Pinned Qwen producer for authorization decisions and residual activations.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from observerbench.tasks.qwen_induction.plant import Qwen2InductionPlant
from observerbench.tasks.qwen_safety.design import (
    ALLOW_ANSWER,
    BLOCK_ANSWER,
    QwenSafetyPrompt,
)


@dataclass(frozen=True)
class QwenSafetyActivationBatch:
    prompt_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    activations: np.ndarray
    candidate_margins: np.ndarray
    block_minus_allow_margins: np.ndarray
    candidate_correct: np.ndarray
    top1_correct: np.ndarray
    sequence_lengths: np.ndarray


class QwenSafetyActivationProducer:
    """Extract selected last-prompt residuals without generating text."""

    def __init__(self, plant: Qwen2InductionPlant) -> None:
        self.plant = plant
        self.model = plant.model
        self.tokenizer = plant.tokenizer
        self.device = plant.device
        self.torch = plant.torch
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.allow_token_id = self._single_token(ALLOW_ANSWER)
        self.block_token_id = self._single_token(BLOCK_ANSWER)
        if self.allow_token_id == self.block_token_id:
            raise ValueError("allow and block answers must use distinct tokens")

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
    ) -> "QwenSafetyActivationProducer":
        plant = Qwen2InductionPlant.from_pretrained(
            model_id,
            revision,
            device=device,
            dtype=dtype,
            attention_implementation=attention_implementation,
            local_files_only=local_files_only,
        )
        return cls(plant)

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if len(tokens) != 1:
            raise ValueError(f"Qwen safety answer {text!r} must encode to one token")
        return int(tokens[0])

    def _tokens(self, prompt: QwenSafetyPrompt) -> tuple[int, ...]:
        tokens = self.tokenizer.apply_chat_template(
            list(prompt.messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], (list, tuple)):
            if len(tokens) != 1:
                raise ValueError("Qwen safety prompt tokenization returned multiple rows")
            tokens = tokens[0]
        result = tuple(map(int, tokens))
        if not result:
            raise ValueError("Qwen safety prompt tokenization returned no tokens")
        return result

    def encode(
        self,
        prompts: Sequence[QwenSafetyPrompt],
        *,
        layer_indices: Sequence[int],
        batch_size: int,
    ) -> QwenSafetyActivationBatch:
        if not prompts:
            raise ValueError("at least one Qwen safety prompt is required")
        layers = tuple(map(int, layer_indices))
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("layer_indices must be nonempty and unique")
        n_hidden_states = int(self.model.config.num_hidden_layers) + 1
        if any(layer < 0 or layer >= n_hidden_states for layer in layers):
            raise ValueError(f"layer indices must lie in [0, {n_hidden_states - 1}]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        all_activations: list[np.ndarray] = []
        all_margins: list[np.ndarray] = []
        all_block_minus_allow_margins: list[np.ndarray] = []
        all_candidate_correct: list[np.ndarray] = []
        all_top1_correct: list[np.ndarray] = []
        all_lengths: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            rows = prompts[start : start + batch_size]
            token_rows = [self._tokens(prompt) for prompt in rows]
            lengths = np.asarray([len(tokens) for tokens in token_rows], dtype=np.int64)
            maximum = int(np.max(lengths))
            input_ids = self.torch.full(
                (len(rows), maximum),
                int(self.tokenizer.pad_token_id),
                dtype=self.torch.long,
                device=self.device,
            )
            attention_mask = self.torch.zeros_like(input_ids)
            for index, tokens in enumerate(token_rows):
                input_ids[index, : len(tokens)] = self.torch.as_tensor(
                    tokens,
                    dtype=self.torch.long,
                    device=self.device,
                )
                attention_mask[index, : len(tokens)] = 1
            positions = self.torch.as_tensor(lengths - 1, dtype=self.torch.long, device=self.device)
            batch_indices = self.torch.arange(len(rows), device=self.device)
            with self.torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            logits = outputs.logits[batch_indices, positions].float()
            allow = logits[:, self.allow_token_id]
            block = logits[:, self.block_token_id]
            unsafe = self.torch.as_tensor(
                [prompt.unsafe_if_allowed for prompt in rows],
                dtype=self.torch.bool,
                device=self.device,
            )
            correct = self.torch.where(unsafe, block, allow)
            incorrect = self.torch.where(unsafe, allow, block)
            correct_token = self.torch.where(
                unsafe,
                self.torch.full_like(positions, self.block_token_id),
                self.torch.full_like(positions, self.allow_token_id),
            )
            activations = self.torch.stack(
                [outputs.hidden_states[layer][batch_indices, positions] for layer in layers],
                dim=1,
            )
            all_activations.append(activations.float().cpu().numpy().astype(np.float16))
            all_margins.append((correct - incorrect).cpu().numpy())
            all_block_minus_allow_margins.append((block - allow).cpu().numpy())
            all_candidate_correct.append((correct > incorrect).cpu().numpy())
            all_top1_correct.append((self.torch.argmax(logits, dim=1) == correct_token).cpu().numpy())
            all_lengths.append(lengths)
            del outputs, logits, activations

        return QwenSafetyActivationBatch(
            prompt_ids=tuple(prompt.prompt_id for prompt in prompts),
            layer_indices=layers,
            activations=np.concatenate(all_activations),
            candidate_margins=np.concatenate(all_margins).astype(np.float32),
            block_minus_allow_margins=np.concatenate(
                all_block_minus_allow_margins
            ).astype(np.float32),
            candidate_correct=np.concatenate(all_candidate_correct).astype(bool),
            top1_correct=np.concatenate(all_top1_correct).astype(bool),
            sequence_lengths=np.concatenate(all_lengths),
        )


__all__ = ["QwenSafetyActivationBatch", "QwenSafetyActivationProducer"]
