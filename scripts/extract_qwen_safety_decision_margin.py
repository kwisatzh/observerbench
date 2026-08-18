"""Extract raw Qwen block-minus-allow margins for the locked safety bank.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from observerbench.core import write_json
from observerbench.provenance import file_sha256
from observerbench.tasks.qwen_safety.design import (
    QwenSafetyDesignConfig,
    build_qwen_safety_design,
)
from observerbench.tasks.qwen_safety.plant import QwenSafetyActivationProducer
from observerbench.tasks.qwen_safety.runner import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    design = build_qwen_safety_design(QwenSafetyDesignConfig(**config["design"]))
    model = config["model"]
    producer = QwenSafetyActivationProducer.from_pretrained(
        model["id"],
        model["revision"],
        device=args.device,
        dtype=model["dtype"],
        attention_implementation=model["attention_implementation"],
    )
    runtime = producer.plant.audit_runtime(
        expected_model_id=model["id"],
        expected_revision=model["revision"],
        expected_layers=model["expected_layers"],
        expected_query_heads=model["expected_query_heads"],
        expected_kv_heads=model["expected_kv_heads"],
        expected_dtype=model["dtype"],
        expected_attention_implementation=model["attention_implementation"],
    )
    prompts = design.prompts_for("locked_test")
    layer = int(config["activations"]["layers"][0])
    batch = producer.encode(
        prompts,
        layer_indices=(layer,),
        batch_size=int(config["runtime"]["batch_size"]),
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    margin_path = output / "locked_test_decision_margins.npz"
    np.savez_compressed(
        margin_path,
        prompt_ids=np.asarray(batch.prompt_ids),
        block_minus_allow_margins=batch.block_minus_allow_margins,
        candidate_margins=batch.candidate_margins,
        candidate_correct=batch.candidate_correct,
        top1_correct=batch.top1_correct,
    )
    write_json(
        output / "locked_test_decision_margins.manifest.json",
        {
            "schema": "observerbench.qwen_safety.decision_margins.v0",
            "status": "post_outcome_secondary_extraction",
            "model_runtime": runtime,
            "design_sha256": design.design_sha256,
            "bank": "locked_test",
            "n_prompts": len(prompts),
            "margin_definition": "logit(block token) - logit(allow token)",
            "margin_file": margin_path.name,
            "margin_sha256": file_sha256(margin_path),
        },
    )
    print(margin_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
