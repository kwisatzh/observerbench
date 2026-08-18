"""Freeze the prospective Phase-6 IOI design without a model forward pass.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.phase5_design import load_legacy_mask_bits
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.phase6_confirmatory import (
    load_phase6_protocol,
    prepare_phase6_design,
    write_phase6_design,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_MASKS = (
    ROOT
    / "results/revision/phase02/inputs/stage2b_mean_end/ioi_stage2b_subset_design.csv",
    ROOT
    / "results/frozen/ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_subset_design.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze fresh Phase-6 templates, name splits, prompts, and masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json",
    )
    parser.add_argument(
        "--phase5-design-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/design",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/design",
    )
    parser.add_argument("--legacy-mask-csv", action="append", type=Path)
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Pinned GPT-2 tokenizer name or local snapshot.",
    )
    parser.add_argument("--allow-tokenizer-network", action="store_true")
    return parser.parse_args()


def _token_encoder(
    tokenizer_name: str,
    *,
    revision: str,
    allow_network: bool,
) -> Callable[[str], Sequence[int]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Phase-6 design preparation requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=revision if not Path(tokenizer_name).exists() else None,
        local_files_only=not allow_network,
    )
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    args = parse_args()
    protocol = load_phase6_protocol(args.config)
    legacy_paths = tuple(args.legacy_mask_csv or DEFAULT_LEGACY_MASKS)
    expected_legacy = protocol["legacy_mask_exclusions"]
    for path in legacy_paths:
        label = path.relative_to(ROOT).as_posix()
        if expected_legacy.get(label) != file_sha256(path):
            raise ValueError(f"legacy mask source differs from protocol: {label}")

    phase5_prompts, phase5_masks, _manifest = load_locked_ioi_design(
        args.phase5_design_dir
    )
    del phase5_prompts
    phase5_names = pd.read_csv(args.phase5_design_dir / "names.csv")
    excluded_bits = set(phase5_masks["mask_bits"].astype(str))
    excluded_bits.update(load_legacy_mask_bits(legacy_paths))
    design = prepare_phase6_design(
        protocol,
        phase5_names=phase5_names,
        phase5_masks=phase5_masks,
        excluded_bits=excluded_bits,
        encode_name=_token_encoder(
            args.tokenizer,
            revision=str(protocol["model_revision"]),
            allow_network=args.allow_tokenizer_network,
        ),
    )
    output = write_phase6_design(
        design,
        args.outdir,
        protocol_path=args.config,
        phase5_design_dir=args.phase5_design_dir,
        exclusion_paths=legacy_paths,
    )
    print(f"Wrote sealed outcome-free Phase-6 design to {output}")


if __name__ == "__main__":
    main()
