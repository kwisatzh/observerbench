"""Prepare the locked Phase 5 IOI prompts and masks without model outcomes.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

from observerbench.tasks.ioi.phase5_design import (
    load_legacy_mask_bits,
    load_phase5_protocol,
    prepare_phase5_design,
    write_phase5_design,
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
        description="Freeze Phase 5 IOI prompts, masks, and leakage diagnostics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase05_confirmatory_v2.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase05/design",
    )
    parser.add_argument(
        "--legacy-mask-csv",
        action="append",
        type=Path,
        help="Legacy subset CSV to exclude. Repeat for multiple banks.",
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional Hugging Face tokenizer name or local snapshot. When omitted, "
            "token validation is deferred to the model runner."
        ),
    )
    parser.add_argument(
        "--allow-tokenizer-network",
        action="store_true",
        help="Allow tokenizer loading outside the local cache.",
    )
    return parser.parse_args()


def _token_encoder(
    tokenizer_name: str | None,
    *,
    revision: str,
    allow_network: bool,
) -> Callable[[str], Sequence[int]] | None:
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "--tokenizer requires transformers; install the IOI optional dependencies"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=revision if not Path(tokenizer_name).exists() else None,
        local_files_only=not allow_network,
    )
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    args = parse_args()
    legacy_paths = tuple(args.legacy_mask_csv or DEFAULT_LEGACY_MASKS)
    protocol = load_phase5_protocol(args.config)
    encode_name = _token_encoder(
        args.tokenizer,
        revision=str(protocol["model_revision"]),
        allow_network=args.allow_tokenizer_network,
    )
    design = prepare_phase5_design(
        protocol,
        legacy_masks=load_legacy_mask_bits(legacy_paths),
        encode_name=encode_name,
    )
    output = write_phase5_design(
        design,
        args.outdir,
        protocol_path=args.config,
        legacy_paths=legacy_paths,
    )
    print(f"Wrote frozen Phase 5 design to {output}")


if __name__ == "__main__":
    main()
