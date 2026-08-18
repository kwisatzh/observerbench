"""Freeze the outcome-free Phase-7 canonical-template design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.phase7_confirmation import (
    load_legacy_exclusions,
    load_phase7_protocol,
    prepare_phase7_design,
    protocol_source_paths,
    verify_protocol_sources,
    write_phase7_design,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE6 = ROOT / "results/revision/phase06/ioi_fresh_confirmation"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze Phase-7 prompts and action pools without a model forward pass."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
    )
    parser.add_argument("--phase5-design-dir", type=Path, default=PHASE5 / "design")
    parser.add_argument("--phase6-design-dir", type=Path, default=PHASE6 / "design")
    parser.add_argument("--outdir", type=Path, default=PHASE7 / "design")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_phase7_protocol(args.protocol)
    verify_protocol_sources(protocol, ROOT)
    phase5_prompts, phase5_masks, _ = load_locked_ioi_design(args.phase5_design_dir)
    phase6_names = pd.read_csv(args.phase6_design_dir / "names.csv")
    phase6_pairs = pd.read_csv(args.phase6_design_dir / "pair_clusters.csv")
    phase6_prompts = pd.read_csv(args.phase6_design_dir / "prompts.csv")
    phase6_candidates = pd.read_csv(
        args.phase6_design_dir / "candidate_masks.csv", dtype={"mask_bits": str}
    )
    excluded = set(phase5_masks["mask_bits"].astype(str).str.zfill(13))
    excluded.update(phase6_candidates["mask_bits"].astype(str).str.zfill(13))
    excluded.update(load_legacy_exclusions(protocol, ROOT))
    design = prepare_phase7_design(
        protocol,
        phase5_prompts=phase5_prompts,
        phase5_masks=phase5_masks,
        phase6_names=phase6_names,
        phase6_pairs=phase6_pairs,
        phase6_prompts=phase6_prompts,
        excluded_bits=excluded,
    )
    sources = protocol_source_paths(protocol, ROOT)
    output = write_phase7_design(
        design,
        args.outdir,
        protocol_path=args.protocol,
        source_paths=sources,
    )
    print(f"Wrote sealed outcome-free Phase-7 design to {output}")


if __name__ == "__main__":
    main()
