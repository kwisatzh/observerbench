"""IOI Stage 2b random head-subset diagnostic smoke runner.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass
class IOIStage2bConfig:
    task: str = "ioi_stage2b"
    mode: str = "quick"
    quick: bool = True
    ablation: str = "mean"
    seed: int = 0


def run_ioi_stage2b(cfg: IOIStage2bConfig, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.ablation != "mean":
        raise ValueError("Mean ablation is primary; zero ablation is robustness only.")
    if not cfg.quick:
        raise RuntimeError("Full IOI Stage 2b requires the optional TransformerLens runner.")
    pd.DataFrame(
        [
            {"model": "additive_head", "mae": 0.32, "mode": "quick_fixture"},
            {"model": "group_interaction", "mae": 0.31, "mode": "quick_fixture"},
        ]
    ).to_csv(out / "ioi_stage2b_fit_summary.csv", index=False)
    (out / "ioi_stage2b_metadata.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    return out
