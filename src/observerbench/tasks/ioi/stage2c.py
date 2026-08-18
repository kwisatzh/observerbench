"""IOI Stage 2c primary-stratified fixture generation and optional runner.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd

from observerbench.tasks.ioi.heads import head_records


@dataclass
class IOIStage2cConfig:
    task: str = "ioi_stage2c"
    mode: str = "quick"
    quick: bool = True
    n_prompts: int = 24
    seed: int = 0
    ablation: str = "mean"


def _mask_from_counts(n_p: int, n_b: int, n_e: int) -> np.ndarray:
    mask = np.zeros(13, dtype=int)
    mask[:n_p] = 1
    mask[3 : 3 + n_b] = 1
    mask[11 : 11 + n_e] = 1
    return mask


def _mask_bits(mask: np.ndarray) -> str:
    return "".join(str(int(x)) for x in mask)


def _drop_formula(n_p: int, n_b: int, n_e: int, mask: np.ndarray) -> float:
    head_effects = np.array([0.95, 0.06, 0.66, 0.01, 0.10, 0.10, 0.23, 0.33, 0.35, -0.08, 0.16, -0.30, -0.53])
    n_p_tot, n_b_tot, n_e_tot = 3, 8, 2
    p = n_p / n_p_tot
    b = n_b / n_b_tot
    e = n_e / n_e_tot
    count_curve = 0.05 * (n_p == 2) - 0.03 * (n_b >= 4) + 0.04 * (n_e == 2)
    pair_effect = 0.08 * p * b + 1.00 * p * e + 0.10 * b * e
    return float(mask @ head_effects + count_curve + pair_effect)


def write_stage2c_fixture(outdir: str | Path, n_prompts: int = 24, seed: int = 0) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    heads = pd.DataFrame(head_records())

    masks: list[np.ndarray] = [np.zeros(13, dtype=int)]
    for idx in range(13):
        m = np.zeros(13, dtype=int)
        m[idx] = 1
        masks.append(m)
    for n_p, n_b, n_e in product([0, 1, 2, 3], [0, 1, 3, 5, 8], [0, 1, 2]):
        masks.append(_mask_from_counts(n_p, n_b, n_e))

    unique: dict[str, np.ndarray] = {}
    for mask in masks:
        unique[_mask_bits(mask)] = mask
    masks = list(unique.values())

    subset_rows = []
    for subset_idx, mask in enumerate(masks):
        labels = [str(heads.loc[i, "label"]) for i, bit in enumerate(mask) if bit]
        n_p = int(mask[:3].sum())
        n_b = int(mask[3:11].sum())
        n_e = int(mask[11:13].sum())
        subset_rows.append(
            {
                "subset_idx": subset_idx,
                "subset_name": "|".join(labels) if labels else "clean",
                "mask_bits": _mask_bits(mask),
                "heads": ",".join(label.split(":", 1)[1] for label in labels),
                "n_heads": int(mask.sum()),
                "n_P": n_p,
                "n_B": n_b,
                "n_E": n_e,
                "has_P": int(n_p > 0),
                "has_B": int(n_b > 0),
                "has_E": int(n_e > 0),
                "P_B": int(n_p > 0 and n_b > 0),
                "P_E": int(n_p > 0 and n_e > 0),
                "B_E": int(n_b > 0 and n_e > 0),
                "P_B_count": (n_p / 3.0) * (n_b / 8.0),
                "P_E_count": (n_p / 3.0) * (n_e / 2.0),
                "B_E_count": (n_b / 8.0) * (n_e / 2.0),
                "high_primary": int(n_p >= 2),
                "full_primary": int(n_p == 3),
            }
        )
    subset = pd.DataFrame(subset_rows)

    rng = np.random.default_rng(seed)
    prompt_offsets = rng.normal(0.0, 0.015, size=n_prompts)
    drop_rows = []
    for row, mask in zip(subset_rows, masks):
        mean_drop = _drop_formula(row["n_P"], row["n_B"], row["n_E"], mask)
        for prompt_idx, offset in enumerate(prompt_offsets):
            drop_rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "subset_idx": row["subset_idx"],
                    "drop_from_clean": float(mean_drop + offset),
                }
            )

    heads.to_csv(out / "ioi_stage2c_head_records.csv", index=False)
    subset.to_csv(out / "ioi_stage2c_subset_design.csv", index=False)
    pd.DataFrame(drop_rows).to_csv(out / "ioi_stage2c_per_prompt_drops.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "fixture",
                "mae": 0.0,
                "note": "Synthetic quick fixture for Stage 2d smoke tests; not a paper result.",
            }
        ]
    ).to_csv(out / "ioi_stage2c_fit_summary.csv", index=False)
    return out


def run_ioi_stage2c(cfg: IOIStage2cConfig, outdir: str | Path) -> Path:
    if cfg.ablation != "mean":
        raise ValueError("Mean ablation is primary; zero ablation is robustness only.")
    if not cfg.quick:
        raise RuntimeError("Full IOI Stage 2c requires the optional TransformerLens runner.")
    out = write_stage2c_fixture(outdir, n_prompts=cfg.n_prompts, seed=cfg.seed)
    (out / "ioi_stage2c_metadata.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    return out
