"""IOI Stage 1 whole-group self-repair diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from observerbench.tasks.ioi.heads import BACKUP_NAME_MOVERS, NAME_MOVERS, format_heads


@dataclass
class IOIStage1Config:
    task: str = "ioi_stage1"
    mode: str = "quick"
    quick: bool = True
    ablation: str = "mean"
    include_zero_robustness: bool = False
    n_prompts: int = 32
    seed: int = 0


def _quick_stage1_values() -> dict[str, float]:
    drop_p = 0.45
    drop_b = 0.20
    drop_pb = 0.85
    interaction = drop_pb - drop_p - drop_b
    return {
        "clean_logit_diff": 2.0,
        "drop_P": drop_p,
        "drop_B": drop_b,
        "drop_P+B": drop_pb,
        "interaction": interaction,
        "interaction_fraction_of_joint": interaction / drop_pb,
        "backup_conditional_amplification": (drop_pb - drop_p) / drop_b,
    }


def run_ioi_stage1(cfg: IOIStage1Config, outdir: str | Path) -> list[dict[str, float]]:
    """Run Stage 1.

    The quick path writes a deterministic smoke fixture with the same columns as
    the paper diagnostic. Full GPT-2 execution is intentionally gated behind a
    future optional IOI runner so default tests never download models.
    """

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.quick:
        raise RuntimeError(
            "Full IOI Stage 1 requires the optional TransformerLens runner and "
            "is not part of the default reproduction smoke path."
        )
    if cfg.ablation != "mean":
        raise ValueError("Mean ablation is the primary IOI Stage 1 mode; zero ablation is robustness only.")

    values = _quick_stage1_values()
    clean = values["clean_logit_diff"]
    condition_rows = [
        {
            "ablation": cfg.ablation,
            "position_mode": "end",
            "condition": "clean",
            "heads": "",
            "logit_diff_mean": clean,
            "logit_diff_std": 0.0,
            "drop_from_clean": 0.0,
            "n_prompts": cfg.n_prompts,
        },
        {
            "ablation": cfg.ablation,
            "position_mode": "end",
            "condition": "P",
            "heads": format_heads(NAME_MOVERS),
            "logit_diff_mean": clean - values["drop_P"],
            "logit_diff_std": 0.0,
            "drop_from_clean": values["drop_P"],
            "n_prompts": cfg.n_prompts,
        },
        {
            "ablation": cfg.ablation,
            "position_mode": "end",
            "condition": "B",
            "heads": format_heads(BACKUP_NAME_MOVERS),
            "logit_diff_mean": clean - values["drop_B"],
            "logit_diff_std": 0.0,
            "drop_from_clean": values["drop_B"],
            "n_prompts": cfg.n_prompts,
        },
        {
            "ablation": cfg.ablation,
            "position_mode": "end",
            "condition": "P+B",
            "heads": format_heads(NAME_MOVERS + BACKUP_NAME_MOVERS),
            "logit_diff_mean": clean - values["drop_P+B"],
            "logit_diff_std": 0.0,
            "drop_from_clean": values["drop_P+B"],
            "n_prompts": cfg.n_prompts,
        },
    ]
    summary = {
        "ablation": cfg.ablation,
        "drop_P": values["drop_P"],
        "drop_B": values["drop_B"],
        "drop_P+B": values["drop_P+B"],
        "interaction": values["interaction"],
        "interaction_fraction_of_joint": values["interaction_fraction_of_joint"],
        "backup_conditional_amplification": values["backup_conditional_amplification"],
        "mode": "quick_fixture",
    }
    pd.DataFrame(condition_rows).to_csv(out / "ioi_stage1_condition_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "ioi_stage1_summary.csv", index=False)
    (out / "ioi_stage1_metadata.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(["P", "B", "P+B", "interaction"], [values["drop_P"], values["drop_B"], values["drop_P+B"], values["interaction"]])
    ax.set_ylabel("drop in IOI logit difference")
    ax.set_title("IOI Stage 1 quick self-repair diagnostic")
    fig.tight_layout()
    fig.savefig(out / "ioi_stage1_bar.png", dpi=160)
    plt.close(fig)
    return [summary]
