"""Run a submitted IOI prediction table through the fixed decision rule on CPU.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from observerbench.tasks.ioi.decision_submission import DEFAULT_PACK, TASK_ID, evaluate_ioi_decision_csv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--predictions", type=Path, default=DEFAULT_PACK / "submissions/atp_calibrated.csv")
parser.add_argument("--observer-card", type=Path, default=DEFAULT_PACK / "submissions/atp_calibrated.json")
parser.add_argument("--outdir", type=Path, default=Path("outputs/ioi-decision"))
args = parser.parse_args()
evaluate_ioi_decision_csv(TASK_ID, pack_root=DEFAULT_PACK, predictions_path=args.predictions,
                          observer_card_path=args.observer_card, outdir=args.outdir)
print((args.outdir / "scorecard.txt").read_text(), end="")
print(f"Selected actions and machine-readable scores: {args.outdir.resolve()}")
