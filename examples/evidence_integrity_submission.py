"""An editable, CPU-only evidence-integrity observer.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
import argparse
from pathlib import Path

from observerbench.core import write_json
from observerbench.tasks.evidence_integrity import (
    EvidenceIntegrityConfig, SELECTORS, VARIANTS, evidence_scores,
    run_evidence_integrity, select_measurements,
)


def choose_measurements(fitting_examples, queries, allowance):
    # Edit this function to choose which evidence is worth inspecting.
    return select_measurements(fitting_examples, queries, allowance, method="uncertainty-severity")


def predict_risk(fitting_examples, measured_queries):
    # Edit this function to turn the available evidence into one risk per case.
    return evidence_scores(fitting_examples, measured_queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--concealment", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path("runs/evidence-integrity-example"))
    args = parser.parse_args()
    cfg = EvidenceIntegrityConfig(variant=args.variant, measurement_fraction=args.budget, concealment_rate=args.concealment)
    args.outdir.mkdir(parents=True, exist_ok=True)
    reports = [run_evidence_integrity(cfg, name=name) for name in SELECTORS]
    reports.append(run_evidence_integrity(cfg, selector=choose_measurements, predictor=predict_risk, name="your-observer", outdir=args.outdir))
    write_json(args.outdir/"comparison.json", reports)
    print(f"Open, inert practice task: {args.variant}; inspect at most {int(cfg.test_examples*args.budget)} pending cases.")
    print(f"{'Observer':<24} {'Missed severity':>16} {'Clean utility':>15} {'Total loss':>12}")
    for row in reports:
        metrics = row["protocol"]["metrics"]
        print(f"{row['observer']:<24} {metrics['severity_weighted_miss_rate']:>16.3f} {metrics['clean_utility_retained']:>15.3f} {row['total_loss_with_measurement']:>12.4f}")
    print("Total loss includes the declared measurement cost. Public generation is not a sealed score.")


if __name__ == "__main__":
    main()
