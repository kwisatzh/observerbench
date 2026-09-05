"""Run the declared integrity grid; retain all conditions, including nulls.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, source_hashes
from observerbench.tasks.evidence_integrity import EvidenceIntegrityConfig, run_evidence_integrity

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/revision/evidence_integrity_v0.json")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.config.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"design": design, "config_sha256": file_sha256(args.config),
                "sources": source_hashes([Path(__file__),ROOT/"src/observerbench/tasks/evidence_integrity.py",ROOT/"src/observerbench/safety.py"], ROOT)}
    write_json(args.outdir/"design.json", manifest)
    rows = []
    for variant in design["variants"]:
        for seed in design["seeds"]:
            for concealment in design["concealment_rates"]:
                for fraction in design["measurement_fractions"]:
                    cfg = EvidenceIntegrityConfig(variant=variant, seed=seed, fit_examples=design["fit_examples"],
                        test_examples=design["test_examples"], concealment_rate=concealment, measurement_fraction=fraction,
                        measurement_cost=design["measurement_cost"], block_fraction=design["block_fraction"], escalate_fraction=design["escalate_fraction"])
                    for method in design["selectors"]:
                        result = run_evidence_integrity(cfg, name=method)
                        rows.append({"variant":variant,"seed":seed,"concealment":concealment,"measurement_fraction":fraction,
                                     "observer":method,"total_loss":result["total_loss_with_measurement"],
                                     "measurements_used":result["measurements_used"],**result["protocol"]["metrics"]})
            print(f"{variant}: completed seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir/"all_conditions.csv",index=False)
    summary = frame.groupby(["variant","concealment","measurement_fraction","observer"],as_index=False).agg(
        total_loss=("total_loss","mean"), loss_min=("total_loss","min"), loss_max=("total_loss","max"),
        missed_severity=("severity_weighted_miss_rate","mean"), clean_utility=("clean_utility_retained","mean"))
    summary.to_csv(args.outdir/"summary.csv",index=False)
    primary = frame[(frame.concealment==design["primary_concealment_rate"]) &
                    (frame.measurement_fraction==design["primary_measurement_fraction"])]
    contrasts=[]
    for variant in design["variants"]:
        pivot = primary[primary.variant==variant].pivot(index="seed",columns="observer",values="total_loss")
        delta=(pivot["uncertainty-severity"]-pivot["random"]).to_numpy()
        rng=np.random.default_rng(9119)
        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
        contrasts.append({"variant":variant,"mean_delta":float(delta.mean()),"paired_seed_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
                          "fraction_better":float(np.mean(delta<0)),"n_independent_generated_seeds":len(delta)})
    write_json(args.outdir/"primary_contrasts.json",contrasts)
    print(json.dumps(contrasts,indent=2))


if __name__ == "__main__":
    main()
