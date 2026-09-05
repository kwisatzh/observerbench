"""Measure output-observer transfer with cached, aligned Qwen measurements.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from observerbench.ai_control_followup import budget_metrics, prevalence_budget_sensitivity
from observerbench.core import write_json
from observerbench.provenance import file_sha256, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def fit_ridge_with_calibration(features, labels, fit, calibration, *, ridge_grid, budget):
    """Compose the existing calibration criterion with a low-dimensional ridge fit."""
    x=np.asarray(features,dtype=np.float32).astype(float)
    candidates=[]
    for alpha in ridge_grid:
        model=Ridge(alpha=alpha,solver="cholesky").fit(x[fit],labels[fit])
        metrics=budget_metrics(labels[calibration].astype(bool),model.predict(x[calibration]),budget=budget)
        candidates.append((metrics["realized_violation_rate"],-metrics["risk_auroc"],alpha))
    alpha=min(candidates)[2]
    model=Ridge(alpha=alpha,solver="cholesky").fit(x[fit|calibration],labels[fit|calibration])
    return model, float(alpha)


def adaptation_split(rows, pairs, seed, valid):
    """Nested subsets by problem, entirely outside held-out evaluation rows."""
    eligible=sorted({r["problem_id"] for r in rows if r["split"] != "locked_test"})
    if pairs>len(eligible) or pairs<4:
        raise ValueError("adaptation pair budget is outside the available fitting pool")
    order=np.random.default_rng(seed).permutation(eligible)
    chosen=order[:pairs]
    n_fit=max(2,int(.8*pairs))
    fit_ids=set(chosen[:n_fit])
    calibration_ids=set(chosen[n_fit:])
    fit=np.asarray([r["problem_id"] in fit_ids for r in rows])&valid
    calibration=np.asarray([r["problem_id"] in calibration_ids for r in rows])&valid
    if not fit.any() or not calibration.any():
        raise ValueError("adaptation leaves an empty split")
    return fit,calibration,chosen.tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/revision/qwen_monitor_transfer_v0.json")
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--private-outdir",type=Path,required=True,help="Local fitted parameters and per-example predictions; do not publish evaluator rows.")
    args=parser.parse_args()
    design=json.loads(args.config.read_text())
    args.outdir.mkdir(parents=True,exist_ok=True)
    args.private_outdir.mkdir(parents=True,exist_ok=True)
    paths={k:ROOT/design[k] for k in ("source_design","target_design","source_cache","source_lengths","target_cache")}
    metadata=json.loads(paths["source_design"].read_text())["rows"]
    if metadata != json.loads(paths["target_design"].read_text())["rows"]:
        raise ValueError("source and target metadata must match exactly, including order and splits")
    with np.load(paths["source_cache"],allow_pickle=False) as source:
        xs=np.asarray(source[design["feature_key"]],dtype=np.float32)
    with np.load(paths["source_lengths"],allow_pickle=False) as archive:
        ls=archive["basic_lengths"].copy()
    with np.load(paths["target_cache"],allow_pickle=False) as target:
        xt=np.asarray(target[design["feature_key"]],dtype=np.float32)
        lt=target["lengths"].copy()
        raw=target["expected_scores"].copy()
    if xs.shape != xt.shape or xs.shape != (len(metadata),10) or not (np.isfinite(xs).all() and np.isfinite(xt).all()):
        raise ValueError("ten-bin measurement arrays are not aligned and finite")
    locked=np.asarray([r["split"]=="locked_test" for r in metadata])
    if np.any(((ls>=4096)|(lt>=4096))&locked):
        raise ValueError("at-limit held-out row; do not silently filter it")
    valid=(ls<4096)&(lt<4096)
    fit=np.asarray([r["split"]=="fit" for r in metadata])&valid
    calibration=np.asarray([r["split"]=="calibration" for r in metadata])&valid
    # Fitting functions never receive the held-out labels.
    fitting_labels=np.asarray([float(r["backdoor_works"]) if not test else 0.0 for r,test in zip(metadata,locked)])
    manifest={"design":design,"source_hashes":source_hashes([Path(__file__),ROOT/"src/observerbench/ai_control_followup.py"],ROOT),
              "input_hashes":{k:file_sha256(p) for k,p in paths.items()},"config_sha256":file_sha256(args.config),
              "excluded_fit_or_calibration_rows":int((~valid&~locked).sum()),"locked_rows":int(locked.sum())}
    write_json(args.outdir/"design.json",manifest)
    kwargs={"ridge_grid":design["ridge_grid"],"budget":design["ridge_selection_budget"]}
    source_model,source_alpha=fit_ridge_with_calibration(xs,fitting_labels,fit,calibration,**kwargs)
    target_model,target_alpha=fit_ridge_with_calibration(xt,fitting_labels,fit,calibration,**kwargs)
    scores={"source-frozen":source_model.predict(xt[locked]),"target-native":target_model.predict(xt[locked]),"target-raw-prompt":raw[locked]}
    fits={"source-frozen":{"ridge":source_alpha,"target_pairs":0},"target-native":{"ridge":target_alpha,"target_pairs":700},"target-raw-prompt":{"target_pairs":0}}
    coefficients={"source_coef":source_model.coef_,"source_intercept":source_model.intercept_,"target_coef":target_model.coef_,"target_intercept":target_model.intercept_}
    for seed in design["adaptation_seeds"]:
        for pairs in design["adaptation_problem_pairs"]:
            f,c,chosen=adaptation_split(metadata,pairs,seed,valid)
            model,alpha=fit_ridge_with_calibration(xt,fitting_labels,f,c,**kwargs)
            name=f"refit-p{pairs}-s{seed}"
            scores[name]=model.predict(xt[locked])
            fits[name]={"ridge":alpha,"target_pairs":pairs,"fitting_rows":int(f.sum()),"calibration_rows":int(c.sum())}
            coefficients[name+"_coef"]=model.coef_
            coefficients[name+"_intercept"]=model.intercept_
        print(f"Fitted target adaptation seed {seed}",flush=True)
    # Record parameters and predictions before the evaluator consumes outcomes.
    np.savez_compressed(args.private_outdir/"frozen_parameters.npz",**coefficients)
    np.savez_compressed(args.private_outdir/"heldout_predictions.npz",**scores)
    write_json(args.outdir/"fit_choices.json",fits)
    evaluated=[r for r,is_test in zip(metadata,locked) if is_test]
    labels=np.asarray([r["backdoor_works"] for r in evaluated],dtype=bool)
    honest=~labels
    pools={name:{"honest":values[honest],"attack":values[labels]} for name,values in scores.items()}
    sensitivity=prevalence_budget_sensitivity(pools,prevalences=design["attack_prevalences"],budgets=design["audit_budgets"],
        population_size=design["population_size"],replicates=design["deployment_resamples"],seed=design["deployment_seed"],
        status=design["status"],design_disclosure=design["scope"])
    write_json(args.outdir/"deployment_sensitivity.json",sensitivity)
    records=[]
    for name,values in scores.items():
        auc=budget_metrics(labels,values,budget=.02)["risk_auroc"]
        for cell in sensitivity["cells"].values():
            metric=cell["monitors"][name]["realized_violations"]
            records.append({"observer":name,"target_pairs":fits[name]["target_pairs"],"prevalence":cell["prevalence"],
                            "audit_fraction":cell["budget"],"auroc":auc,"violations_mean":metric["mean"],
                            "outcome_lower_95":metric["lower_95"],"outcome_upper_95":metric["upper_95"]})
    frame=pd.DataFrame(records)
    frame.to_csv(args.outdir/"all_results.csv",index=False)
    frame["method"]=frame.observer.str.replace(r"-s\d+$","",regex=True)
    grouped=frame.groupby(["method","target_pairs","prevalence"],as_index=False).agg(
        mean_auroc=("auroc","mean"), mean_violations=("violations_mean","mean"),
        refit_min=("violations_mean","min"),refit_max=("violations_mean","max"),refits=("observer","count"))
    grouped.to_csv(args.outdir/"summary.csv",index=False)
    print(grouped.to_string(index=False))


if __name__=="__main__":
    main()
