"""Reproduce the cost table from recorded stage timings, without a GPU.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from observerbench.core import write_json


def summarize(result):
    stages = {
        "basic_prompt": ("warm_basic_output_measurement",),
        "detailed_prompt": ("warm_detailed_output_measurement",),
        "dense_neutral": ("warm_dense_neutral_measurement", "warm_dense_projection_and_readout"),
        "sae_neutral": ("warm_sae_neutral_measurement", "warm_sae_encoding_with_weight_transfer", "warm_sae_readout"),
    }
    rows = []
    for observer, names in stages.items():
        totals, peaks = [], []
        for repeat in range(result["protocol"]["warm_inference_repeats"]):
            records = [r for r in result["timings"] if r["stage"] in names and r.get("repeat") == repeat]
            if len(records) != len(names) or {r["stage"] for r in records} != set(names):
                raise ValueError(f"Missing or duplicate timing: {observer}, repeat {repeat}")
            if {r["n_rows"] for r in records} != {64}:
                raise ValueError("The published timing panel contains 64 rows")
            totals.append(sum(r["seconds"] for r in records))
            peaks.append(max(r["gpu_peak_allocated_bytes"] for r in records))
        rows.append({"observer": observer, "n_rows": 64, "repeat_seconds": totals,
                     "median_seconds": median(totals), "min_seconds": min(totals),
                     "max_seconds": max(totals), "max_gpu_allocated_gib": max(peaks) / 2**30})
    return {"schema": "observerbench.gemma_observer_cost.summary.v1", "rows": rows,
            "scope": "Sum stages within each repeat, then take the median. Warm times include tokenization, full model forward, extraction and any encoder/readout. Prompt rendering, cold loading and cached audit selection are reported separately. This is a post-outcome cost follow-up, not a new quality ranking."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.results.read_text()))
    write_json(args.out, report)
    for row in report["rows"]:
        print(f"{row['observer']}: {row['median_seconds']:.3f}s "
              f"[{row['min_seconds']:.3f}, {row['max_seconds']:.3f}], "
              f"{row['max_gpu_allocated_gib']:.2f} GiB")
