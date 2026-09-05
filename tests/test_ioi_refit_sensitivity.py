# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ioi_refit",ROOT/"scripts/run_ioi_refit_sensitivity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_halves_are_disjoint_complete_and_grouped():
    prompts = pd.DataFrame({"io_name":["a","b","c","d","e","f"],"s_name":["b","a","d","c","f","e"]})
    splits = list(module.training_halves(prompts,[11,12]))
    assert splits[0][0]=="full"
    for a,b in (splits[1:3],splits[3:5]):
        assert set(a[1]).isdisjoint(b[1])
        assert set(a[1])|set(b[1])==set(range(6))
        assert all((i in a[1])==(i+1 in a[1]) for i in (0,2,4))


def test_full_fit_reproduces_existing_forward_predictions():
    pack = ROOT/"practice/ioi_decision_v1"
    calibration = pd.read_csv(pack/"calibration_measurements.csv",dtype={"mask_bits":str})
    queries = pd.read_csv(pack/"queries.csv",dtype={"mask_bits":str})
    # Enrich the interchange table with the original basis counts.
    from observerbench.tasks.ioi.heads import head_records
    groups = [r["group"] for r in head_records()]
    for frame,id_column in ((calibration,"measurement_id"),(queries,"query_id")):
        frame["mask_id"] = frame[id_column]
        for group in ("P","B","E"):
            frame[f"n_{group}"] = [sum(int(bit) for bit,g in zip(bits,groups) if g==group) for bits in frame.mask_bits]
    predictions,_ = module.fit_predictions(calibration,queries,calibration.observed_effect.to_numpy(),np.ones(13),1e-6)
    for name in ("additive","all_pairs"):
        expected = pd.read_csv(pack/f"submissions/{name}.csv").set_index("query_id").loc[queries.query_id].predicted_effect
        np.testing.assert_allclose(predictions[name],expected,atol=1e-6)
