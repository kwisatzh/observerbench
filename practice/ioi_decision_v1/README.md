# From predicted effects to chosen actions

**Submit a prediction table. See the actions it chooses and what they cost.**

This CPU-only task uses stored measurements from GPT-2-small (124M parameters).
It includes raw and scalar-calibrated attribution patching, additive ridge,
and an interaction-aware ridge observer. No model download or API key is needed
to score their tables or your own.

## Run the example

From the repository root, install the small numerical stack once:

```bash
python -m pip install numpy pandas matplotlib
python examples/ioi_decision_submission.py
```

The example scores the supplied calibrated attribution-patching table. It
writes to `outputs/ioi-decision/`:

- `scorecard.txt`: prediction error, action loss, and comparisons with the
  four reference observers and doing nothing.
- `selected_actions.csv`: the action chosen in each pool, its predicted
  effect, and its measured outcome.
- `decision_evaluation.json`: the same scores with paired uncertainty
  intervals, task identity, and file hashes.
- Prediction metrics, the TaskCard, and the ObserverCard.

Scoring is inference-free. Producing the original attribution-patching table
required white-box gradients over 192 training prompts; its calibrated variant
also used 160 finite measurements. Cached scoring does not erase that access cost.

## Try your own observer

Fit using `calibration_measurements.csv`: 160 masks and their mean effects over
training prompts. Predict one mean effect for each of the 320 masks in
`queries.csv`. A `1` in `mask_bits` means that head is ablated. The order is the
three primary Name Movers, eight Backup Name Movers, and two Negative Name
Movers, as specified in the [canonical head list](../../src/observerbench/tasks/ioi/heads.py).

Keep the existing effect-table format:

```csv
schema_version,query_id,predicted_effect
observerbench.effect_predictions.v0,mask_id_from_queries,0.73
```

Copy a JSON ObserverCard from `submissions/`, then describe **your** method,
fitting procedure, and information access. Run:

```bash
python examples/ioi_decision_submission.py \
  --predictions my_predictions.csv \
  --observer-card my_observer.json \
  --outdir outputs/my-ioi-observer
```

The evaluator accepts data, not executable observer code. It rejects missing,
duplicate, unexpected, or non-finite predictions. These are predictions of
**mean effects**, not expected action losses; substituting one for the other
would change the decision rule.

## What the task fixes

There are ten pools, each with 32 measured masks plus exact no-op. For each
target (0.5, 1, and 1.5 logits), the controller chooses the predicted mean
closest to the target. Ties go to fewer ablated heads, then mask ID. It makes
one choice per pool for all 256 held-out prompts; it cannot inspect each
prompt's outcome and pick a different action afterward.

The score is the average absolute distance between the selected action's effect
and the target. A mask that produces effects 0 and 2 has mean 1 but loss 1 at
target 1. A mask that always produces 0.8 has loss 0.2. This is why the scorecard
reports prediction quality **and** decision quality.

Doing nothing produces effect zero, so its loss is exactly the target.
The displayed comparison uses target 1; the scorecard always reports all three.
Intervals resample 107 unordered-name-pair clusters and ten pools, keeping the
fitted observers and selected actions fixed. They do not measure variation
from retraining the observers.

## What this result can claim

This is an **open replay**, not a sealed leaderboard submission or a new
confirmatory experiment. The evaluator responses are public in `responses.npz`
so anyone can reproduce the scores. Do not fit to them and describe the score
as held out. The pack reuses the original Phase 5 panel and adds exact no-op;
it is separate from the later clean-task Phase 7 confirmation in the paper.

The [decision comparison](../../leaderboards/decision/ioi-gpt2-small-action-selection-b160/)
records the bundled rows. New local submissions receive comparisons immediately;
they are not automatically published or admitted to a sealed ranking.

To rebuild the pack from the original local experiment files and regenerate
all four scorecards, use `python scripts/build_ioi_decision_replay.py --score`.
That maintenance command needs the archived Phase 5 files. The example above
needs only this pack and the repository code.
