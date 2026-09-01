# Qwen effect-prediction practice task

This is a small, fully open view of
`induction-qwen2-5-7b-finite-effects@copy-v2-b040`. It uses measurements that
were already produced with the pinned Qwen2.5-7B model. You do not need the
model, a GPU, an API key, or third-party Python packages.

The question is simple: after seeing 40 interventions, can your observer
predict 128 new intervention effects? More importantly, do those predictions
choose good interventions?

## Try it

From the repository root:

```bash
python3 practice/qwen_copy_v2_b040/baseline.py \
  --output /tmp/qwen-practice-predictions.csv
python3 practice/qwen_copy_v2_b040/score.py \
  /tmp/qwen-practice-predictions.csv
```

The first command fits a nine-parameter additive model. Edit `baseline.py`, or
write predictions with the same three-column CSV format, and score again.

The scorer reports two different outcomes:

- **MAE and RMSE:** how closely the observer predicts held-out mean effects.
- **Action loss and regret:** how well those predictions choose an intervention
  from each fixed candidate pool.

Every action set contains eight intervention masks and the exact no-op. For a
target `t`, the controller chooses the smallest predicted `|effect - t|`. The
reported action loss is the held-out prompt average `E|Y_i(mask) - t|`, not the
error of the mean effect. No-op has effect zero and loss exactly `t`.

## Files

- `calibration_measurements.csv`: 40 public mask features and mean effects.
- `queries.csv`: features for 128 held-out masks in 16 candidate pools.
- `targets.csv`: public held-out mean effects used for instant practice scoring.
- `action_outcomes.csv`: prompt-level realized loss aggregated for each fixed
  target, pool, and action, including no-op.
- `task_card.json`: model identity, head mapping, contracts, source hashes, and
  scope limits.
- `baseline.py`: dependency-free example observer.
- `score.py`: dependency-free open scorer.

## Status

This is an **open practice task**. Its targets are public, so its results are
not eligible for a sealed leaderboard and should not be described as such. It
exists to make the ObserverBench contract easy to inspect, modify, and run.

The builder at `scripts/build_qwen_copy_v2_b040_practice_pack.py` verifies the
frozen scientific manifests and source hashes before recreating these aggregate
files. It never runs the model or changes the source artifacts.
