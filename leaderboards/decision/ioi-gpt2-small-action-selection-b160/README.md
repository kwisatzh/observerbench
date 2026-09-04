# GPT-2-small IOI: submitted effects scored as actions

This panel compares four existing observers through one fixed mean-based action
rule. Each row comes from the same CSV submission path available to outside
users. The comparison includes exact no-op and reports all three targets.

**Primary displayed metric:** action loss at target 1. Lower is better.
Prediction MAE is a separate metric, not a proxy for this loss. White-box
attribution patching and forward-only fitted observers have different access
costs; their position in this table does not establish equal-cost superiority.

At target 1, none of the four observers establishes an improvement over no-op
under the paired 95% intervals. All lose to no-op at target 0.5 and beat it at
target 1.5. Those intervals resample held-out name-pair clusters and action
pools, not independent observer fits. Read the full scorecards under
[`results/revision/ioi_decision_replay_v1`](../../../results/revision/ioi_decision_replay_v1/).

This is a secondary **open replay** of the original Phase 5 panel, not the
later Phase 7 confirmation and not a sealed score. Historical experiment
results remain unchanged.

[Run your own table](../../../practice/ioi_decision_v1/README.md).
