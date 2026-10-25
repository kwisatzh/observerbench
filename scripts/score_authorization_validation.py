#!/usr/bin/env python3
"""Score a predictions CSV on public validation using the leaderboard controller."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from observerbench.safety import (SafetyTask, SafetyTaskCard, SafetyPolicy, SafetyMeasurement,
    SafetyQuery, SafetyTarget, read_safety_predictions, evaluate_safety_scores)

DEFAULT_PACK = ROOT / 'task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/validation.json'


def load_validation(path=DEFAULT_PACK):
    p = json.loads(Path(path).read_text())
    if p.get('schema_version') != 'observerbench.public_validation.v1':
        raise ValueError('Only explicitly public validation packs are accepted')
    return SafetyTask(name=p['task_name'], version=p['task_version'],
        measurements=tuple(SafetyMeasurement(**r) for r in p['measurements']),
        queries=tuple(SafetyQuery(**r) for r in p['queries']),
        targets=tuple(SafetyTarget(**r) for r in p['labels']),
        policy=SafetyPolicy(**p['policy']), card=SafetyTaskCard(**p['task_card']))


def score(predictions, pack=DEFAULT_PACK):
    task = load_validation(pack)
    values = read_safety_predictions(predictions)
    if set(values) != {q.query_id for q in task.queries}:
        raise ValueError('CSV must cover exactly the public validation query IDs, not private-test IDs')
    result = evaluate_safety_scores(task, [values[q.query_id] for q in task.queries], observer_name='local-validation')
    return {'status': 'public validation only; not leaderboard eligible',
        'task_version': task.version, 'n_queries': result.n_queries,
        'metrics': dict(result.metrics), 'action_counts': dict(result.action_counts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--pack', type=Path, default=DEFAULT_PACK)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(score(args.predictions, args.pack), indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
