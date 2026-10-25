#!/usr/bin/env python3
"""Derive public validation exclusively from the already-public calibration bank."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1'


def build(source=PACK / 'safety_task.json', output=PACK / 'validation.json'):
    raw = source.read_bytes()
    original = json.loads(raw)
    fit = [r for r in original['measurements'] if r['metadata']['measurement_bank'] == 'fit']
    validation = [r for r in original['measurements'] if r['metadata']['measurement_bank'] == 'calibration']
    if not fit or not validation or len(fit) + len(validation) != len(original['measurements']):
        raise ValueError('Expected exactly the public fit and calibration banks')
    # Test IDs, pairs, and exact inputs are checked without loading any test labels.
    groups = [fit, validation, original['queries']]
    for field in ('pair_id',):
        sets = [{r[field] for r in rows} for rows in groups]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Split pair overlap')
    ids = [{r.get('measurement_id', r.get('query_id')) for r in rows} for rows in groups]
    prompts = [{json.dumps(r['features'], sort_keys=True) for r in rows} for rows in groups]
    for sets in (ids, prompts):
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Split identity or input overlap')
    result = {
        'schema_version': 'observerbench.public_validation.v1',
        'source_task_sha256': hashlib.sha256(raw).hexdigest(),
        'task_name': original['task_name'],
        'task_version': 'paired-scope-v1-public-validation-20261024',
        'policy': original['policy'],
        'task_card': {**original['task_card'],
            'task_version': 'paired-scope-v1-public-validation-20261024',
            'train_split': 'public fit bank only',
            'evaluation_split': 'already-public calibration bank; development only'},
        'measurements': fit,
        'queries': [{ 'query_id': r['measurement_id'], **{k:r[k] for k in ('features','family_id','pair_id','metadata')}} for r in validation],
        'labels': [{ 'query_id': r['measurement_id'], **{k:r[k] for k in ('unsafe_if_allowed','severity','benign_value')}} for r in validation],
        'scope': 'Development feedback, not a hidden-test leaderboard result. Labels were already public. Do not fit on these validation rows when reporting validation error.',
        'disjointness': {'ids': True, 'pairs': True, 'exact_inputs': True, 'private_labels_read': False},
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'fit':len(fit),'validation':len(validation),'test':len(original['queries']), 'disjointness': result['disjointness']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PACK / 'safety_task.json')
    parser.add_argument('--output', type=Path, default=PACK / 'validation.json')
    args = parser.parse_args()
    build(args.source, args.output)
