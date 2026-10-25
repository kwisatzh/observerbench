#!/usr/bin/env python3
"""Maintainer-only, network-free authorization scoring. Never runs submitted code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from validate_public_submission import validate_submission
from observerbench.safety import read_safety_predictions, evaluate_safety_scores
from observerbench.safety_task_pack import load_safety_task_pack

METRICS = ('protocol_loss_mean', 'protocol_loss_cvar', 'risk_auroc',
           'unsafe_allowed_rate', 'severity_weighted_miss_rate', 'false_block_rate',
           'clean_utility_retained', 'escalation_rate', 'mean_risk_score', 'worst_family_protocol_loss')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def score(predictions, card, task, targets, output, expected):
    paths = dict(predictions=Path(predictions), card=Path(card), task=Path(task), targets=Path(targets))
    output = Path(output)
    for path in (paths['targets'], output):
        if path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Protected inputs and draft outputs must stay outside the public checkout')
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or digest(path) != expected[name]:
            raise ValueError('Input seal mismatch')
    if paths['predictions'].stat().st_size > 5 * 1024 * 1024 or paths['card'].stat().st_size > 65536:
        raise ValueError('Submission exceeds size limit')
    # Copy inert bytes into an isolated bundle; never import anything supplied by a submitter.
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        bundle = base / 'safety' / 'authorization' / 'offline'
        bundle.mkdir(parents=True)
        (bundle / 'predictions.csv').write_bytes(paths['predictions'].read_bytes())
        (bundle / 'observer_card.json').write_bytes(paths['card'].read_bytes())
        preflight = validate_submission(bundle, root=base)
        predictions_by_id = read_safety_predictions(bundle / 'predictions.csv')
        observer_card = json.loads((bundle / 'observer_card.json').read_text())
        # Bind the bytes actually validated, not just the original paths.
        if digest(bundle / 'predictions.csv') != expected['predictions'] or digest(bundle / 'observer_card.json') != expected['card']:
            raise ValueError('Submission changed during validation')
    loaded = load_safety_task_pack(paths['task'], paths['targets'])
    ids = {q.query_id for q in loaded.queries}
    if set(predictions_by_id) != ids or {t.query_id for t in loaded.targets} != ids:
        raise ValueError('Exact test coverage required')
    result = evaluate_safety_scores(loaded, [predictions_by_id[q.query_id] for q in loaded.queries], observer_name=preflight['observer_name'])
    if any(digest(path) != expected[name] for name, path in paths.items()):
        raise ValueError('Inputs changed during scoring')
    row = {k: observer_card[k] for k in ('observer_name', 'observer_version', 'observer_family', 'access_regime', 'fit_procedure')}
    row.update(result_status='draft offline score; pending maintainer approval',
               metrics={k: float(result.metrics[k]) for k in METRICS if k in result.metrics})
    draft = {'status': 'DRAFT; no publication performed', 'task_id': f'{loaded.name}@{loaded.version}',
             'submission_sha256': {k: expected[k] for k in ('predictions', 'card', 'task')},
             'n_queries': len(ids), 'additional_rows': [row]}
    # Only aggregate allowlisted values. No labels, per-query results, target hash, or private paths.
    with output.open('x', encoding='utf-8') as handle:
        output.chmod(0o600)
        handle.write(json.dumps(draft, indent=2, allow_nan=False) + '\n')
    return draft

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'card', 'task', 'targets'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        score(args.predictions, args.card, args.task, args.targets, args.output,
              {k: getattr(args, k + '_sha256') for k in ('predictions', 'card', 'task', 'targets')})
    except Exception:
        # Do not echo exceptions from protected parsing or evaluation into shareable logs.
        print('Offline scoring failed; no result should be published. Check local inputs and seals.', file=sys.stderr)
        return 1
    print('Draft aggregate row written locally. Review before publication.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
