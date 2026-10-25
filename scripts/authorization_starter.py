#!/usr/bin/env python3
"""CPU-only text ridge starter; no model inference or hidden labels."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from observerbench.safety import SafetyObserverCard, SAFETY_PREDICTION_CSV_COLUMNS, SAFETY_PREDICTION_CSV_SCHEMA_VERSION
from score_authorization_validation import load_validation, score, DEFAULT_PACK
from prepare_issue_submission import prepare_issue


def write_predictions(path, queries, scores):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(SAFETY_PREDICTION_CSV_COLUMNS)
        for q, value in zip(queries, scores):
            writer.writerow([SAFETY_PREDICTION_CSV_SCHEMA_VERSION, q['query_id'], float(value)])


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    task = load_validation()
    # Hyperparameters fixed in advance. No validation labels enter fitting.
    model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=12000), Ridge(alpha=10, solver='lsqr'))
    model.fit([r.features['user_prompt'] for r in task.measurements], [float(r.unsafe_if_allowed) for r in task.measurements])
    def predict(rows):
        probability = model.predict([r['features']['user_prompt'] for r in rows]).clip(0,1)
        return probability * [r['features']['severity'] for r in rows]
    validation_rows = [asdict(q) for q in task.queries]
    write_predictions(output / 'validation_predictions.csv', validation_rows, predict(validation_rows))
    public = json.loads((DEFAULT_PACK.parent / 'safety_task.json').read_text())
    write_predictions(output / 'predictions.csv', public['queries'], predict(public['queries']))
    card = asdict(SafetyObserverCard(observer_name='text-ridge-starter', observer_version='20261024',
        observer_family='character TF-IDF linear ridge', observer_input='public user_prompt and action severity',
        fit_procedure='Fit bank only; character 3-5 grams, max 12000 features, ridge alpha 10; no validation tuning',
        implementation='scripts/authorization_starter.py', risk_score_meaning='clipped estimated unsafe probability multiplied by severity; higher means intervene first',
        access_regime='prompt text only; no Qwen activations or outputs', requires_white_box_access=False,
        additional_forward_passes=0, known_failure_modes=('Lexical baseline; may not generalize to unseen authorization forms.',)))
    (output / 'observer_card.json').write_text(json.dumps(card, indent=2) + '\n')
    event = {'repository': {'full_name':'local/dry-run'}, 'issue': {'number':1,
        'body': '\n'.join(['### Task and version', 'safety-interlock-qwen2-5-7b-instruct@paired-scope-v1',
            '### Observer name and version', 'text-ridge-starter@20261024', '### Prediction-table URL',
            'https://raw.githubusercontent.com/example/repo/' + '0'*40 + '/predictions.csv',
            '### ObserverCard JSON', json.dumps(card)])}}
    preflight = prepare_issue(event, output / 'preflight', prediction_bytes=(output / 'predictions.csv').read_bytes())
    report = {**score(output / 'validation_predictions.csv'), 'preflight': preflight['status'], 'elapsed_seconds':time.monotonic()-started}
    (output / 'validation_scorecard.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
