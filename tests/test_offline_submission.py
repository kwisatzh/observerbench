"""Offline tests use synthetic targets only, never the authorization hidden test."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_submission_offline import score, digest, METRICS
from observerbench.tasks.safety_interlock import SafetyInterlockConfig, make_safety_interlock_task
from observerbench.safety_task_pack import write_safety_task_pack
from observerbench.safety import evaluate_safety_scores

def test_offline_scoring_and_rejections(tmp_path):
    task = make_safety_interlock_task(SafetyInterlockConfig(n_train_pairs=8, n_test_pairs=10, seed=7))
    public, targets = write_safety_task_pack(task, tmp_path / 'synthetic')
    card = tmp_path / 'card.json'
    card.write_text(json.dumps(dict(schema_version='observerbench.safety_observer_card.v0', observer_name='test',
        observer_version='1', observer_family='test', observer_input='synthetic', fit_procedure='none',
        implementation='test', risk_score_meaning='higher is riskier', access_regime='synthetic')))
    predictions = tmp_path / 'predictions.csv'
    predictions.write_text('schema_version,query_id,predicted_risk\n' + ''.join(
        f'observerbench.safety_predictions.v0,{q.query_id},0.5\n' for q in task.queries))
    paths = dict(predictions=predictions, card=card, task=public, targets=targets)
    hashes = {k:digest(p) for k,p in paths.items()}
    output = tmp_path / 'draft.json'
    actual = score(**paths, output=output, expected=hashes)
    expected = evaluate_safety_scores(task, [.5] * len(task.queries), observer_name='test')
    assert actual['additional_rows'][0]['metrics'] == {k:expected.metrics[k] for k in METRICS}
    assert 'targets' not in actual['submission_sha256']
    assert not any(q.query_id in output.read_text() for q in task.queries)
    with pytest.raises(FileExistsError): score(**paths, output=output, expected=hashes)
    with pytest.raises(ValueError): score(**paths, output=tmp_path/'bad.json', expected={**hashes, 'card':'0'*64})
    with pytest.raises(ValueError): score(**paths, output=ROOT/'bad.json', expected=hashes)
    predictions.write_text('\n'.join(predictions.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError, match='coverage'):
        score(**paths, output=tmp_path/'missing.json', expected={**hashes,'predictions':digest(predictions)})

def test_no_event_triggered_workflows_or_dispatch_secrets():
    workflows=ROOT/'.github/workflows'
    assert not (workflows/'issue-submission.yml').exists()
    assert not (workflows/'submission-preflight.yml').exists()
    assert not (workflows/'queue-sealed-evaluation.yml').exists()
    for path in workflows.glob('*.yml'):
        text=path.read_text()
        assert 'workflow_dispatch:' in text
        for trigger in ('pull_request:', 'issues:', 'push:', 'repository_dispatch:', 'workflow_run:'):
            assert trigger not in text
    assert not (ROOT/'task-packs/ai-control/controlarena-apps-open-replay').exists()

@pytest.mark.skipif(not shutil.which('node'), reason='Node required')
def test_gemma_secondary_row_is_not_numbered():
    script = '''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('site/assets/site.js','utf8').replace(
 'if (page === "leaderboards") setupLeaderboards();','globalThis.api={eligibleOrders};');
const ctx={document:{body:{dataset:{}}}};vm.runInNewContext(source,ctx);
const rows=JSON.parse(fs.readFileSync('leaderboards/safety/controlarena-apps-gemma2-9b-it-v0/results.json')).rows;
rows.push({observer_name:'full-output-post-panel',result_status:'post-panel matched control; not pre-specified',metrics:{mean_realized_violations_at_p01_b02:2.173}});
rows.sort((a,b)=>a.metrics.mean_realized_violations_at_p01_b02-b.metrics.mean_realized_violations_at_p01_b02);
const orders=ctx.api.eligibleOrders(rows);
assert.equal(orders[rows.findIndex(r=>r.observer_name==='full-output-post-panel')],'—');
assert.equal(orders[rows.findIndex(r=>r.observer_name==='gemma-verdict-adjacent-residual-probe')],'1');
'''
    subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True)
