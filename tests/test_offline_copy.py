"""Copy contract for the local-only October 24 revision."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_monthly_draft_and_no_hosted_promises():
    submit = (ROOT/'site/submit/index.html').read_text()
    assert 'Monthly batches, scored within 30 days' in submit
    assert 'three submissions per person per task every 30 days' in submit
    assert 'pending approval' not in submit
    assert 'checks this limit by hand' in submit
    assert '<details id="formats">' in submit
    for name in ['index.html','runners/index.html','submit/index.html']:
        page = (ROOT/'site'/name).read_text()
        for obsolete in ['Public submission checks are automatic','Requires the private-test evaluator',
                         'enabled later','Automatic preflight currently accepts']:
            assert obsolete not in page

@pytest.mark.skipif(not shutil.which('node'), reason='Node required')
def test_plain_badges_preserve_status_and_order():
    subprocess.run(['node','-e',r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('site/assets/site.js','utf8').replace(
 'if (page === "leaderboards") setupLeaderboards();',
 'globalThis.api={statusLabel,eligibleOrders};');
const context={document:{body:{dataset:{}}}};vm.runInNewContext(source,context);
const rows=[{result_status:'post-panel matched control, added 2026-10-24; not pre-specified'},
 {result_status:'prospectively frozen matched replication'}, {result_status:'evaluator-only bound'}];
const before=JSON.stringify(rows);
assert.equal(JSON.stringify(rows.map(context.api.statusLabel)),JSON.stringify(['added later','planned','oracle']));
assert.equal(JSON.stringify(context.api.eligibleOrders(rows)),JSON.stringify(['—','1','—']));
assert.equal(before,JSON.stringify(rows));
assert(source.includes('badge.title = [row.result_status'));
'''],cwd=ROOT,check=True,capture_output=True,text=True)
