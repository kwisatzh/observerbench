from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_visible_cleanup():
    for p in (ROOT/'site').rglob('*.html'):
        text = p.read_text()
        assert 'Do better predictions lead to better decisions?' in text.split('<footer',1)[1]
        assert 'schema valid' not in text.lower()
        assert 'pending approval' not in text.lower()
    run = (ROOT/'site/runners/index.html').read_text().lower()
    for phrase in ['choose a boundary','public boundary','read the boundary','no runner needed','inference-free','local scores','python contract']:
        assert phrase not in run
    submit = (ROOT/'site/submit/index.html').read_text()
    assert 'We compute file hashes on receipt.' in submit
    assert 'Include the task version and file SHA-256' not in submit
    js = (ROOT/'site/assets/site.js').read_text()
    assert 'panel.id === "qwen-paired-scope-v0"' in js
    assert 'return "Average decision cost"' in js
    assert 'return "AUROC (ranking accuracy)"' in js
    assert '["Panel",' not in js and '["Global rank",' not in js

def test_shorter_try_preserves_full_scoring():
    js = (ROOT/'site/try/playground.js').read_text()
    assert 'selectedResult.ranked.slice(0, 8).forEach' in js
    assert 'task.rows.slice().sort' in js
    page = (ROOT/'site/try/index.html').read_text()
    assert 'scores above use all 16' in page
    assert 'make demo-qwen' in page
