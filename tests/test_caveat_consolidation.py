from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def test_one_about_section_and_all_footer_links():
    pages = list((ROOT/'site').rglob('*.html'))
    assert len(pages) == 5
    assert sum(p.read_text().count('id="about-results"') for p in pages) == 1
    for p in pages:
        footer = p.read_text().split('<footer',1)[1]
        assert '#about-results' in footer
    board = (ROOT/'site/leaderboards/index.html').read_text()
    about = board.split('id="about-results"',1)[1].split('</section>',1)[0]
    assert about.count('<li>') == 6
    assert 'APPS submissions are paused' in about
    assert 'RESULT_QUALIFICATIONS_20261024.md' in about

def test_submission_instructions_stay():
    page = (ROOT/'site/submit/index.html').read_text()
    for phrase in ['Do not train on validation examples','Use validation IDs for local scoring and test IDs for submissions',
                   'Supply every query ID exactly once','no labels, index column, missing rows, NaN, or infinity']:
        assert phrase in page

def test_removed_warnings_live_in_record_not_pages():
    text = '\n'.join(p.read_text() for p in (ROOT/'site').rglob('*.html'))
    for phrase in ['not an independent replication','not a measured human audit','not a safety-monitor comparison',
                   'This is a simplified teaching file','Public validation scores do not become']:
        assert phrase not in text
    assert 'The scores are supplied; nothing runs in your browser.' in text
    js = (ROOT/'site/assets/site.js').read_text()
    assert 'shortPanelNote(panelNote' in js
    assert 'details.href = `${root}/${activePanel.data_url}`' in js
    assert '["Global rank", "never issued"]' not in js
