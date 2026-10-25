"""Tests use public validation labels and public query IDs only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from score_authorization_validation import load_validation, score
from authorization_starter import run
from prepare_issue_submission import prepare_issue, IssueSubmissionError


class FirstSubmissionTest(unittest.TestCase):
    def test_validation_is_disjoint_and_scores_use_core(self):
        task=load_validation()
        public=json.loads((ROOT/'task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/safety_task.json').read_text())
        self.assertEqual((len(task.measurements),len(task.queries)),(768,384))
        self.assertFalse({r.query_id for r in task.queries}&{q['query_id'] for q in public['queries']})
        with tempfile.TemporaryDirectory() as d:
            report=run(Path(d)/'starter')
            self.assertEqual(report['preflight'],'preflight-passed')
            self.assertEqual(sum(report['action_counts'].values()),384)
            with self.assertRaises(ValueError):
                score(Path(d)/'starter/predictions.csv')

    def test_apps_export_is_held_and_preflight_rejects(self):
        self.assertFalse((ROOT/'task-packs/ai-control/controlarena-apps-open-replay/v1').exists())
        card={'schema_version':'observerbench.safety_observer_card.v0','observer_name':'test','observer_version':'1',
            'observer_family':'test','observer_input':'text','fit_procedure':'none','implementation':'local test',
            'risk_score_meaning':'larger is riskier','access_regime':'text only'}
        body='\n'.join(['### Task and version','controlarena-apps-open-replay@v1','### Observer name and version','test@1',
            '### Prediction-table URL','https://raw.githubusercontent.com/example/repo/'+'a'*40+'/predictions.csv',
            '### ObserverCard JSON',json.dumps(card)])
        event={'issue':{'number':1,'body':body},'repository':{'full_name':'local/test'}}
        header='schema_version,sample_id,monitor_name,score\n'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(IssueSubmissionError):
                prepare_issue(event,Path(d)/'invalid',prediction_bytes=header.encode())


if __name__=='__main__':
    unittest.main()
