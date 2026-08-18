from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_issue_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_issue_submission", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def card() -> dict[str, object]:
    return {
        "schema_version": "observerbench.safety_observer_card.v0",
        "observer_name": "issue-observer",
        "observer_version": "1.0.0",
        "observer_family": "test",
        "observer_input": "output scores",
        "fit_procedure": "fixed",
        "implementation": "https://example.org/observer",
        "risk_score_meaning": "larger is riskier",
        "known_failure_modes": [],
        "access_regime": "output only",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "metadata": {},
    }


def event(prediction_url: str) -> dict[str, object]:
    body = "\n\n".join(
        (
            "### Task and version\n\nsafety-interlock-qwen2-5-7b-instruct@paired-scope-v1",
            "### Observer name and version\n\nissue-observer@1.0.0",
            f"### Prediction-table URL\n\n{prediction_url}",
            "### ObserverCard JSON\n\n```json\n" + json.dumps(card()) + "\n```",
        )
    )
    return {
        "issue": {
            "number": 17,
            "html_url": "https://github.com/example/observerbench/issues/17",
            "body": body,
            "user": {"login": "researcher"},
        },
        "repository": {"full_name": "example/observerbench"},
    }


class IssueSubmissionTest(unittest.TestCase):
    def test_prepares_valid_issue_without_network_in_test(self) -> None:
        url = "https://raw.githubusercontent.com/example/repo/" + "a" * 40 + "/predictions.csv"
        predictions = (
            "schema_version,query_id,predicted_risk\n"
            "observerbench.safety_predictions.v0,q-1,0.7\n"
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = module.prepare_issue(event(url), output, prediction_bytes=predictions)
            self.assertEqual(result["status"], "preflight-passed")
            self.assertEqual(result["issue_number"], 17)
            self.assertEqual(len(result["prediction_sha256"]), 64)

    def test_rejects_unpinned_url(self) -> None:
        with self.assertRaisesRegex(module.IssueSubmissionError, "immutable"):
            module.validate_prediction_url(
                "https://raw.githubusercontent.com/example/repo/main/predictions.csv"
            )

    def test_rejects_non_github_url(self) -> None:
        with self.assertRaisesRegex(module.IssueSubmissionError, "raw.githubusercontent"):
            module.validate_prediction_url("https://example.org/predictions.csv")

    def test_rejects_identity_mismatch(self) -> None:
        url = "https://raw.githubusercontent.com/example/repo/" + "b" * 40 + "/predictions.csv"
        payload = event(url)
        payload["issue"]["body"] = payload["issue"]["body"].replace(
            "issue-observer@1.0.0", "somebody-else@1.0.0", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(module.IssueSubmissionError, "must match"):
                module.prepare_issue(
                    payload,
                    Path(directory),
                    prediction_bytes=(
                        "schema_version,query_id,predicted_risk\n"
                        "observerbench.safety_predictions.v0,q-1,0.7\n"
                    ).encode(),
                )


if __name__ == "__main__":
    unittest.main()
