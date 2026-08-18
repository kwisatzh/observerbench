from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_public_submission.py"
SPEC = importlib.util.spec_from_file_location("validate_public_submission", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def safety_card() -> dict[str, object]:
    return {
        "schema_version": "observerbench.safety_observer_card.v0",
        "observer_name": "test-observer",
        "observer_version": "1.0.0",
        "observer_family": "test family",
        "observer_input": "public fixture",
        "fit_procedure": "fixed",
        "implementation": "https://example.org/test",
        "risk_score_meaning": "larger is riskier",
        "known_failure_modes": [],
        "access_regime": "output only",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "metadata": {},
    }


class PublicSubmissionTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        bundle = root / "safety" / "task@v0" / "test-observer-v1"
        bundle.mkdir(parents=True)
        (bundle / "observer_card.json").write_text(
            json.dumps(safety_card()), encoding="utf-8"
        )
        (bundle / "predictions.csv").write_text(
            "schema_version,query_id,predicted_risk\n"
            "observerbench.safety_predictions.v0,q-1,0.2\n"
            "observerbench.safety_predictions.v0,q-2,1.3\n",
            encoding="utf-8",
        )
        return bundle

    def test_valid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_bundle(root)
            result = validator.validate_root(root)
            self.assertEqual(result[0]["prediction_count"], 2)
            self.assertEqual(result[0]["status"], "preflight-passed")

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            (bundle / "predictions.csv").write_text(
                "schema_version,query_id,predicted_risk\n"
                "observerbench.safety_predictions.v0,q-1,0.2\n"
                "observerbench.safety_predictions.v0,q-1,1.3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(validator.SubmissionError, "duplicate"):
                validator.validate_root(root)

    def test_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            (bundle / "run.py").write_text("print('never execute me')", encoding="utf-8")
            with self.assertRaisesRegex(validator.SubmissionError, "expected only"):
                validator.validate_root(root)

    def test_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            (bundle / "predictions.csv").write_text(
                "schema_version,query_id,predicted_risk\n"
                "observerbench.safety_predictions.v0,q-1,nan\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(validator.SubmissionError, "finite"):
                validator.validate_root(root)

    def test_rejects_two_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_bundle(root)
            second = first.parent / "second"
            second.mkdir()
            for name in ("predictions.csv", "observer_card.json"):
                (second / name).write_bytes((first / name).read_bytes())
            with self.assertRaisesRegex(validator.SubmissionError, "exactly one"):
                validator.validate_root(root)


if __name__ == "__main__":
    unittest.main()
