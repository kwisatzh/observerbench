"""Compatibility check against ObserverBench's existing safety-pack loader.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_blinded_pack_contract", ROOT / "scripts" / "build_blinded_pack.py"
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

try:
    from observerbench.safety import evaluate_safety_predictor
    from observerbench.safety_task_pack import (
        load_safety_task_pack,
        validate_public_safety_task_pack,
    )
except ImportError:
    evaluate_safety_predictor = None
    load_safety_task_pack = None
    validate_public_safety_task_pack = None


@unittest.skipUnless(
    load_safety_task_pack is not None,
    "run with the ObserverBench source and dependencies on PYTHONPATH",
)
class ActualSafetyPackContractTests(unittest.TestCase):
    def test_builder_output_validates_and_loads_with_observerbench(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "evaluator.key"
            secret.write_bytes(b"observerbench-contract-test-secret-00001")
            outdir = root / "pack"
            BUILDER.build_qwen_safety_pack(
                source=ROOT / "tests" / "fixtures" / "qwen_prompts.csv",
                secret_path=secret,
                outdir=outdir,
                task_id=BUILDER.QWEN_PUBLIC_TASK_ID,
            )

            public_path = outdir / "public" / "safety_task.json"
            targets_path = outdir / "private" / "evaluator_targets.json"
            summary = validate_public_safety_task_pack(public_path)
            self.assertEqual(summary["task_id"], BUILDER.QWEN_PUBLIC_TASK_ID)
            self.assertEqual(summary["n_measurements"], 3)
            self.assertEqual(summary["n_queries"], 2)

            task = load_safety_task_pack(public_path, targets_path)
            self.assertEqual(task.name, BUILDER.QWEN_PUBLIC_TASK_NAME)
            self.assertEqual(task.version, BUILDER.QWEN_PUBLIC_TASK_VERSION)
            self.assertEqual(len(task.measurements), 3)
            self.assertEqual(len(task.queries), 2)
            self.assertEqual(len(task.targets), 2)
            self.assertEqual(task.policy.block_budget_fraction, 0.10)
            self.assertEqual(task.policy.escalation_budget_fraction, 0.10)
            self.assertEqual(task.policy.escalation_cost, 0.05)
            self.assertEqual(task.policy.cvar_alpha, 0.90)

            class SeverityOnlyObserver:
                name = "severity-only-contract-smoke"

                def fit(self, measurements: object) -> None:
                    self.measurement_count = len(measurements)

                def predict_risk(self, queries: object) -> list[float]:
                    return [float(query.features["severity"]) for query in queries]

            observer = SeverityOnlyObserver()
            result = evaluate_safety_predictor(task, observer)
            self.assertEqual(observer.measurement_count, 3)
            self.assertEqual(result.observer_name, observer.name)
            self.assertIn("protocol_loss_mean", result.metrics)


if __name__ == "__main__":
    unittest.main()
