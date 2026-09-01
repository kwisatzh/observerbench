from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data" / "safety_tutorial.json"
SCRIPT = ROOT / "demo" / "safety_tutorial.py"
SPEC = importlib.util.spec_from_file_location("safety_tutorial", SCRIPT)
assert SPEC and SPEC.loader
SAFETY_TUTORIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY_TUTORIAL)


class SafetyTutorialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = json.loads(DATA.read_text(encoding="utf-8"))
        self.rows = self.task["rows"]

    def test_fixed_example_separates_classification_from_action_loss(self) -> None:
        probability = SAFETY_TUTORIAL.evaluate(
            self.rows,
            lambda row: float(row["violation_probability"]),
            self.task["audit_budget"],
            self.task["false_audit_cost"],
        )
        consequence = SAFETY_TUTORIAL.evaluate(
            self.rows,
            SAFETY_TUTORIAL.score,
            self.task["audit_budget"],
            self.task["false_audit_cost"],
        )

        self.assertAlmostEqual(probability["auroc"], 0.95)
        self.assertAlmostEqual(consequence["auroc"], 0.70)
        self.assertGreater(probability["auroc"], consequence["auroc"])
        self.assertAlmostEqual(probability["decision_loss"], 24.25)
        self.assertAlmostEqual(consequence["decision_loss"], 6.25)
        self.assertGreater(probability["decision_loss"], consequence["decision_loss"])

    def test_demo_uses_system_python_without_package_imports(self) -> None:
        completed = subprocess.run(
            ["python3", str(SCRIPT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Higher classification AUROC produced the worse safety decision", completed.stdout)
        self.assertIn("Public targets; this run is not leaderboard eligible", completed.stdout)
        source = SCRIPT.read_text(encoding="utf-8")
        for dependency in ("numpy", "pandas", "torch", "observerbench"):
            self.assertNotIn(f"import {dependency}", source)

    def test_browser_playground_is_static_and_explicitly_a_tutorial(self) -> None:
        html = (ROOT / "site" / "try" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "site" / "try" / "playground.js").read_text(encoding="utf-8")

        self.assertIn("not leaderboard eligible", html)
        self.assertIn("One score per request", html)
        self.assertIn('src="playground.js"', html)
        self.assertNotIn('<script src="http', html)
        self.assertIn('../data/safety_tutorial.json', javascript)
        self.assertNotIn("eval(", javascript)
        self.assertNotIn("new Function", javascript)

    def test_homepage_and_readme_link_to_tutorial(self) -> None:
        homepage = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('href="try/">Try it in one minute</a>', homepage)
        self.assertIn("## Sixty-second demo", readme)
        self.assertIn("make demo", readme)


if __name__ == "__main__":
    unittest.main()
