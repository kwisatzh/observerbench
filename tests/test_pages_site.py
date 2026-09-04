from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
assert SPEC and SPEC.loader
BUILD_SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_SITE)


class BuildSiteTest(unittest.TestCase):
    def test_ioi_decision_replay_is_a_separate_action_panel(self) -> None:
        path = ROOT / "leaderboards/decision/ioi-gpt2-small-action-selection-b160/results.json"
        payload = json.loads(path.read_text())
        self.assertEqual(payload["participation_class"], "open_replay")
        self.assertEqual(payload["primary_target"], 1.0)
        self.assertEqual(BUILD_SITE._primary_metric(payload["metric_direction"]), "action_loss")
        self.assertEqual(len(payload["rows"]), 5)
        noop = next(row for row in payload["rows"] if row["observer_name"] == "analytic_noop")
        self.assertEqual(noop["metrics"]["action_loss"], 1.0)
        self.assertNotIn("mae", noop["metrics"])

    def test_public_paper_is_packaged_and_linked(self) -> None:
        paper = ROOT / "site" / "downloads" / "observerbench.pdf"
        self.assertGreater(paper.stat().st_size, 100_000)
        self.assertIn(
            'href="downloads/observerbench.pdf"',
            (ROOT / "site" / "index.html").read_text(encoding="utf-8"),
        )
        for page in ("leaderboards", "runners", "submit"):
            self.assertIn(
                'href="../downloads/observerbench.pdf"',
                (ROOT / "site" / page / "index.html").read_text(encoding="utf-8"),
            )

    def test_runner_page_is_linked_from_submission_path(self) -> None:
        runner_page = ROOT / "site" / "runners" / "index.html"
        self.assertTrue(runner_page.is_file())
        runner_guide = (ROOT / "docs" / "RUNNERS.md").read_text(encoding="utf-8")
        self.assertIn("Pick the boundary that matches your method", runner_guide)

        submit = (ROOT / "site" / "submit" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('href="../runners/">View runners</a>', submit)

    def test_gemma_panel_title_and_primary_metric_are_explicit(self) -> None:
        self.assertEqual(
            BUILD_SITE._title(
                "controlarena-apps-verified-backdoors@gemma-2-9b-it-v0",
                "controlarena-apps-gemma2-9b-it-v0",
            ),
            "Gemma-2-9B-it APPS verified-backdoor monitoring",
        )
        self.assertEqual(
            BUILD_SITE._primary_metric(
                {
                    "risk_auroc": "higher",
                    "mean_realized_violations_at_p01_b02": "lower",
                }
            ),
            "mean_realized_violations_at_p01_b02",
        )

        javascript = (ROOT / "site/assets/site.js").read_text(encoding="utf-8")
        decimal_case = (
            'if (name === "mean_realized_violations_at_p01_b02") '
            'return number.toFixed(3);'
        )
        self.assertIn(decimal_case, javascript)
        self.assertLess(
            javascript.index(decimal_case),
            javascript.index('if (name.includes("violations"))'),
        )
        self.assertIn('status.includes("prospectively frozen")', javascript)
        self.assertEqual(
            BUILD_SITE._title(
                "controlarena-apps-verified-backdoors@qwen2.5-7b-v1",
                "controlarena-apps-qwen-v1",
            ),
            "Qwen2.5-7B-Instruct APPS verified-backdoor monitoring",
        )
        self.assertEqual(
            BUILD_SITE._title(
                "safety-interlock-qwen2-5-7b-instruct@paired-scope-v0",
                "qwen-paired-scope-v0",
            ),
            "Qwen2.5-7B-Instruct authorization triage",
        )
        self.assertEqual(
            BUILD_SITE._title(
                "controlarena-apps-verified-backdoors@qwen3.5-9b-v0",
                "controlarena-apps-qwen3-5-9b-v0",
            ),
            "Qwen3.5-9B APPS verified-backdoor monitoring",
        )

    def test_effect_panel_title_states_measurement_budget(self) -> None:
        self.assertEqual(
            BUILD_SITE._title(
                "ioi-gpt2-small-finite-effects@phase5-test-v1-b040",
                "ioi-gpt2-small-finite-effects-b040",
            ),
            "GPT-2 IOI finite effects — 40 measurements",
        )

        panels = [
            {
                "track": "observer",
                "title": f"GPT-2 IOI finite effects — {budget} measurements",
                "panel": f"ioi-gpt2-small-finite-effects-b{budget:03d}",
            }
            for budget in (160, 20, 80, 40)
        ]
        self.assertEqual(
            [item["panel"] for item in sorted(panels, key=BUILD_SITE._catalog_sort_key)],
            [
                "ioi-gpt2-small-finite-effects-b020",
                "ioi-gpt2-small-finite-effects-b040",
                "ioi-gpt2-small-finite-effects-b080",
                "ioi-gpt2-small-finite-effects-b160",
            ],
        )

        self.assertEqual(
            BUILD_SITE._title(
                "induction-qwen2-5-7b-finite-effects@copy-v2-b064",
                "induction-qwen2-5-7b-finite-effects-b064",
            ),
            "Qwen2.5-7B induction-copy finite effects — 64 measurements",
        )
        qwen_panels = [
            {
                "track": "observer",
                "title": f"Qwen finite effects — {budget}",
                "panel": f"induction-qwen2-5-7b-finite-effects-b{budget:03d}",
            }
            for budget in (128, 16, 64, 40)
        ]
        self.assertEqual(
            [
                item["panel"]
                for item in sorted(qwen_panels, key=BUILD_SITE._catalog_sort_key)
            ],
            [
                "induction-qwen2-5-7b-finite-effects-b016",
                "induction-qwen2-5-7b-finite-effects-b040",
                "induction-qwen2-5-7b-finite-effects-b064",
                "induction-qwen2-5-7b-finite-effects-b128",
            ],
        )

    def test_qwen35_apps_is_the_featured_default_panel(self) -> None:
        panels = [
            {
                "track": "observer",
                "title": "GPT-2 IOI finite effects — 20 measurements",
                "panel": "ioi-gpt2-small-finite-effects-b020",
            },
            {
                "track": "safety",
                "title": "Gemma-2-9B-it APPS verified-backdoor monitoring",
                "panel": "controlarena-apps-gemma2-9b-it-v0",
            },
            {
                "track": "safety",
                "title": "Qwen3.5-9B APPS verified-backdoor monitoring",
                "panel": "controlarena-apps-qwen3-5-9b-v0",
            },
        ]

        ordered = sorted(panels, key=BUILD_SITE._catalog_sort_key)
        self.assertEqual(
            ordered[0]["panel"], "controlarena-apps-qwen3-5-9b-v0"
        )

    def test_homepage_catalog_features_current_model_results(self) -> None:
        homepage = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        catalog = homepage.split(
            '<section class="section task-section"', 1
        )[1].split("</section>", 1)[0]

        for expected in (
            "Qwen3.5-9B",
            "Gemma-2-9B-it",
            "Compression without a better decision.",
            "lowers held-out MAE from 0.121 to 0.040",
        ):
            self.assertIn(expected, catalog)
        self.assertNotIn("GPT-2", catalog)

    def test_build_copies_site_and_creates_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            repo = temp / "repo"
            panel = repo / "leaderboards" / "safety" / "example-v1"
            panel.mkdir(parents=True)
            task_pack = repo / "task-packs" / "safety" / "example-v1"
            task_pack.mkdir(parents=True)
            (task_pack / "safety_task.json").write_text('{"targets": false}', encoding="utf-8")
            (panel / "results.json").write_text(
                json.dumps(
                    {
                        "schema_version": "example.v1",
                        "metric_direction": {"action_loss": "lower", "risk_auroc": "higher"},
                        "rows": [
                            {
                                "task_id": "example@v1",
                                "display_name": "Baseline",
                                "result_status": "checked baseline",
                                "access_regime": "public inputs",
                                "metrics": {"action_loss": 2.0, "risk_auroc": 0.8},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = temp / "pages"
            BUILD_SITE.build(repo, ROOT / "site", output)

            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "runners" / "index.html").is_file())
            self.assertGreater(
                (output / "downloads" / "observerbench.pdf").stat().st_size,
                100_000,
            )
            self.assertTrue(
                (output / "task-packs" / "safety" / "example-v1" / "safety_task.json").is_file()
            )
            self.assertTrue((output / "data" / "leaderboards" / "example-v1.json").is_file())
            catalog = json.loads((output / "data" / "catalog.json").read_text())
            self.assertEqual(catalog["schema_version"], "observerbench.site_catalog.v0")
            self.assertEqual(catalog["leaderboards"][0]["task_id"], "example@v1")
            self.assertEqual(catalog["leaderboards"][0]["primary_metric"], "action_loss")
            self.assertFalse(catalog["ranking_policy"]["global_rank"])

    def test_empty_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            path.write_text('{"rows": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty rows"):
                BUILD_SITE._load_payload(path)

    def test_private_task_pack_file_stops_public_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            repo = temp / "repo"
            pack = repo / "task-packs" / "safety" / "example"
            pack.mkdir(parents=True)
            (pack / "evaluator_targets.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to publish"):
                BUILD_SITE._copy_public_task_packs(repo, temp / "output")


if __name__ == "__main__":
    unittest.main()
