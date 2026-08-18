from __future__ import annotations

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
