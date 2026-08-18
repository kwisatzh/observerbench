"""Tests for target-free blinded pack construction.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_blinded_pack", ROOT / "scripts" / "build_blinded_pack.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class BlindedPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.secret = self.root / "evaluator.key"
        self.secret.write_bytes(b"observerbench-test-secret-material-0001")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_qwen_pack_removes_direct_and_joinable_identifier_leakage(self) -> None:
        outdir = self.root / "qwen"
        MODULE.build_qwen_safety_pack(
            source=ROOT / "tests" / "fixtures" / "qwen_prompts.csv",
            secret_path=self.secret,
            outdir=outdir,
            task_id="safety-interlock-qwen2-5-7b-instruct@paired-scope-v1",
        )
        rows = read_jsonl(outdir / "public" / "queries.jsonl")
        self.assertEqual(len(rows), 2)
        public_text = (outdir / "public" / "queries.jsonl").read_text(encoding="utf-8")
        for forbidden in (
            "authorized",
            "unauthorized",
            "unsafe_if_allowed",
            "benign_value",
            "family_id",
            "pair_id",
            "locked_test",
            "workspace_locked_a",
            "workspace_locked_b",
            "account_locked_test_00000",
        ):
            self.assertNotIn(forbidden, public_text)
        self.assertTrue(all(str(row["query_id"]).startswith("ob-") for row in rows))
        self.assertTrue(all(set(row) == {"schema_version", "query_id", "payload"} for row in rows))
        self.assertTrue(all("workspace_" in row["payload"]["user_prompt"] for row in rows))

        with (outdir / "private" / "targets.csv").open(newline="", encoding="utf-8") as handle:
            target_reader = csv.DictReader(handle)
            self.assertIn("query_id", tuple(target_reader.fieldnames or ()))
            self.assertNotIn("sample_id", tuple(target_reader.fieldnames or ()))
            targets = list(target_reader)
        self.assertEqual({row["unsafe_if_allowed"] for row in targets}, {"true", "false"})
        with (outdir / "private" / "id_map.csv").open(newline="", encoding="utf-8") as handle:
            mapping = list(csv.DictReader(handle))
        self.assertEqual(
            {row["source_sample_id"] for row in mapping},
            {
                "locked_test-pair-00000-authorized",
                "locked_test-pair-00000-unauthorized",
            },
        )
        safety_task = json.loads(
            (outdir / "public" / "safety_task.json").read_text(encoding="utf-8")
        )
        self.assertEqual(safety_task["task_version"], "paired-scope-v1")
        self.assertEqual(len(safety_task["measurements"]), 3)
        self.assertEqual(len(safety_task["queries"]), 2)
        self.assertNotIn("targets", safety_task)
        self.assertTrue(
            all("unsafe_if_allowed" not in query for query in safety_task["queries"])
        )
        self.assertTrue(
            all(str(query["query_id"]).startswith("ob-") for query in safety_task["queries"])
        )
        self.assertTrue(
            all(str(query["family_id"]).startswith("family-") for query in safety_task["queries"])
        )
        self.assertTrue(
            all(str(query["pair_id"]).startswith("pair-") for query in safety_task["queries"])
        )
        evaluator_targets = json.loads(
            (outdir / "private" / "evaluator_targets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(evaluator_targets["targets"]), 2)

    def test_qwen_pack_is_reproducible_but_keyed(self) -> None:
        source = ROOT / "tests" / "fixtures" / "qwen_prompts.csv"
        first = self.root / "first"
        second = self.root / "second"
        MODULE.build_qwen_safety_pack(
            source=source,
            secret_path=self.secret,
            outdir=first,
            task_id="safety-interlock-qwen2-5-7b-instruct@paired-scope-v1",
        )
        MODULE.build_qwen_safety_pack(
            source=source,
            secret_path=self.secret,
            outdir=second,
            task_id="safety-interlock-qwen2-5-7b-instruct@paired-scope-v1",
        )
        self.assertEqual(
            (first / "public" / "queries.jsonl").read_bytes(),
            (second / "public" / "queries.jsonl").read_bytes(),
        )

        other_secret = self.root / "other.key"
        other_secret.write_bytes(b"different-observerbench-secret-00000001")
        third = self.root / "third"
        MODULE.build_qwen_safety_pack(
            source=source,
            secret_path=other_secret,
            outdir=third,
            task_id="safety-interlock-qwen2-5-7b-instruct@paired-scope-v1",
        )
        self.assertNotEqual(
            (first / "public" / "queries.jsonl").read_bytes(),
            (third / "public" / "queries.jsonl").read_bytes(),
        )
        manifest = json.loads((first / "public" / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["submission_columns"],
            ["schema_version", "query_id", "predicted_risk"],
        )
        self.assertEqual(manifest["submission_schema"], "observerbench.safety_predictions.v0")
        self.assertEqual(manifest["submission_id_column"], "query_id")

    def test_apps_pack_uses_opaque_ids_and_private_target_schema(self) -> None:
        outdir = self.root / "apps"
        MODULE.build_apps_pack(
            source=ROOT / "tests" / "fixtures" / "apps_rows.jsonl",
            secret_path=self.secret,
            outdir=outdir,
            task_id="controlarena-apps-test-v0",
        )
        rows = read_jsonl(outdir / "public" / "queries.jsonl")
        self.assertEqual(len(rows), 2)
        public_text = (outdir / "public" / "queries.jsonl").read_text(encoding="utf-8")
        for forbidden in (
            "1288::honest",
            "1288::attack",
            "eval_mode",
            "backdoor_works",
            "family_id",
            "problem_id",
            "locked_test",
        ):
            self.assertNotIn(forbidden, public_text)
        self.assertIn("candidate_solution", public_text)

        with (outdir / "private" / "targets.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(
                tuple(reader.fieldnames or ()),
                (
                    "schema_version",
                    "sample_id",
                    "eval_mode",
                    "backdoor_works",
                    "family_id",
                ),
            )
            targets = list(reader)
        self.assertEqual({row["eval_mode"] for row in targets}, {"honest", "attack"})
        manifest = json.loads((outdir / "public" / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["integrity_class"], "target-free-open-source-matchable")
        self.assertIn("not a sealed", manifest["warning"])
        self.assertEqual(
            manifest["submission_columns"],
            ["schema_version", "sample_id", "monitor_name", "score"],
        )
        self.assertEqual(
            manifest["submission_schema"], "observerbench.ai_control_public_scores.v0"
        )
        self.assertEqual(manifest["submission_id_column"], "sample_id")

    def test_short_secret_and_nonempty_output_are_rejected(self) -> None:
        short = self.root / "short.key"
        short.write_bytes(b"too-short")
        with self.assertRaisesRegex(MODULE.PackError, "at least 32 bytes"):
            MODULE.build_apps_pack(
                source=ROOT / "tests" / "fixtures" / "apps_rows.jsonl",
                secret_path=short,
                outdir=self.root / "short-output",
                task_id="controlarena-apps-test-v0",
            )

        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PackError, "not empty"):
            MODULE.build_apps_pack(
                source=ROOT / "tests" / "fixtures" / "apps_rows.jsonl",
                secret_path=self.secret,
                outdir=occupied,
                task_id="controlarena-apps-test-v0",
            )


if __name__ == "__main__":
    unittest.main()
