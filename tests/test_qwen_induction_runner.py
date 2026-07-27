"""Model-free orchestration tests for the staged Qwen Phase-09 runner.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.qwen_induction.artifacts import load_phase09_config
from observerbench.tasks.qwen_induction.plant import HeadAblationMeans
from observerbench.tasks.qwen_induction.runner import Phase09Runner, SCIENTIFIC_STAGES


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = REPO_ROOT / "configs/revision/phase09/qwen2_5_7b_induction_full.json"
SMOKE_CONFIG = REPO_ROOT / "configs/revision/phase09/qwen2_5_0_5b_induction_smoke.json"


class FakeTokenizer:
    vocab_size = 3000
    all_special_ids = (0,)

    def __len__(self):
        return self.vocab_size

    def decode(self, values, **_kwargs):
        return chr(0x1000 + int(values[0]))

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(text) - 0x1000]


class FakePlant:
    fail_after: int | None = None
    yielded_masks: list[str] = []
    implementations: list[str] = []
    events: list[str] = []
    clean_calls: list[tuple[str, tuple[str, ...], int]] = []

    def __init__(self, attention_implementation: str):
        self.attention_implementation = attention_implementation
        type(self).implementations.append(attention_implementation)
        self.architecture = SimpleNamespace(
            n_layers=28,
            n_query_heads=28,
            n_kv_heads=4,
            head_dim=2,
            hidden_size=56,
            qkv_bias_present=True,
            output_bias_present=False,
        )

    def audit_runtime(self, **expected):
        assert expected["expected_model_id"] == "Qwen/Qwen2.5-7B"
        assert expected["expected_layers"] == 28
        assert expected["expected_query_heads"] == 28
        assert expected["expected_kv_heads"] == 4
        assert expected["expected_dtype"] == "bfloat16"
        assert expected["expected_attention_implementation"] == self.attention_implementation
        return dict(expected)

    def score_clean(self, records, *, batch_size):
        type(self).clean_calls.append(
            (
                self.attention_implementation,
                tuple(sorted({str(row.bank) for row in records})),
                int(batch_size),
            )
        )
        return [
            SimpleNamespace(
                prompt_id=row.prompt_id,
                family_id=row.family_id,
                candidate_margin=3.0,
                candidate_correct=True,
                top1_correct=True,
                target_nll=0.01,
                target_token_id=row.target_value_token,
                candidate_predicted_token_id=row.target_value_token,
                predicted_token_id=row.target_value_token,
                candidate_logits_finite=True,
            )
            for row in records
        ]

    def scan_attention(self, records, *, batch_size):
        del batch_size
        n_prompts = len(records)
        rows = []
        for layer in range(28):
            for head in range(28):
                if layer < 8 and head < 4:
                    specificity = 100.0 - (4 * layer + head)
                elif layer < 8 and head in (4, 5, 6):
                    specificity = -1000.0 - head
                else:
                    specificity = -100.0 - layer - head
                rows.append(
                    SimpleNamespace(
                        layer=layer,
                        head=head,
                        kv_group=head // 7,
                        attention_to_target=specificity + 1.0,
                        attention_to_distractors=1.0,
                        attention_specificity=specificity,
                        direct_logit_attribution=specificity / 10.0,
                        output_norm=1.0 + head / 100.0,
                        n_prompts=n_prompts,
                    )
                )
        return rows

    def capture_reference_means(self, records, heads, *, batch_size):
        del batch_size
        families = tuple(sorted({row.family_id for row in records}))
        return HeadAblationMeans(
            families,
            tuple(heads),
            np.zeros((len(families), len(heads), 2), dtype=np.float32),
            tuple(sum(row.family_id == family for row in records) for family in families),
        )

    def noop_hook_parity(self, records, heads, means, *, batch_size):
        n_heads = len(heads)
        del records, heads, means, batch_size
        type(self).events.append(f"noop:{n_heads}")
        return {"maximum_margin_error": 0.0, "prediction_parity": True}

    @staticmethod
    def _effect(heads, bits, prompt_index):
        if not any(bits):
            return 0.0
        if len(heads) == 32:
            component = bits.index(1)
            head = heads[component]
            return 1.0 - 0.01 * head.layer - 0.001 * head.head
        if len(heads) == 16:
            return 0.20 * sum(bits[:8]) + 0.01 * sum(bits[8:])
        linear = sum((index + 1) * 0.025 * bit for index, bit in enumerate(bits))
        pair = 0.08 * bits[1] * bits[6] if len(bits) == 8 else 0.0
        dispersion = 0.0025 * (prompt_index - 1.5) * sum(bits)
        return linear + pair + dispersion

    def iter_mask_effects(
        self,
        records,
        heads,
        masks,
        means,
        *,
        batch_size,
        mode="mean",
        position_scope="final",
    ):
        del means, batch_size, mode, position_scope
        type(self).events.append(f"effects:{len(heads)}")
        completed = 0
        for mask in masks:
            if self.fail_after is not None and completed == self.fail_after:
                raise RuntimeError("injected interruption")
            bits = tuple(map(int, mask.bits))
            rows = []
            for prompt_index, record in enumerate(records):
                effect = self._effect(heads, bits, prompt_index)
                rows.append(
                    SimpleNamespace(
                        prompt_id=record.prompt_id,
                        family_id=record.family_id,
                        mask_id=mask.mask_id,
                        mask_bits="".join(map(str, bits)),
                        clean_margin=3.0,
                        ablated_margin=3.0 - effect,
                        drop_from_clean=effect,
                        clean_candidate_correct=True,
                        ablated_candidate_correct=bool(effect < 2.5),
                        clean_top1_correct=True,
                        ablated_top1_correct=bool(effect < 2.0),
                        clean_target_nll=0.05,
                        ablated_target_nll=0.05 + max(0.0, effect),
                    )
                )
            type(self).yielded_masks.append(mask.mask_id)
            completed += 1
            yield mask.mask_id, rows


def _fake_collateral_iterator(
    plant,
    controls,
    heads,
    masks,
    means,
    *,
    batch_size,
    include_total_variation,
):
    del plant, heads, means, batch_size
    assert include_total_variation
    for mask in masks:
        magnitude = 0.002 * sum(mask.bits)
        yield mask.mask_id, [
            SimpleNamespace(
                prompt_id=control.prompt_id,
                source_prompt_id=control.source_prompt_id,
                family_id=control.family_id,
                mask_id=mask.mask_id,
                mask_bits="".join(map(str, mask.bits)),
                kl_clean_to_intervened=magnitude,
                total_variation=min(1.0, magnitude * 2.0),
            )
            for control in controls
        ]


def _scientific_config() -> dict:
    config = deepcopy(load_phase09_config(FULL_CONFIG, require_scientific=True))
    config["token_pool"]["per_split_size"] = 64
    for bank in config["sequence_design"]["prompts_per_family"]:
        config["sequence_design"]["prompts_per_family"][bank] = 1
    config["head_discovery"]["bootstrap_repeats"] = 20
    config["uncertainty"]["bootstrap_repeats"] = 20
    config["runtime"]["discovery_batch_size"] = 2
    config["runtime"]["measurement_batch_size"] = 2
    return config


def _smoke_config() -> dict:
    config = deepcopy(load_phase09_config(SMOKE_CONFIG))
    config["token_pool"]["per_split_size"] = 64
    for bank in config["sequence_design"]["prompts_per_family"]:
        config["sequence_design"]["prompts_per_family"][bank] = 1
    return config


def _runner(config, root, *, resume: bool = False):
    return Phase09Runner(
        config,
        root,
        tokenizer_factory=lambda _config: FakeTokenizer(),
        plant_factory=lambda implementation, _config: FakePlant(implementation),
        collateral_iterator=_fake_collateral_iterator,
        allow_injected_test_runtime=True,
        resume=resume,
    )


def test_production_runner_accepts_only_the_frozen_config_and_source_seal(
    tmp_path: Path,
) -> None:
    # Copy-v1 remains executable only from its immutable preregistration tag.
    # The Copy-v2 branch changes transitive producers and must fail closed if
    # someone tries to present it as the sealed Copy-v1 implementation.
    with pytest.raises(ValueError, match="producer source differs"):
        Phase09Runner(
            FULL_CONFIG,
            tmp_path,
            device="cpu",
            local_files_only=True,
        )


def test_scientific_stage_order_resume_blind_freeze_and_complete_evaluation(
    tmp_path: Path,
) -> None:
    FakePlant.fail_after = None
    FakePlant.yielded_masks = []
    FakePlant.implementations = []
    FakePlant.events = []
    runner = _runner(_scientific_config(), tmp_path)

    with pytest.raises(RuntimeError, match="out of order"):
        runner.discover()
    assert runner.prepare().status == "complete"
    prepared = tmp_path / "design/prompts_all.csv"
    original = prepared.read_bytes()
    prepared.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="completed stage artifact changed"):
        runner.discover()
    prepared.write_bytes(original)
    assert runner.discover().status == "complete"
    assert FakePlant.events.index("noop:32") < FakePlant.events.index("effects:32")
    assert runner.confirm().status == "complete"
    assert runner.freeze_design().status == "complete"

    FakePlant.fail_after = 3
    with pytest.raises(RuntimeError, match="injected interruption"):
        runner.measure_calibration()
    checkpoint_dir = tmp_path / "work/checkpoints/calibration"
    completed_before_resume = {path.stem for path in checkpoint_dir.glob("*.csv")}
    assert len(completed_before_resume) == 3
    checkpoint_manifest = json.loads(
        (checkpoint_dir / "checkpoint_manifest.json").read_text()
    )
    assert len(checkpoint_manifest["checkpoint_hashes"]) == 3
    retained_path = next(checkpoint_dir.glob("*.csv"))
    retained_original = retained_path.read_bytes()
    retained = pd.read_csv(retained_path)
    assert {
        "clean_candidate_correct",
        "ablated_candidate_correct",
        "clean_top1_correct",
        "ablated_top1_correct",
        "clean_target_nll",
        "ablated_target_nll",
    }.issubset(retained)
    retained.loc[0, "ablated_target_nll"] = -1.0
    retained.to_csv(retained_path, index=False)
    resume_runner = _runner(_scientific_config(), tmp_path, resume=True)
    with pytest.raises(ValueError, match="checkpoint hash changed"):
        resume_runner.measure_calibration()
    retained_path.write_bytes(retained_original)

    FakePlant.fail_after = None
    assert resume_runner.measure_calibration().status == "complete"
    runner = resume_runner
    for mask_id in completed_before_resume:
        assert FakePlant.yielded_masks.count(mask_id) == 1

    leaked = tmp_path / "work/checkpoints/locked_test/forbidden.csv"
    leaked.parent.mkdir(parents=True)
    leaked.write_text("locked outcome", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked-test outcomes exist"):
        runner.freeze_observers()
    leaked.unlink()

    assert runner.freeze_observers().status == "complete"
    freeze = json.loads(
        (tmp_path / "work/observer_freeze/observer_freeze_manifest.json").read_text()
    )
    assert freeze["locked_outcomes_read"] is False
    assert runner.measure_locked_test().status == "complete"
    assert runner.evaluate().status == "complete"
    assert runner.measure_collateral().status == "complete"

    audit = json.loads((tmp_path / "work/stage_audit.json").read_text())
    assert tuple(audit["completed_stages"]) == SCIENTIFIC_STAGES
    assert (tmp_path / "effects/effect_manifest.json").is_file()
    assert (tmp_path / "work/evaluation/prediction_bootstrap_contrasts.csv").is_file()
    assert (tmp_path / "work/evaluation/action_bootstrap_contrasts.csv").is_file()
    assert (tmp_path / "work/evaluation/mean_effect_density_summary.csv").is_file()
    assert (
        tmp_path / "work/evaluation/mean_effect_prompt_cell_residual_summary.csv"
    ).is_file()
    assert (
        tmp_path / "work/evaluation/aggregate_action_bootstrap_contrasts.csv"
    ).is_file()
    assert (tmp_path / "work/evaluation/intervention_outcomes_by_mask.csv").is_file()
    assert (tmp_path / "work/evaluation/intervention_outcomes_by_density.csv").is_file()
    assert (tmp_path / "work/evaluation/intervention_outcomes_by_family.csv").is_file()
    assert (tmp_path / "work/collateral/collateral_shifts_raw.csv").is_file()
    collateral_summary = json.loads(
        (tmp_path / "work/collateral/summary.json").read_text()
    )
    assert collateral_summary["analytic_noop_exact_zero"] is True
    assert collateral_summary["affects_primary_gate"] is False
    locked = (tmp_path / "work/measurements/locked_test_effects_raw.csv").read_text()
    assert "cluster_id" in locked.splitlines()[0]

    # Completed stages are immutable, hash-checked no-ops.
    assert runner.evaluate().status == "complete"


def test_engineering_smoke_is_namespaced_and_cannot_emit_scientific_artifacts(
    tmp_path: Path,
) -> None:
    FakePlant.fail_after = None
    FakePlant.implementations = []
    FakePlant.events = []
    runner = _runner(_smoke_config(), tmp_path)

    with pytest.raises(ValueError, match="smoke config"):
        runner.prepare()
    result = runner.engineering_smoke()
    assert result.status == "complete"
    summary = json.loads(
        (tmp_path / "engineering_smoke/smoke_outputs/summary.json").read_text()
    )
    assert summary["scientific_claim_allowed"] is False
    assert FakePlant.implementations == ["eager", "sdpa"]
    assert "noop:2" in FakePlant.events
    assert not (tmp_path / "design/design_manifest.json").exists()
    assert not (tmp_path / "effects").exists()
