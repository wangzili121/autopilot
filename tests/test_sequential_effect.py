from __future__ import annotations

from dataclasses import replace
import math
import unittest

from inference_autopilot.calibration.models import CalibrationSpec, canonical_sha256
from inference_autopilot.sequential_effect import (
    SequentialEffectAssessment,
    SequentialEffectSpec,
    assess_sequential_effect,
    calibration_spec_from_sequential_effect,
    confidence_interval,
    context_bindings_from_calibration_spec,
)


def _calibration() -> CalibrationSpec:
    return CalibrationSpec.from_dict(
        {
            "schema_version": "1.0",
            "campaign_id": "sequential-test",
            "strong_baseline": True,
            "protocol": {"pattern": "ABBA", "blocks": 1, "pair_seeds": [1, 2]},
            "semantic_contract": {
                "algorithm_id": "conditional_is_small_proposal",
                "semantic_class": "exact",
                "graph_sha256": "a" * 64,
                "invariants": {"candidate_count": 8},
            },
            "workload_contract": {
                "workload_id": "medium2k",
                "dataset_sha256": "b" * 64,
                "arrival_trace_sha256": "c" * 64,
                "parameters": {"requests": 32, "context_tokens": 2048},
            },
            "environment_contract": {
                "environment_id": "npu-test",
                "hardware": {"device": "Ascend"},
                "software": {"vllm": "0.18"},
                "models": {"base": "test", "proposal": "test"},
            },
            "objective": {
                "primary_metric": "completed_qps",
                "direction": "maximize",
                "constraints": [{"metric": "accuracy", "operator": ">=", "value": 0.5}],
            },
            "required_metrics": ["completed_qps", "accuracy"],
            "baseline": {
                "configuration_id": "baseline",
                "settings": {"base_max_num_seqs": 40, "proposal_tokens": 12288},
                "description": "control",
            },
            "candidates": [
                {
                    "configuration_id": "replay",
                    "settings": {"base_max_num_seqs": 40, "proposal_tokens": 12288},
                    "description": "identical replay",
                },
                {
                    "configuration_id": "candidate",
                    "settings": {"base_max_num_seqs": 40, "proposal_tokens": 16384},
                    "description": "candidate",
                },
            ],
        }
    )


def _spec(
    mode: str = "prospective",
    method: str = "betting_mixture",
    maximum_pair_count: int = 20,
) -> SequentialEffectSpec:
    calibration = _calibration()
    return SequentialEffectSpec(
        effect_id="medium2k-proposal16k",
        analysis_mode=mode,
        baseline_configuration_id="baseline",
        replay_control_configuration_id="replay",
        candidate_configuration_id="candidate",
        primary_metric="completed_qps",
        direction="maximize",
        **context_bindings_from_calibration_spec(calibration, "candidate"),
        lower_log_effect_bound=-0.25,
        upper_log_effect_bound=0.25,
        confidence_alpha=0.05,
        confidence_method=method,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 0.9),
        minimum_improvement_fraction=0.03,
        minimum_decision_pairs=2,
        maximum_pair_count=maximum_pair_count,
        replication_batch_pairs=2,
        replication_seed_pool=tuple(range(101, 201)),
        require_quality_constraints=True,
    )


def _record(configuration_id: str, group_id: str, seed: int) -> dict:
    calibration = _calibration()
    configuration = {
        item.configuration_id: item
        for item in (calibration.baseline, *calibration.candidates)
    }[configuration_id]
    return {
        "algorithm": {
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact",
            "graph_sha256": "a" * 64,
            "candidate_count": 8,
        },
        "workload": {
            "workload_id": "medium2k",
            "dataset_sha256": "b" * 64,
            "arrival_trace_sha256": "c" * 64,
            "requests": 32,
            "context_tokens": 2048,
            "comparison_group_id": group_id,
            "workload_seed": seed,
            "pair_index": 0,
            "sequence_index": 0,
        },
        "environment": calibration.environment_contract.to_dict(),
        "configuration": {
            "configuration_id": configuration_id,
            **dict(configuration.settings),
        },
    }


def _effect(
    configuration_id: str,
    group_id: str,
    seeds: list[int],
    log_effects: list[float],
    replay_control: bool,
    quality: bool = True,
) -> dict:
    pairs = []
    for index, (seed, log_effect) in enumerate(zip(seeds, log_effects, strict=True)):
        ratio = math.exp(log_effect)
        pairs.append(
            {
                "pair_index": index,
                "workload_seed": seed,
                "baseline_run_id": f"{group_id}-{index}-baseline",
                "candidate_run_id": f"{group_id}-{index}-candidate",
                "baseline_value": 1.0,
                "candidate_value": ratio,
                "candidate_over_baseline_ratio": ratio,
                "directional_relative_improvement": ratio - 1.0,
            }
        )
    return {
        "comparison_group_id": group_id,
        "candidate_configuration_id": configuration_id,
        "primary_metric": "completed_qps",
        "direction": "maximize",
        "replay_control": replay_control,
        "formal_group": True,
        "quality_constraints_satisfied": quality,
        "expected_pair_count": len(seeds),
        "complete_pair_count": len(seeds),
        "pair_effects": pairs,
        "candidate_over_baseline_geomean_ratio": math.exp(sum(log_effects) / len(log_effects)),
        "median_directional_relative_improvement": 0.0,
        "effect_exceeds_replay_noise": None if replay_control else False,
        "effect_outside_replay_noise": None if replay_control else False,
    }


def _assessment(
    seeds: list[int],
    replay_log_effects: list[float],
    candidate_log_effects: list[float],
    quality: bool = True,
    plan_digest: str = "d" * 64,
) -> dict:
    replay_group = "test-replay"
    candidate_group = "test-candidate"
    records = []
    for seed in seeds:
        records.extend(
            [
                _record("baseline", replay_group, seed),
                _record("replay", replay_group, seed),
                _record("baseline", candidate_group, seed),
                _record("candidate", candidate_group, seed),
            ]
        )
    return {
        "schema_version": "1.0",
        "plan_sha256": plan_digest,
        "formal_complete": True,
        "issues": [],
        "effects": [
            _effect("replay", replay_group, seeds, replay_log_effects, True),
            _effect(
                "candidate",
                candidate_group,
                seeds,
                candidate_log_effects,
                False,
                quality,
            ),
        ],
        "ledger": {"records": records},
    }


class SequentialEffectTest(unittest.TestCase):
    def test_retrospective_data_is_diagnostic_only(self) -> None:
        spec = _spec(mode="retrospective_diagnostic")
        raw = _assessment(
            [1, 2],
            [math.log(1.0649380387), math.log(1.0228653137)],
            [math.log(0.9465839647), math.log(1.0128006024)],
        )
        result = assess_sequential_effect(spec, [("e" * 64, raw)])
        payload = result.to_dict()

        self.assertEqual(result.status, "diagnostic_only")
        self.assertEqual(payload["audit"]["pair_count"], 2)
        self.assertAlmostEqual(
            payload["observations"][0]["adjusted_log_effect"],
            math.log(0.9465839647) - math.log(1.0649380387),
        )
        self.assertEqual(
            payload["decision"]["action"],
            "freeze_prospective_spec_before_collecting_new_pairs",
        )

    def test_prospective_inconclusive_result_requests_fresh_seeds(self) -> None:
        result = assess_sequential_effect(
            _spec(),
            [("e" * 64, _assessment([1, 2], [0.02, -0.01], [0.03, 0.0]))],
        )

        self.assertEqual(result.status, "replicate")
        self.assertEqual(result.to_dict()["decision"]["next_pair_seeds"], [101, 102])
        self.assertEqual(
            result.to_dict()["summary"]["projected_at_max_pairs_assumption"],
            "future_effects_equal_current_sample_mean",
        )
        self.assertEqual(
            result.to_dict()["summary"]["projected_boundary_at_current_mean"],
            "no_boundary",
        )

    def test_betting_sequence_can_promote_and_close(self) -> None:
        seeds = list(range(1, 41))
        positive = assess_sequential_effect(
            _spec(maximum_pair_count=40),
            [("e" * 64, _assessment(seeds, [0.0] * 40, [0.15] * 40))],
        )
        negative = assess_sequential_effect(
            _spec(maximum_pair_count=40),
            [("f" * 64, _assessment(seeds, [0.0] * 40, [-0.15] * 40))],
        )

        self.assertEqual(positive.status, "promote")
        self.assertGreater(
            positive.to_dict()["summary"]["interval_lower_improvement_fraction"],
            0.03,
        )
        self.assertEqual(negative.status, "close_direction")
        self.assertLess(
            negative.to_dict()["summary"]["interval_upper_log_effect"], 0
        )

    def test_bound_violation_invalidates_instead_of_clipping(self) -> None:
        result = assess_sequential_effect(
            _spec(),
            [("e" * 64, _assessment([1, 2], [0.0, 0.0], [0.3, 0.0]))],
        )

        self.assertEqual(result.status, "invalid_evidence")
        self.assertEqual(result.to_dict()["audit"]["bound_violation_count"], 1)
        self.assertIn(
            "predeclared_effect_bound_violated",
            result.to_dict()["decision"]["reasons"],
        )

    def test_quality_failure_rejects_candidate(self) -> None:
        result = assess_sequential_effect(
            _spec(),
            [("e" * 64, _assessment([1, 2], [0.0, 0.0], [0.1, 0.1], False))],
        )
        self.assertEqual(result.status, "quality_rejected")

    def test_finite_horizon_reference_shrinks_with_samples(self) -> None:
        spec = _spec(method="finite_horizon_hoeffding", maximum_pair_count=40)
        short = confidence_interval([0.1] * 2, spec)
        long = confidence_interval([0.1] * 40, spec)

        self.assertLess(long[1] - long[0], short[1] - short[0])
        self.assertLessEqual(long[0], 0.1)
        self.assertGreaterEqual(long[1], 0.1)

    def test_rehashed_observation_tampering_is_rejected(self) -> None:
        payload = assess_sequential_effect(
            _spec(mode="retrospective_diagnostic"),
            [("e" * 64, _assessment([1, 2], [0.0, 0.0], [0.1, 0.1]))],
        ).to_dict()
        payload["observations"][0]["adjusted_log_effect"] = 0.2
        unsigned = dict(payload)
        unsigned.pop("sequential_effect_assessment_sha256")
        payload["sequential_effect_assessment_sha256"] = canonical_sha256(unsigned)

        with self.assertRaisesRegex(ValueError, "component effects"):
            SequentialEffectAssessment.from_dict(payload)

    def test_replication_spec_preserves_exact_settings_and_frozen_seeds(self) -> None:
        updated = calibration_spec_from_sequential_effect(
            _calibration(),
            _spec(),
            "sequential-test-r2",
            [101, 102],
        )

        self.assertEqual(updated.protocol.pair_seeds, (101, 102))
        self.assertEqual(updated.candidates[0].settings, updated.baseline.settings)
        self.assertEqual(updated.candidates[1].configuration_id, "candidate")
        self.assertEqual(updated.candidates[1].settings["proposal_tokens"], 16384)

    def test_context_or_duplicate_seed_mismatch_is_invalid_evidence(self) -> None:
        first = _assessment([1, 2], [0.0, 0.0], [0.01, 0.01])
        second = _assessment(
            [1, 3], [0.0, 0.0], [0.01, 0.01], plan_digest="f" * 64
        )
        duplicate = assess_sequential_effect(
            _spec(), [("d" * 64, first), ("e" * 64, second)]
        )
        changed = _assessment([1, 2], [0.0, 0.0], [0.01, 0.01])
        changed["ledger"]["records"][0]["workload"]["requests"] = 64
        mismatch = assess_sequential_effect(_spec(), [("f" * 64, changed)])

        self.assertEqual(duplicate.status, "invalid_evidence")
        self.assertEqual(mismatch.status, "invalid_evidence")


if __name__ == "__main__":
    unittest.main()
