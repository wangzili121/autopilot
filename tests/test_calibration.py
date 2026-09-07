from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.calibration.assessment import (
    LoadedObservation,
    assess_calibration,
    load_replay_noise_reference,
)
from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    EnvironmentContract,
    MetricConstraint,
    ObjectiveSpec,
    ProtocolSpec,
    RunObservation,
    SemanticContract,
    WorkloadContract,
)
from inference_autopilot.calibration.planning import build_plan, build_run_manifest
from inference_autopilot.evidence import EvidenceGrade


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def calibration_spec(*, strong_baseline: bool = True) -> CalibrationSpec:
    return CalibrationSpec(
        campaign_id="cis-short-p96",
        strong_baseline=strong_baseline,
        protocol=ProtocolSpec("ABBA", 1, (101, 202)),
        semantic_contract=SemanticContract(
            "conditional_is_small_proposal",
            "exact",
            _DIGEST_A,
            {
                "candidate_count": 15,
                "rollout_count": 3,
                "block_size": 128,
                "total_length": 512,
                "apply_importance_correction": True,
            },
        ),
        workload_contract=WorkloadContract(
            "gsm8k-short-p96",
            _DIGEST_B,
            _DIGEST_C,
            {"requests": 96, "arrival_qps": 0.0, "prefix_mode": "unique"},
        ),
        environment_contract=EnvironmentContract(
            "ascend-a2-single-node",
            {"device": "Ascend", "count": 1},
            {"inference_scaling_commit": "deadbeef", "vllm": "0.18"},
            {"base": "Qwen2.5-1.5B", "proposal": "Qwen2.5-0.5B"},
        ),
        objective=ObjectiveSpec(
            "completed_qps",
            "maximize",
            (
                MetricConstraint("latency_seconds.p95", "<=", 120.0),
                MetricConstraint("accuracy", ">=", 0.45),
            ),
        ),
        required_metrics=("completed_qps", "latency_seconds.p95", "accuracy"),
        baseline=ConfigurationSpec(
            "manual-128-768",
            {"base_max_num_seqs": 128, "proposal_max_num_seqs": 768},
            "Strong manually swept baseline",
        ),
        candidates=(
            ConfigurationSpec(
                "candidate-256-896",
                {"base_max_num_seqs": 256, "proposal_max_num_seqs": 896},
                "Candidate capacity point",
            ),
        ),
    )


def loaded_observations(
    plan,
    *,
    status_by_run: dict[str, str] | None = None,
    accuracy_by_run: dict[str, float] | None = None,
    qps_by_run: dict[str, float] | None = None,
    start_by_run: dict[str, float] | None = None,
) -> tuple[LoadedObservation, ...]:
    status_by_run = status_by_run or {}
    accuracy_by_run = accuracy_by_run or {}
    qps_by_run = qps_by_run or {}
    start_by_run = start_by_run or {}
    loaded: list[LoadedObservation] = []
    for run in plan.runs:
        status = status_by_run.get(run.run_id, "success")
        metrics = (
            {
                "completed_qps": qps_by_run.get(
                    run.run_id,
                    0.8 if run.variant_role == "baseline" else 0.9,
                ),
                "latency_seconds": {"p95": 100.0},
                "accuracy": accuracy_by_run.get(run.run_id, 0.5),
            }
            if status == "success"
            else {"failure": status}
        )
        start = start_by_run.get(run.run_id, float(run.sequence_index * 10))
        observation = RunObservation(
            run_id=run.run_id,
            run_manifest_sha256=build_run_manifest(plan, run.run_id)[
                "run_manifest_sha256"
            ],
            started_at_unix=start,
            finished_at_unix=start + 5.0,
            status=status,
            metrics=metrics,
        )
        digest = hashlib.sha256(run.run_id.encode("utf-8")).hexdigest()
        loaded.append(LoadedObservation(observation, f"{run.run_id}.json", digest))
    return tuple(loaded)


class CalibrationPlanningTest(unittest.TestCase):
    def test_abba_schedule_pairs_workload_seeds(self) -> None:
        plan = build_plan(calibration_spec())
        self.assertEqual(
            [run.variant_role for run in plan.runs],
            ["baseline", "candidate", "candidate", "baseline"],
        )
        self.assertEqual([run.workload_seed for run in plan.runs], [101, 101, 202, 202])
        self.assertEqual([run.pair_index for run in plan.runs], [0, 0, 1, 1])

    def test_run_manifest_is_deterministic_and_binds_configuration(self) -> None:
        plan = build_plan(calibration_spec())
        manifest = build_run_manifest(plan, plan.runs[1].run_id)
        repeated = build_run_manifest(plan, plan.runs[1].run_id)
        self.assertEqual(manifest, repeated)
        self.assertEqual(
            manifest["configuration"]["configuration_id"], "candidate-256-896"
        )
        self.assertEqual(len(manifest["run_manifest_sha256"]), 64)

    def test_deployment_configuration_cannot_override_semantic_invariant(self) -> None:
        spec = calibration_spec()
        conflicting = ConfigurationSpec(
            "candidate-semantic-change",
            {"candidate_count": 8, "base_max_num_seqs": 256},
        )
        with self.assertRaisesRegex(ValueError, "semantic invariants"):
            replace(spec, candidates=(conflicting,))


class CalibrationAssessmentTest(unittest.TestCase):
    def test_complete_group_is_promoted_to_grade_a(self) -> None:
        plan = build_plan(calibration_spec())
        assessment = assess_calibration(plan, loaded_observations(plan))
        self.assertTrue(assessment.formal_complete)
        self.assertEqual(assessment.formal_group_count, 1)
        self.assertEqual(len(assessment.effects), 1)
        self.assertAlmostEqual(
            assessment.effects[0].candidate_over_baseline_geomean_ratio,
            1.125,
        )
        self.assertIsNone(assessment.replay_noise_envelope)
        self.assertTrue(
            all(
                record.quality.grade == EvidenceGrade.A_FORMAL_PAIRED
                for record in assessment.ledger.records
            )
        )

    def test_constraint_violation_does_not_destroy_formal_evidence(self) -> None:
        plan = build_plan(calibration_spec())
        candidate_run = next(
            run for run in plan.runs if run.variant_role == "candidate"
        )
        observations = loaded_observations(
            plan, accuracy_by_run={candidate_run.run_id: 0.40}
        )
        assessment = assess_calibration(plan, observations)
        record = next(
            item
            for item in assessment.ledger.records
            if item.source.locator == candidate_run.run_id
        )
        self.assertTrue(assessment.formal_complete)
        self.assertEqual(record.quality.grade, EvidenceGrade.A_FORMAL_PAIRED)
        self.assertIn("constraint_violation", record.tags)

    def test_out_of_order_group_is_downgraded(self) -> None:
        plan = build_plan(calibration_spec())
        starts = {
            plan.runs[0].run_id: 10.0,
            plan.runs[1].run_id: 0.0,
            plan.runs[2].run_id: 20.0,
            plan.runs[3].run_id: 30.0,
        }
        assessment = assess_calibration(
            plan, loaded_observations(plan, start_by_run=starts)
        )
        self.assertFalse(assessment.formal_complete)
        self.assertIn("run_order_mismatch", {issue.code for issue in assessment.issues})
        self.assertTrue(
            all(
                record.quality.grade == EvidenceGrade.B_CONTROLLED_SINGLE
                for record in assessment.ledger.records
            )
        )

    def test_oom_is_constraint_and_partner_runs_stay_calibration_only(self) -> None:
        plan = build_plan(calibration_spec())
        failed_run = plan.runs[1]
        assessment = assess_calibration(
            plan, loaded_observations(plan, status_by_run={failed_run.run_id: "oom"})
        )
        grades = {
            record.source.locator: record.quality.grade
            for record in assessment.ledger.records
        }
        self.assertEqual(grades[failed_run.run_id], EvidenceGrade.X_EXCLUDED)
        self.assertEqual(
            {grade for run_id, grade in grades.items() if run_id != failed_run.run_id},
            {EvidenceGrade.B_CONTROLLED_SINGLE},
        )

    def test_replay_control_defines_noise_envelope_for_candidate_effect(self) -> None:
        spec = calibration_spec()
        replay = ConfigurationSpec(
            "replay-control",
            dict(spec.baseline.settings),
            "Identical baseline replay",
        )
        plan = build_plan(replace(spec, candidates=(replay, *spec.candidates)))
        qps_by_run: dict[str, float] = {}
        for run in plan.runs:
            if run.configuration_id == "replay-control":
                if run.variant_role == "baseline":
                    qps_by_run[run.run_id] = 1.0
                else:
                    qps_by_run[run.run_id] = 1.02 if run.pair_index == 0 else 0.99
            elif run.variant_role == "baseline":
                qps_by_run[run.run_id] = 1.0
            else:
                qps_by_run[run.run_id] = 1.08

        assessment = assess_calibration(
            plan,
            loaded_observations(plan, qps_by_run=qps_by_run),
        )
        effects = {
            effect.candidate_configuration_id: effect
            for effect in assessment.effects
        }

        self.assertAlmostEqual(assessment.replay_noise_envelope, 0.02)
        self.assertTrue(effects["replay-control"].replay_control)
        self.assertIsNone(effects["replay-control"].effect_exceeds_replay_noise)
        self.assertTrue(
            effects["candidate-256-896"].effect_exceeds_replay_noise
        )
        self.assertTrue(
            effects["candidate-256-896"].effect_outside_replay_noise
        )

    def test_formal_replay_assessment_can_gate_a_later_campaign(self) -> None:
        spec = calibration_spec()
        replay = ConfigurationSpec(
            "replay-control",
            dict(spec.baseline.settings),
            "Identical baseline replay",
        )
        replay_plan = build_plan(replace(spec, candidates=(replay,)))
        replay_qps = {
            run.run_id: (
                1.0
                if run.variant_role == "baseline"
                else (1.02 if run.pair_index == 0 else 0.99)
            )
            for run in replay_plan.runs
        }
        replay_assessment = assess_calibration(
            replay_plan,
            loaded_observations(replay_plan, qps_by_run=replay_qps),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay-assessment.json"
            path.write_text(
                json.dumps(replay_assessment.to_dict()),
                encoding="utf-8",
            )
            reference = load_replay_noise_reference(path)

        candidate_plan = build_plan(spec)
        candidate_qps = {
            run.run_id: 1.0 if run.variant_role == "baseline" else 0.88
            for run in candidate_plan.runs
        }
        assessment = assess_calibration(
            candidate_plan,
            loaded_observations(candidate_plan, qps_by_run=candidate_qps),
            replay_noise_reference=reference,
        )
        effect = assessment.effects[0]

        self.assertIsNone(assessment.local_replay_noise_envelope)
        self.assertAlmostEqual(assessment.replay_noise_envelope, 0.02)
        self.assertEqual(
            assessment.replay_noise_reference.assessment_sha256,
            reference.assessment_sha256,
        )
        self.assertFalse(effect.effect_exceeds_replay_noise)
        self.assertTrue(effect.effect_outside_replay_noise)

    def test_replay_reference_rejects_a_different_environment(self) -> None:
        spec = calibration_spec()
        replay = ConfigurationSpec(
            "replay-control",
            dict(spec.baseline.settings),
            "Identical baseline replay",
        )
        replay_plan = build_plan(replace(spec, candidates=(replay,)))
        replay_assessment = assess_calibration(
            replay_plan, loaded_observations(replay_plan)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay-assessment.json"
            path.write_text(
                json.dumps(replay_assessment.to_dict()), encoding="utf-8"
            )
            reference = load_replay_noise_reference(path)

        other_environment = replace(
            spec.environment_contract,
            environment_id="different-ascend-host",
        )
        candidate_plan = build_plan(
            replace(spec, environment_contract=other_environment)
        )

        with self.assertRaisesRegex(ValueError, "environment_context_sha256"):
            assess_calibration(
                candidate_plan,
                loaded_observations(candidate_plan),
                replay_noise_reference=reference,
            )


if __name__ == "__main__":
    unittest.main()
