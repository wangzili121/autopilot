from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ProtocolSpec,
    WorkloadContract,
    canonical_sha256,
)
from inference_autopilot.calibration.planning import build_plan
from inference_autopilot.candidate_planning import NumericRange, SelectionContext
from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    EvidenceRecord,
    QualityAssessment,
    SourceArtifact,
)
from inference_autopilot.features import features_from_ledger
from inference_autopilot.policy_selection import (
    LocalResponseModelSpec,
    PolicySelectionSpec,
    ResponseObjective,
    select_policy,
)
from inference_autopilot.policy_transfer import (
    PolicyTransferPlan,
    PolicyTransferSpec,
    calibration_spec_from_policy_transfer_plan,
    plan_policy_transfer,
)
from inference_autopilot.transfer_validation import (
    PolicyTransferAssessment,
    PolicyTransferAssessmentSpec,
    assess_policy_transfer,
)
from inference_autopilot.runners.chang import arrival_trace_sha256
from inference_autopilot.search_space import DeploymentSearchSpace, compile_search_space


GRAPH_SHA256 = "a" * 64


def _record(capacity: int, qps: float) -> EvidenceRecord:
    record_id = f"capacity-{capacity}"
    return EvidenceRecord(
        record_id=record_id,
        campaign="transfer-source",
        variant=record_id,
        source=SourceArtifact(f"{record_id}.json", "b" * 64, "test", record_id),
        workload={
            "workload_id": "short-p16",
            "requests": 16,
            "workers": 16,
            "arrival_qps": 0.0,
            "context_tokens": 0,
        },
        configuration={
            "model_runner": "MRV1",
            "base_max_num_seqs": 40,
            "proposal_max_num_seqs": capacity,
            "base_max_num_batched_tokens": 10240,
            "proposal_max_num_batched_tokens": 12288,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact",
            "graph_sha256": GRAPH_SHA256,
            "candidate_count": 8,
            "rollout_count": 3,
            "block_size": 48,
            "total_length": 192,
            "apply_importance_correction": True,
        },
        environment={"environment_id": "npu6"},
        metrics={
            "completed_qps": qps,
            "p95_seconds": 10.0,
            "accuracy": 0.8,
        },
        quality=QualityAssessment(
            grade=EvidenceGrade.A_FORMAL_PAIRED,
            purpose=EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
            claim_eligible=True,
            reasons=("synthetic formal evidence",),
        ),
    )


def _policy():
    space = DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "transfer-space",
            "algorithm_id": "conditional_is_small_proposal",
            "fixed_settings": {
                "model_runner": "MRV1",
                "base_max_num_seqs": 40,
                "base_max_num_batched_tokens": 10240,
                "proposal_max_num_batched_tokens": 12288,
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
            },
            "knobs": [
                {
                    "name": "proposal.max_num_seqs",
                    "setting_name": "proposal_max_num_seqs",
                    "value_type": "integer",
                    "domain": {"kind": "choices", "values": [32, 48, 64]},
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "description": "proposal capacity",
                }
            ],
            "constraints": [],
        }
    )
    compiled = compile_search_space(space)
    features = features_from_ledger(
        EvidenceLedger((_record(32, 0.7), _record(48, 0.9), _record(64, 1.2)))
    )
    selection = PolicySelectionSpec(
        policy_id="short-p16-policy",
        compiled_space_sha256=compiled.to_dict()["compiled_space_sha256"],
        selection_context=SelectionContext(
            algorithm_id="conditional_is_small_proposal",
            semantic_cohort_id="conditional_is_small_proposal",
            graph_sha256=GRAPH_SHA256,
            accepted_evidence_semantic_classes=("exact",),
            workload_id="short-p16",
            environment_id="npu6",
            static_features={"workload.workers": 16},
            static_feature_ranges={"workload.requests": NumericRange(16, 64)},
        ),
        query_static_features={"workload.requests": 16, "workload.workers": 16},
        baseline_settings={
            "model_runner": "MRV1",
            "base_max_num_seqs": 40,
            "proposal_max_num_seqs": 48,
            "base_max_num_batched_tokens": 10240,
            "proposal_max_num_batched_tokens": 12288,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
        },
        objective=ResponseObjective(
            target="performance.completed_qps",
            direction="maximize",
            constraints=(),
        ),
        model=LocalResponseModelSpec(
            feature_names=(
                "deployment.proposal_max_num_seqs",
                "workload.requests",
            ),
            neighbors=3,
            distance_power=2.0,
            maximum_normalized_distance=1.0,
            uncertainty_multiplier=0.0,
            maximum_failure_probability=0.1,
            minimum_response_rows=3,
            minimum_distinct_configurations=3,
            minimum_paired_replicates=2,
            ranked_candidate_limit=3,
        ),
    )
    return select_policy(selection, compiled, features)


def _template() -> CalibrationSpec:
    return CalibrationSpec.from_dict(
        {
            "schema_version": "1.0",
            "campaign_id": "transfer-template",
            "strong_baseline": True,
            "protocol": {"pattern": "ABBA", "blocks": 1, "pair_seeds": [1, 2]},
            "semantic_contract": {
                "algorithm_id": "conditional_is_small_proposal",
                "semantic_class": "exact",
                "graph_sha256": GRAPH_SHA256,
                "invariants": {"candidate_count": 8},
            },
            "workload_contract": {
                "workload_id": "short-p16",
                "dataset_sha256": "c" * 64,
                "arrival_trace_sha256": arrival_trace_sha256(
                    {"requests": 16, "workers": 16, "arrival_qps": 0.0}
                ),
                "parameters": {
                    "requests": 16,
                    "workers": 16,
                    "arrival_qps": 0.0,
                },
            },
            "environment_contract": {
                "environment_id": "npu6",
                "hardware": {"device": "Ascend"},
                "software": {"runtime": "test"},
                "models": {"base": "test"},
            },
            "objective": {
                "primary_metric": "completed_qps",
                "direction": "maximize",
                "constraints": [],
            },
            "required_metrics": ["completed_qps"],
            "baseline": {
                "configuration_id": "manual-48",
                "settings": {
                    "model_runner": "MRV1",
                    "base_max_num_seqs": 40,
                    "proposal_max_num_seqs": 48,
                    "base_max_num_batched_tokens": 10240,
                    "proposal_max_num_batched_tokens": 12288,
                    "base_memory_fraction": 0.54,
                    "proposal_memory_fraction": 0.36,
                    "proposal_graph_capture_sizes": [1, 48],
                },
                "description": "strong fallback",
            },
            "candidates": [
                {
                    "configuration_id": "placeholder",
                    "settings": {"proposal_max_num_seqs": 48},
                    "description": "replaced by transfer planner",
                }
            ],
        }
    )


def _transfer_spec(policy_digest: str) -> PolicyTransferSpec:
    parameters = {"requests": 32, "workers": 32, "arrival_qps": 0.0}
    return PolicyTransferSpec(
        transfer_id="short-p16-to-p32",
        policy_bundle_sha256=policy_digest,
        campaign_id="short-p32-transfer-r1",
        protocol=ProtocolSpec("ABBA", 1, (11, 12)),
        target_workload_contract=WorkloadContract(
            workload_id="short-p32",
            dataset_sha256="c" * 64,
            arrival_trace_sha256=arrival_trace_sha256(parameters),
            parameters=parameters,
        ),
        allowed_guard_deviations=(
            "exact_static_feature:workload.workers",
            "workload_id",
        ),
    )


class PolicyTransferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _policy()
        self.template = _template()
        self.spec = _transfer_spec(self.policy.to_dict()["policy_bundle_sha256"])

    def test_plans_explicit_workload_transfer_with_replay(self) -> None:
        plan = plan_policy_transfer(self.spec, self.policy, self.template)
        calibration = calibration_spec_from_policy_transfer_plan(plan)

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.to_dict()["guard_evaluation"]["unallowed_deviations"], [])
        self.assertEqual(calibration.workload_contract.workload_id, "short-p32")
        self.assertEqual(calibration.protocol.pair_seeds, (11, 12))
        self.assertEqual(len(calibration.candidates), 2)
        self.assertEqual(
            calibration.candidates[0].settings, calibration.baseline.settings
        )
        self.assertEqual(
            calibration.candidates[1].settings["proposal_max_num_seqs"], 64
        )
        self.assertEqual(
            calibration.candidates[1].settings["proposal_graph_capture_sizes"],
            [1, 48],
        )
        self.assertEqual(PolicyTransferPlan.from_dict(plan.to_dict()), plan)

    def test_blocks_an_unapproved_guard_deviation(self) -> None:
        spec = replace(
            self.spec,
            allowed_guard_deviations=("workload_id",),
        )
        plan = plan_policy_transfer(spec, self.policy, self.template)

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.to_dict()["calibration_spec"], None)
        self.assertEqual(
            plan.to_dict()["guard_evaluation"]["unallowed_deviations"],
            ["exact_static_feature:workload.workers"],
        )
        with self.assertRaisesRegex(ValueError, "blocked policy transfer"):
            calibration_spec_from_policy_transfer_plan(plan)

    def test_rejects_wrong_arrival_trace_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "arrival trace SHA256"):
            replace(
                self.spec,
                target_workload_contract=replace(
                    self.spec.target_workload_contract,
                    arrival_trace_sha256="d" * 64,
                ),
            )

    def test_rejects_plan_tampering(self) -> None:
        payload = plan_policy_transfer(self.spec, self.policy, self.template).to_dict()
        payload["calibration_spec"]["workload_contract"]["parameters"]["workers"] = 96
        with self.assertRaisesRegex(ValueError, "SHA256"):
            PolicyTransferPlan.from_dict(payload)

    def test_assessment_separates_positive_evidence_from_activation(self) -> None:
        plan = plan_policy_transfer(self.spec, self.policy, self.template)
        calibration = calibration_spec_from_policy_transfer_plan(plan)
        calibration_plan = build_plan(calibration)
        candidate_id = calibration.candidates[-1].configuration_id
        calibration_assessment = {
            "schema_version": "1.0",
            "plan_sha256": canonical_sha256(calibration_plan.to_dict()),
            "formal_complete": True,
            "issues": [],
            "effects": [
                {
                    "candidate_configuration_id": candidate_id,
                    "primary_metric": "completed_qps",
                    "direction": "maximize",
                    "replay_control": False,
                    "formal_group": True,
                    "quality_constraints_satisfied": True,
                    "complete_pair_count": 2,
                    "median_directional_relative_improvement": 0.12,
                    "candidate_over_baseline_geomean_ratio": 1.12,
                    "effect_outside_replay_noise": True,
                }
            ],
            "ledger": EvidenceLedger(()).to_dict(),
        }
        assessment_spec = PolicyTransferAssessmentSpec(
            assessment_id="short-p32-transfer-assessment",
            policy_transfer_plan_sha256=plan.payload["policy_transfer_plan_sha256"],
            minimum_complete_pairs=2,
            minimum_median_improvement_fraction=0.03,
        )
        assessment = assess_policy_transfer(
            assessment_spec, plan, calibration_assessment
        )

        self.assertEqual(assessment.status, "positive_transfer_evidence")
        self.assertTrue(assessment.payload["eligible_for_target_policy_evidence"])
        self.assertTrue(assessment.payload["target_policy_required"])
        self.assertFalse(assessment.payload["source_policy_activation_eligible"])
        self.assertEqual(
            PolicyTransferAssessment.from_dict(assessment.to_dict()).to_dict(),
            assessment.to_dict(),
        )

        calibration_assessment["effects"][0][
            "median_directional_relative_improvement"
        ] = 0.01
        calibration_assessment["effects"][0]["effect_outside_replay_noise"] = False
        rejected = assess_policy_transfer(assessment_spec, plan, calibration_assessment)
        self.assertEqual(rejected.status, "rejected")
        self.assertIn(
            "minimum_transfer_improvement_not_met", rejected.payload["reasons"]
        )
        self.assertIn("effect_not_outside_replay_noise", rejected.payload["reasons"])


if __name__ == "__main__":
    unittest.main()
