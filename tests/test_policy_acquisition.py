from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.calibration.models import CalibrationSpec, canonical_sha256
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
from inference_autopilot.policy_acquisition import (
    AcquisitionWeights,
    PolicyAcquisitionSpec,
    PolicyExperimentAssessment,
    PolicyExperimentAssessmentSpec,
    PolicyExperimentPlan,
    _changed_knobs,
    _directional_regression_blockers,
    _inactive_changed_knobs,
    assess_policy_experiment,
    calibration_spec_from_policy_experiment_plan,
    plan_policy_experiments,
)
from inference_autopilot.policy_selection import (
    LocalResponseModelSpec,
    PolicySelectionSpec,
    ResponseConstraint,
    ResponseObjective,
    select_policy,
)
from inference_autopilot.search_space import (
    CompiledCandidate,
    DeploymentSearchSpace,
    compile_search_space,
)


GRAPH_SHA256 = "c" * 64
ENVIRONMENT_ID = "ascend-acquisition-test"


def _space() -> DeploymentSearchSpace:
    return DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "policy-acquisition-test",
            "algorithm_id": "conditional_is_small_proposal",
            "fixed_settings": {
                "model_runner": "MRV1",
                "proposal_max_num_seqs": 96,
                "base_max_num_batched_tokens": 10240,
                "proposal_max_num_batched_tokens": 12288,
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
            },
            "knobs": [
                {
                    "name": "base.max_num_seqs",
                    "setting_name": "base_max_num_seqs",
                    "value_type": "integer",
                    "domain": {
                        "kind": "choices",
                        "values": [64, 96, 128, 256],
                    },
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "description": "Base scheduler capacity",
                }
            ],
            "constraints": [],
        }
    )


def _record(record_id: str, capacity: int, qps: float, p95: float) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        campaign="synthetic-acquisition",
        variant=record_id,
        source=SourceArtifact(f"{record_id}.json", "a" * 64, "test", record_id),
        workload={
            "workload_id": "synthetic-p128",
            "method": "conditional_is_small_proposal",
            "requests": 96,
            "workers": 96,
            "prompt_tokens_mean": 128,
        },
        configuration={
            "model_runner": "MRV1",
            "base_max_num_seqs": capacity,
            "proposal_max_num_seqs": 96,
            "base_max_num_batched_tokens": 10240,
            "proposal_max_num_batched_tokens": 12288,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact_algorithm",
            "graph_sha256": GRAPH_SHA256,
            "candidate_count": 8,
            "rollout_count": 3,
            "block_size": 48,
            "total_length": 192,
            "apply_importance_correction": True,
        },
        environment={"environment_id": ENVIRONMENT_ID},
        metrics={
            "completed_qps": qps,
            "p95_seconds": p95,
            "accuracy": 0.8,
        },
        quality=QualityAssessment(
            grade=EvidenceGrade.A_FORMAL_PAIRED,
            purpose=EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
            claim_eligible=True,
            reasons=("synthetic formal evidence",),
        ),
    )


class PolicyAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = compile_search_space(_space())
        self.features = features_from_ledger(
            EvidenceLedger(
                (
                    _record("capacity-64", 64, 0.8, 105.0),
                    _record("capacity-128", 128, 1.2, 90.0),
                    _record("capacity-256", 256, 1.0, 110.0),
                )
            )
        )
        selection_spec = PolicySelectionSpec(
            policy_id="synthetic-acquisition-policy",
            compiled_space_sha256=self.compiled.to_dict()["compiled_space_sha256"],
            selection_context=SelectionContext(
                algorithm_id="conditional_is_small_proposal",
                semantic_cohort_id="conditional_is_small_proposal",
                graph_sha256=GRAPH_SHA256,
                accepted_evidence_semantic_classes=("exact_algorithm",),
                workload_id="synthetic-p128",
                environment_id=ENVIRONMENT_ID,
                static_features={"workload.requests": 96},
                static_feature_ranges={
                    "workload.prompt_tokens_mean": NumericRange(64, 256)
                },
            ),
            query_static_features={
                "workload.prompt_tokens_mean": 128,
                "workload.requests": 96,
            },
            baseline_settings={
                "model_runner": "MRV1",
                "base_max_num_seqs": 64,
                "proposal_max_num_seqs": 96,
                "base_max_num_batched_tokens": 10240,
                "proposal_max_num_batched_tokens": 12288,
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
            },
            objective=ResponseObjective(
                target="performance.completed_qps",
                direction="maximize",
                constraints=(
                    ResponseConstraint("latency.p95_seconds", "<=", 130.0),
                    ResponseConstraint("quality.accuracy", ">=", 0.6),
                ),
            ),
            model=LocalResponseModelSpec(
                feature_names=(
                    "deployment.base_max_num_seqs",
                    "workload.prompt_tokens_mean",
                    "workload.requests",
                ),
                neighbors=3,
                distance_power=2.0,
                maximum_normalized_distance=0.6,
                uncertainty_multiplier=0.5,
                maximum_failure_probability=0.1,
                minimum_response_rows=3,
                minimum_distinct_configurations=3,
                minimum_paired_replicates=2,
                ranked_candidate_limit=4,
            ),
        )
        self.policy = select_policy(selection_spec, self.compiled, self.features)
        self.spec = PolicyAcquisitionSpec(
            acquisition_id="synthetic-acquisition-round-1",
            policy_bundle_sha256=self.policy.to_dict()["policy_bundle_sha256"],
            candidate_budget=1,
            require_policy_eligible=True,
            exclude_exactly_observed=True,
            maximum_changed_knobs=1,
            maximum_normalized_distance=0.3,
            minimum_normalized_optimistic_improvement=0.0,
            weights=AcquisitionWeights(1.0, 0.25, 1.0, 0.1),
        )

    def test_selects_only_unobserved_candidate_in_trust_region(self) -> None:
        plan = plan_policy_experiments(
            self.spec, self.policy, self.compiled, self.features
        )
        payload = plan.to_dict()

        self.assertEqual(plan.status, "planned")
        self.assertEqual(
            payload["selections"][0]["deployment_settings"]["base_max_num_seqs"],
            96,
        )
        self.assertEqual(
            payload["selections"][0]["rationale_tags"],
            [
                "decision_uncertainty",
                "one_factor_from_control",
                "policy_slo_feasible",
                "positive_optimistic_improvement",
                "unmeasured_exact_configuration",
            ],
        )
        self.assertEqual(PolicyExperimentPlan.from_dict(payload).to_dict(), payload)

    def test_digest_mismatch_is_rejected(self) -> None:
        bad_spec = replace(self.spec, policy_bundle_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "does not match policy bundle"):
            plan_policy_experiments(
                bad_spec, self.policy, self.compiled, self.features
            )

    def test_calibration_overlay_preserves_settings_outside_search_space(self) -> None:
        plan = plan_policy_experiments(
            self.spec, self.policy, self.compiled, self.features
        )
        baseline = {
            **plan.to_dict()["control"]["deployment_settings"],
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 32, 64],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 48, 96],
        }
        calibration = CalibrationSpec.from_dict(
            {
                "schema_version": "1.0",
                "campaign_id": "synthetic-acquisition-calibration",
                "strong_baseline": True,
                "protocol": {
                    "pattern": "ABBA",
                    "blocks": 1,
                    "pair_seeds": [1, 2],
                },
                "semantic_contract": {
                    "algorithm_id": "conditional_is_small_proposal",
                    "semantic_class": "exact",
                    "graph_sha256": GRAPH_SHA256,
                    "invariants": {"candidate_count": 8},
                },
                "workload_contract": {
                    "workload_id": "synthetic-p128",
                    "dataset_sha256": "d" * 64,
                    "arrival_trace_sha256": "e" * 64,
                    "parameters": {"requests": 96},
                },
                "environment_contract": {
                    "environment_id": ENVIRONMENT_ID,
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
                    "configuration_id": "manual-baseline",
                    "settings": baseline,
                    "description": "synthetic baseline",
                },
                "candidates": [
                    {
                        "configuration_id": "placeholder",
                        "settings": {"base_max_num_seqs": 64},
                        "description": "replaced by acquisition",
                    }
                ],
            }
        )

        updated = calibration_spec_from_policy_experiment_plan(
            calibration,
            plan,
            include_replay_control=True,
            campaign_id="synthetic-acquisition-next",
            pair_seeds=(31, 32),
        )
        replay, candidate = updated.candidates

        self.assertEqual(updated.campaign_id, "synthetic-acquisition-next")
        self.assertEqual(updated.protocol.pair_seeds, (31, 32))
        self.assertEqual(replay.settings, calibration.baseline.settings)
        self.assertEqual(candidate.settings["base_max_num_seqs"], 96)
        self.assertEqual(candidate.settings["base_graph_capture_sizes"], [1, 32, 64])
        self.assertEqual(
            candidate.settings["proposal_graph_capture_sizes"], [1, 48, 96]
        )

    def test_replicates_do_not_crow_distinct_support_points_from_neighbors(self) -> None:
        features = features_from_ledger(
            EvidenceLedger(
                (
                    *(
                        _record(f"capacity-64-r{index}", 64, 0.8, 105.0)
                        for index in range(5)
                    ),
                    _record("capacity-128", 128, 1.2, 90.0),
                    _record("capacity-256", 256, 1.0, 110.0),
                )
            )
        )
        selection_spec = PolicySelectionSpec.from_dict(
            self.policy.to_dict()["selection_spec"]
        )
        policy = select_policy(selection_spec, self.compiled, features).to_dict()
        candidate = next(
            item
            for item in policy["ranked_candidates"]
            if item["deployment_settings"]["base_max_num_seqs"] == 96
        )
        rows_by_id = {row.row_id: row for row in features.rows}
        neighbor_capacities = {
            rows_by_id[row_id].static_features["deployment.base_max_num_seqs"]
            for row_id in candidate["response_neighbor_row_ids"]
        }

        self.assertEqual(neighbor_capacities, {64, 128, 256})

    def test_returns_no_candidate_after_all_points_are_observed(self) -> None:
        features = features_from_ledger(
            EvidenceLedger(
                (
                    _record("capacity-64", 64, 0.8, 105.0),
                    _record("capacity-96", 96, 1.0, 98.0),
                    _record("capacity-128", 128, 1.2, 90.0),
                    _record("capacity-256", 256, 1.0, 110.0),
                )
            )
        )
        selection_spec = PolicySelectionSpec.from_dict(
            self.policy.to_dict()["selection_spec"]
        )
        policy = select_policy(selection_spec, self.compiled, features)
        spec = replace(
            self.spec,
            policy_bundle_sha256=policy.to_dict()["policy_bundle_sha256"],
        )

        plan = plan_policy_experiments(spec, policy, self.compiled, features)

        self.assertEqual(plan.status, "no_candidate")
        self.assertEqual(plan.to_dict()["selections"], [])

    def test_recomputes_candidates_omitted_from_policy_report(self) -> None:
        selection_spec = PolicySelectionSpec.from_dict(
            self.policy.to_dict()["selection_spec"]
        )
        limited_spec = replace(
            selection_spec,
            model=replace(selection_spec.model, ranked_candidate_limit=1),
        )
        limited_policy = select_policy(limited_spec, self.compiled, self.features)
        acquisition_spec = replace(
            self.spec,
            policy_bundle_sha256=limited_policy.to_dict()["policy_bundle_sha256"],
        )

        plan = plan_policy_experiments(
            acquisition_spec, limited_policy, self.compiled, self.features
        ).to_dict()

        self.assertEqual(plan["audit"]["policy_reported_candidate_count"], 2)
        self.assertEqual(plan["audit"]["surrogate_evaluated_candidate_count"], 4)
        self.assertNotIn(
            "missing_policy_evaluation", plan["audit"]["rejections_by_reason"]
        )
        self.assertEqual(
            plan["selections"][0]["deployment_settings"]["base_max_num_seqs"],
            96,
        )

    def test_unreachable_graph_bucket_change_is_inactive(self) -> None:
        control = _compiled_graph_candidate("control", (1, 32, 40), 40)
        candidate = _compiled_graph_candidate("candidate", (1, 32, 40, 48), 40)
        changes = _changed_knobs(control, candidate)

        self.assertEqual(
            _inactive_changed_knobs(control, candidate, changes),
            ["base.capture_sizes"],
        )

    def test_reachable_graph_bucket_change_remains_active(self) -> None:
        control = _compiled_graph_candidate("control", (1, 24, 40), 40)
        candidate = _compiled_graph_candidate("candidate", (1, 24, 32, 40), 40)
        changes = _changed_knobs(control, candidate)

        self.assertEqual(_inactive_changed_knobs(control, candidate, changes), [])

    def test_regressive_inner_probe_blocks_farther_numeric_candidate(self) -> None:
        control = _compiled_capacity_candidate("control", 10240)
        failed_probe = _compiled_capacity_candidate("failed-probe", 12288)
        farther = _compiled_capacity_candidate("farther", 16384)
        blockers = _directional_regression_blockers(
            control,
            farther,
            (control, failed_probe, farther),
            {
                "failed-probe": {
                    "paired_objective_effect": {
                        "upper_directional_relative_improvement": -0.01
                    }
                }
            },
        )

        self.assertEqual([item["candidate_id"] for item in blockers], ["failed-probe"])

    def test_assessment_validates_signal_and_retains_formal_regression(self) -> None:
        plan = plan_policy_experiments(
            self.spec, self.policy, self.compiled, self.features
        )
        candidate_id = plan.to_dict()["selections"][0]["candidate_id"]
        spec = PolicyExperimentAssessmentSpec(
            assessment_id="synthetic-acquisition-assessment",
            policy_experiment_plan_sha256=plan.to_dict()[
                "policy_experiment_plan_sha256"
            ],
            calibration_plan_sha256="e" * 64,
            candidate_configuration_id=candidate_id,
            minimum_complete_pairs=2,
            minimum_median_improvement_fraction=0.03,
            require_effect_outside_replay_noise=True,
            require_quality_constraints=True,
        )
        calibration = {
            "plan_sha256": "e" * 64,
            "formal_complete": True,
            "issues": [],
            "effects": [
                {
                    "candidate_configuration_id": candidate_id,
                    "replay_control": False,
                    "primary_metric": "completed_qps",
                    "direction": "maximize",
                    "formal_group": True,
                    "complete_pair_count": 2,
                    "median_directional_relative_improvement": 0.05,
                    "candidate_over_baseline_geomean_ratio": 1.05,
                    "effect_outside_replay_noise": True,
                    "quality_constraints_satisfied": True,
                }
            ],
        }

        accepted = assess_policy_experiment(spec, plan, calibration)
        self.assertEqual(accepted.status, "validated_signal")
        self.assertTrue(accepted.to_dict()["eligible_for_response_model"])
        self.assertEqual(
            PolicyExperimentAssessment.from_dict(accepted.to_dict()).to_dict(),
            accepted.to_dict(),
        )

        calibration["effects"][0]["median_directional_relative_improvement"] = -0.02
        rejected = assess_policy_experiment(spec, plan, calibration)
        self.assertEqual(rejected.status, "rejected")
        self.assertTrue(rejected.to_dict()["eligible_for_response_model"])
        self.assertFalse(rejected.to_dict()["validated_improvement"])

        calibration["effects"][0]["effect_outside_replay_noise"] = False
        inconclusive = assess_policy_experiment(spec, plan, calibration)
        self.assertEqual(inconclusive.status, "inconclusive")
        self.assertEqual(
            inconclusive.to_dict()["next_action"],
            "collect_additional_pairs_before_expanding_search",
        )

        payload = rejected.to_dict()
        payload["validated_improvement"] = True
        unhashed = dict(payload)
        unhashed.pop("policy_experiment_assessment_sha256")
        payload["policy_experiment_assessment_sha256"] = canonical_sha256(unhashed)
        with self.assertRaisesRegex(ValueError, "validated improvement"):
            PolicyExperimentAssessment.from_dict(payload)


def _compiled_graph_candidate(
    candidate_id: str, capture_sizes: tuple[int, ...], capacity: int
) -> CompiledCandidate:
    return CompiledCandidate(
        candidate_id=candidate_id,
        semantic_cohort_id="conditional_is_small_proposal",
        knob_values={
            "base.capture_sizes": capture_sizes,
            "base.max_num_seqs": capacity,
        },
        deployment_settings={
            "base_graph_capture_sizes": capture_sizes,
            "base_max_num_seqs": capacity,
        },
        semantic_settings={},
        change_scopes=("engine_restart",),
        tuning_layers=("graph_capture", "static_deployment"),
        requires_engine_restart=True,
    )


def _compiled_capacity_candidate(
    candidate_id: str, token_capacity: int
) -> CompiledCandidate:
    return CompiledCandidate(
        candidate_id=candidate_id,
        semantic_cohort_id="conditional_is_small_proposal",
        knob_values={"base.max_num_batched_tokens": token_capacity},
        deployment_settings={"base_max_num_batched_tokens": token_capacity},
        semantic_settings={},
        change_scopes=("engine_restart",),
        tuning_layers=("static_deployment",),
        requires_engine_restart=True,
    )
