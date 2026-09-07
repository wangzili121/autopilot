from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.calibration.models import canonical_sha256
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
from inference_autopilot.guarded_routing import (
    RuntimePolicyEndpointSpec,
    RuntimePolicyPool,
    RuntimePolicyPoolSpec,
    RuntimeRoutingRequest,
    build_runtime_routing_request,
    compile_runtime_policy_pool,
    route_runtime_policy,
)
from inference_autopilot.policy_selection import (
    LocalResponseModelSpec,
    PolicySelectionSpec,
    ResponseConstraint,
    ResponseObjective,
    select_policy,
)
from inference_autopilot.policy_validation import (
    PolicyHoldoutAssessment,
    PolicyHoldoutSpec,
    assess_policy_holdout,
)
from inference_autopilot.search_space import DeploymentSearchSpace, compile_search_space


GRAPH_SHA256 = "d" * 64


def _record(record_id: str, capacity: int, qps: float) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        campaign="runtime-policy-test",
        variant=record_id,
        source=SourceArtifact(record_id, "a" * 64, "test", record_id),
        workload={
            "workload_id": "short-p32",
            "requests": 32,
            "workers": 32,
            "prompt_tokens_mean": 128,
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
        environment={"environment_id": "npu2"},
        metrics={"completed_qps": qps, "p95_seconds": 10.0, "accuracy": 0.8},
        quality=QualityAssessment(
            grade=EvidenceGrade.A_FORMAL_PAIRED,
            purpose=EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
            claim_eligible=True,
            reasons=("synthetic runtime policy evidence",),
        ),
    )


def _validated_policy():
    compiled = compile_search_space(
        DeploymentSearchSpace.from_dict(
            {
                "schema_version": "1.0",
                "space_id": "runtime-policy-test",
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
                        "domain": {"kind": "choices", "values": [48, 64]},
                        "change_scope": "engine_restart",
                        "semantic_effect": "preserves_algorithm",
                        "description": "proposal capacity",
                    }
                ],
                "constraints": [],
            }
        )
    )
    selection = PolicySelectionSpec(
        policy_id="short-p32-runtime-test",
        compiled_space_sha256=compiled.to_dict()["compiled_space_sha256"],
        selection_context=SelectionContext(
            algorithm_id="conditional_is_small_proposal",
            semantic_cohort_id="conditional_is_small_proposal",
            graph_sha256=GRAPH_SHA256,
            accepted_evidence_semantic_classes=("exact",),
            workload_id="short-p32",
            environment_id="npu2",
            static_features={"workload.requests": 32, "workload.workers": 32},
            static_feature_ranges={
                "workload.prompt_tokens_mean": NumericRange(64, 256)
            },
        ),
        query_static_features={
            "workload.prompt_tokens_mean": 128,
            "workload.requests": 32,
        },
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
            constraints=(
                ResponseConstraint("latency.p95_seconds", "<=", 12.0),
                ResponseConstraint("quality.accuracy", ">=", 0.5),
            ),
        ),
        model=LocalResponseModelSpec(
            feature_names=(
                "deployment.proposal_max_num_seqs",
                "workload.prompt_tokens_mean",
                "workload.requests",
            ),
            neighbors=2,
            distance_power=2.0,
            maximum_normalized_distance=1.0,
            uncertainty_multiplier=0.0,
            maximum_failure_probability=0.1,
            minimum_response_rows=2,
            minimum_distinct_configurations=2,
            minimum_paired_replicates=1,
            ranked_candidate_limit=2,
        ),
    )
    training = features_from_ledger(
        EvidenceLedger((_record("train-48", 48, 1.0), _record("train-64", 64, 1.2)))
    )
    policy = select_policy(selection, compiled, training)
    holdout = features_from_ledger(
        EvidenceLedger(
            (
                _record("holdout-48-a", 48, 1.0),
                _record("holdout-48-b", 48, 1.02),
                _record("holdout-64-a", 64, 1.3),
                _record("holdout-64-b", 64, 1.32),
            )
        )
    )
    assessment = assess_policy_holdout(
        PolicyHoldoutSpec(
            assessment_id="short-p32-runtime-holdout",
            policy_bundle_sha256=policy.payload["policy_bundle_sha256"],
            minimum_successful_replicates=2,
            maximum_regret_fraction=0.01,
            minimum_improvement_over_fallback_fraction=0.05,
        ),
        policy,
        compiled,
        holdout,
    )
    return policy, assessment


class GuardedRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy, self.assessment = _validated_policy()
        self.spec = RuntimePolicyPoolSpec(
            pool_id="short-p32-runtime-pool",
            algorithm_id="conditional_is_small_proposal",
            semantic_cohort_id="conditional_is_small_proposal",
            graph_sha256=GRAPH_SHA256,
            accepted_semantic_classes=("exact",),
            fallback_endpoint_id="fallback-48",
            fallback_deployment_settings=self.policy.payload["fallback"][
                "deployment_settings"
            ],
            policy_endpoints=(
                RuntimePolicyEndpointSpec(
                    endpoint_id="selected-64",
                    policy_bundle_sha256=self.policy.payload["policy_bundle_sha256"],
                    policy_holdout_assessment_sha256=self.assessment.payload[
                        "policy_holdout_assessment_sha256"
                    ],
                    priority=100,
                ),
            ),
            require_safe_boundary=True,
            minimum_live_metric_samples=8,
            maximum_consecutive_failures=0,
            live_constraints=(ResponseConstraint("latency.p95_seconds", "<=", 12.0),),
        )
        self.pool = compile_runtime_policy_pool(
            self.spec, (self.policy,), (self.assessment,)
        )

    def _request(
        self,
        *,
        prompt_tokens: int = 128,
        safe_boundary: bool = True,
        selected_overrides=None,
        fallback_overrides=None,
    ):
        selected = self.pool.payload["policy_endpoints"][0]
        fallback = self.pool.payload["fallback_endpoint"]
        selected_state = {
            "endpoint_id": selected["endpoint_id"],
            "configuration_sha256": selected["configuration_sha256"],
            "ready": True,
            "accepting_requests": True,
            "consecutive_failures": 0,
            "live_metric_samples": 0,
            "live_metrics": {},
            **(selected_overrides or {}),
        }
        fallback_state = {
            "endpoint_id": fallback["endpoint_id"],
            "configuration_sha256": fallback["configuration_sha256"],
            "ready": True,
            "accepting_requests": True,
            "consecutive_failures": 0,
            "live_metric_samples": 0,
            "live_metrics": {},
            **(fallback_overrides or {}),
        }
        return build_runtime_routing_request(
            self.pool,
            decision_id="route-short-p32",
            safe_boundary=safe_boundary,
            context={
                "algorithm_id": "conditional_is_small_proposal",
                "semantic_cohort_id": "conditional_is_small_proposal",
                "semantic_class": "exact",
                "graph_sha256": GRAPH_SHA256,
                "workload_id": "short-p32",
                "environment_id": "npu2",
                "static_features": {
                    "workload.requests": 32,
                    "workload.workers": 32,
                    "workload.prompt_tokens_mean": prompt_tokens,
                },
            },
            endpoint_states=(selected_state, fallback_state),
        )

    def test_routes_only_to_independently_validated_matching_endpoint(self) -> None:
        decision = route_runtime_policy(self.pool, self._request())
        self.assertEqual(decision.status, "selected")
        self.assertEqual(decision.payload["selected_endpoint_id"], "selected-64")
        self.assertFalse(decision.payload["fallback_used"])
        self.assertEqual(
            RuntimePolicyPool.from_dict(self.pool.to_dict()).to_dict(),
            self.pool.to_dict(),
        )

    def test_context_boundary_and_live_slo_violation_fall_back(self) -> None:
        outside = route_runtime_policy(self.pool, self._request(prompt_tokens=512))
        self.assertEqual(outside.status, "fallback")
        self.assertIn(
            "range_feature_out_of_bounds:workload.prompt_tokens_mean",
            outside.payload["endpoint_evaluations"][0]["reasons"],
        )

        unhealthy = route_runtime_policy(
            self.pool,
            self._request(
                selected_overrides={
                    "live_metric_samples": 8,
                    "live_metrics": {"latency.p95_seconds": 13.0},
                }
            ),
        )
        self.assertEqual(unhealthy.status, "fallback")
        self.assertIn(
            "live_constraint_violation:latency.p95_seconds",
            unhealthy.payload["endpoint_evaluations"][0]["reasons"],
        )

    def test_unsafe_boundary_and_configuration_mismatch_fall_back(self) -> None:
        unsafe = route_runtime_policy(self.pool, self._request(safe_boundary=False))
        self.assertEqual(unsafe.status, "fallback")
        mismatch = route_runtime_policy(
            self.pool,
            self._request(selected_overrides={"configuration_sha256": "f" * 64}),
        )
        self.assertEqual(mismatch.status, "fallback")
        self.assertIn(
            "endpoint_configuration_mismatch",
            mismatch.payload["endpoint_evaluations"][0]["reasons"],
        )

    def test_returns_unavailable_when_fallback_is_not_healthy(self) -> None:
        decision = route_runtime_policy(
            self.pool,
            self._request(
                prompt_tokens=512,
                fallback_overrides={"accepting_requests": False},
            ),
        )
        self.assertEqual(decision.status, "unavailable")
        self.assertIsNone(decision.payload["selected_endpoint_id"])
        self.assertIn(
            "fallback_endpoint_not_accepting_requests", decision.payload["reasons"]
        )

    def test_pool_requires_validated_holdout_and_unambiguous_overlap(self) -> None:
        rejected_raw = self.assessment.to_dict()
        rejected_raw["status"] = "rejected"
        rejected_raw["reasons"] = ["synthetic_rejection"]
        rejected_raw.pop("policy_holdout_assessment_sha256")
        rejected_raw["policy_holdout_assessment_sha256"] = canonical_sha256(
            rejected_raw
        )
        rejected = PolicyHoldoutAssessment.from_dict(rejected_raw)
        rejected_binding = replace(
            self.spec.policy_endpoints[0],
            policy_holdout_assessment_sha256=rejected.payload[
                "policy_holdout_assessment_sha256"
            ],
        )
        rejected_spec = replace(self.spec, policy_endpoints=(rejected_binding,))
        with self.assertRaisesRegex(ValueError, "lacks validated holdout"):
            compile_runtime_policy_pool(rejected_spec, (self.policy,), (rejected,))

        overlap_spec = replace(
            self.spec,
            policy_endpoints=(
                replace(self.spec.policy_endpoints[0], endpoint_id="selected-64-a"),
                replace(self.spec.policy_endpoints[0], endpoint_id="selected-64-b"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "distinct priorities"):
            compile_runtime_policy_pool(
                overlap_spec, (self.policy,), (self.assessment,)
            )

    def test_request_and_pool_digests_detect_tampering(self) -> None:
        request_raw = self._request().to_dict()
        request_raw["context"]["static_features"]["workload.requests"] = 64
        with self.assertRaisesRegex(ValueError, "request SHA256"):
            RuntimeRoutingRequest.from_dict(request_raw)
        pool_raw = self.pool.to_dict()
        pool_raw["policy_endpoints"][0]["priority"] = 99
        with self.assertRaisesRegex(ValueError, "pool SHA256"):
            RuntimePolicyPool.from_dict(pool_raw)


if __name__ == "__main__":
    unittest.main()
