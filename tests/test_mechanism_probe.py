from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.adapters import build_conditional_is_small_proposal_graph
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
from inference_autopilot.mechanism_probe import (
    MechanismProbeAssessment,
    MechanismProbeAssessmentSpec,
    MechanismProbePlan,
    MechanismProbeSpec,
    MechanismThresholds,
    assess_mechanism_probe,
    plan_mechanism_probes,
)
from inference_autopilot.search_space import DeploymentSearchSpace, compile_search_space


ENVIRONMENT_ID = "ascend-mechanism-test"
WORKLOAD_ID = "gsm8k-medium2k-p32-test"


def _graph():
    return build_conditional_is_small_proposal_graph(
        candidate_count=8,
        rollout_count=3,
        block_size=48,
        total_length=192,
    )


def _space() -> DeploymentSearchSpace:
    return DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "medium2k-mechanism-test",
            "algorithm_id": "conditional_is_small_proposal",
            "fixed_settings": {
                "model_runner": "MRV1",
                "base_max_num_seqs": 40,
                "proposal_max_num_seqs": 48,
                "proposal_max_num_batched_tokens": 12288,
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
                "base_batch_wait_seconds": 0.0,
                "proposal_batch_wait_seconds": 0.02,
                "base_score_priority": 1,
                "base_graph_mode": "FULL_DECODE_ONLY",
                "base_graph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40],
                "proposal_graph_mode": "FULL_DECODE_ONLY",
                "proposal_graph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48],
            },
            "knobs": [
                {
                    "name": "base.max_num_batched_tokens",
                    "setting_name": "base_max_num_batched_tokens",
                    "value_type": "integer",
                    "domain": {"kind": "choices", "values": [10240, 12288, 16384]},
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "tuning_layer": "static_deployment",
                    "applies_to_stages": ["candidate_generate", "target_score"],
                    "description": "Base token batching budget",
                }
            ],
            "constraints": [],
        }
    )


def _record(record_id: str, qps: float, waiting: float) -> EvidenceRecord:
    graph = _graph()
    return EvidenceRecord(
        record_id=record_id,
        campaign="medium2k-mechanism-evidence",
        variant="fallback",
        source=SourceArtifact(f"{record_id}.json", "a" * 64, "test", record_id),
        workload={
            "workload_id": WORKLOAD_ID,
            "method": "conditional_is_small_proposal",
            "requests": 32,
            "workers": 32,
            "arrival_qps": 0.0,
            "prompt_tokens": {"mean": 2184.0},
        },
        configuration={
            "model_runner": "MRV1",
            "base_max_num_seqs": 40,
            "proposal_max_num_seqs": 48,
            "base_max_num_batched_tokens": 10240,
            "proposal_max_num_batched_tokens": 12288,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
            "base_batch_wait_seconds": 0.0,
            "proposal_batch_wait_seconds": 0.02,
            "base_score_priority": 1,
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48],
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact",
            "graph_sha256": canonical_sha256(graph.to_dict()),
            "candidate_count": 8,
            "rollout_count": 3,
            "block_size": 48,
            "total_length": 192,
            "apply_importance_correction": True,
        },
        environment={"environment_id": ENVIRONMENT_ID},
        metrics={
            "run_status": "success",
            "completed_qps": qps,
            "latency_seconds": {"p95": 150.0},
            "accuracy": 0.5,
            "compute": {
                "base_backend": {
                    "estimated_dense_forward_flops": 9_700.0,
                    "score_forward_token_slots": 8_800.0,
                },
                "proposal_backend": {
                    "estimated_dense_forward_flops": 300.0,
                    "generation_forward_token_slots": 700.0,
                },
                "total_forward_token_slots": 10_000.0,
            },
            "continuous_batching": {
                "base": {"maximum_sample_batch": 8, "maximum_score_batch": 24},
                "proposal": {"maximum_sample_batch": 48, "maximum_score_batch": 0},
            },
            "vllm_runtime_metrics": {
                "proposal": {
                    "vllm:num_requests_running": {"mean": 42.0},
                    "vllm:num_requests_waiting": {"mean": waiting},
                    "vllm:kv_cache_usage_perc": {"maximum": 0.015},
                    "vllm:num_preemptions": {"maximum": 0.0},
                }
            },
        },
        quality=QualityAssessment(
            grade=EvidenceGrade.A_FORMAL_PAIRED,
            purpose=EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
            claim_eligible=True,
            reasons=("formal paired mechanism evidence",),
        ),
    )


class MechanismProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _graph()
        self.space = _space()
        self.space.validate_against_graph(self.graph)
        self.compiled = compile_search_space(self.space)
        self.features = features_from_ledger(
            EvidenceLedger(
                (
                    _record("fallback-a", 0.205, 260.0),
                    _record("fallback-b", 0.210, 280.0),
                )
            )
        )
        self.spec = MechanismProbeSpec(
            probe_id="medium2k-base-token-round1",
            compiled_space_sha256=self.compiled.to_dict()["compiled_space_sha256"],
            selection_context=SelectionContext(
                algorithm_id="conditional_is_small_proposal",
                semantic_cohort_id="conditional_is_small_proposal",
                graph_sha256=canonical_sha256(self.graph.to_dict()),
                accepted_evidence_semantic_classes=("exact",),
                workload_id=WORKLOAD_ID,
                environment_id=ENVIRONMENT_ID,
                static_features={"workload.requests": 32, "workload.workers": 32},
                static_feature_ranges={
                    "workload.prompt_tokens.mean": NumericRange(2000, 2400)
                },
            ),
            control_settings={
                "base_max_num_seqs": 40,
                "proposal_max_num_seqs": 48,
                "base_max_num_batched_tokens": 10240,
                "proposal_max_num_batched_tokens": 12288,
            },
            candidate_budget=1,
            maximum_changed_knobs=1,
            minimum_formal_control_rows=2,
            exclude_exactly_observed=True,
            thresholds=MechanismThresholds(0.6, 0.8, 1.0, 0.8, 0.05),
        )

    def test_selects_nearest_base_token_probe_for_base_score_dominance(self) -> None:
        plan = plan_mechanism_probes(
            self.spec, self.graph, self.compiled, self.features
        )
        payload = plan.to_dict()

        self.assertEqual(plan.status, "planned")
        self.assertEqual(
            payload["selections"][0]["deployment_settings"][
                "base_max_num_batched_tokens"
            ],
            12288,
        )
        self.assertIn("base_compute_dominant", payload["diagnostics"]["pressure_tags"])
        self.assertIn(
            "base_score_token_dominant", payload["diagnostics"]["pressure_tags"]
        )
        self.assertIn(
            "base_token_capacity_pressure", payload["diagnostics"]["pressure_tags"]
        )
        self.assertEqual(
            payload["diagnostics"]["roles"]["base"][
                "estimated_token_capacity_utilization"
            ],
            1.0,
        )
        self.assertEqual(
            payload["selections"][0]["stage_bindings"]["base.max_num_batched_tokens"],
            ["candidate_generate", "target_score"],
        )

    def test_does_not_expand_token_budget_with_observed_headroom(self) -> None:
        rows = tuple(
            replace(
                row,
                static_features={
                    **row.static_features,
                    "workload.prompt_tokens.mean": 118.0,
                },
            )
            for row in self.features.rows
        )
        features = type(self.features)(rows)
        context = replace(
            self.spec.selection_context,
            static_feature_ranges={
                "workload.prompt_tokens.mean": NumericRange(100, 140)
            },
        )
        plan = plan_mechanism_probes(
            replace(self.spec, selection_context=context),
            self.graph,
            self.compiled,
            features,
        )
        payload = plan.to_dict()

        self.assertEqual(plan.status, "no_candidate")
        self.assertNotIn(
            "base_token_capacity_pressure", payload["diagnostics"]["pressure_tags"]
        )
        self.assertAlmostEqual(
            payload["diagnostics"]["roles"]["base"][
                "estimated_token_capacity_utilization"
            ],
            0.7265625,
        )
        self.assertTrue(
            all(
                "no_positive_mechanism" in item["rejection_reasons"]
                for item in payload["ranked_candidates"]
                if item["candidate_id"]
                != payload["control"]["candidate_id"]
            )
        )
        self.assertTrue(
            all(
                "no_observed_token_capacity_pressure" in item["risk_tags"]
                for item in payload["ranked_candidates"]
                if item["candidate_id"]
                != payload["control"]["candidate_id"]
            )
        )

    def test_does_not_reduce_token_budget_without_resource_pressure(self) -> None:
        raw = self.space.to_dict()
        raw["knobs"][0]["domain"]["values"] = [8192, 10240, 12288]
        space = DeploymentSearchSpace.from_dict(raw)
        compiled = compile_search_space(space)
        spec = replace(
            self.spec,
            compiled_space_sha256=compiled.to_dict()["compiled_space_sha256"],
        )
        plan = plan_mechanism_probes(spec, self.graph, compiled, self.features)
        payload = plan.to_dict()
        reduction = next(
            item
            for item in payload["ranked_candidates"]
            if item["knob_values"]["base.max_num_batched_tokens"] == 8192
        )

        self.assertFalse(reduction["eligible"])
        self.assertIn("no_positive_mechanism", reduction["rejection_reasons"])
        self.assertIn("no_observed_resource_pressure", reduction["risk_tags"])
        self.assertEqual(
            reduction["mechanism_signals"][0]["mechanism"],
            "token_resource_headroom_no_reduction",
        )
        self.assertEqual(MechanismProbePlan.from_dict(payload).to_dict(), payload)

    def test_requires_predeclared_formal_control_support(self) -> None:
        strict = replace(self.spec, minimum_formal_control_rows=3)
        plan = plan_mechanism_probes(strict, self.graph, self.compiled, self.features)
        self.assertEqual(plan.status, "insufficient_evidence")
        self.assertEqual(plan.to_dict()["selections"], [])

    def test_uses_attested_runtime_graph_ceiling_when_policy_is_implicit(self) -> None:
        rows = tuple(
            replace(
                row,
                static_features={
                    **row.static_features,
                    "runtime.proposal.graph_capture_ceiling": 32,
                    "runtime.proposal.uncovered_graph_capacity": 16,
                },
            )
            for row in self.features.rows
        )
        features = type(self.features)(rows)
        plan = plan_mechanism_probes(
            self.spec, self.graph, self.compiled, features
        )

        diagnostics = plan.to_dict()["diagnostics"]
        self.assertEqual(
            diagnostics["roles"]["proposal"]["graph_capture_ceiling"], 48.0
        )
        self.assertNotIn(
            "proposal_graph_coverage_gap", diagnostics["pressure_tags"]
        )

        implicit_raw = self.space.to_dict()
        for name in (
            "base_graph_mode",
            "base_graph_capture_sizes",
            "proposal_graph_mode",
            "proposal_graph_capture_sizes",
        ):
            implicit_raw["fixed_settings"].pop(name)
        implicit_space = DeploymentSearchSpace.from_dict(implicit_raw)
        implicit_compiled = compile_search_space(implicit_space)
        implicit_spec = replace(
            self.spec,
            compiled_space_sha256=implicit_compiled.to_dict()[
                "compiled_space_sha256"
            ],
        )
        implicit_plan = plan_mechanism_probes(
            implicit_spec, self.graph, implicit_compiled, features
        )
        implicit_diagnostics = implicit_plan.to_dict()["diagnostics"]
        self.assertEqual(
            implicit_diagnostics["roles"]["proposal"][
                "graph_capture_ceiling"
            ],
            32.0,
        )
        self.assertIn(
            "proposal_graph_coverage_gap",
            implicit_diagnostics["pressure_tags"],
        )

    def test_rejects_tampered_plan(self) -> None:
        payload = plan_mechanism_probes(
            self.spec, self.graph, self.compiled, self.features
        ).to_dict()
        payload["diagnostics"]["prompt_tokens_mean"] = 1.0
        with self.assertRaisesRegex(ValueError, "SHA256"):
            MechanismProbePlan.from_dict(payload)

    def test_rejects_graph_identity_mismatch(self) -> None:
        bad_context = replace(self.spec.selection_context, graph_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "graph content"):
            plan_mechanism_probes(
                replace(self.spec, selection_context=bad_context),
                self.graph,
                self.compiled,
                self.features,
            )

    def _assessment_inputs(self, improvement: float, outside_noise: bool):
        plan = plan_mechanism_probes(
            self.spec, self.graph, self.compiled, self.features
        )
        candidate_id = plan.to_dict()["selections"][0]["candidate_id"]
        calibration_plan_sha256 = "d" * 64
        spec = MechanismProbeAssessmentSpec(
            assessment_id="medium2k-base-token-round1-result",
            mechanism_probe_plan_sha256=plan.to_dict()["mechanism_probe_plan_sha256"],
            calibration_plan_sha256=calibration_plan_sha256,
            candidate_configuration_id=candidate_id,
            minimum_complete_pairs=2,
            minimum_median_improvement_fraction=0.03,
            require_effect_outside_replay_noise=True,
            require_quality_constraints=True,
        )
        assessment = {
            "plan_sha256": calibration_plan_sha256,
            "formal_complete": True,
            "issues": [],
            "effects": [
                {
                    "candidate_configuration_id": candidate_id,
                    "replay_control": False,
                    "formal_group": True,
                    "complete_pair_count": 2,
                    "primary_metric": "completed_qps",
                    "direction": "maximize",
                    "median_directional_relative_improvement": improvement,
                    "candidate_over_baseline_geomean_ratio": 1.0 + improvement,
                    "effect_outside_replay_noise": outside_noise,
                    "quality_constraints_satisfied": True,
                }
            ],
        }
        return spec, plan, assessment

    def test_assessment_validates_signal_with_frozen_gate(self) -> None:
        spec, plan, calibration = self._assessment_inputs(0.08, True)
        assessment = assess_mechanism_probe(spec, plan, calibration)

        self.assertEqual(assessment.status, "validated_signal")
        self.assertTrue(assessment.to_dict()["eligible_for_response_model"])
        self.assertTrue(assessment.to_dict()["validated_improvement"])
        self.assertEqual(
            MechanismProbeAssessment.from_dict(assessment.to_dict()).to_dict(),
            assessment.to_dict(),
        )

    def test_formal_negative_probe_is_kept_for_response_model(self) -> None:
        spec, plan, calibration = self._assessment_inputs(-0.02, True)
        assessment = assess_mechanism_probe(spec, plan, calibration)
        payload = assessment.to_dict()

        self.assertEqual(assessment.status, "rejected")
        self.assertTrue(payload["eligible_for_response_model"])
        self.assertFalse(payload["validated_improvement"])
        self.assertEqual(
            payload["next_action"], "retain_control_and_refit_response_model"
        )
        self.assertIn("minimum_probe_improvement_not_met", payload["reasons"])

    def test_rehashed_assessment_cannot_change_effect_without_recomputing_gate(
        self,
    ) -> None:
        spec, plan, calibration = self._assessment_inputs(0.08, True)
        payload = assess_mechanism_probe(spec, plan, calibration).to_dict()
        payload["effect"]["median_directional_relative_improvement"] = -0.5
        unsigned = dict(payload)
        unsigned.pop("mechanism_probe_assessment_sha256")
        payload["mechanism_probe_assessment_sha256"] = canonical_sha256(unsigned)

        with self.assertRaisesRegex(ValueError, "reasons"):
            MechanismProbeAssessment.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
