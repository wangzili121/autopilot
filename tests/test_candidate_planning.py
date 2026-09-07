from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.candidate_planning import (
    CandidateDesignSpec,
    CandidatePlan,
    EvidenceAnchorPolicy,
    NumericRange,
    SelectionContext,
    build_candidate_plan,
    configurations_from_plan,
)
from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    EvidenceRecord,
    QualityAssessment,
    SourceArtifact,
)
from inference_autopilot.features import features_from_ledger
from inference_autopilot.search_space import DeploymentSearchSpace, compile_search_space


def _space() -> DeploymentSearchSpace:
    return DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "candidate-design-test",
            "algorithm_id": "conditional_is_small_proposal",
            "fixed_settings": {"batch_wait_seconds": 0.01},
            "knobs": [
                {
                    "name": "base.capacity",
                    "setting_name": "base_capacity",
                    "value_type": "integer",
                    "domain": {"kind": "choices", "values": [1, 2, 3]},
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "description": "Base capacity",
                },
                {
                    "name": "proposal.capacity",
                    "setting_name": "proposal_capacity",
                    "value_type": "integer",
                    "domain": {"kind": "choices", "values": [10, 20, 30]},
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "description": "Proposal capacity",
                },
            ],
            "constraints": [],
        }
    )


def _design(compiled_digest: str) -> CandidateDesignSpec:
    return CandidateDesignSpec(
        design_id="candidate-design-test",
        compiled_space_sha256=compiled_digest,
        candidate_budget=4,
        seed=7,
        selection_context=SelectionContext(
            algorithm_id="conditional_is_small_proposal",
            semantic_cohort_id="conditional_is_small_proposal",
            graph_sha256="b" * 64,
            accepted_evidence_semantic_classes=(
                "exact",
                "exact_algorithm",
            ),
            workload_id="short-p96",
            environment_id="test-environment",
            static_features={"workload.requests": 96},
            static_feature_ranges={
                "workload.prompt_tokens_mean": NumericRange(0, 512)
            },
        ),
        baseline_settings={
            "batch_wait_seconds": 0.01,
            "base_capacity": 2,
            "proposal_capacity": 20,
        },
        evidence_policy=EvidenceAnchorPolicy(
            grades=("A_formal_paired", "B_controlled_single"),
            minimum_matched_deployment_settings=2,
            maximum_anchor_candidates=1,
        ),
        maximum_boundary_candidates=1,
    )


def _record(
    record_id: str,
    *,
    requests: int,
    prompt_tokens: int,
    base_capacity: int,
    proposal_capacity: int,
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        campaign="historical",
        variant=record_id,
        source=SourceArtifact(
            f"{record_id}.json", "a" * 64, "test", record_id
        ),
        workload={
            "workload_id": "short-p96",
            "method": "conditional_is_small_proposal",
            "requests": requests,
            "workers": requests,
            "prompt_tokens_mean": prompt_tokens,
        },
        configuration={
            "base_capacity": base_capacity,
            "proposal_capacity": proposal_capacity,
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "autopilot_semantic_class": "exact_algorithm",
        },
        environment={},
        metrics={"completed_qps": 0.8, "p95_seconds": 10.0},
        quality=QualityAssessment(
            EvidenceGrade.B_CONTROLLED_SINGLE,
            EvidencePurpose.CALIBRATION_ONLY,
            False,
            ("controlled historical point",),
        ),
    )


class CandidatePlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.space = _space()
        self.compiled = compile_search_space(self.space)
        self.design = _design(self.compiled.to_dict()["compiled_space_sha256"])
        self.features = features_from_ledger(
            EvidenceLedger(
                (
                    _record(
                        "good-anchor",
                        requests=96,
                        prompt_tokens=100,
                        base_capacity=3,
                        proposal_capacity=30,
                    ),
                    _record(
                        "long-context",
                        requests=96,
                        prompt_tokens=2000,
                        base_capacity=1,
                        proposal_capacity=10,
                    ),
                    _record(
                        "wrong-load",
                        requests=32,
                        prompt_tokens=100,
                        base_capacity=1,
                        proposal_capacity=30,
                    ),
                )
            )
        )

    def test_plan_layers_baseline_anchor_boundary_and_space_filling(self) -> None:
        plan = build_candidate_plan(
            self.design, self.space, self.compiled, self.features
        )

        self.assertEqual(
            [selection.reason for selection in plan.selections],
            ["baseline", "evidence_anchor", "domain_boundary", "space_filling"],
        )
        self.assertEqual(
            plan.selections[1].evidence_row_ids,
            ("good-anchor",),
        )
        self.assertEqual(plan.evidence_audit["rows_seen"], 3)
        self.assertEqual(plan.evidence_audit["eligible_anchor_rows"], 1)
        self.assertEqual(plan.evidence_audit["context_conflict"], 2)
        self.assertEqual(plan.audit()["unselected_candidate_count"], 5)
        self.assertTrue(plan.boundary_tags_covered)

    def test_plan_round_trip_and_content_digest_detect_tampering(self) -> None:
        plan = build_candidate_plan(
            self.design, self.space, self.compiled, self.features
        )
        payload = plan.to_dict()
        self.assertEqual(CandidatePlan.from_dict(payload), plan)

        payload["selections"][0]["reason"] = "space_filling"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            CandidatePlan.from_dict(payload)

    def test_plan_is_deterministic_for_reordered_evidence(self) -> None:
        forward = build_candidate_plan(
            self.design, self.space, self.compiled, self.features
        )
        reversed_features = features_from_ledger(
            EvidenceLedger(tuple(reversed(self.features_to_records())))
        )
        reverse = build_candidate_plan(
            self.design, self.space, self.compiled, reversed_features
        )

        self.assertEqual(
            [selection.candidate_id for selection in forward.selections],
            [selection.candidate_id for selection in reverse.selections],
        )

    def features_to_records(self) -> tuple[EvidenceRecord, ...]:
        return (
            _record(
                "good-anchor",
                requests=96,
                prompt_tokens=100,
                base_capacity=3,
                proposal_capacity=30,
            ),
            _record(
                "long-context",
                requests=96,
                prompt_tokens=2000,
                base_capacity=1,
                proposal_capacity=10,
            ),
            _record(
                "wrong-load",
                requests=32,
                prompt_tokens=100,
                base_capacity=1,
                proposal_capacity=30,
            ),
        )

    def test_exported_configurations_exclude_baseline_by_default(self) -> None:
        plan = build_candidate_plan(
            self.design, self.space, self.compiled, self.features
        )
        configurations = configurations_from_plan(plan)

        self.assertEqual(len(configurations), 3)
        self.assertEqual(
            {configuration.configuration_id for configuration in configurations},
            {selection.candidate_id for selection in plan.selections[1:]},
        )

    def test_design_must_bind_compiled_artifact_and_exact_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "not bound"):
            build_candidate_plan(
                replace(self.design, compiled_space_sha256="f" * 64),
                self.space,
                self.compiled,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_candidate_plan(
                replace(
                    self.design,
                    baseline_settings={
                        "batch_wait_seconds": 0.01,
                        "base_capacity": 999,
                        "proposal_capacity": 20,
                    },
                ),
                self.space,
                self.compiled,
            )


if __name__ == "__main__":
    unittest.main()
