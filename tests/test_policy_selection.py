from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.candidate_planning import NumericRange, SelectionContext
from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.cli import main
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
    PolicyBundle,
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


GRAPH_SHA256 = "b" * 64
ENVIRONMENT_ID = "ascend-a2-test"


def _space() -> DeploymentSearchSpace:
    return DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "policy-selection-test",
            "algorithm_id": "conditional_is_small_proposal",
            "fixed_settings": {
                "model_runner": "MRV1",
                "proposal_max_num_seqs": 896,
                "base_max_num_batched_tokens": 65536,
                "proposal_max_num_batched_tokens": 147456,
                "base_memory_fraction": 0.54,
                "proposal_memory_fraction": 0.36,
            },
            "knobs": [
                {
                    "name": "base.max_num_seqs",
                    "setting_name": "base_max_num_seqs",
                    "value_type": "integer",
                    "domain": {"kind": "choices", "values": [64, 128, 256]},
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "description": "Base scheduler capacity",
                }
            ],
            "constraints": [],
        }
    )


def _quality(grade: EvidenceGrade) -> QualityAssessment:
    purpose = {
        EvidenceGrade.A_FORMAL_PAIRED: EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
        EvidenceGrade.B_CONTROLLED_SINGLE: EvidencePurpose.CALIBRATION_ONLY,
        EvidenceGrade.X_EXCLUDED: EvidencePurpose.CONSTRAINT_ONLY,
    }[grade]
    return QualityAssessment(
        grade=grade,
        purpose=purpose,
        claim_eligible=grade == EvidenceGrade.A_FORMAL_PAIRED,
        reasons=("synthetic selector validation",),
    )


def _record(
    record_id: str,
    base_max_num_seqs: int,
    *,
    qps: float = 1.0,
    p95: float = 100.0,
    accuracy: float = 0.7,
    prompt_tokens: int = 128,
    grade: EvidenceGrade = EvidenceGrade.A_FORMAL_PAIRED,
    failure: str | None = None,
) -> EvidenceRecord:
    metrics = (
        {"failure": failure}
        if failure is not None
        else {
            "completed_qps": qps,
            "p95_seconds": p95,
            "accuracy": accuracy,
        }
    )
    return EvidenceRecord(
        record_id=record_id,
        campaign="synthetic-policy-replay",
        variant=record_id,
        source=SourceArtifact(
            f"{record_id}.json", "a" * 64, "test", record_id
        ),
        workload={
            "workload_id": f"synthetic-p{prompt_tokens}",
            "method": "conditional_is_small_proposal",
            "requests": 96,
            "workers": 96,
            "prompt_tokens_mean": prompt_tokens,
        },
        configuration={
            "model_runner": "MRV1",
            "base_max_num_seqs": base_max_num_seqs,
            "proposal_max_num_seqs": 896,
            "base_max_num_batched_tokens": 65536,
            "proposal_max_num_batched_tokens": 147456,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact_algorithm",
            "graph_sha256": GRAPH_SHA256,
            "candidate_count": 15,
            "rollout_count": 3,
            "block_size": 128,
            "total_length": 512,
            "apply_importance_correction": True,
        },
        environment={"environment_id": ENVIRONMENT_ID},
        metrics=metrics,
        quality=_quality(grade),
    )


def _paired_record(
    record_id: str,
    base_max_num_seqs: int,
    *,
    group_id: str,
    pair_index: int,
    variant_role: str,
    qps: float,
) -> EvidenceRecord:
    record = _record(record_id, base_max_num_seqs, qps=qps)
    return replace(
        record,
        workload={
            **record.workload,
            "comparison_group_id": group_id,
            "pair_index": pair_index,
            "workload_seed": 1000 + pair_index,
        },
        tags=("calibration", "formal_group", variant_role),
    )


def _spec(compiled_digest: str, *, maximum_failure_probability: float = 0.4) -> PolicySelectionSpec:
    return PolicySelectionSpec(
        policy_id="conditional-is-p128-policy",
        compiled_space_sha256=compiled_digest,
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
            "proposal_max_num_seqs": 896,
            "base_max_num_batched_tokens": 65536,
            "proposal_max_num_batched_tokens": 147456,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
        },
        objective=ResponseObjective(
            target="performance.completed_qps",
            direction="maximize",
            constraints=(
                ResponseConstraint("latency.p95_seconds", "<=", 120.0),
                ResponseConstraint("quality.accuracy", ">=", 0.65),
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
            uncertainty_multiplier=1.0,
            maximum_failure_probability=maximum_failure_probability,
            minimum_response_rows=3,
            minimum_distinct_configurations=3,
            minimum_paired_replicates=2,
            ranked_candidate_limit=3,
        ),
    )


def _joint_capacity_graph_space() -> DeploymentSearchSpace:
    graph48 = [1, 2, 4, 8, 16, 24, 32, 40, 48]
    graph64 = [*graph48, 64]
    return DeploymentSearchSpace.from_dict(
        {
            "schema_version": "1.0",
            "space_id": "joint-capacity-graph-test",
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
                    "description": "Proposal scheduler capacity",
                },
                {
                    "name": "proposal.graph_capture_sizes",
                    "setting_name": "proposal_graph_capture_sizes",
                    "value_type": "integer_sequence",
                    "domain": {
                        "kind": "choices",
                        "values": [graph48, graph64],
                    },
                    "change_scope": "engine_restart",
                    "semantic_effect": "preserves_algorithm",
                    "tuning_layer": "graph_capture",
                    "description": "Proposal ACL Graph buckets",
                },
            ],
            "constraints": [
                {
                    "kind": "allowed_combinations",
                    "parameters": [
                        "proposal.max_num_seqs",
                        "proposal.graph_capture_sizes",
                    ],
                    "values": [
                        [48, graph48],
                        [64, graph48],
                        [64, graph64],
                    ],
                }
            ],
        }
    )


def _joint_capacity_graph_record(
    record_id: str,
    capacity: int,
    captures: list[int],
    qps: float,
) -> EvidenceRecord:
    record = _record(record_id, 64, qps=qps, p95=50.0)
    return replace(
        record,
        configuration={
            "model_runner": "MRV1",
            "base_max_num_seqs": 40,
            "proposal_max_num_seqs": capacity,
            "base_max_num_batched_tokens": 10240,
            "proposal_max_num_batched_tokens": 12288,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
            "proposal_graph_capture_sizes": captures,
        },
    )


class PolicySelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = compile_search_space(_space())
        self.spec = _spec(self.compiled.to_dict()["compiled_space_sha256"])

    def _features(self, *extra: EvidenceRecord):
        return features_from_ledger(
            EvidenceLedger(
                (
                    _record("capacity-64", 64, qps=0.8, p95=105.0),
                    _record("capacity-128", 128, qps=1.2, p95=90.0),
                    _record("capacity-256", 256, qps=1.6, p95=140.0),
                    *extra,
                )
            )
        )

    def test_selects_conservative_best_candidate_within_slos(self) -> None:
        bundle = select_policy(self.spec, self.compiled, self._features())
        payload = bundle.to_dict()

        self.assertEqual(bundle.status, "selected")
        self.assertEqual(
            payload["selected"]["deployment_settings"]["base_max_num_seqs"],
            128,
        )
        rejected = next(
            item
            for item in payload["ranked_candidates"]
            if item["deployment_settings"]["base_max_num_seqs"] == 256
        )
        self.assertIn("constraint:latency.p95_seconds", rejected["rejection_reasons"])
        self.assertEqual(payload["activation_guard"]["on_violation"], "fallback")
        self.assertEqual(
            PolicyBundle.from_dict(payload).to_dict(),
            payload,
        )

    def test_joint_capacity_graph_policy_uses_derived_graph_features(self) -> None:
        graph48 = [1, 2, 4, 8, 16, 24, 32, 40, 48]
        graph64 = [*graph48, 64]
        compiled = compile_search_space(_joint_capacity_graph_space())
        spec = PolicySelectionSpec(
            policy_id="joint-capacity-graph-policy",
            compiled_space_sha256=compiled.to_dict()["compiled_space_sha256"],
            selection_context=SelectionContext(
                algorithm_id="conditional_is_small_proposal",
                semantic_cohort_id="conditional_is_small_proposal",
                graph_sha256=GRAPH_SHA256,
                accepted_evidence_semantic_classes=("exact_algorithm",),
                workload_id="synthetic-p128",
                environment_id=ENVIRONMENT_ID,
                static_features={"workload.requests": 96},
                static_feature_ranges={},
            ),
            query_static_features={"workload.requests": 96},
            baseline_settings={
                "proposal_max_num_seqs": 48,
                "proposal_graph_capture_sizes": graph48,
            },
            objective=ResponseObjective(
                target="performance.completed_qps",
                direction="maximize",
                constraints=(),
            ),
            model=LocalResponseModelSpec(
                feature_names=(
                    "deployment.proposal_graph_capture_ceiling",
                    "deployment.proposal_graph_uncaptured_capacity",
                    "deployment.proposal_max_num_seqs",
                ),
                neighbors=1,
                distance_power=2.0,
                maximum_normalized_distance=1.0,
                uncertainty_multiplier=0.0,
                maximum_failure_probability=1.0,
                minimum_response_rows=3,
                minimum_distinct_configurations=3,
                minimum_paired_replicates=2,
                ranked_candidate_limit=3,
            ),
        )
        features = features_from_ledger(
            EvidenceLedger(
                (
                    _joint_capacity_graph_record("graph-48-cap-48", 48, graph48, 1.0),
                    _joint_capacity_graph_record("graph-48-cap-64", 64, graph48, 0.7),
                    _joint_capacity_graph_record("graph-64-cap-64", 64, graph64, 1.2),
                )
            )
        )

        bundle = select_policy(spec, compiled, features)
        payload = bundle.to_dict()

        self.assertEqual(bundle.status, "selected")
        self.assertEqual(
            payload["selected"]["deployment_settings"][
                "proposal_graph_capture_sizes"
            ],
            graph64,
        )
        self.assertEqual(
            payload["selected"]["deployment_settings"]["proposal_max_num_seqs"],
            64,
        )
        self.assertEqual(PolicyBundle.from_dict(payload).to_dict(), payload)

    def test_exact_failure_observation_rejects_an_otherwise_best_candidate(self) -> None:
        failure = _record(
            "capacity-128-oom",
            128,
            grade=EvidenceGrade.X_EXCLUDED,
            failure="OOM",
        )
        bundle = select_policy(self.spec, self.compiled, self._features(failure))
        payload = bundle.to_dict()

        self.assertEqual(
            payload["selected"]["deployment_settings"]["base_max_num_seqs"],
            64,
        )
        rejected = next(
            item
            for item in payload["ranked_candidates"]
            if item["deployment_settings"]["base_max_num_seqs"] == 128
        )
        self.assertEqual(rejected["failure_probability"], 0.5)
        self.assertIn("failure_probability", rejected["rejection_reasons"])

    def test_replicate_noise_floor_prevents_zero_uncertainty_from_two_equal_runs(self) -> None:
        records = (
            _record("capacity-64-a", 64, qps=0.8),
            _record("capacity-64-b", 64, qps=1.2),
            _record("capacity-128-a", 128, qps=1.1),
            _record("capacity-128-b", 128, qps=1.1),
            _record("capacity-256-a", 256, qps=0.7, p95=140.0),
            _record("capacity-256-b", 256, qps=0.7, p95=140.0),
        )
        bundle = select_policy(
            self.spec,
            self.compiled,
            features_from_ledger(EvidenceLedger(records)),
        ).to_dict()
        candidate = next(
            item
            for item in bundle["ranked_candidates"]
            if item["deployment_settings"]["base_max_num_seqs"] == 128
        )

        self.assertAlmostEqual(
            bundle["response_model"]["replicate_noise_floors"][
                "performance.completed_qps"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            candidate["target_estimates"]["performance.completed_qps"][
                "uncertainty"
            ],
            0.2,
        )

    def test_formal_paired_regression_vetoes_absolute_model_promotion(self) -> None:
        records = (
            _record("old-capacity-64-a", 64, qps=0.5),
            _record("old-capacity-64-b", 64, qps=0.5),
            _paired_record(
                "pair-0-baseline",
                64,
                group_id="capacity-128-abba",
                pair_index=0,
                variant_role="baseline",
                qps=1.2,
            ),
            _paired_record(
                "pair-0-candidate",
                128,
                group_id="capacity-128-abba",
                pair_index=0,
                variant_role="candidate",
                qps=0.9,
            ),
            _paired_record(
                "pair-1-candidate",
                128,
                group_id="capacity-128-abba",
                pair_index=1,
                variant_role="candidate",
                qps=0.95,
            ),
            _paired_record(
                "pair-1-baseline",
                64,
                group_id="capacity-128-abba",
                pair_index=1,
                variant_role="baseline",
                qps=1.2,
            ),
            _record("capacity-256", 256, qps=0.7, p95=140.0),
        )
        payload = select_policy(
            self.spec,
            self.compiled,
            features_from_ledger(EvidenceLedger(records)),
        ).to_dict()
        candidate = next(
            item
            for item in payload["ranked_candidates"]
            if item["deployment_settings"]["base_max_num_seqs"] == 128
        )

        self.assertFalse(candidate["eligible"])
        self.assertIn("paired_objective_regression", candidate["rejection_reasons"])
        self.assertEqual(candidate["paired_objective_effect"]["pair_count"], 2)
        self.assertLess(
            candidate["paired_objective_effect"][
                "upper_directional_relative_improvement"
            ],
            0,
        )

    def test_low_grade_evidence_fails_closed(self) -> None:
        records = tuple(
            replace(
                record,
                quality=_quality(EvidenceGrade.B_CONTROLLED_SINGLE),
            )
            for record in (
                _record("capacity-64", 64),
                _record("capacity-128", 128),
                _record("capacity-256", 256),
            )
        )
        bundle = select_policy(
            self.spec,
            self.compiled,
            features_from_ledger(EvidenceLedger(records)),
        )

        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertIsNone(bundle.to_dict()["selected"])
        self.assertEqual(bundle.to_dict()["ranked_candidates"], [])

    def test_policy_digest_detects_tampering(self) -> None:
        payload = select_policy(
            self.spec, self.compiled, self._features()
        ).to_dict()
        payload["selected"]["deployment_settings"]["base_max_num_seqs"] = 256

        with self.assertRaisesRegex(ValueError, "SHA256"):
            PolicyBundle.from_dict(payload)

    def test_rehashed_guard_cannot_diverge_from_selection_contract(self) -> None:
        payload = select_policy(
            self.spec, self.compiled, self._features()
        ).to_dict()
        payload["activation_guard"]["maximum_failure_probability"] = 1.0
        unsigned = dict(payload)
        unsigned.pop("policy_bundle_sha256")
        payload["policy_bundle_sha256"] = canonical_sha256(unsigned)

        with self.assertRaisesRegex(ValueError, "activation thresholds"):
            PolicyBundle.from_dict(payload)

    def test_cli_emits_and_audits_policy_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            compiled_path = root / "compiled.json"
            features_path = root / "features.json"
            policy_path = root / "policy.json"
            spec_path.write_text(json.dumps(self.spec.to_dict()), encoding="utf-8")
            compiled_path.write_text(
                json.dumps(self.compiled.to_dict()), encoding="utf-8"
            )
            features_path.write_text(
                json.dumps(self._features().to_dict()), encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "select-policy",
                        str(spec_path),
                        str(compiled_path),
                        str(features_path),
                        "--output",
                        str(policy_path),
                    ]
                )
                audit_status = main(["audit-policy", str(policy_path)])
            policy = PolicyBundle.from_dict(
                json.loads(policy_path.read_text(encoding="utf-8"))
            )

        self.assertEqual(status, 0)
        self.assertEqual(audit_status, 0)
        self.assertEqual(policy.status, "selected")

    def test_independent_context_holdout_beats_baseline_and_matches_oracle(self) -> None:
        training = features_from_ledger(
            EvidenceLedger(
                (
                    _record("train-p64-c64", 64, prompt_tokens=64, qps=0.9),
                    _record("train-p64-c128", 128, prompt_tokens=64, qps=1.2),
                    _record(
                        "train-p64-c256",
                        256,
                        prompt_tokens=64,
                        qps=1.3,
                        p95=135.0,
                    ),
                    _record("train-p256-c64", 64, prompt_tokens=256, qps=0.6),
                    _record("train-p256-c128", 128, prompt_tokens=256, qps=1.0),
                    _record("train-p256-c256", 256, prompt_tokens=256, qps=0.8),
                )
            )
        )
        policy = select_policy(self.spec, self.compiled, training)
        self.assertEqual(
            policy.to_dict()["selected"]["deployment_settings"]["base_max_num_seqs"],
            128,
        )
        holdout_records = []
        for replicate in (1, 2):
            holdout_records.extend(
                (
                    _record(f"holdout-r{replicate}-c64", 64, qps=0.8),
                    _record(f"holdout-r{replicate}-c128", 128, qps=1.15),
                    _record(f"holdout-r{replicate}-c256", 256, qps=1.05),
                )
            )
        holdout = features_from_ledger(EvidenceLedger(tuple(holdout_records)))
        holdout_spec = PolicyHoldoutSpec(
            assessment_id="synthetic-p128-holdout",
            policy_bundle_sha256=policy.to_dict()["policy_bundle_sha256"],
            minimum_successful_replicates=2,
            maximum_regret_fraction=0.05,
            minimum_improvement_over_fallback_fraction=0.1,
        )
        assessment = assess_policy_holdout(
            holdout_spec, policy, self.compiled, holdout
        )
        payload = assessment.to_dict()

        self.assertEqual(assessment.status, "validated")
        self.assertEqual(payload["comparison"]["regret_to_measured_oracle_fraction"], 0.0)
        self.assertGreater(
            payload["comparison"]["improvement_over_fallback_fraction"], 0.4
        )
        self.assertEqual(
            PolicyHoldoutAssessment.from_dict(payload).to_dict(), payload
        )

        leaked = features_from_ledger(
            EvidenceLedger(
                (
                    _record("train-p64-c64", 64, qps=0.8),
                    *tuple(holdout_records),
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "leaks training rows"):
            assess_policy_holdout(
                holdout_spec, policy, self.compiled, leaked
            )


if __name__ == "__main__":
    unittest.main()
