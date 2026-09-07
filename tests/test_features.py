from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.calibration.models import (
    ArtifactReference,
    CalibrationSpec,
    RunObservation,
)
from inference_autopilot.calibration.planning import build_plan, build_run_manifest
from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    EvidenceRecord,
    QualityAssessment,
    SourceArtifact,
)
from inference_autopilot.features import (
    SelectorFeatureTable,
    features_from_ledger,
    features_from_record,
    features_from_run,
)


def _quality(grade: EvidenceGrade) -> QualityAssessment:
    purpose = {
        EvidenceGrade.A_FORMAL_PAIRED: EvidencePurpose.SELECTOR_FIT_AND_CLAIM,
        EvidenceGrade.B_CONTROLLED_SINGLE: EvidencePurpose.CALIBRATION_ONLY,
        EvidenceGrade.C_DIAGNOSTIC: EvidencePurpose.DIAGNOSTIC_ONLY,
        EvidenceGrade.X_EXCLUDED: EvidencePurpose.CONSTRAINT_ONLY,
    }[grade]
    return QualityAssessment(
        grade=grade,
        purpose=purpose,
        claim_eligible=grade == EvidenceGrade.A_FORMAL_PAIRED,
        reasons=("test evidence",),
    )


def _record(grade: EvidenceGrade = EvidenceGrade.A_FORMAL_PAIRED) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=f"record-{grade.value}",
        campaign="campaign",
        variant="config",
        source=SourceArtifact("run.json", "a" * 64, "test", "run"),
        workload={
            "workload_id": "gsm8k-short",
            "method": "conditional_is_small_proposal",
            "requests": 96,
            "workers": 96,
            "arrival_qps": 0.0,
            "prompt_length_regime": "short",
        },
        configuration={
            "base_max_num_seqs": 256,
            "proposal_max_num_seqs": 896,
            "base_max_num_batched_tokens": 65536,
            "proposal_max_num_batched_tokens": 147456,
            "base_memory_fraction": 0.54,
            "proposal_memory_fraction": 0.36,
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128, 256],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 128, 512],
        },
        algorithm={
            "algorithm_id": "conditional_is_small_proposal",
            "semantic_class": "exact",
            "graph_sha256": "b" * 64,
            "candidate_count": 15,
            "rollout_count": 3,
            "block_size": 128,
            "total_length": 512,
            "apply_importance_correction": True,
        },
        environment={"environment_id": "ascend-a2", "hardware": {"device_count": 1}},
        metrics={
            "completed_qps": 0.8,
            "elapsed_seconds": 120.0,
            "latency_seconds": {"p50": 80.0, "p95": 100.0, "p99": 110.0},
            "accuracy": 0.5,
            "preemptions": 0,
            "proposal_kv_peak_fraction": 0.16,
            "workload": {
                "prompt_tokens": {"minimum": 80, "mean": 100.0, "maximum": 120}
            },
            "continuous_batching": {
                "base": {"sample_batches": 8, "maximum_sample_batch": 15}
            },
            "vllm_runtime_metrics": {
                "proposal": {
                    "vllm:num_requests_waiting": {"mean": 2.0, "maximum": 8.0}
                }
            },
            "compute": {"total_forward_token_slots": 50000},
        },
        quality=_quality(grade),
        tags=("test",),
    )


class SelectorFeatureTest(unittest.TestCase):
    def test_exact_graph_demand_and_telemetry_are_separated_from_targets(self) -> None:
        row = features_from_record(_record())

        self.assertEqual(row.static_features["graph.guidance_steps_upper"], 4)
        self.assertEqual(
            row.static_features["graph.candidate_sequences_per_request_upper"], 60
        )
        self.assertEqual(
            row.static_features["graph.proposal_sequences_per_request_upper"], 180
        )
        self.assertEqual(row.static_features["graph.parallel_width_upper"], 45)
        self.assertEqual(
            row.static_features["graph.candidate_token_slots_per_request_upper"],
            7680,
        )
        self.assertEqual(
            row.static_features["graph.proposal_token_slots_per_request_upper"],
            34560,
        )
        self.assertEqual(
            row.static_features["graph.target_score_token_slots_per_request_upper"],
            34560,
        )
        self.assertEqual(
            row.static_features["deployment.base_graph_capture_ceiling"], 256
        )
        self.assertEqual(
            row.static_features["deployment.proposal_graph_capture_bucket_count"], 3
        )
        self.assertAlmostEqual(
            row.static_features[
                "deployment.proposal_graph_capacity_coverage_ratio"
            ],
            512 / 896,
        )
        self.assertEqual(
            row.static_features["deployment.proposal_graph_uncaptured_capacity"],
            384,
        )
        self.assertFalse(
            row.static_features[
                "deployment.proposal_graph_covers_scheduler_capacity"
            ]
        )
        self.assertEqual(row.targets["performance.completed_qps"], 0.8)
        self.assertNotIn("performance.completed_qps", row.static_features)
        self.assertEqual(
            row.telemetry_features["telemetry.batching.base.sample_batches"], 8
        )
        self.assertTrue(row.eligibility.response_model_fit)
        self.assertTrue(row.eligibility.performance_claim)
        self.assertFalse(row.missing["selector_requirements"])

    def test_evidence_grade_controls_training_role(self) -> None:
        grade_b = features_from_record(_record(EvidenceGrade.B_CONTROLLED_SINGLE))
        grade_c = features_from_record(_record(EvidenceGrade.C_DIAGNOSTIC))
        failure = replace(
            _record(EvidenceGrade.X_EXCLUDED),
            workload={
                "method": "conditional_is_small_proposal",
                "requests": 96,
                "prompt_tokens_approx": 8400,
            },
            metrics={"failure": "OOM"},
        )
        grade_x = features_from_record(failure)

        self.assertTrue(grade_b.eligibility.prior_construction)
        self.assertFalse(grade_b.eligibility.response_model_fit)
        self.assertTrue(grade_c.eligibility.diagnostic_analysis)
        self.assertTrue(grade_x.eligibility.feasibility_model)
        self.assertIn(
            "workload.load_descriptor", grade_x.missing["prior_requirements"]
        )
        self.assertFalse(grade_x.missing["feasibility_requirements"])
        self.assertFalse(grade_x.targets["run.success"])

    def test_incomplete_context_blocks_fit_without_revoking_grade_a_claim(self) -> None:
        record = replace(
            _record(),
            workload={
                "method": "conditional_is_small_proposal",
                "requests": 96,
                "workers": 96,
            },
            metrics={
                "completed_qps": 0.8,
                "latency_seconds": {"p95": 100.0},
            },
        )
        row = features_from_record(record)

        self.assertFalse(row.eligibility.response_model_fit)
        self.assertTrue(row.eligibility.performance_claim)
        self.assertIn(
            "workload.context_descriptor", row.missing["selector_requirements"]
        )

    def test_legacy_kv_percent_is_normalized(self) -> None:
        record = replace(
            _record(EvidenceGrade.B_CONTROLLED_SINGLE),
            metrics={
                "completed_qps": 0.8,
                "p95_seconds": 10.0,
                "proposal_kv_peak_percent": 25.0,
            },
        )
        row = features_from_record(record)
        self.assertEqual(row.targets["resource.proposal_kv_peak_fraction"], 0.25)

    def test_table_round_trip_checks_derived_catalog_and_audit(self) -> None:
        table = features_from_ledger(EvidenceLedger((_record(),)))
        payload = table.to_dict()
        self.assertEqual(SelectorFeatureTable.from_dict(payload), table)

        tampered_role = table.to_dict()
        tampered_role["rows"][0]["eligibility"]["prior_construction"] = True
        with self.assertRaisesRegex(ValueError, "eligibility does not match"):
            SelectorFeatureTable.from_dict(tampered_role)

        payload["audit"]["row_count"] = 2
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            SelectorFeatureTable.from_dict(payload)

    def test_manifest_observation_stays_ungraded_until_assessment(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = CalibrationSpec.from_dict(
            json.loads(
                (root / "examples/conditional-is-short-p96.calibration.example.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        plan = build_plan(spec)
        manifest = build_run_manifest(plan, plan.runs[0].run_id)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "native.json"
            artifact.write_text("{}", encoding="utf-8")
            observation = RunObservation(
                run_id=plan.runs[0].run_id,
                run_manifest_sha256=manifest["run_manifest_sha256"],
                started_at_unix=1.0,
                finished_at_unix=2.0,
                status="success",
                metrics={
                    "completed_qps": 0.8,
                    "latency_seconds": {"p95": 100.0},
                    "workload": {"prompt_tokens": {"mean": 96.0}},
                },
                artifact=ArtifactReference(str(artifact), "c" * 64),
            )
            row = features_from_run(manifest, observation).rows[0]

        self.assertEqual(row.evidence["grade"], "ungraded")
        self.assertTrue(row.eligibility.diagnostic_analysis)
        self.assertFalse(row.eligibility.response_model_fit)
        self.assertEqual(row.static_features["workload.prompt_tokens.mean"], 96.0)
        self.assertEqual(row.static_features["graph.parallel_width_upper"], 45)

        tampered = dict(manifest)
        tampered["required_metrics"] = []
        with self.assertRaisesRegex(ValueError, "SHA256"):
            features_from_run(tampered, observation)


if __name__ == "__main__":
    unittest.main()
