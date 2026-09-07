from __future__ import annotations

import unittest

from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.execution_readiness import (
    ExecutionReadinessAssessment,
    ExecutionReadinessSpec,
    _derive_summary_and_decision,
)


def _spec() -> ExecutionReadinessSpec:
    return ExecutionReadinessSpec(
        readiness_id="medium2k-npu2-readiness",
        minimum_attempt_count=5,
        maximum_environment_retries=2,
        wilson_z=1.6448536269514722,
        minimum_clean_probability_lower_bound=0.5,
        minimum_campaign_completion_probability=0.8,
        maximum_expected_npu_hours=2.0,
        duration_safety_factor=1.15,
    )


def _attempt(index: int, outcome: str, duration: float = 300.0) -> dict[str, object]:
    return {
        "campaign_id": "history-a",
        "logical_run_id": f"run-{index:03d}",
        "attempt_index": 0,
        "outcome": outcome,
        "duration_seconds": duration,
        "record_sha256": f"{index + 1:064x}",
    }


def _assessment_payload(
    spec: ExecutionReadinessSpec, attempts: list[dict[str, object]]
) -> dict[str, object]:
    summary, decision, status = _derive_summary_and_decision(
        spec, attempts, planned_run_count=4
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "readiness_id": spec.readiness_id,
        "status": status,
        "requirements": spec.to_dict(),
        "source": {
            "requirements_sha256": spec.sha256,
            "target_plan_file_sha256": "a" * 64,
            "target_plan_sha256": "b" * 64,
            "target_context_sha256": "c" * 64,
            "history_campaigns": [
                {
                    "campaign_id": "history-a",
                    "campaign_plan_file_sha256": "d" * 64,
                    "attempt_ledger_file_sha256": "e" * 64,
                    "attempt_ledger_final_record_sha256": attempts[-1][
                        "record_sha256"
                    ],
                    "attempt_ledger_status": "complete",
                    "attempt_count": len(attempts),
                }
            ],
        },
        "attempts": attempts,
        "summary": summary,
        "decision": decision,
    }
    return {
        **payload,
        "execution_readiness_assessment_sha256": canonical_sha256(payload),
    }


class ExecutionReadinessTest(unittest.TestCase):
    def test_low_clean_rate_defers_an_expensive_campaign(self) -> None:
        attempts = [
            _attempt(0, "accepted", 310),
            _attempt(1, "accepted", 320),
            _attempt(2, "environment_contaminated", 330),
            _attempt(3, "environment_contaminated", 340),
            _attempt(4, "environment_contaminated", 350),
        ]

        assessment = ExecutionReadinessAssessment(
            _assessment_payload(_spec(), attempts)
        )

        self.assertEqual(assessment.status, "defer")
        self.assertIn(
            "clean_probability_below_threshold",
            assessment.payload["decision"]["reasons"],
        )
        self.assertIn(
            "campaign_completion_probability_below_threshold",
            assessment.payload["decision"]["reasons"],
        )
        self.assertLess(
            assessment.payload["summary"][
                "campaign_completion_probability_lower_bound"
            ],
            0.1,
        )

    def test_clean_history_launches_within_npu_budget(self) -> None:
        attempts = [_attempt(index, "accepted") for index in range(20)]

        assessment = ExecutionReadinessAssessment(
            _assessment_payload(_spec(), attempts)
        )

        self.assertEqual(assessment.status, "launch")
        self.assertEqual(assessment.payload["decision"]["reasons"], [])
        self.assertGreater(
            assessment.payload["summary"][
                "campaign_completion_probability_lower_bound"
            ],
            0.8,
        )
        self.assertLess(
            assessment.payload["summary"][
                "expected_npu_hours_conservative_estimate"
            ],
            2.0,
        )

    def test_rehashed_summary_tampering_is_rejected(self) -> None:
        attempts = [_attempt(index, "accepted") for index in range(20)]
        raw = _assessment_payload(_spec(), attempts)
        raw["summary"]["accepted_attempt_count"] = 2
        payload = dict(raw)
        payload.pop("execution_readiness_assessment_sha256")
        raw["execution_readiness_assessment_sha256"] = canonical_sha256(payload)

        with self.assertRaisesRegex(ValueError, "summary does not match"):
            ExecutionReadinessAssessment(raw)

    def test_non_environment_failure_blocks_launch(self) -> None:
        spec = _spec()
        attempts = [_attempt(index, "accepted") for index in range(20)]
        attempts[-1] = _attempt(19, "run_failed")

        assessment = ExecutionReadinessAssessment(
            _assessment_payload(spec, attempts)
        )

        self.assertEqual(assessment.status, "defer")
        self.assertIn(
            "non_environment_failures_observed",
            assessment.payload["decision"]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
