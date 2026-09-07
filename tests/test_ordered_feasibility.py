from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.cli import main
from inference_autopilot.evidence import SourceArtifact
from inference_autopilot.ordered_feasibility import (
    OrderedFeasibilityPlan,
    OrderedFeasibilitySpec,
    OrderedProbeObservation,
    plan_ordered_feasibility,
)


_VALUES = (1024, 2048, 4096, 6144, 8192, 10240)


def _observation(attempt_id: str, value: int, outcome: str):
    return OrderedProbeObservation(
        attempt_id=attempt_id,
        value=value,
        outcome=outcome,
        source=SourceArtifact(
            path=f"{attempt_id}.json",
            sha256=(f"{value:064x}"[-64:]),
            imported_format="ordered_capacity_probe",
            locator=attempt_id,
        ),
        details={"prompt_tokens_mean": 8311.5},
    )


def _spec(*observations, confirmations: int = 1, values=_VALUES):
    return OrderedFeasibilitySpec(
        probe_id="long8k-base-token-budget",
        parameter_name="base.max_num_batched_tokens",
        ordered_values=values,
        context={
            "algorithm_id": "conditional_is_small_proposal",
            "environment_id": "ascend-910b3-vllm-0.18",
            "prompt_prefix_tokens": 8192,
            "requests": 16,
        },
        required_success_observations=confirmations,
        required_resource_exhausted_observations=confirmations,
        observations=tuple(observations),
    )


class OrderedFeasibilityTest(unittest.TestCase):
    def test_brackets_resource_boundary_with_midpoint_probes(self) -> None:
        first = plan_ordered_feasibility(
            _spec(
                _observation("success-2048", 2048, "success"),
                _observation("oom-10240", 10240, "resource_exhausted"),
            )
        )
        second = plan_ordered_feasibility(
            _spec(
                *first.observations,
                _observation("oom-6144", 6144, "resource_exhausted"),
            )
        )
        complete = plan_ordered_feasibility(
            _spec(
                *second.observations,
                _observation("success-4096", 4096, "success"),
            )
        )

        self.assertEqual(first.next_probe_value, 6144)
        self.assertEqual(first.provisional_recommendation, 2048)
        self.assertEqual(first.inferred_resource_exhausted_values, ())
        self.assertEqual(second.next_probe_value, 4096)
        self.assertTrue(complete.search_complete)
        self.assertIsNone(complete.next_probe_value)
        self.assertEqual(complete.provisional_recommendation, 4096)
        self.assertEqual(complete.inferred_resource_exhausted_values, (8192,))

    def test_requires_confirmation_before_moving_boundary(self) -> None:
        plan = plan_ordered_feasibility(
            _spec(
                _observation("success-2048-a", 2048, "success"),
                _observation("oom-10240-a", 10240, "resource_exhausted"),
                confirmations=2,
            )
        )

        self.assertTrue(plan.eligible)
        self.assertEqual(plan.next_probe_value, 2048)
        self.assertIsNone(plan.provisional_recommendation)

    def test_confirmation_ignores_points_dominated_by_confirmed_bounds(self) -> None:
        values = (1024, 2048, 3072, 3584, 4096, 6144, 10240)
        observations = (
            _observation("success-2048-a", 2048, "success"),
            _observation("success-3072-a", 3072, "success"),
            _observation("oom-3584-a", 3584, "resource_exhausted"),
            _observation("oom-4096-a", 4096, "resource_exhausted"),
        )
        confirm_success = plan_ordered_feasibility(
            _spec(*observations, confirmations=2, values=values)
        )
        confirm_oom = plan_ordered_feasibility(
            _spec(
                *observations,
                _observation("success-3072-b", 3072, "success"),
                confirmations=2,
                values=values,
            )
        )
        complete = plan_ordered_feasibility(
            _spec(
                *confirm_oom.observations,
                _observation("oom-3584-b", 3584, "resource_exhausted"),
                confirmations=2,
                values=values,
            )
        )

        self.assertEqual(confirm_success.next_probe_value, 3072)
        self.assertEqual(confirm_oom.next_probe_value, 3584)
        self.assertTrue(complete.search_complete)
        self.assertIsNone(complete.next_probe_value)
        self.assertEqual(complete.provisional_recommendation, 3072)

    def test_rejects_non_monotone_or_mixed_evidence(self) -> None:
        non_monotone = plan_ordered_feasibility(
            _spec(
                _observation("oom-4096", 4096, "resource_exhausted"),
                _observation("success-6144", 6144, "success"),
            )
        )
        mixed = plan_ordered_feasibility(
            _spec(
                _observation("success-2048", 2048, "success"),
                _observation("oom-2048", 2048, "resource_exhausted"),
            )
        )

        self.assertFalse(non_monotone.eligible)
        self.assertEqual(
            non_monotone.rejection_reasons,
            ("non_monotone_resource_boundary",),
        )
        self.assertIsNone(non_monotone.next_probe_value)
        self.assertFalse(mixed.eligible)
        self.assertEqual(mixed.rejection_reasons, ("mixed_outcomes_at_value",))

    def test_round_trip_digest_and_cli(self) -> None:
        spec = _spec(
            _observation("success-2048", 2048, "success"),
            _observation("oom-10240", 10240, "resource_exhausted"),
        )
        plan = plan_ordered_feasibility(spec)
        self.assertEqual(OrderedFeasibilityPlan.from_dict(plan.to_dict()), plan)

        tampered = plan.to_dict()
        tampered["next_probe_value"] = 4096
        with self.assertRaisesRegex(ValueError, "SHA256"):
            OrderedFeasibilityPlan.from_dict(tampered)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            plan_path = root / "plan.json"
            spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "plan-ordered-feasibility",
                        str(spec_path),
                        "--output",
                        str(plan_path),
                    ]
                )
                audit_status = main(
                    ["audit-ordered-feasibility", str(plan_path)]
                )
            payload = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(audit_status, 0)
        self.assertEqual(payload["next_probe_value"], 6144)


if __name__ == "__main__":
    unittest.main()
