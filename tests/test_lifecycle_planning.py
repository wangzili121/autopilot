from __future__ import annotations

import copy
import unittest

from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.lifecycle_planning import (
    EngineLifecyclePlan,
    EngineLifecyclePlanningSpec,
    _build_payload,
)


def _spec(**overrides: object) -> EngineLifecyclePlanningSpec:
    values = {
        "lifecycle_id": "medium2k-role-sticky-v1",
        "maximum_resident_memory_fraction": 0.95,
        "preserve_run_order": True,
        "required_epoch_resets": (
            "assert_no_running_requests",
            "rebuild_continuous_batchers",
            "rebuild_score_caches",
            "reseed_workload",
            "reset_backend_metric_snapshots",
            "reset_prefix_cache",
            "reset_request_ids",
            "synchronize_device",
        ),
        "validation_pair_seeds": (611, 612),
    }
    values.update(overrides)
    return EngineLifecyclePlanningSpec(**values)  # type: ignore[arg-type]


def _runs() -> list[dict[str, object]]:
    result = []
    for index in range(8):
        proposal = "c" * 64 if index in (5, 6) else "b" * 64
        result.append(
            {
                "run_id": f"run-{index}",
                "sequence_index": index,
                "configuration_id": "candidate" if index in (5, 6) else "baseline",
                "workload_seed": 600 + index,
                "engines": [
                    {
                        "role": "base",
                        "engine_fingerprint_sha256": "a" * 64,
                        "engine_init_seconds": 57.0,
                        "reserved_memory_fraction": 0.54,
                    },
                    {
                        "role": "proposal",
                        "engine_fingerprint_sha256": proposal,
                        "engine_init_seconds": 31.0,
                        "reserved_memory_fraction": 0.36,
                    },
                ],
            }
        )
    return result


class EngineLifecyclePlanningTests(unittest.TestCase):
    def _payload(self, spec: EngineLifecyclePlanningSpec | None = None) -> dict:
        return _build_payload(spec or _spec(), _runs(), "d" * 64, "e" * 64)

    def test_role_sticky_preserves_order_and_reduces_engine_starts(self) -> None:
        payload = self._payload()

        self.assertEqual(payload["selected_strategy"], "role_sticky")
        self.assertEqual(payload["audit"]["measured_engine_start_count"], 16)
        self.assertEqual(payload["audit"]["planned_engine_start_count"], 4)
        self.assertEqual(payload["audit"]["projected_startup_seconds"], 150.0)
        self.assertEqual(payload["audit"]["projected_startup_savings_seconds"], 554.0)
        self.assertFalse(payload["audit"]["formal_execution_eligible"])
        self.assertEqual(
            [action["sequence_index"] for action in payload["actions"]],
            list(range(8)),
        )
        self.assertTrue(
            all(
                action["actions_before_measurement"][-1]["action"]
                == "reset_epoch_state"
                for action in payload["actions"]
            )
        )

    def test_fully_resident_is_blocked_by_measured_memory(self) -> None:
        payload = self._payload()
        strategies = {row["strategy"]: row for row in payload["strategies"]}

        self.assertEqual(
            strategies["isolated_process"]["projected_startup_seconds"], 704.0
        )
        self.assertEqual(
            strategies["isolated_process"]["projected_startup_savings_seconds"],
            0.0,
        )
        self.assertEqual(
            strategies["fully_resident"]["peak_reserved_memory_fraction"], 1.26
        )
        self.assertFalse(strategies["fully_resident"]["eligible"])
        self.assertEqual(
            strategies["fully_resident"]["blockers"],
            ["resident_memory_fraction_exceeds_limit"],
        )

    def test_rehashed_derived_schedule_tampering_is_rejected(self) -> None:
        payload = self._payload()
        tampered = copy.deepcopy(payload)
        tampered["audit"]["planned_engine_start_count"] = 3
        unhashed = dict(tampered)
        unhashed.pop("engine_lifecycle_plan_sha256")
        tampered["engine_lifecycle_plan_sha256"] = canonical_sha256(unhashed)

        with self.assertRaisesRegex(ValueError, "does not match derived schedule"):
            EngineLifecyclePlan.from_dict(tampered)

    def test_reset_contract_cannot_omit_stateful_subsystems(self) -> None:
        resets = tuple(
            value for value in _spec().required_epoch_resets if value != "reset_prefix_cache"
        )
        with self.assertRaisesRegex(ValueError, "reset_prefix_cache"):
            _spec(required_epoch_resets=resets)


if __name__ == "__main__":
    unittest.main()
