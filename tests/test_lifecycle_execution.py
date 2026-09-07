from __future__ import annotations

import copy
import hashlib
import unittest

from inference_autopilot.calibration.models import canonical_sha256
from inference_autopilot.lifecycle_execution import (
    EpochMeasurement,
    LifecycleExecutionReceipt,
    audit_lifecycle_execution,
    execute_lifecycle_plan,
)
from inference_autopilot.lifecycle_planning import (
    EngineLifecyclePlan,
    EngineLifecyclePlanningSpec,
    _build_payload,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _plan() -> EngineLifecyclePlan:
    spec = EngineLifecyclePlanningSpec(
        lifecycle_id="execution-test",
        maximum_resident_memory_fraction=0.95,
        preserve_run_order=True,
        required_epoch_resets=(
            "assert_no_running_requests",
            "rebuild_continuous_batchers",
            "rebuild_score_caches",
            "reseed_workload",
            "reset_backend_metric_snapshots",
            "reset_prefix_cache",
            "reset_request_ids",
            "synchronize_device",
        ),
        validation_pair_seeds=(611, 612),
    )
    runs = []
    for index, proposal in enumerate(("b" * 64, "c" * 64, "b" * 64)):
        runs.append(
            {
                "run_id": f"run-{index}",
                "sequence_index": index,
                "configuration_id": "candidate" if index == 1 else "baseline",
                "workload_seed": 700 + index,
                "engines": [
                    {
                        "role": "base",
                        "engine_fingerprint_sha256": "a" * 64,
                        "engine_init_seconds": 50.0,
                        "reserved_memory_fraction": 0.54,
                    },
                    {
                        "role": "proposal",
                        "engine_fingerprint_sha256": proposal,
                        "engine_init_seconds": 30.0,
                        "reserved_memory_fraction": 0.36,
                    },
                ],
            }
        )
    return EngineLifecyclePlan(_build_payload(spec, runs, "d" * 64, "e" * 64))


class _Driver:
    def __init__(self, *, omit_reset: str | None = None) -> None:
        self.omit_reset = omit_reset
        self.starts: list[tuple[str, str]] = []
        self.stops: list[tuple[str, str]] = []
        self.resets: list[str] = []
        self.measurements: list[str] = []

    def start_engine(self, role: str, fingerprint: str) -> None:
        self.starts.append((role, fingerprint))

    def stop_engine(self, role: str, fingerprint: str) -> None:
        self.stops.append((role, fingerprint))

    def reset_epoch(self, run_id, workload_seed, requirements, bindings):
        self.resets.append(run_id)
        return {
            requirement: True
            for requirement in requirements
            if requirement != self.omit_reset
        }

    def measure_epoch(self, run_id, workload_seed, bindings):
        self.measurements.append(run_id)
        return EpochMeasurement(
            native_result_sha256=_digest(f"native:{run_id}"),
            observation_sha256=_digest(f"observation:{run_id}"),
            output_token_ids_sha256=_digest(f"tokens:{run_id}:{workload_seed}"),
        )


class LifecycleExecutionTests(unittest.TestCase):
    def test_role_sticky_state_machine_executes_and_audits(self) -> None:
        plan = _plan()
        driver = _Driver()
        receipt = execute_lifecycle_plan(plan, driver, validation_only=True)
        audit = audit_lifecycle_execution(plan, receipt)

        self.assertEqual(len(driver.starts), 4)
        self.assertEqual(len(driver.stops), 4)
        self.assertEqual(driver.resets, ["run-0", "run-1", "run-2"])
        self.assertEqual(driver.measurements, ["run-0", "run-1", "run-2"])
        self.assertEqual(audit["engine_start_count"], 4)
        self.assertEqual(audit["measurement_count"], 3)
        self.assertTrue(audit["all_resets_acknowledged"])
        self.assertFalse(audit["formal_execution_eligible"])

    def test_missing_reset_acknowledgement_fails_before_measurement(self) -> None:
        driver = _Driver(omit_reset="reset_prefix_cache")
        with self.assertRaisesRegex(ValueError, "do not match requirements"):
            execute_lifecycle_plan(_plan(), driver, validation_only=True)

        self.assertEqual(driver.measurements, [])
        self.assertCountEqual(driver.starts, driver.stops)

    def test_explicit_validation_mode_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit validation mode"):
            execute_lifecycle_plan(_plan(), _Driver(), validation_only=False)

    def test_rehashed_summary_tampering_is_rejected(self) -> None:
        payload = execute_lifecycle_plan(
            _plan(), _Driver(), validation_only=True
        ).to_dict()
        tampered = copy.deepcopy(payload)
        tampered["summary"]["engine_start_count"] = 3
        unhashed = dict(tampered)
        unhashed.pop("lifecycle_execution_receipt_sha256")
        tampered["lifecycle_execution_receipt_sha256"] = canonical_sha256(unhashed)

        with self.assertRaisesRegex(ValueError, "summary does not match"):
            LifecycleExecutionReceipt.from_dict(tampered)

    def test_rehashed_action_tampering_is_rejected_against_plan(self) -> None:
        plan = _plan()
        payload = execute_lifecycle_plan(
            plan, _Driver(), validation_only=True
        ).to_dict()
        tampered = copy.deepcopy(payload)
        tampered["epochs"][1]["action_receipts"][0]["action"] = (
            "bind_resident_engine"
        )
        unhashed = dict(tampered)
        unhashed.pop("lifecycle_execution_receipt_sha256")
        tampered["lifecycle_execution_receipt_sha256"] = canonical_sha256(unhashed)
        receipt = LifecycleExecutionReceipt.from_dict(tampered)

        with self.assertRaisesRegex(ValueError, "differ from the frozen schedule"):
            audit_lifecycle_execution(plan, receipt)


if __name__ == "__main__":
    unittest.main()
