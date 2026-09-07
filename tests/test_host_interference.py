from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference_autopilot.host_interference import (
    CampaignInterferencePolicy,
    HostInterferencePolicy,
    assess_campaign_host_reports,
    assess_host_telemetry,
    assess_host_telemetry_file,
    capture_window,
    parse_npu_smi_info,
)


NPU_SMI_OUTPUT = """
+------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 118.7       40                0    / 0             |
| 0                         | 0000:C1:00.0  | 24          0    / 0          61105/ 65536         |
+===========================+===============+====================================================+
| 2     910B3               | OK            | 89.1        35                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          3409 / 65536         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 1572711       | VLLMWorker_TP            | 57746                   |
+===========================+===============+====================================================+
| No running processes found in NPU 2                                                            |
"""


def _sample(
    timestamp: float,
    *,
    total: int,
    idle: int,
    sibling_pid: int = 100,
    npu_status: str = "success",
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "captured_at_unix": timestamp,
        "capture_duration_seconds": 0.1,
        "cpu": {
            "total_jiffies": total,
            "idle_jiffies": idle,
            "iowait_jiffies": 0,
            "logical_cpu_count": 8,
        },
        "load": {
            "load1": 1.0,
            "load5": 1.0,
            "load15": 1.0,
            "runnable_processes": 2,
            "total_processes": 100,
        },
        "memory": {
            "total_kib": 1000,
            "available_kib": 800,
            "swap_total_kib": 0,
            "swap_free_kib": 0,
        },
        "npu": {
            "status": npu_status,
            "raw_sha256": "a" * 64,
            "devices": [
                {
                    "physical_device_id": 0,
                    "aicore_percent": 25.0,
                    "processes": [
                        {"pid": sibling_pid, "name": "worker", "memory_mb": 100}
                    ],
                },
                {
                    "physical_device_id": 2,
                    "aicore_percent": 90.0,
                    "temperature_c": 40.0,
                    "power_watts": 90.0,
                    "hbm_used_mb": 3400,
                    "processes": [
                        {"pid": 200 + int(timestamp), "name": "autopilot", "memory_mb": 500}
                    ],
                },
            ],
        },
    }


class HostInterferenceTest(unittest.TestCase):
    def test_parses_device_metrics_and_processes(self) -> None:
        devices = parse_npu_smi_info(NPU_SMI_OUTPUT)

        self.assertEqual([device["physical_device_id"] for device in devices], [0, 2])
        self.assertEqual(devices[0]["health"], "OK")
        self.assertEqual(devices[0]["aicore_percent"], 24.0)
        self.assertEqual(devices[0]["hbm_used_mb"], 61105)
        self.assertEqual(devices[0]["process_memory_mb"], 57746)
        self.assertEqual(devices[1]["processes"], [])

    def test_target_process_churn_is_not_external_interference(self) -> None:
        samples = [
            _sample(0.0, total=100, idle=80),
            _sample(10.0, total=200, idle=160),
            _sample(20.0, total=300, idle=240),
        ]

        report = assess_host_telemetry(
            samples,
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="b" * 64,
        )

        self.assertEqual(report["status"], "clean")
        self.assertEqual(report["hard_findings"], [])
        self.assertEqual(report["warnings"][0]["code"], "sibling_npu_active")
        self.assertEqual(len(report["report_sha256"]), 64)

    def test_admission_policy_requires_an_idle_target(self) -> None:
        report = assess_host_telemetry(
            [
                _sample(0.0, total=100, idle=80),
                _sample(10.0, total=200, idle=160),
                _sample(20.0, total=300, idle=240),
            ],
            target_npu_id=2,
            policy=HostInterferencePolicy(require_target_idle=True),
            telemetry_sha256="b" * 64,
        )

        self.assertEqual(report["status"], "contaminated")
        self.assertIn(
            "target_npu_busy",
            {finding["code"] for finding in report["hard_findings"]},
        )

    def test_bounded_window_captures_both_endpoints(self) -> None:
        now = [0.0]

        def monotonic() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        def capture(_path: Path) -> dict[str, object]:
            return _sample(
                now[0],
                total=100 + int(now[0] * 10),
                idle=80 + int(now[0] * 8),
            )

        with tempfile.TemporaryDirectory() as directory:
            telemetry = Path(directory) / "window.jsonl"
            capture_window(
                telemetry,
                duration_seconds=20.0,
                interval_seconds=10.0,
                npu_smi_path=Path("/unused"),
                capture=capture,
                monotonic=monotonic,
                sleep=sleep,
            )
            samples = [json.loads(line) for line in telemetry.read_text().splitlines()]

        self.assertEqual([sample["captured_at_unix"] for sample in samples], [0.0, 10.0, 20.0])

    def test_sibling_process_churn_contaminates_run(self) -> None:
        samples = [
            _sample(0.0, total=100, idle=80),
            _sample(10.0, total=200, idle=160, sibling_pid=101),
            _sample(20.0, total=300, idle=240, sibling_pid=101),
        ]

        report = assess_host_telemetry(
            samples,
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="b" * 64,
        )

        self.assertEqual(report["status"], "contaminated")
        self.assertIn(
            "sibling_npu_process_churn",
            {finding["code"] for finding in report["hard_findings"]},
        )

    def test_cpu_pressure_and_missing_npu_samples_fail_closed(self) -> None:
        samples = [
            _sample(0.0, total=100, idle=80),
            _sample(10.0, total=200, idle=81, npu_status="failed"),
            _sample(20.0, total=300, idle=82),
        ]

        report = assess_host_telemetry(
            samples,
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="b" * 64,
        )

        self.assertEqual(report["status"], "insufficient_telemetry")
        codes = {finding["code"] for finding in report["hard_findings"]}
        self.assertIn("host_cpu_pressure", codes)
        self.assertIn("missing_npu_telemetry", codes)

    def test_file_assessment_binds_raw_telemetry_digest(self) -> None:
        samples = [
            _sample(0.0, total=100, idle=80),
            _sample(10.0, total=200, idle=160),
            _sample(20.0, total=300, idle=240),
        ]
        raw = "".join(json.dumps(sample) + "\n" for sample in samples).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            path.write_bytes(raw)

            report = assess_host_telemetry_file(
                path,
                target_npu_id=2,
                policy=HostInterferencePolicy(),
            )

        self.assertEqual(report["telemetry_sha256"], hashlib.sha256(raw).hexdigest())

    def test_campaign_detects_process_changes_between_runs(self) -> None:
        first = assess_host_telemetry(
            [
                _sample(0.0, total=100, idle=80, sibling_pid=100),
                _sample(10.0, total=200, idle=160, sibling_pid=100),
                _sample(20.0, total=300, idle=240, sibling_pid=100),
            ],
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="b" * 64,
        )
        second = assess_host_telemetry(
            [
                _sample(30.0, total=400, idle=320, sibling_pid=101),
                _sample(40.0, total=500, idle=400, sibling_pid=101),
                _sample(50.0, total=600, idle=480, sibling_pid=101),
            ],
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="c" * 64,
        )

        report = assess_campaign_host_reports(
            {"run-0": first, "run-1": second},
            expected_run_ids=("run-0", "run-1"),
            plan_sha256="d" * 64,
            policy=CampaignInterferencePolicy(),
        )

        self.assertEqual(report["status"], "contaminated")
        self.assertIn(
            "sibling_npu_process_change_between_runs",
            {finding["code"] for finding in report["hard_findings"]},
        )

    def test_campaign_fails_closed_for_missing_or_mutated_report(self) -> None:
        clean = assess_host_telemetry(
            [
                _sample(0.0, total=100, idle=80),
                _sample(10.0, total=200, idle=160),
                _sample(20.0, total=300, idle=240),
            ],
            target_npu_id=2,
            policy=HostInterferencePolicy(),
            telemetry_sha256="b" * 64,
        )
        mutated = {**clean, "status": "contaminated"}

        report = assess_campaign_host_reports(
            {"run-0": mutated},
            expected_run_ids=("run-0", "run-1"),
            plan_sha256="d" * 64,
            policy=CampaignInterferencePolicy(),
        )

        self.assertEqual(report["status"], "insufficient_telemetry")
        codes = {finding["code"] for finding in report["hard_findings"]}
        self.assertIn("missing_run_host_report", codes)
        self.assertIn("invalid_run_host_report", codes)

    def test_campaign_detects_different_target_thermal_start(self) -> None:
        first_samples = [
            _sample(0.0, total=100, idle=80),
            _sample(10.0, total=200, idle=160),
            _sample(20.0, total=300, idle=240),
        ]
        second_samples = [
            _sample(30.0, total=400, idle=320),
            _sample(40.0, total=500, idle=400),
            _sample(50.0, total=600, idle=480),
        ]
        second_samples[0]["npu"]["devices"][1]["temperature_c"] = 55.0
        reports = {
            "run-0": assess_host_telemetry(
                first_samples,
                target_npu_id=2,
                policy=HostInterferencePolicy(),
                telemetry_sha256="b" * 64,
            ),
            "run-1": assess_host_telemetry(
                second_samples,
                target_npu_id=2,
                policy=HostInterferencePolicy(),
                telemetry_sha256="c" * 64,
            ),
        }

        report = assess_campaign_host_reports(
            reports,
            expected_run_ids=("run-0", "run-1"),
            plan_sha256="d" * 64,
            policy=CampaignInterferencePolicy(),
        )

        self.assertEqual(report["status"], "contaminated")
        self.assertIn(
            "target_npu_initial_temperature_shift",
            {finding["code"] for finding in report["hard_findings"]},
        )


if __name__ == "__main__":
    unittest.main()
