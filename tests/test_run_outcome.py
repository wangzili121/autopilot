from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


class RunOutcomeTest(unittest.TestCase):
    def _classify(
        self,
        status: int,
        log: str,
        *,
        result: bool = False,
        initial_process_mb: int = 0,
        final_process_mb: int = 0,
        maximum_existing_process_mb: int = 1024,
        host_integrity_status: str = "clean",
    ) -> str:
        root = Path(__file__).resolve().parents[1]
        helper = root / "scripts" / "lib" / "run_outcome.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            log_path = temporary / "runner.log"
            result_path = temporary / "result.json"
            log_path.write_text(log, encoding="utf-8")
            if result:
                result_path.write_text("{}\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; classify_run_outcome "$2" "$3" "$4" "$5" "$6" "$7" "$8"',
                    "classify-run",
                    str(helper),
                    str(status),
                    str(result_path),
                    str(log_path),
                    str(initial_process_mb),
                    str(final_process_mb),
                    str(maximum_existing_process_mb),
                    host_integrity_status,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()

    def test_success_requires_result_artifact(self) -> None:
        self.assertEqual(self._classify(0, "completed", result=True), "success")
        self.assertEqual(self._classify(0, "completed"), "failed")

    def test_generic_memory_warning_is_not_resource_exhaustion(self) -> None:
        warning = "A known problem may cause (Out of Memory) errors"
        self.assertEqual(self._classify(1, warning), "failed")

    def test_runtime_oom_is_resource_exhaustion(self) -> None:
        error = "RuntimeError: NPU out of memory. Tried to allocate 2.03 GiB"
        self.assertEqual(self._classify(1, error), "resource_exhausted")

    def test_vllm_startup_memory_failure_is_resource_exhaustion(self) -> None:
        error = (
            "Free memory on device (1.2/60.96 GiB) on startup is less than "
            "desired GPU memory utilization (0.36, 21.94 GiB)"
        )
        self.assertEqual(self._classify(2, error), "resource_exhausted")

    def test_late_external_process_marks_environment_contamination(self) -> None:
        self.assertEqual(
            self._classify(
                2,
                "failed to initialize engine",
                initial_process_mb=0,
                final_process_mb=57534,
            ),
            "environment_contaminated",
        )

    def test_host_integrity_failure_overrides_successful_runner(self) -> None:
        for status in ("contaminated", "insufficient_telemetry"):
            with self.subTest(status=status):
                self.assertEqual(
                    self._classify(
                        0,
                        "completed",
                        result=True,
                        host_integrity_status=status,
                    ),
                    "environment_contaminated",
                )
        self.assertEqual(
            self._classify(
                0,
                "completed",
                result=True,
                initial_process_mb=0,
                final_process_mb=57534,
            ),
            "environment_contaminated",
        )


if __name__ == "__main__":
    unittest.main()
