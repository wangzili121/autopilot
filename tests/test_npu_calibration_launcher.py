import unittest
from pathlib import Path


class NPUCalibrationLauncherTest(unittest.TestCase):
    def test_read_only_source_has_writable_artifact_hash_cache(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_npu_calibration_bundle.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('"${SOURCE_REPO}/.cache"', script)
        self.assertIn('"${CACHE_ROOT}/source-artifact-cache"', script)
        self.assertIn(
            '-v "${CACHE_ROOT}/source-artifact-cache:/workspace/.cache"',
            script,
        )
        self.assertIn(
            'NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE:-isolated_device_mapping}"',
            script,
        )
        self.assertIn('privileged_visible_devices)', script)
        self.assertIn('NPU_DOCKER_DEVICE_ARGS=(--privileged)', script)
        self.assertIn('CONTAINER_VISIBLE_DEVICES="${NPU_ID}"', script)
        self.assertIn(
            '-e ASCEND_RT_VISIBLE_DEVICES="${CONTAINER_VISIBLE_DEVICES}"',
            script,
        )
        self.assertIn('STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG:-1}"', script)
        self.assertIn('> "${RUNNER_STDOUT}"', script)
        self.assertIn('npu-processes-before.txt', script)
        self.assertIn('final_process_mb=${final_mb}', script)
        self.assertIn('classify_run_outcome', script)
        self.assertIn('src/inference_autopilot/host_interference.py', script)
        self.assertIn('host-telemetry.jsonl', script)
        self.assertIn('host-interference-report.json', script)
        self.assertIn('host_integrity_status=${host_integrity_status}', script)

    def test_campaign_forwards_manifest_bound_device_mode(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_npu_calibration_campaign.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE:-isolated_device_mapping}"',
            script,
        )
        self.assertIn(
            'NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE}"',
            script,
        )
        self.assertIn('STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG}"', script)
        self.assertIn('STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG:-0}"', script)
        self.assertIn('HOST_TELEMETRY_INTERVAL_SECONDS="${HOST_TELEMETRY_INTERVAL_SECONDS:-10}"', script)
        self.assertIn('HOST_TELEMETRY_INTERVAL_SECONDS="${HOST_TELEMETRY_INTERVAL_SECONDS}"', script)
        self.assertIn('assess-campaign "${CAMPAIGN_DIR}"', script)
        self.assertIn('campaign-host-interference-report.json', script)
        self.assertIn('if (( host_campaign_assessment_status != 0 ))', script)
        self.assertIn('MAX_ENVIRONMENT_RETRIES="${MAX_ENVIRONMENT_RETRIES:-2}"', script)
        self.assertIn('RESUME_CAMPAIGN="${RESUME_CAMPAIGN:-0}"', script)
        self.assertIn(
            'EXECUTION_READINESS_ASSESSMENT="${EXECUTION_READINESS_ASSESSMENT:-}"',
            script,
        )
        self.assertIn('audit-execution-readiness', script)
        self.assertIn('execution_readiness_plan_mismatch', script)
        self.assertIn('execution_readiness_deferred', script)
        self.assertIn('--allow-incomplete', script)
        self.assertIn('run_already_accepted=', script)
        self.assertIn('check-window', script)
        self.assertIn('admission_attempt_index=0', script)
        self.assertIn('admission_window_index=$((admission_window_index + 1))', script)
        self.assertIn('attempt-ledger.jsonl', script)
        self.assertIn('--outcome environment_contaminated', script)
        self.assertIn('record_early_terminal_status "environment_retries_exhausted"', script)
        self.assertIn('echo "terminal_status=complete"', script)
        host_campaign_gate = script.index('if (( host_campaign_assessment_status != 0 ))')
        formal_assessment = script.index('assessment_args=')
        self.assertLess(host_campaign_gate, formal_assessment)
        self.assertIn('[[ "${observation_status}" != "success" ]]', script)
        self.assertIn('run failed without an environment-only retry', script)
        self.assertIn('host_run_outcome=', script)
        self.assertIn('"${host_run_outcome}" == "environment_contaminated"', script)
        contamination_gate = script.index('"${host_run_outcome}" == "environment_contaminated"')
        observation_copy = script.index('cp "${accepted_bundle}/observation.json"')
        self.assertLess(contamination_gate, observation_copy)
        ledger_gate = script.index('if (( attempt_ledger_status != 0 ))')
        self.assertLess(ledger_gate, formal_assessment)
        readiness_gate = script.index('if (( readiness_status == 2 ))')
        run_loop = script.index('for run_id in "${RUN_IDS[@]}"; do')
        self.assertLess(readiness_gate, run_loop)

    def test_graph_profiler_accepts_an_explicit_bfloat16_stack(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_npu_graph_metrics_smoke.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('DTYPE="${DTYPE:-float16}"', script)
        self.assertIn('float16|bfloat16)', script)
        self.assertIn("--dtype '${DTYPE}'", script)
        self.assertIn('echo "dtype=${DTYPE}"', script)
        self.assertIn('RUN_SEED="${RUN_SEED:-}"', script)
        self.assertIn('SUBSET_SEED="${SUBSET_SEED:-}"', script)
        self.assertIn(
            '-e "INFERENCE_AUTOPILOT_RUN_SEED=${RUN_SEED}"', script
        )
        self.assertIn(
            '-e "INFERENCE_AUTOPILOT_SUBSET_SEED=${SUBSET_SEED}"', script
        )
        self.assertIn('"${CACHE_ROOT}/source-artifact-cache"', script)
        self.assertIn(
            '-v "${CACHE_ROOT}/source-artifact-cache:/workspace/.cache"',
            script,
        )
        self.assertIn('GRAPH_PHASE_METRICS="${GRAPH_PHASE_METRICS:-0}"', script)
        self.assertIn(
            "GRAPH_PHASE_METRICS currently requires MODEL_RUNNER=MRV1",
            script,
        )
        self.assertIn("vllm_core_graph_phase_patch_sha256", script)
        self.assertIn("vllm_ascend_graph_phase_patch_sha256", script)
        self.assertIn(
            'PROPOSAL_MAX_NUM_BATCHED_TOKENS="${PROPOSAL_MAX_NUM_BATCHED_TOKENS:-}"',
            script,
        )
        self.assertIn(
            '-e "INFERENCE_AUTOPILOT_PROPOSAL_MAX_NUM_BATCHED_TOKENS=', script
        )
        self.assertIn('PROPOSAL_STAGE_WAVEFRONT="${PROPOSAL_STAGE_WAVEFRONT:-off}"', script)
        self.assertIn(
            "auto PROPOSAL_STAGE_WAVEFRONT requires a positive ", script
        )
        self.assertIn(
            '-e "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT=', script
        )
        self.assertIn('echo "proposal_graph_capture_ceiling=', script)


if __name__ == "__main__":
    unittest.main()
