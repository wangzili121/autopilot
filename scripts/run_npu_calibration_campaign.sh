#!/usr/bin/env bash
set -euo pipefail

AUTOPILOT_ROOT="${AUTOPILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CALIBRATION_SPEC="${CALIBRATION_SPEC:?set CALIBRATION_SPEC to a frozen calibration spec}"
SOURCE_REPO="${SOURCE_REPO:?set SOURCE_REPO to the inference-scaling worktree}"
SOURCE_CONFIG="${SOURCE_CONFIG:?set SOURCE_CONFIG to a TOML file under SOURCE_REPO}"
DATA="${DATA:-${SOURCE_REPO}/data/gsm8k/test.jsonl}"
NPU_ID="${NPU_ID:-0}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
MAX_EXISTING_PROCESS_MB="${MAX_EXISTING_PROCESS_MB:-1024}"
REPLAY_NOISE_ASSESSMENT="${REPLAY_NOISE_ASSESSMENT:-}"
EXECUTION_READINESS_ASSESSMENT="${EXECUTION_READINESS_ASSESSMENT:-}"
NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE:-isolated_device_mapping}"
STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG:-0}"
HOST_TELEMETRY_INTERVAL_SECONDS="${HOST_TELEMETRY_INTERVAL_SECONDS:-10}"
HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS="${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS:-30}"
HOST_MAX_CPU_BUSY_FRACTION="${HOST_MAX_CPU_BUSY_FRACTION:-0.90}"
HOST_MAX_IOWAIT_FRACTION="${HOST_MAX_IOWAIT_FRACTION:-0.15}"
HOST_MAX_RUN_QUEUE_PER_CPU="${HOST_MAX_RUN_QUEUE_PER_CPU:-1.0}"
HOST_MIN_MEMORY_AVAILABLE_FRACTION="${HOST_MIN_MEMORY_AVAILABLE_FRACTION:-0.05}"
HOST_TELEMETRY_PYTHON="${HOST_TELEMETRY_PYTHON:-python3}"
HOST_CAMPAIGN_MAX_CPU_BUSY_P95_RANGE="${HOST_CAMPAIGN_MAX_CPU_BUSY_P95_RANGE:-0.20}"
HOST_CAMPAIGN_MAX_IOWAIT_P95_RANGE="${HOST_CAMPAIGN_MAX_IOWAIT_P95_RANGE:-0.10}"
HOST_CAMPAIGN_MAX_RUN_QUEUE_P95_RANGE="${HOST_CAMPAIGN_MAX_RUN_QUEUE_P95_RANGE:-0.25}"
HOST_CAMPAIGN_MAX_MEMORY_AVAILABLE_RANGE="${HOST_CAMPAIGN_MAX_MEMORY_AVAILABLE_RANGE:-0.10}"
HOST_CAMPAIGN_MAX_SIBLING_AICORE_MEAN_RANGE="${HOST_CAMPAIGN_MAX_SIBLING_AICORE_MEAN_RANGE:-20.0}"
HOST_CAMPAIGN_MAX_TARGET_INITIAL_TEMPERATURE_RANGE="${HOST_CAMPAIGN_MAX_TARGET_INITIAL_TEMPERATURE_RANGE:-10.0}"
HOST_CAMPAIGN_MAX_TARGET_INITIAL_POWER_RANGE="${HOST_CAMPAIGN_MAX_TARGET_INITIAL_POWER_RANGE:-30.0}"
HOST_ADMISSION_WINDOW_SECONDS="${HOST_ADMISSION_WINDOW_SECONDS:-35}"
HOST_ADMISSION_INTERVAL_SECONDS="${HOST_ADMISSION_INTERVAL_SECONDS:-10}"
MAX_ADMISSION_ATTEMPTS="${MAX_ADMISSION_ATTEMPTS:-6}"
MAX_ENVIRONMENT_RETRIES="${MAX_ENVIRONMENT_RETRIES:-2}"
RESUME_CAMPAIGN="${RESUME_CAMPAIGN:-0}"

if [[ ! "${MAX_ADMISSION_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ADMISSION_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_ENVIRONMENT_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "MAX_ENVIRONMENT_RETRIES must be a non-negative integer" >&2
  exit 2
fi
if [[ "${RESUME_CAMPAIGN}" != "0" && "${RESUME_CAMPAIGN}" != "1" ]]; then
  echo "RESUME_CAMPAIGN must be 0 or 1" >&2
  exit 2
fi

for path in "${CALIBRATION_SPEC}" "${SOURCE_CONFIG}" "${DATA}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing campaign input: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${SOURCE_REPO}" ]]; then
  echo "missing source repository: ${SOURCE_REPO}" >&2
  exit 2
fi
if [[ -n "${REPLAY_NOISE_ASSESSMENT}" && ! -f "${REPLAY_NOISE_ASSESSMENT}" ]]; then
  echo "missing replay-noise assessment: ${REPLAY_NOISE_ASSESSMENT}" >&2
  exit 2
fi
if [[ -n "${EXECUTION_READINESS_ASSESSMENT}" && ! -f "${EXECUTION_READINESS_ASSESSMENT}" ]]; then
  echo "missing execution-readiness assessment: ${EXECUTION_READINESS_ASSESSMENT}" >&2
  exit 2
fi

CAMPAIGN_ID="$({
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_id"])' \
    "${CALIBRATION_SPEC}"
})"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${AUTOPILOT_ROOT}/artifacts/${CAMPAIGN_ID}}"
CAMPAIGN_LOCK="/tmp/inference-autopilot-campaign-npu-${NPU_ID}.lock"
if [[ -e "${CAMPAIGN_DIR}" && "${RESUME_CAMPAIGN}" != "1" ]]; then
  echo "refusing to overwrite campaign directory: ${CAMPAIGN_DIR}" >&2
  exit 2
fi
if [[ ! -d "${CAMPAIGN_DIR}" && "${RESUME_CAMPAIGN}" == "1" ]]; then
  echo "cannot resume a missing campaign directory: ${CAMPAIGN_DIR}" >&2
  exit 2
fi
if ! mkdir "${CAMPAIGN_LOCK}" 2>/dev/null; then
  echo "NPU ${NPU_ID} is already reserved by another Autopilot campaign" >&2
  exit 2
fi
trap 'rmdir "${CAMPAIGN_LOCK}" 2>/dev/null || true' EXIT

mkdir -p \
  "${CAMPAIGN_DIR}/admission" \
  "${CAMPAIGN_DIR}/attempts" \
  "${CAMPAIGN_DIR}/observations"
PLAN="${CAMPAIGN_DIR}/plan.json"
ATTEMPT_LEDGER="${CAMPAIGN_DIR}/attempt-ledger.jsonl"
ATTEMPT_LEDGER_PROGRAM="${AUTOPILOT_ROOT}/src/inference_autopilot/attempt_ledger.py"
if [[ "${RESUME_CAMPAIGN}" == "1" ]]; then
  for path in "${CAMPAIGN_DIR}/calibration-spec.json" "${PLAN}"; do
    if [[ ! -f "${path}" ]]; then
      echo "cannot resume campaign with missing artifact: ${path}" >&2
      exit 2
    fi
  done
  if ! cmp -s "${CALIBRATION_SPEC}" "${CAMPAIGN_DIR}/calibration-spec.json"; then
    echo "resume calibration spec differs from the frozen campaign spec" >&2
    exit 2
  fi
  if [[ -e "${CAMPAIGN_DIR}/assessment.json" ]]; then
    echo "refusing to resume an already assessed campaign" >&2
    exit 2
  fi
else
  cp "${CALIBRATION_SPEC}" "${CAMPAIGN_DIR}/calibration-spec.json"
fi
SOURCE_CONFIG_RELATIVE="$(realpath --relative-to="${SOURCE_REPO}" "${SOURCE_CONFIG}")"
DATA_RELATIVE="$(realpath --relative-to="${SOURCE_REPO}" "${DATA}")"
for relative in "${SOURCE_CONFIG_RELATIVE}" "${DATA_RELATIVE}"; do
  case "${relative}" in
    ../*|..)
      echo "source config and data must be under SOURCE_REPO" >&2
      exit 2
      ;;
  esac
done

CONTROL_CONTAINER=(
  docker run --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH=/autopilot/src
  -v "${AUTOPILOT_ROOT}:/autopilot:ro"
  -v "${SOURCE_REPO}:/source:ro"
  -v "${CAMPAIGN_DIR}:/campaign"
  -w /autopilot
  --entrypoint python3
  "${IMAGE}"
  -m inference_autopilot.cli
)
if [[ "${RESUME_CAMPAIGN}" == "1" ]]; then
  RESUME_LEDGER_ASSESSMENT="${CAMPAIGN_DIR}/resume-attempt-ledger-assessment-$(date +%s).json"
  set +e
  "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" audit \
    "${ATTEMPT_LEDGER}" \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --plan "${PLAN}" \
    --output "${RESUME_LEDGER_ASSESSMENT}" \
    --allow-incomplete \
    >/dev/null
  resume_ledger_status=$?
  set -e
  if (( resume_ledger_status != 0 )); then
    echo "resume attempt-ledger audit failed" >&2
    exit 3
  fi
else
  "${CONTROL_CONTAINER[@]}" plan-calibration \
    /campaign/calibration-spec.json --output /campaign/plan.json
fi

mapfile -t RUN_IDS < <(
  python3 -c \
    'import json, sys; print("\n".join(run["run_id"] for run in json.load(open(sys.argv[1], encoding="utf-8"))["runs"]))' \
    "${PLAN}"
)
if (( ${#RUN_IDS[@]} == 0 )); then
  echo "calibration plan contains no runs" >&2
  exit 2
fi

if [[ "${RESUME_CAMPAIGN}" == "1" ]]; then
  {
    echo "resumed_at=$(date --iso-8601=seconds)"
    echo "resume_attempt_ledger_assessment_sha256=$(sha256sum "${RESUME_LEDGER_ASSESSMENT}" | awk '{print $1}')"
  } >> "${CAMPAIGN_DIR}/campaign-execution.meta"
else
  {
    echo "campaign_id=${CAMPAIGN_ID}"
    echo "host=$(hostname)"
    echo "npu_id=${NPU_ID}"
    echo "source_repo=${SOURCE_REPO}"
    echo "source_config=${SOURCE_CONFIG}"
    echo "data=${DATA}"
    echo "image=${IMAGE}"
    echo "container_device_mode=${NPU_CONTAINER_DEVICE_MODE}"
    echo "stream_runner_log=${STREAM_RUNNER_LOG}"
    echo "host_telemetry_interval_seconds=${HOST_TELEMETRY_INTERVAL_SECONDS}"
    echo "host_telemetry_max_sample_gap_seconds=${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS}"
    echo "host_max_cpu_busy_fraction=${HOST_MAX_CPU_BUSY_FRACTION}"
    echo "host_max_iowait_fraction=${HOST_MAX_IOWAIT_FRACTION}"
    echo "host_max_run_queue_per_cpu=${HOST_MAX_RUN_QUEUE_PER_CPU}"
    echo "host_min_memory_available_fraction=${HOST_MIN_MEMORY_AVAILABLE_FRACTION}"
    echo "host_telemetry_python=${HOST_TELEMETRY_PYTHON}"
    echo "host_campaign_max_cpu_busy_p95_range=${HOST_CAMPAIGN_MAX_CPU_BUSY_P95_RANGE}"
    echo "host_campaign_max_iowait_p95_range=${HOST_CAMPAIGN_MAX_IOWAIT_P95_RANGE}"
    echo "host_campaign_max_run_queue_p95_range=${HOST_CAMPAIGN_MAX_RUN_QUEUE_P95_RANGE}"
    echo "host_campaign_max_memory_available_range=${HOST_CAMPAIGN_MAX_MEMORY_AVAILABLE_RANGE}"
    echo "host_campaign_max_sibling_aicore_mean_range=${HOST_CAMPAIGN_MAX_SIBLING_AICORE_MEAN_RANGE}"
    echo "host_campaign_max_target_initial_temperature_range=${HOST_CAMPAIGN_MAX_TARGET_INITIAL_TEMPERATURE_RANGE}"
    echo "host_campaign_max_target_initial_power_range=${HOST_CAMPAIGN_MAX_TARGET_INITIAL_POWER_RANGE}"
    echo "host_admission_window_seconds=${HOST_ADMISSION_WINDOW_SECONDS}"
    echo "host_admission_interval_seconds=${HOST_ADMISSION_INTERVAL_SECONDS}"
    echo "max_admission_attempts=${MAX_ADMISSION_ATTEMPTS}"
    echo "max_environment_retries=${MAX_ENVIRONMENT_RETRIES}"
    echo "run_count=${#RUN_IDS[@]}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${CAMPAIGN_DIR}/campaign-execution.meta"
  : > "${CAMPAIGN_DIR}/run-status.tsv"
  : > "${CAMPAIGN_DIR}/attempt-status.tsv"
  : > "${CAMPAIGN_DIR}/admission-status.tsv"
fi
record_early_terminal_status() {
  {
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "assessment_exit_code=not_run"
    echo "terminal_status=$1"
  } >> "${CAMPAIGN_DIR}/campaign-execution.meta"
}

if [[ -n "${EXECUTION_READINESS_ASSESSMENT}" ]]; then
  READINESS_SNAPSHOT="${CAMPAIGN_DIR}/execution-readiness-preflight.json"
  if [[ "$(realpath "${EXECUTION_READINESS_ASSESSMENT}")" != "$(realpath -m "${READINESS_SNAPSHOT}")" ]]; then
    cp "${EXECUTION_READINESS_ASSESSMENT}" "${READINESS_SNAPSHOT}"
  fi
  readiness_target_plan_sha256="$(
    python3 -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source"]["target_plan_file_sha256"])' \
      "${READINESS_SNAPSHOT}"
  )"
  actual_plan_sha256="$(sha256sum "${PLAN}" | awk '{print $1}')"
  if [[ "${readiness_target_plan_sha256}" != "${actual_plan_sha256}" ]]; then
    record_early_terminal_status "execution_readiness_plan_mismatch"
    echo "execution-readiness assessment does not bind the campaign plan" >&2
    exit 3
  fi
  set +e
  "${CONTROL_CONTAINER[@]}" audit-execution-readiness \
    /campaign/execution-readiness-preflight.json \
    > "${CAMPAIGN_DIR}/execution-readiness-preflight.audit.json"
  readiness_status=$?
  set -e
  {
    echo "execution_readiness_assessment_sha256=$(sha256sum "${READINESS_SNAPSHOT}" | awk '{print $1}')"
    echo "execution_readiness_audit_exit_code=${readiness_status}"
  } >> "${CAMPAIGN_DIR}/campaign-execution.meta"
  if (( readiness_status == 2 )); then
    record_early_terminal_status "execution_readiness_deferred"
    echo "execution-readiness gate deferred this campaign before NPU execution" >&2
    exit 4
  fi
  if (( readiness_status != 0 )); then
    record_early_terminal_status "execution_readiness_invalid"
    echo "execution-readiness assessment audit failed" >&2
    exit 3
  fi
fi

for run_id in "${RUN_IDS[@]}"; do
  run_state="$(
    "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" state \
      "${ATTEMPT_LEDGER}" --logical-run-id "${run_id}"
  )"
  run_already_accepted="$(
    python3 -c 'import json, sys; print(str(json.loads(sys.argv[1])["accepted"]).lower())' \
      "${run_state}"
  )"
  attempt_index="$(
    python3 -c 'import json, sys; print(json.loads(sys.argv[1])["attempt_count"])' \
      "${run_state}"
  )"
  if [[ "${run_already_accepted}" == "true" ]]; then
    cp "${CAMPAIGN_DIR}/${run_id}/observation.json" \
      "${CAMPAIGN_DIR}/observations/${run_id}.json"
    continue
  fi
  admission_window_index="$({
    find "${CAMPAIGN_DIR}/admission/${run_id}" -maxdepth 1 \
      -type f -name 'window-*.report.json' 2>/dev/null | wc -l
  })"
  admission_window_index="${admission_window_index//[[:space:]]/}"
  while true; do
    attempt_id="$(printf 'attempt-%03d' "${attempt_index}")"
    admission_attempt_index=0
    while true; do
      window_id="$(printf 'window-%03d' "${admission_window_index}")"
      admission_dir="${CAMPAIGN_DIR}/admission/${run_id}"
      admission_telemetry="${admission_dir}/${window_id}.jsonl"
      admission_report="${admission_dir}/${window_id}.report.json"
      mkdir -p "${admission_dir}"
      set +e
      "${HOST_TELEMETRY_PYTHON}" \
        "${AUTOPILOT_ROOT}/src/inference_autopilot/host_interference.py" \
        check-window \
        --output-telemetry "${admission_telemetry}" \
        --output "${admission_report}" \
        --target-npu "${NPU_ID}" \
        --duration-seconds "${HOST_ADMISSION_WINDOW_SECONDS}" \
        --interval-seconds "${HOST_ADMISSION_INTERVAL_SECONDS}" \
        --npu-smi /usr/local/bin/npu-smi \
        --max-sample-gap-seconds "${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS}" \
        --max-host-cpu-busy-fraction "${HOST_MAX_CPU_BUSY_FRACTION}" \
        --max-host-iowait-fraction "${HOST_MAX_IOWAIT_FRACTION}" \
        --max-run-queue-per-cpu "${HOST_MAX_RUN_QUEUE_PER_CPU}" \
        --min-memory-available-fraction "${HOST_MIN_MEMORY_AVAILABLE_FRACTION}" \
        >/dev/null
      admission_status=$?
      set -e
      if [[ -f "${admission_report}" ]]; then
        admission_outcome="$(
          python3 -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
            "${admission_report}"
        )"
      else
        admission_outcome="missing_report"
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${run_id}" "${attempt_id}" "${window_id}" \
        "${admission_status}" "${admission_outcome}" \
        >> "${CAMPAIGN_DIR}/admission-status.tsv"
      admission_window_index=$((admission_window_index + 1))
      admission_attempt_index=$((admission_attempt_index + 1))
      if (( admission_status == 0 )); then
        break
      fi
      if (( admission_attempt_index >= MAX_ADMISSION_ATTEMPTS )); then
        printf '%s\t%s\t%s\n' \
          "${run_id}" "${admission_status}" "admission_rejected" \
          >> "${CAMPAIGN_DIR}/run-status.tsv"
        echo "no stable host admission window; stopping campaign: ${run_id}" >&2
        record_early_terminal_status "admission_rejected"
        exit 3
      fi
    done

    bundle="${CAMPAIGN_DIR}/attempts/${run_id}/${attempt_id}"
    "${CONTROL_CONTAINER[@]}" prepare-run \
      /campaign/plan.json "${run_id}" \
      --source-repo /source \
      --source-config "/source/${SOURCE_CONFIG_RELATIVE}" \
      --data "/source/${DATA_RELATIVE}" \
      --output-dir "/campaign/attempts/${run_id}/${attempt_id}" \
      --require-formal

    set +e
    AUTOPILOT_ROOT="${AUTOPILOT_ROOT}" \
      BUNDLE_DIR="${bundle}" \
      SOURCE_REPO="${SOURCE_REPO}" \
      SOURCE_CONFIG="${SOURCE_CONFIG}" \
      DATA="${DATA}" \
      NPU_ID="${NPU_ID}" \
      IMAGE="${IMAGE}" \
      MAX_EXISTING_PROCESS_MB="${MAX_EXISTING_PROCESS_MB}" \
      NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE}" \
      STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG}" \
      HOST_TELEMETRY_INTERVAL_SECONDS="${HOST_TELEMETRY_INTERVAL_SECONDS}" \
      HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS="${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS}" \
      HOST_MAX_CPU_BUSY_FRACTION="${HOST_MAX_CPU_BUSY_FRACTION}" \
      HOST_MAX_IOWAIT_FRACTION="${HOST_MAX_IOWAIT_FRACTION}" \
      HOST_MAX_RUN_QUEUE_PER_CPU="${HOST_MAX_RUN_QUEUE_PER_CPU}" \
      HOST_MIN_MEMORY_AVAILABLE_FRACTION="${HOST_MIN_MEMORY_AVAILABLE_FRACTION}" \
      HOST_TELEMETRY_PYTHON="${HOST_TELEMETRY_PYTHON}" \
      "${AUTOPILOT_ROOT}/scripts/run_npu_calibration_bundle.sh"
    run_status=$?
    set -e

    host_run_outcome="missing_execution_metadata"
    if [[ -f "${bundle}/execution.meta" ]]; then
      host_run_outcome="$({
        awk -F= '$1 == "run_outcome" { value = $2 } END { print value }' \
          "${bundle}/execution.meta"
      })"
      host_run_outcome="${host_run_outcome:-unclassified_failure}"
    fi
    if [[ "${host_run_outcome}" == "environment_contaminated" ]]; then
      "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" append \
        "${ATTEMPT_LEDGER}" \
        --campaign-dir "${CAMPAIGN_DIR}" \
        --bundle-dir "${bundle}" \
        --logical-run-id "${run_id}" \
        --attempt-index "${attempt_index}" \
        --outcome environment_contaminated \
        --admission-report "${admission_report}" \
        >/dev/null
      printf '%s\t%s\t%s\t%s\n' \
        "${run_id}" "${attempt_id}" "${run_status}" "${host_run_outcome}" \
        >> "${CAMPAIGN_DIR}/attempt-status.tsv"
      if (( attempt_index >= MAX_ENVIRONMENT_RETRIES )); then
        printf '%s\t%s\t%s\n' \
          "${run_id}" "${run_status}" "environment_retries_exhausted" \
          >> "${CAMPAIGN_DIR}/run-status.tsv"
        echo "environment retries exhausted; stopping campaign: ${run_id}" >&2
        record_early_terminal_status "environment_retries_exhausted"
        exit 3
      fi
      attempt_index=$((attempt_index + 1))
      continue
    fi

    if [[ ! -f "${bundle}/observation.json" ]]; then
      "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" append \
        "${ATTEMPT_LEDGER}" \
        --campaign-dir "${CAMPAIGN_DIR}" \
        --bundle-dir "${bundle}" \
        --logical-run-id "${run_id}" \
        --attempt-index "${attempt_index}" \
        --outcome missing_observation \
        --admission-report "${admission_report}" \
        >/dev/null
      printf '%s\t%s\t%s\t%s\n' \
        "${run_id}" "${attempt_id}" "${run_status}" "missing_observation" \
        >> "${CAMPAIGN_DIR}/attempt-status.tsv"
      printf '%s\t%s\t%s\n' "${run_id}" "${run_status}" "missing_observation" \
        >> "${CAMPAIGN_DIR}/run-status.tsv"
      echo "run did not emit an observation; stopping campaign: ${run_id}" >&2
      record_early_terminal_status "missing_observation"
      if (( run_status == 0 )); then
        exit 2
      fi
      exit "${run_status}"
    fi

    observation_status="$({
      python3 -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
        "${bundle}/observation.json"
    })"
    if (( run_status != 0 )) || [[ "${host_run_outcome}" != "success" ]] || \
      [[ "${observation_status}" != "success" ]]; then
      "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" append \
        "${ATTEMPT_LEDGER}" \
        --campaign-dir "${CAMPAIGN_DIR}" \
        --bundle-dir "${bundle}" \
        --logical-run-id "${run_id}" \
        --attempt-index "${attempt_index}" \
        --outcome run_failed \
        --admission-report "${admission_report}" \
        >/dev/null
      printf '%s\t%s\t%s\t%s\n' \
        "${run_id}" "${attempt_id}" "${run_status}" "${host_run_outcome}" \
        >> "${CAMPAIGN_DIR}/attempt-status.tsv"
      printf '%s\t%s\t%s\n' "${run_id}" "${run_status}" "${observation_status}" \
        >> "${CAMPAIGN_DIR}/run-status.tsv"
      echo "run failed without an environment-only retry: ${run_id} (${host_run_outcome})" >&2
      record_early_terminal_status "run_${host_run_outcome}"
      if (( run_status == 0 )); then
        exit 2
      fi
      exit "${run_status}"
    fi

    accepted_bundle="${CAMPAIGN_DIR}/${run_id}"
    mv "${bundle}" "${accepted_bundle}"
    "${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" append \
      "${ATTEMPT_LEDGER}" \
      --campaign-dir "${CAMPAIGN_DIR}" \
      --bundle-dir "${accepted_bundle}" \
      --logical-run-id "${run_id}" \
      --attempt-index "${attempt_index}" \
      --outcome accepted \
      --admission-report "${admission_report}" \
      >/dev/null
    cp "${accepted_bundle}/observation.json" \
      "${CAMPAIGN_DIR}/observations/${run_id}.json"
    printf '%s\t%s\t%s\t%s\n' \
      "${run_id}" "${attempt_id}" "${run_status}" "accepted" \
      >> "${CAMPAIGN_DIR}/attempt-status.tsv"
    printf '%s\t%s\t%s\n' "${run_id}" "${run_status}" "${observation_status}" \
      >> "${CAMPAIGN_DIR}/run-status.tsv"
    break
  done
done

set +e
"${HOST_TELEMETRY_PYTHON}" "${ATTEMPT_LEDGER_PROGRAM}" audit \
  "${ATTEMPT_LEDGER}" \
  --campaign-dir "${CAMPAIGN_DIR}" \
  --plan "${PLAN}" \
  --output "${CAMPAIGN_DIR}/attempt-ledger-assessment.json" \
  >/dev/null
attempt_ledger_status=$?
set -e
{
  echo "attempt_ledger_assessment_exit_code=${attempt_ledger_status}"
  echo "attempt_ledger_sha256=$(sha256sum "${ATTEMPT_LEDGER}" | awk '{print $1}')"
  echo "attempt_ledger_assessment_sha256=$(sha256sum "${CAMPAIGN_DIR}/attempt-ledger-assessment.json" | awk '{print $1}')"
} >> "${CAMPAIGN_DIR}/campaign-execution.meta"
if (( attempt_ledger_status != 0 )); then
  echo "attempt ledger audit failed; refusing formal assessment" >&2
  record_early_terminal_status "invalid_attempt_ledger"
  exit 3
fi

set +e
"${HOST_TELEMETRY_PYTHON}" \
  "${AUTOPILOT_ROOT}/src/inference_autopilot/host_interference.py" \
  assess-campaign "${CAMPAIGN_DIR}" \
  --plan "${PLAN}" \
  --output "${CAMPAIGN_DIR}/campaign-host-interference-report.json" \
  --max-host-cpu-busy-p95-range "${HOST_CAMPAIGN_MAX_CPU_BUSY_P95_RANGE}" \
  --max-host-iowait-p95-range "${HOST_CAMPAIGN_MAX_IOWAIT_P95_RANGE}" \
  --max-run-queue-per-cpu-p95-range "${HOST_CAMPAIGN_MAX_RUN_QUEUE_P95_RANGE}" \
  --max-memory-available-fraction-range "${HOST_CAMPAIGN_MAX_MEMORY_AVAILABLE_RANGE}" \
  --max-sibling-aicore-mean-range-percent "${HOST_CAMPAIGN_MAX_SIBLING_AICORE_MEAN_RANGE}" \
  --max-target-initial-temperature-range-c "${HOST_CAMPAIGN_MAX_TARGET_INITIAL_TEMPERATURE_RANGE}" \
  --max-target-initial-power-range-watts "${HOST_CAMPAIGN_MAX_TARGET_INITIAL_POWER_RANGE}" \
  >/dev/null
host_campaign_assessment_status=$?
set -e
{
  echo "host_campaign_assessment_exit_code=${host_campaign_assessment_status}"
  echo "host_campaign_interference_report_sha256=$(sha256sum "${CAMPAIGN_DIR}/campaign-host-interference-report.json" | awk '{print $1}')"
} >> "${CAMPAIGN_DIR}/campaign-execution.meta"
if (( host_campaign_assessment_status != 0 )); then
  echo "campaign host integrity gate failed; refusing formal assessment" >&2
  record_early_terminal_status "campaign_environment_contaminated"
  exit 3
fi

assessment_args=(/campaign/plan.json /campaign/observations --output /campaign/assessment.json)
if [[ -n "${REPLAY_NOISE_ASSESSMENT}" ]]; then
  cp "${REPLAY_NOISE_ASSESSMENT}" \
    "${CAMPAIGN_DIR}/replay-noise-assessment.json"
  assessment_args+=(
    --replay-noise-assessment /campaign/replay-noise-assessment.json
  )
fi
set +e
"${CONTROL_CONTAINER[@]}" assess-calibration "${assessment_args[@]}"
assessment_status=$?
set -e

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "assessment_exit_code=${assessment_status}"
  echo "plan_sha256=$(sha256sum "${PLAN}" | awk '{print $1}')"
  echo "assessment_sha256=$(sha256sum "${CAMPAIGN_DIR}/assessment.json" | awk '{print $1}')"
  echo "terminal_status=complete"
} >> "${CAMPAIGN_DIR}/campaign-execution.meta"
exit "${assessment_status}"
