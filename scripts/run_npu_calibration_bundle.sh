#!/usr/bin/env bash
set -euo pipefail

AUTOPILOT_ROOT="${AUTOPILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR to a prepared chang run bundle}"
SOURCE_REPO="${SOURCE_REPO:?set SOURCE_REPO to the inference-scaling worktree}"
SOURCE_CONFIG="${SOURCE_CONFIG:?set SOURCE_CONFIG to a TOML file under SOURCE_REPO}"
DATA="${DATA:-${SOURCE_REPO}/data/gsm8k/test.jsonl}"
NPU_ID="${NPU_ID:-0}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
MAX_EXISTING_PROCESS_MB="${MAX_EXISTING_PROCESS_MB:-1024}"
CACHE_ROOT="${CACHE_ROOT:-${AUTOPILOT_ROOT}/.cache/vllm-npu${NPU_ID}}"
NPU_CONTAINER_DEVICE_MODE="${NPU_CONTAINER_DEVICE_MODE:-isolated_device_mapping}"
STREAM_RUNNER_LOG="${STREAM_RUNNER_LOG:-1}"
HOST_TELEMETRY_INTERVAL_SECONDS="${HOST_TELEMETRY_INTERVAL_SECONDS:-10}"
HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS="${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS:-30}"
HOST_MAX_CPU_BUSY_FRACTION="${HOST_MAX_CPU_BUSY_FRACTION:-0.90}"
HOST_MAX_IOWAIT_FRACTION="${HOST_MAX_IOWAIT_FRACTION:-0.15}"
HOST_MAX_RUN_QUEUE_PER_CPU="${HOST_MAX_RUN_QUEUE_PER_CPU:-1.0}"
HOST_MIN_MEMORY_AVAILABLE_FRACTION="${HOST_MIN_MEMORY_AVAILABLE_FRACTION:-0.05}"
HOST_TELEMETRY_PYTHON="${HOST_TELEMETRY_PYTHON:-python3}"
BUNDLE_BASENAME="$(basename "${BUNDLE_DIR}")"
case "${BUNDLE_BASENAME}" in
  attempt-[0-9]*) ATTEMPT_ID="${BUNDLE_BASENAME}" ;;
  *) ATTEMPT_ID="${ATTEMPT_ID:-attempt-000}" ;;
esac
LOCK_DIR="/tmp/inference-autopilot-npu-${NPU_ID}.lock"
HOST_TELEMETRY="${BUNDLE_DIR}/host-telemetry.jsonl"
HOST_INTERFERENCE_REPORT="${BUNDLE_DIR}/host-interference-report.json"
HOST_TELEMETRY_PROGRAM="${AUTOPILOT_ROOT}/src/inference_autopilot/host_interference.py"
host_monitor_pid=""

source "${AUTOPILOT_ROOT}/scripts/lib/run_outcome.sh"

for path in \
  "${BUNDLE_DIR}/run-manifest.json" \
  "${BUNDLE_DIR}/launch.json" \
  "${SOURCE_CONFIG}" \
  "${DATA}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing calibration input: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${SOURCE_REPO}" ]]; then
  echo "missing source repository: ${SOURCE_REPO}" >&2
  exit 2
fi
RUN_ID="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run"]["run_id"])' \
    "${BUNDLE_DIR}/run-manifest.json"
)"
RUN_MANIFEST_SHA256="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_manifest_sha256"])' \
    "${BUNDLE_DIR}/run-manifest.json"
)"
CONTAINER_NAME="inference-autopilot-calibration-${RUN_ID}-${ATTEMPT_ID}-npu${NPU_ID}"
for output in \
  native-result.json \
  observation.json \
  runner.log \
  host-telemetry.jsonl \
  host-interference-report.json; do
  if [[ -e "${BUNDLE_DIR}/${output}" ]]; then
    echo "refusing to overwrite calibration output: ${BUNDLE_DIR}/${output}" >&2
    exit 2
  fi
done

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

read_launch_value() {
  local key="$1"
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source"][sys.argv[2]])' \
    "${BUNDLE_DIR}/launch.json" "${key}"
}

if [[ "$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["compatibility"]["formal_eligible"]).lower())' "${BUNDLE_DIR}/launch.json")" != "true" ]]; then
  echo "refusing a calibration bundle that is not formal-eligible" >&2
  exit 2
fi

EXPECTED_CONFIG_SHA256="$(read_launch_value source_config_sha256)"
EXPECTED_SNAPSHOT_SHA256="$(read_launch_value snapshot_sha256)"
EXPECTED_NPU_ID="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["environment_contract"]["hardware"]["physical_device_id"])' \
    "${BUNDLE_DIR}/run-manifest.json"
)"
EXPECTED_IMAGE="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["environment_contract"]["software"]["container_image"])' \
    "${BUNDLE_DIR}/run-manifest.json"
)"
EXPECTED_CONTAINER_DEVICE_MODE="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["environment_contract"]["software"].get("npu_container_device_mode", "isolated_device_mapping"))' \
    "${BUNDLE_DIR}/run-manifest.json"
)"
if [[ "${NPU_ID}" != "${EXPECTED_NPU_ID}" ]]; then
  echo "refusing physical NPU ${NPU_ID}: manifest requires ${EXPECTED_NPU_ID}" >&2
  exit 2
fi
if [[ "${IMAGE}" != "${EXPECTED_IMAGE}" ]]; then
  echo "refusing image ${IMAGE}: manifest requires ${EXPECTED_IMAGE}" >&2
  exit 2
fi
if [[ "${NPU_CONTAINER_DEVICE_MODE}" != "${EXPECTED_CONTAINER_DEVICE_MODE}" ]]; then
  echo "refusing container device mode ${NPU_CONTAINER_DEVICE_MODE}: manifest requires ${EXPECTED_CONTAINER_DEVICE_MODE}" >&2
  exit 2
fi
case "${NPU_CONTAINER_DEVICE_MODE}" in
  isolated_device_mapping)
    NPU_DOCKER_DEVICE_ARGS=(
      --device "/dev/davinci${NPU_ID}:/dev/davinci0"
      --device /dev/davinci_manager
      --device /dev/devmm_svm
      --device /dev/hisi_hdc
    )
    CONTAINER_VISIBLE_DEVICES=0
    ;;
  privileged_visible_devices)
    NPU_DOCKER_DEVICE_ARGS=(--privileged)
    CONTAINER_VISIBLE_DEVICES="${NPU_ID}"
    ;;
  *)
    echo "unsupported container device mode: ${NPU_CONTAINER_DEVICE_MODE}" >&2
    exit 2
    ;;
esac
case "${STREAM_RUNNER_LOG}" in
  0)
    RUNNER_STDOUT=/dev/null
    ;;
  1)
    RUNNER_STDOUT=/dev/stdout
    ;;
  *)
    echo "STREAM_RUNNER_LOG must be 0 or 1" >&2
    exit 2
    ;;
esac

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "NPU ${NPU_ID} is already locked by another Autopilot run" >&2
  exit 2
fi
stop_host_monitor() {
  if [[ -n "${host_monitor_pid}" ]] && kill -0 "${host_monitor_pid}" 2>/dev/null; then
    kill -TERM "${host_monitor_pid}" 2>/dev/null || true
    wait "${host_monitor_pid}" 2>/dev/null || true
  fi
  host_monitor_pid=""
}
cleanup() {
  stop_host_monitor
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

maximum_process_mb() {
  awk '
        /Process memory\(MB\):/ {
          value = $0
          sub(/^.*Process memory\(MB\):[[:space:]]*/, "", value)
          sub(/[[:space:]].*$/, "", value)
          if ((value + 0) > maximum) maximum = value + 0
        }
        END { print maximum + 0 }
      ' "$1"
}

NPU_PROCESSES_BEFORE="${BUNDLE_DIR}/npu-processes-before.txt"
/usr/local/bin/npu-smi info -t proc-mem -i "${NPU_ID}" -c 0 \
  > "${NPU_PROCESSES_BEFORE}" 2>&1 || true
existing_mb="$(maximum_process_mb "${NPU_PROCESSES_BEFORE}")"
if (( existing_mb > MAX_EXISTING_PROCESS_MB )); then
  echo "refusing NPU ${NPU_ID}: existing process uses ${existing_mb} MB" >&2
  exit 2
fi
mkdir -p \
  "${SOURCE_REPO}/.cache" \
  "${CACHE_ROOT}" \
  "${CACHE_ROOT}/source-artifact-cache"

{
  echo "run_id=${RUN_ID}"
  echo "logical_run_id=${RUN_ID}"
  echo "attempt_id=${ATTEMPT_ID}"
  echo "run_manifest_sha256=${RUN_MANIFEST_SHA256}"
  echo "host=$(hostname)"
  echo "npu_id=${NPU_ID}"
  echo "expected_npu_id=${EXPECTED_NPU_ID}"
  echo "existing_process_mb=${existing_mb}"
  echo "source_repo=${SOURCE_REPO}"
  echo "source_config=${SOURCE_CONFIG_RELATIVE}"
  echo "data=${DATA_RELATIVE}"
  echo "image=${IMAGE}"
  echo "expected_image=${EXPECTED_IMAGE}"
  echo "container_device_mode=${NPU_CONTAINER_DEVICE_MODE}"
  echo "stream_runner_log=${STREAM_RUNNER_LOG}"
  echo "container_visible_devices=${CONTAINER_VISIBLE_DEVICES}"
  echo "expected_source_config_sha256=${EXPECTED_CONFIG_SHA256}"
  echo "expected_source_snapshot_sha256=${EXPECTED_SNAPSHOT_SHA256}"
  date --iso-8601=seconds
} > "${BUNDLE_DIR}/execution.meta"

"${HOST_TELEMETRY_PYTHON}" "${HOST_TELEMETRY_PROGRAM}" \
  monitor \
  --output "${HOST_TELEMETRY}" \
  --interval-seconds "${HOST_TELEMETRY_INTERVAL_SECONDS}" \
  --npu-smi /usr/local/bin/npu-smi &
host_monitor_pid=$!
for _ in $(seq 1 100); do
  if [[ -s "${HOST_TELEMETRY}" ]]; then
    break
  fi
  if ! kill -0 "${host_monitor_pid}" 2>/dev/null; then
    echo "host telemetry monitor exited before its first sample" >&2
    exit 2
  fi
  sleep 0.1
done
if [[ ! -s "${HOST_TELEMETRY}" ]]; then
  echo "host telemetry monitor did not emit its first sample" >&2
  exit 2
fi

set +e
docker run --rm --network host --ipc host \
  --name "${CONTAINER_NAME}" \
  "${NPU_DOCKER_DEVICE_ARGS[@]}" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "${CACHE_ROOT}:/root/.cache/vllm" \
  -v "${SOURCE_REPO}:/workspace:ro" \
  -v "${CACHE_ROOT}/source-artifact-cache:/workspace/.cache" \
  -v "${AUTOPILOT_ROOT}:/autopilot:ro" \
  -v "${BUNDLE_DIR}:/bundle" \
  -w /workspace \
  -e ASCEND_RT_VISIBLE_DEVICES="${CONTAINER_VISIBLE_DEVICES}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/autopilot/src:/workspace/src:/workspace \
  --entrypoint bash \
  "${IMAGE}" \
  -lc \
  "python3 -m inference_autopilot.runners.chang_pressure execute \
    --launch-manifest /bundle/launch.json \
    --run-manifest /bundle/run-manifest.json \
    --source-repo /workspace \
    --source-config '/workspace/${SOURCE_CONFIG_RELATIVE}' \
    --data '/workspace/${DATA_RELATIVE}' \
    --native-result /bundle/native-result.json \
    --observation /bundle/observation.json \
    --expected-source-config-sha256 '${EXPECTED_CONFIG_SHA256}' \
    --expected-source-snapshot-sha256 '${EXPECTED_SNAPSHOT_SHA256}'" \
  2>&1 | tee "${BUNDLE_DIR}/runner.log" > "${RUNNER_STDOUT}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
runner_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
stop_host_monitor

set +e
"${HOST_TELEMETRY_PYTHON}" "${HOST_TELEMETRY_PROGRAM}" \
  assess "${HOST_TELEMETRY}" \
  --target-npu "${NPU_ID}" \
  --output "${HOST_INTERFERENCE_REPORT}" \
  --max-sample-gap-seconds "${HOST_TELEMETRY_MAX_SAMPLE_GAP_SECONDS}" \
  --max-host-cpu-busy-fraction "${HOST_MAX_CPU_BUSY_FRACTION}" \
  --max-host-iowait-fraction "${HOST_MAX_IOWAIT_FRACTION}" \
  --max-run-queue-per-cpu "${HOST_MAX_RUN_QUEUE_PER_CPU}" \
  --min-memory-available-fraction "${HOST_MIN_MEMORY_AVAILABLE_FRACTION}" \
  >/dev/null
host_assessment_status=$?
set -e
if [[ -f "${HOST_INTERFERENCE_REPORT}" ]]; then
  host_integrity_status="$(
    python3 -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
      "${HOST_INTERFERENCE_REPORT}"
  )"
else
  host_integrity_status="insufficient_telemetry"
fi

/usr/local/bin/npu-smi info -t proc-mem -i "${NPU_ID}" -c 0 \
  > "${BUNDLE_DIR}/npu-processes-after.txt" 2>&1 || true
final_mb="$(maximum_process_mb "${BUNDLE_DIR}/npu-processes-after.txt")"
run_outcome="$(
  classify_run_outcome \
    "${runner_status}" \
    "${BUNDLE_DIR}/native-result.json" \
    "${BUNDLE_DIR}/runner.log" \
    "${existing_mb}" \
    "${final_mb}" \
    "${MAX_EXISTING_PROCESS_MB}" \
    "${host_integrity_status}"
)"
{
  echo "run_outcome=${run_outcome}"
  echo "host_integrity_status=${host_integrity_status}"
  echo "host_assessment_exit_code=${host_assessment_status}"
  echo "final_process_mb=${final_mb}"
  echo "host_telemetry_sha256=$(sha256sum "${HOST_TELEMETRY}" | awk '{print $1}')"
  if [[ -f "${HOST_INTERFERENCE_REPORT}" ]]; then
    echo "host_interference_report_sha256=$(sha256sum "${HOST_INTERFERENCE_REPORT}" | awk '{print $1}')"
  else
    echo "host_interference_report_sha256=missing"
  fi
  echo "npu_processes_before_sha256=$(sha256sum "${NPU_PROCESSES_BEFORE}" | awk '{print $1}')"
  echo "npu_processes_after_sha256=$(sha256sum "${BUNDLE_DIR}/npu-processes-after.txt" | awk '{print $1}')"
  echo "runner_exit_code=${runner_status}"
  echo "tee_exit_code=${tee_status}"
  echo "runner_log_sha256=$(sha256sum "${BUNDLE_DIR}/runner.log" | awk '{print $1}')"
  if [[ -f "${BUNDLE_DIR}/observation.json" ]]; then
    echo "observation_sha256=$(sha256sum "${BUNDLE_DIR}/observation.json" | awk '{print $1}')"
  else
    echo "observation_sha256=missing"
  fi
} >> "${BUNDLE_DIR}/execution.meta"

if (( tee_status != 0 )); then
  exit "${tee_status}"
fi
if (( runner_status == 0 )) && [[ "${run_outcome}" != "success" ]]; then
  exit 3
fi
exit "${runner_status}"
