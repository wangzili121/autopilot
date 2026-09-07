#!/usr/bin/env bash
set -euo pipefail

AUTOPILOT_ROOT="${AUTOPILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_REPO="${SOURCE_REPO:?set SOURCE_REPO to the inference-scaling worktree}"
NPU_ID="${NPU_ID:-4}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
DTYPE="${DTYPE:-float16}"
REQUESTS="${REQUESTS:-4}"
WORKERS="${WORKERS:-4}"
ARRIVAL_QPS="${ARRIVAL_QPS:-0}"
RUN_SEED="${RUN_SEED:-}"
SUBSET_SEED="${SUBSET_SEED:-}"
PROMPT_PREFIX_TOKENS="${PROMPT_PREFIX_TOKENS:-0}"
BASE_MAX_NUM_BATCHED_TOKENS="${BASE_MAX_NUM_BATCHED_TOKENS:-}"
PROPOSAL_MAX_NUM_BATCHED_TOKENS="${PROPOSAL_MAX_NUM_BATCHED_TOKENS:-}"
PROPOSAL_STAGE_WAVEFRONT="${PROPOSAL_STAGE_WAVEFRONT:-off}"
PROPOSAL_GRAPH_CAPTURE_CEILING="${PROPOSAL_GRAPH_CAPTURE_CEILING:-}"
PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS="${PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS:-0.5}"
PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION="${PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION:-0.65}"
MODEL_RUNNER="${MODEL_RUNNER:-AUTO}"
VLLM_ASCEND_MRV2_COMPAT_PATCH="${VLLM_ASCEND_MRV2_COMPAT_PATCH:-}"
GRAPH_PHASE_METRICS="${GRAPH_PHASE_METRICS:-0}"
VLLM_CORE_GRAPH_PHASE_PATCH="${VLLM_CORE_GRAPH_PHASE_PATCH:-${AUTOPILOT_ROOT}/patches/vllm-v0.18-cudagraph-execution-phase.patch}"
VLLM_ASCEND_GRAPH_PHASE_PATCH="${VLLM_ASCEND_GRAPH_PHASE_PATCH:-${AUTOPILOT_ROOT}/patches/vllm-ascend-v0.18-cudagraph-execution-phase.patch}"
CONFIG="${CONFIG:-${AUTOPILOT_ROOT}/examples/npu/conditional-is-graph-metrics-smoke.toml}"
GRAPH_PLAN="${GRAPH_PLAN:-}"
GRAPH_RUN_ID="${GRAPH_RUN_ID:-}"
MAX_EXISTING_PROCESS_MB="${MAX_EXISTING_PROCESS_MB:-1024}"
RUN_ID="${RUN_ID:-graph-metrics-smoke-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUTOPILOT_ROOT}/artifacts/${RUN_ID}}"
CACHE_ROOT="${CACHE_ROOT:-${AUTOPILOT_ROOT}/.cache/vllm-npu${NPU_ID}}"
CONTAINER_NAME="inference-autopilot-${RUN_ID}-npu${NPU_ID}"
LOCK_DIR="/tmp/inference-autopilot-npu-${NPU_ID}.lock"

source "${AUTOPILOT_ROOT}/scripts/lib/run_outcome.sh"

source_snapshot_sha256() {
  (
    cd "${SOURCE_REPO}"
    find experiments/arllm src/inference_scaling -type f -name '*.py' -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

if [[ ! -f "${SOURCE_REPO}/experiments/arllm/run_small_proposal_pressure.py" ]]; then
  echo "missing pressure runner under SOURCE_REPO" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_REPO}/data/gsm8k/test.jsonl" ]]; then
  echo "missing GSM8K data under SOURCE_REPO" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing graph metrics config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! "${PROMPT_PREFIX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "PROMPT_PREFIX_TOKENS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${ARRIVAL_QPS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ARRIVAL_QPS must be a non-negative number" >&2
  exit 2
fi
case "${DTYPE}" in
  float16|bfloat16)
    ;;
  *)
    echo "DTYPE must be float16 or bfloat16" >&2
    exit 2
    ;;
esac
case "${GRAPH_PHASE_METRICS}" in
  0|1)
    ;;
  *)
    echo "GRAPH_PHASE_METRICS must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${PROPOSAL_STAGE_WAVEFRONT}" in
  off|auto)
    ;;
  *)
    echo "PROPOSAL_STAGE_WAVEFRONT must be off or auto" >&2
    exit 2
    ;;
esac
if [[ "${PROPOSAL_STAGE_WAVEFRONT}" == "auto" ]] \
    && [[ ! "${PROPOSAL_GRAPH_CAPTURE_CEILING}" =~ ^[1-9][0-9]*$ ]]; then
  echo "auto PROPOSAL_STAGE_WAVEFRONT requires a positive PROPOSAL_GRAPH_CAPTURE_CEILING" >&2
  exit 2
fi
for value_name in \
  PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS \
  PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${value_name} must be a non-negative number" >&2
    exit 2
  fi
done
for seed_name in RUN_SEED SUBSET_SEED; do
  seed_value="${!seed_name}"
  if [[ -n "${seed_value}" ]] && [[ ! "${seed_value}" =~ ^[0-9]+$ ]]; then
    echo "${seed_name} must be a non-negative integer" >&2
    exit 2
  fi
done
PROFILE_SEED_ENV_ARGS=()
if [[ -n "${RUN_SEED}" ]]; then
  PROFILE_SEED_ENV_ARGS+=(
    -e "INFERENCE_AUTOPILOT_RUN_SEED=${RUN_SEED}"
  )
fi
if [[ -n "${SUBSET_SEED}" ]]; then
  PROFILE_SEED_ENV_ARGS+=(
    -e "INFERENCE_AUTOPILOT_SUBSET_SEED=${SUBSET_SEED}"
  )
fi
MODEL_RUNNER_ENV_ARGS=()
case "${MODEL_RUNNER}" in
  AUTO)
    ;;
  MRV1)
    MODEL_RUNNER_ENV_ARGS=(-e VLLM_USE_V2_MODEL_RUNNER=0)
    ;;
  MRV2)
    MODEL_RUNNER_ENV_ARGS=(-e VLLM_USE_V2_MODEL_RUNNER=1)
    ;;
  *)
    echo "MODEL_RUNNER must be AUTO, MRV1, or MRV2" >&2
    exit 2
    ;;
esac
RUNTIME_PATCH_COMMAND=""
RUNTIME_PATCH_RELATIVE=""
if [[ -n "${VLLM_ASCEND_MRV2_COMPAT_PATCH}" ]]; then
  if [[ "${MODEL_RUNNER}" != "MRV2" ]]; then
    echo "VLLM_ASCEND_MRV2_COMPAT_PATCH requires MODEL_RUNNER=MRV2" >&2
    exit 2
  fi
  if [[ ! -f "${VLLM_ASCEND_MRV2_COMPAT_PATCH}" ]]; then
    echo "missing MRV2 compatibility patch: ${VLLM_ASCEND_MRV2_COMPAT_PATCH}" >&2
    exit 2
  fi
  RUNTIME_PATCH_RELATIVE="$(
    cd "${AUTOPILOT_ROOT}"
    realpath --relative-to="${AUTOPILOT_ROOT}" "${VLLM_ASCEND_MRV2_COMPAT_PATCH}"
  )"
  case "${RUNTIME_PATCH_RELATIVE}" in
    ../*|..)
      echo "VLLM_ASCEND_MRV2_COMPAT_PATCH must be inside AUTOPILOT_ROOT" >&2
      exit 2
      ;;
  esac
  RUNTIME_PATCH_COMMAND="patch --fuzz=0 -d /vllm-workspace/vllm-ascend -p1 --forward < '/autopilot/${RUNTIME_PATCH_RELATIVE}' &&"
fi
CORE_GRAPH_PHASE_PATCH_RELATIVE=""
ASCEND_GRAPH_PHASE_PATCH_RELATIVE=""
if [[ "${GRAPH_PHASE_METRICS}" == "1" ]]; then
  if [[ "${MODEL_RUNNER}" != "MRV1" ]]; then
    echo "GRAPH_PHASE_METRICS currently requires MODEL_RUNNER=MRV1" >&2
    exit 2
  fi
  for patch_path in \
    "${VLLM_CORE_GRAPH_PHASE_PATCH}" \
    "${VLLM_ASCEND_GRAPH_PHASE_PATCH}"; do
    if [[ ! -f "${patch_path}" ]]; then
      echo "missing graph phase metrics patch: ${patch_path}" >&2
      exit 2
    fi
  done
  CORE_GRAPH_PHASE_PATCH_RELATIVE="$(
    cd "${AUTOPILOT_ROOT}"
    realpath --relative-to="${AUTOPILOT_ROOT}" "${VLLM_CORE_GRAPH_PHASE_PATCH}"
  )"
  ASCEND_GRAPH_PHASE_PATCH_RELATIVE="$(
    cd "${AUTOPILOT_ROOT}"
    realpath --relative-to="${AUTOPILOT_ROOT}" "${VLLM_ASCEND_GRAPH_PHASE_PATCH}"
  )"
  for patch_relative in \
    "${CORE_GRAPH_PHASE_PATCH_RELATIVE}" \
    "${ASCEND_GRAPH_PHASE_PATCH_RELATIVE}"; do
    case "${patch_relative}" in
      ../*|..)
        echo "graph phase metrics patches must be inside AUTOPILOT_ROOT" >&2
        exit 2
        ;;
    esac
  done
  RUNTIME_PATCH_COMMAND="${RUNTIME_PATCH_COMMAND} git -C /vllm-workspace/vllm apply '/autopilot/${CORE_GRAPH_PHASE_PATCH_RELATIVE}' && git -C /vllm-workspace/vllm-ascend apply '/autopilot/${ASCEND_GRAPH_PHASE_PATCH_RELATIVE}' &&"
fi
BASE_BATCHED_TOKENS_ARG=""
if [[ -n "${BASE_MAX_NUM_BATCHED_TOKENS}" ]]; then
  if [[ ! "${BASE_MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BASE_MAX_NUM_BATCHED_TOKENS must be a positive integer" >&2
    exit 2
  fi
  BASE_BATCHED_TOKENS_ARG="--base-max-num-batched-tokens '${BASE_MAX_NUM_BATCHED_TOKENS}'"
fi
if [[ -n "${PROPOSAL_MAX_NUM_BATCHED_TOKENS}" ]] \
    && [[ ! "${PROPOSAL_MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROPOSAL_MAX_NUM_BATCHED_TOKENS must be a positive integer" >&2
  exit 2
fi
PROPOSAL_TOKEN_ENV_ARGS=()
if [[ -n "${PROPOSAL_MAX_NUM_BATCHED_TOKENS}" ]]; then
  PROPOSAL_TOKEN_ENV_ARGS=(
    -e "INFERENCE_AUTOPILOT_PROPOSAL_MAX_NUM_BATCHED_TOKENS=${PROPOSAL_MAX_NUM_BATCHED_TOKENS}"
  )
fi
STAGE_WAVEFRONT_ENV_ARGS=(
  -e "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT=${PROPOSAL_STAGE_WAVEFRONT}"
  -e "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS=${PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS}"
  -e "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION=${PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION}"
)
if [[ -n "${PROPOSAL_GRAPH_CAPTURE_CEILING}" ]]; then
  STAGE_WAVEFRONT_ENV_ARGS+=(
    -e "INFERENCE_AUTOPILOT_PROPOSAL_GRAPH_CAPTURE_CEILING=${PROPOSAL_GRAPH_CAPTURE_CEILING}"
  )
fi
case "$(cd "${AUTOPILOT_ROOT}" && realpath --relative-to="${AUTOPILOT_ROOT}" "${CONFIG}")" in
  ../*|..)
    echo "CONFIG must be inside AUTOPILOT_ROOT for the read-only container mount" >&2
    exit 2
    ;;
esac
CONFIG_RELATIVE="$(
  cd "${AUTOPILOT_ROOT}"
  realpath --relative-to="${AUTOPILOT_ROOT}" "${CONFIG}"
)"
GRAPH_ENV_ARGS=()
GRAPH_PLAN_RELATIVE=""
if [[ -n "${GRAPH_PLAN}" || -n "${GRAPH_RUN_ID}" ]]; then
  if [[ -z "${GRAPH_PLAN}" || -z "${GRAPH_RUN_ID}" ]]; then
    echo "GRAPH_PLAN and GRAPH_RUN_ID must be provided together" >&2
    exit 2
  fi
  if [[ ! -f "${GRAPH_PLAN}" ]]; then
    echo "missing graph experiment plan: ${GRAPH_PLAN}" >&2
    exit 2
  fi
  GRAPH_PLAN_RELATIVE="$(
    cd "${AUTOPILOT_ROOT}"
    realpath --relative-to="${AUTOPILOT_ROOT}" "${GRAPH_PLAN}"
  )"
  case "${GRAPH_PLAN_RELATIVE}" in
    ../*|..)
      echo "GRAPH_PLAN must be inside AUTOPILOT_ROOT" >&2
      exit 2
      ;;
  esac
  GRAPH_ENV_ARGS=(
    -e "INFERENCE_AUTOPILOT_GRAPH_PLAN=/autopilot/${GRAPH_PLAN_RELATIVE}"
    -e "INFERENCE_AUTOPILOT_GRAPH_RUN_ID=${GRAPH_RUN_ID}"
  )
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "NPU ${NPU_ID} is already locked by another Autopilot run" >&2
  exit 2
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

existing_process_mb() {
  /usr/local/bin/npu-smi info -t proc-mem -i "${NPU_ID}" -c 0 2>/dev/null \
    | awk '
        /Process memory\(MB\):/ {
          value = $0
          sub(/^.*Process memory\(MB\):[[:space:]]*/, "", value)
          sub(/[[:space:]].*$/, "", value)
          if ((value + 0) > maximum) maximum = value + 0
        }
        END { print maximum + 0 }
      '
}

existing_mb="$(existing_process_mb)"
if (( existing_mb > MAX_EXISTING_PROCESS_MB )); then
  echo "refusing NPU ${NPU_ID}: existing process uses ${existing_mb} MB" >&2
  exit 2
fi

if [[ -d "${OUTPUT_DIR}" ]] \
    && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to reuse non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}" "${CACHE_ROOT}/source-artifact-cache"
{
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "npu_id=${NPU_ID}"
  echo "existing_process_mb=${existing_mb}"
  echo "source_repo=${SOURCE_REPO}"
  echo "source_revision=$(git -C "${SOURCE_REPO}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "source_snapshot_sha256=$(source_snapshot_sha256)"
  echo "image=${IMAGE}"
  echo "dtype=${DTYPE}"
  echo "requests=${REQUESTS}"
  echo "workers=${WORKERS}"
  echo "arrival_qps=${ARRIVAL_QPS}"
  echo "run_seed=${RUN_SEED:-config-default}"
  echo "subset_seed=${SUBSET_SEED:-config-default}"
  echo "prompt_prefix_tokens=${PROMPT_PREFIX_TOKENS}"
  echo "base_max_num_batched_tokens=${BASE_MAX_NUM_BATCHED_TOKENS:-config-default}"
  echo "proposal_max_num_batched_tokens=${PROPOSAL_MAX_NUM_BATCHED_TOKENS:-config-default}"
  echo "proposal_stage_wavefront=${PROPOSAL_STAGE_WAVEFRONT}"
  echo "proposal_graph_capture_ceiling=${PROPOSAL_GRAPH_CAPTURE_CEILING:-none}"
  echo "proposal_stage_wavefront_max_wait_seconds=${PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS}"
  echo "proposal_stage_wavefront_min_utilization=${PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION}"
  echo "model_runner=${MODEL_RUNNER}"
  echo "graph_phase_metrics=${GRAPH_PHASE_METRICS}"
  echo "vllm_core_graph_phase_patch=${CORE_GRAPH_PHASE_PATCH_RELATIVE:-none}"
  echo "vllm_ascend_graph_phase_patch=${ASCEND_GRAPH_PHASE_PATCH_RELATIVE:-none}"
  if [[ "${GRAPH_PHASE_METRICS}" == "1" ]]; then
    echo "vllm_core_graph_phase_patch_sha256=$(sha256sum "${VLLM_CORE_GRAPH_PHASE_PATCH}" | awk '{print $1}')"
    echo "vllm_ascend_graph_phase_patch_sha256=$(sha256sum "${VLLM_ASCEND_GRAPH_PHASE_PATCH}" | awk '{print $1}')"
  else
    echo "vllm_core_graph_phase_patch_sha256=none"
    echo "vllm_ascend_graph_phase_patch_sha256=none"
  fi
  echo "mrv2_compat_patch=${RUNTIME_PATCH_RELATIVE:-none}"
  if [[ -n "${RUNTIME_PATCH_RELATIVE}" ]]; then
    echo "mrv2_compat_patch_sha256=$(sha256sum "${VLLM_ASCEND_MRV2_COMPAT_PATCH}" | awk '{print $1}')"
  else
    echo "mrv2_compat_patch_sha256=none"
  fi
  echo "config=${CONFIG_RELATIVE}"
  echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
  echo "dataset_sha256=$(sha256sum "${SOURCE_REPO}/data/gsm8k/test.jsonl" | awk '{print $1}')"
  echo "wrapper_sha256=$(sha256sum "${AUTOPILOT_ROOT}/scripts/run_chang_graph_profile.py" | awk '{print $1}')"
  echo "graph_plan=${GRAPH_PLAN_RELATIVE:-none}"
  echo "graph_run_id=${GRAPH_RUN_ID:-none}"
  date --iso-8601=seconds
} > "${OUTPUT_DIR}/run.meta"

set +e
docker run --rm --network host --ipc host --privileged \
  --security-opt label=disable \
  --name "${CONTAINER_NAME}" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "${CACHE_ROOT}:/root/.cache/vllm" \
  -v "${SOURCE_REPO}:/workspace:ro" \
  -v "${CACHE_ROOT}/source-artifact-cache:/workspace/.cache" \
  -v "${AUTOPILOT_ROOT}:/autopilot:ro" \
  -v "${OUTPUT_DIR}:/artifacts" \
  -w /workspace \
  -e ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/autopilot/src:/workspace/src:/workspace \
  -e INFERENCE_AUTOPILOT_PROMPT_PREFIX_TOKENS="${PROMPT_PREFIX_TOKENS}" \
  "${PROPOSAL_TOKEN_ENV_ARGS[@]}" \
  "${STAGE_WAVEFRONT_ENV_ARGS[@]}" \
  "${MODEL_RUNNER_ENV_ARGS[@]}" \
  "${PROFILE_SEED_ENV_ARGS[@]}" \
  "${GRAPH_ENV_ARGS[@]}" \
  --entrypoint bash \
  "${IMAGE}" -lc \
  "${RUNTIME_PATCH_COMMAND} python /autopilot/scripts/run_chang_graph_profile.py \
    --config '/autopilot/${CONFIG_RELATIVE}' \
    --data /workspace/data/gsm8k/test.jsonl \
    --dtype '${DTYPE}' \
    --requests '${REQUESTS}' \
    --arrival-qps '${ARRIVAL_QPS}' \
    --workers '${WORKERS}' \
    ${BASE_BATCHED_TOKENS_ARG} \
    --output /artifacts/result.json" \
  2>&1 | tee "${OUTPUT_DIR}/runner.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
runner_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"

/usr/local/bin/npu-smi info -t proc-mem -i "${NPU_ID}" -c 0 \
  > "${OUTPUT_DIR}/npu-processes-after.txt" 2>&1 || true

run_outcome="$(
  classify_run_outcome \
    "${runner_status}" \
    "${OUTPUT_DIR}/result.json" \
    "${OUTPUT_DIR}/runner.log"
)"

{
  echo "run_outcome=${run_outcome}"
  echo "runner_exit_code=${runner_status}"
  echo "tee_exit_code=${tee_status}"
  echo "runner_log_sha256=$(sha256sum "${OUTPUT_DIR}/runner.log" | awk '{print $1}')"
  if [[ -f "${OUTPUT_DIR}/result.json" ]]; then
    echo "result_sha256=$(sha256sum "${OUTPUT_DIR}/result.json" | awk '{print $1}')"
  else
    echo "result_sha256=missing"
  fi
} >> "${OUTPUT_DIR}/run.meta"

if (( tee_status != 0 )); then
  exit "${tee_status}"
fi
if (( runner_status == 0 )) && [[ "${run_outcome}" != "success" ]]; then
  exit 3
fi
exit "${runner_status}"
