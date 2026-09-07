#!/usr/bin/env bash
set -euo pipefail

AUTOPILOT_ROOT="${AUTOPILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NPU_ID="${NPU_ID:-4}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
POSITIONS="${POSITIONS:-512}"
VOCAB_SIZE="${VOCAB_SIZE:-151936}"
TOP_K="${TOP_K:-5}"
TOKEN_CHUNK_SIZES="${TOKEN_CHUNK_SIZES:-64,128,256}"
VOCAB_TILE_SIZES="${VOCAB_TILE_SIZES:-2048,4096,8192}"
REPEATS="${REPEATS:-1}"
MAX_EXISTING_PROCESS_MB="${MAX_EXISTING_PROCESS_MB:-1024}"
RUN_ID="${RUN_ID:-score-reduction-probe-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUTOPILOT_ROOT}/artifacts/${RUN_ID}}"
CONTAINER_NAME="inference-autopilot-${RUN_ID}-npu${NPU_ID}"
LOCK_DIR="/tmp/inference-autopilot-npu-${NPU_ID}.lock"

source "${AUTOPILOT_ROOT}/scripts/lib/run_outcome.sh"

for value in "${NPU_ID}" "${POSITIONS}" "${VOCAB_SIZE}" "${TOP_K}" "${REPEATS}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "numeric probe settings must be non-negative integers" >&2
    exit 2
  fi
done
if (( POSITIONS == 0 || VOCAB_SIZE == 0 || REPEATS == 0 )); then
  echo "POSITIONS, VOCAB_SIZE, and REPEATS must be positive" >&2
  exit 2
fi
if (( TOP_K > VOCAB_SIZE )); then
  echo "TOP_K cannot exceed VOCAB_SIZE" >&2
  exit 2
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
mkdir -p "${OUTPUT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "npu_id=${NPU_ID}"
  echo "existing_process_mb=${existing_mb}"
  echo "image=${IMAGE}"
  echo "positions=${POSITIONS}"
  echo "vocab_size=${VOCAB_SIZE}"
  echo "top_k=${TOP_K}"
  echo "token_chunk_sizes=${TOKEN_CHUNK_SIZES}"
  echo "vocab_tile_sizes=${VOCAB_TILE_SIZES}"
  echo "repeats=${REPEATS}"
  echo "probe_sha256=$(sha256sum "${AUTOPILOT_ROOT}/scripts/benchmark_npu_score_reduction.py" | awk '{print $1}')"
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
  -v "${AUTOPILOT_ROOT}:/autopilot:ro" \
  -v "${OUTPUT_DIR}:/artifacts" \
  -e ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python \
  "${IMAGE}" /autopilot/scripts/benchmark_npu_score_reduction.py \
    --positions "${POSITIONS}" \
    --vocab-size "${VOCAB_SIZE}" \
    --top-k "${TOP_K}" \
    --token-chunk-sizes "${TOKEN_CHUNK_SIZES}" \
    --vocab-tile-sizes "${VOCAB_TILE_SIZES}" \
    --repeats "${REPEATS}" \
    --output /artifacts/result.json \
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
  fi
} >> "${OUTPUT_DIR}/run.meta"

exit "${runner_status}"
