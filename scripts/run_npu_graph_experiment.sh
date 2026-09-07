#!/usr/bin/env bash
set -euo pipefail

AUTOPILOT_ROOT="${AUTOPILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_REPO="${SOURCE_REPO:?set SOURCE_REPO to the inference-scaling worktree}"
GRAPH_PLAN="${GRAPH_PLAN:?set GRAPH_PLAN to a graph experiment plan}"
COMPARISON_ID="${COMPARISON_ID:-}"
NPU_ID="${NPU_ID:-4}"
REQUESTS="${REQUESTS:-4}"
WORKERS="${WORKERS:-4}"
ARRIVAL_QPS="${ARRIVAL_QPS:-0}"
PROMPT_PREFIX_TOKENS="${PROMPT_PREFIX_TOKENS:-0}"
BASE_MAX_NUM_BATCHED_TOKENS="${BASE_MAX_NUM_BATCHED_TOKENS:-}"
CONFIG="${CONFIG:-${AUTOPILOT_ROOT}/examples/npu/conditional-is-graph-metrics-smoke.toml}"

if [[ ! -f "${GRAPH_PLAN}" ]]; then
  echo "missing graph experiment plan: ${GRAPH_PLAN}" >&2
  exit 2
fi

run_id_output="$(
  python3 - "${GRAPH_PLAN}" "${COMPARISON_ID}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw = json.loads(Path(sys.argv[1]).read_text())
expected_digest = raw.pop("graph_experiment_plan_sha256", "")
serialized = json.dumps(
    raw, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
)
actual_digest = hashlib.sha256(serialized.encode()).hexdigest()
if actual_digest != expected_digest:
    raise SystemExit("graph experiment plan SHA256 does not match its content")
comparison_id = sys.argv[2]
matched = [
    run["run_id"]
    for run in raw["runs"]
    if not comparison_id or run["comparison_id"] == comparison_id
]
if not matched:
    raise SystemExit(f"no runs match comparison id: {comparison_id}")
print("\n".join(matched))
PY
)"
RUN_IDS=()
while IFS= read -r graph_run_id; do
  [[ -n "${graph_run_id}" ]] && RUN_IDS+=("${graph_run_id}")
done <<< "${run_id_output}"
if (( ${#RUN_IDS[@]} == 0 )); then
  echo "graph experiment plan selected no runs" >&2
  exit 2
fi

for graph_run_id in "${RUN_IDS[@]}"; do
  output_dir="${AUTOPILOT_ROOT}/artifacts/${graph_run_id}"
  if [[ -f "${output_dir}/result.json" ]]; then
    if python3 - "${output_dir}/result.json" "${graph_run_id}" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
actual = result.get("graph_experiment", {}).get("run", {}).get("run_id")
raise SystemExit(0 if actual == sys.argv[2] else 1)
PY
    then
      echo "skipping completed graph experiment run: ${graph_run_id}"
      continue
    fi
    echo "refusing mismatched existing result: ${output_dir}/result.json" >&2
    exit 2
  fi

  echo "starting graph experiment run: ${graph_run_id}"
  AUTOPILOT_ROOT="${AUTOPILOT_ROOT}" \
    SOURCE_REPO="${SOURCE_REPO}" \
    GRAPH_PLAN="${GRAPH_PLAN}" \
    GRAPH_RUN_ID="${graph_run_id}" \
    NPU_ID="${NPU_ID}" \
    REQUESTS="${REQUESTS}" \
    WORKERS="${WORKERS}" \
    ARRIVAL_QPS="${ARRIVAL_QPS}" \
    PROMPT_PREFIX_TOKENS="${PROMPT_PREFIX_TOKENS}" \
    BASE_MAX_NUM_BATCHED_TOKENS="${BASE_MAX_NUM_BATCHED_TOKENS}" \
    CONFIG="${CONFIG}" \
    RUN_ID="${graph_run_id}" \
    OUTPUT_DIR="${output_dir}" \
    bash "${AUTOPILOT_ROOT}/scripts/run_npu_graph_metrics_smoke.sh"
done
