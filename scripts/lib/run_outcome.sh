#!/usr/bin/env bash

classify_run_outcome() {
  local runner_status="$1"
  local result_path="$2"
  local log_path="$3"
  local initial_process_mb="${4:-0}"
  local final_process_mb="${5:-0}"
  local maximum_existing_process_mb="${6:-1024}"
  local host_integrity_status="${7:-clean}"

  if [[ "${host_integrity_status}" != "clean" ]]; then
    echo environment_contaminated
  elif (( initial_process_mb <= maximum_existing_process_mb )) \
      && (( final_process_mb > maximum_existing_process_mb )); then
    echo environment_contaminated
  elif (( runner_status == 0 )) && [[ -f "${result_path}" ]]; then
    echo success
  elif grep -qiE \
      '(NPU out of memory|CUDA out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED|resource exhausted|Free memory on device .* is less than desired .* memory utilization)' \
      "${log_path}"; then
    echo resource_exhausted
  else
    echo failed
  fi
}
