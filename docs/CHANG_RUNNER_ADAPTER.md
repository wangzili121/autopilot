# Chang Runner Adapter

## Scope

`chang-pressure-v1` executes the real AR
`conditional_is_small_proposal` graph: base-model candidates, proposal-model
rollouts, base-model target scoring, importance correction and candidate
selection. It drives chang's own algorithm and backend APIs; it does not
reimplement Conditional IS or vLLM scheduling.

The adapter does not use `benchmark_rollout_infra.py`. That benchmark's
`conditional_fixed` arm creates only a base backend, so its graph is not the
1.5B/0.5B small-proposal path. It also does not use the unmodified async-v5
report as the formal result because that report has aggregate wall time but no
per-request P95/P99 latency.

## Dry Run

Generate the calibration plan, then prepare runs in ascending `sequence_index`:

```bash
inference-autopilot prepare-run calibration-plan.json RUN_ID \
  --source-repo /path/to/inference_scaling \
  --source-config configs/gsm8k_full.toml \
  --data data/gsm8k/test.jsonl \
  --output-dir artifacts/RUN_ID
```

This step does not import torch, load a model or reserve an NPU. It produces:

- `run-manifest.json`: immutable experiment contract;
- `effective-config.json`: explicit semantic, workload and deployment mapping;
- `launch.json`: argv, paths, source and plugin hashes, and compatibility checks.

Use `--require-formal` in automation. It returns nonzero when any required
static check fails, while still writing the bundle for diagnosis.

The generated worker refuses a non-formal launch manifest unless it is invoked
manually with `--allow-nonformal`. That escape hatch is for integration
debugging only; its failed observation is not eligible for formal assessment.

Checks include the exact algorithm ID, dual-engine small-proposal source path,
all deployment knobs, memory split, source-config semantic invariants, dataset
and deterministic arrival hashes, model paths, environment attestation, and
runtime support for base scoring priority. The worker rechecks the source config,
chang implementation snapshot, and complete `inference_autopilot` Python package
snapshot before loading a model, so either repository drifting after preparation
cannot silently enter a pair.

## NPU Execution

When a card is available, prepare the bundle on the machine that owns the
actual source, data and model paths. Install Inference Autopilot in that Python
environment and execute the `command` array from `launch.json` with the bundle's
`working_directory`.

The worker applies only deployment settings from the manifest. It verifies and
keeps candidate count, rollout count, block size, total length, importance
correction, APC and chunked prefill fixed. The paired workload seed replaces
only `run.seed`; `subset_seed` remains fixed so paired runs use the same GSM8K
problems.

Graph calibration may additionally provide all four role-specific settings:
`base_graph_mode`, `base_graph_capture_sizes`, `proposal_graph_mode`, and
`proposal_graph_capture_sizes`. They are optional as a group, validated during
preflight, injected into each role's vLLM compilation config, and echoed in the
manifest-bound native runtime settings. Partial policies, unsorted/duplicate
buckets, empty enabled policies and buckets attached to `NONE` are rejected.

Proposal stage-wavefront admission may be selected with a complete four-setting
group: mode, graph capture ceiling, maximum tail wait, and minimum utilization.
`auto` additionally requires an explicit proposal graph policy whose largest
capture size equals the declared ceiling. `prefix_sharded` also requires shard
sequence and prefill-token caps, a per-parent fairness cap, and an in-flight wave
cap. It preserves
contiguous repeated-prefix rollout runs, request-local seeds, output order, and
the original parent completion barrier while permitting shards from independent
parents to share a wave. The worker records the offline plan and every realized
two-dimensional wave in the native result. Formal sharded results fail closed on
missing telemetry, cancelled shards, oversized atomic requests, incomplete
parent barriers, or fairness violations.

`model_runner` may select `MRV1` or `MRV2`. The worker maps it to
`VLLM_USE_V2_MODEL_RUNNER` before importing the source runtime. This setting is
still capability-gated: the checked-in Ascend 0.18 profile permits MRV1 only.
The NPU smoke launcher can apply an explicitly named, hashed compatibility
patch for diagnostics, but patched runs do not establish stock-image support.

Successful execution writes `native-result.json` and `observation.json`.
Crashes and caught OOMs write a failed observation rather than fabricated
performance values. A process-level kill can still prevent finalization and
must be recorded by the outer job controller.

For a complete ordered campaign, use the campaign wrapper instead of invoking
four bundles manually:

```bash
CALIBRATION_SPEC=/autopilot/calibration.json \
SOURCE_REPO=/data/inference_scaling \
SOURCE_CONFIG=/data/inference_scaling/configs/gsm8k.toml \
NPU_ID=6 \
REPLAY_NOISE_ASSESSMENT=/autopilot/replay/assessment.json \
scripts/run_npu_calibration_campaign.sh
```

It creates the plan, prepares each formal bundle as the host user, executes in
manifest order, collects observations, and assesses the campaign. It owns a
campaign-level card lock, while every individual run rechecks process memory
before model loading. The physical NPU ID and container image must exactly match
the environment contract in every manifest. Existing campaign directories are
never overwritten. A run-level failure with a valid observation remains in the
protocol; a launch that emits no observation stops the campaign.

## Recorded Signals

The native result includes:

- completed QPS and P50/P95/P99 request latency;
- queue wait and service time;
- exact-answer accuracy and request outputs;
- base/proposal continuous-batching occupancy;
- running/waiting requests, KV peak and preemptions when exposed by vLLM;
- base/proposal token-slot and FLOP accounting;
- request and model-call timelines;
- stage-wavefront plans, realized waves and limit violations when enabled;
- source config, dataset, manifest, chang implementation and plugin binding.

`observe-run` can repeat the strict conversion independently:

```bash
inference-autopilot observe-run run-manifest.json native-result.json \
  --output observation.json
```

The converter refuses a result whose manifest digest, runtime settings,
workload parameters, semantic invariants or algorithm ID differ from the
planned run.

## Current Local Audit

The checkout at `/Users/li/Desktop/xchang_original/inference_scaling` is useful
for API compatibility but is not formal-run ready. The dry-run confirms its
dual-engine small-proposal path and semantic config, but finds:

- no local model directories or GSM8K data file;
- no `score_priority` runtime interface required by the retained baseline;
- no `runtime_metrics` interface for formal KV/preemption observations;
- a dirty source checkout whose revision cannot by itself attest the code used;
- placeholder graph, dataset, arrival-trace and environment attestations in the
  example calibration spec.

Those are explicit prerequisites for the remote NPU campaign, not reasons to
weaken the baseline.
