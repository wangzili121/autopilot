# Harness Cost And Cache Evidence

## Why this is a separate objective

The deployment objective measures steady-state request throughput, latency and
quality. An autotuner has another cost: loading engines, compiling graphs,
capturing ACL/CUDA graphs and repeating that work for every isolated trial.
Mixing startup with request latency would corrupt the serving comparison, while
ignoring it can make a broad search operationally impractical.

`assess-harness-cost` creates a content-addressed cost artifact beside the
ordinary calibration assessment. It reads only the frozen plan and each run's:

- `run-manifest.json`;
- `effective-config.json`;
- `runner.log`.

The source files are hashed. Manifest-to-plan and effective-to-requested
configuration equality are mandatory; incomplete runs remain explicit issues.

## Engine fingerprint

Cache compatibility cannot be inferred from a directory name. Each base or
proposal engine receives a SHA256 fingerprint over:

- role and model identity/weight attestation;
- container, vLLM, vLLM-Ascend, driver and source snapshot identity;
- source configuration digest;
- role-specific scheduler, memory and graph settings;
- shared model-runner, prefix-cache and chunked-prefill semantics.

This is deliberately stricter than vLLM's visible short cache key. It prevents
the harness from treating artifacts produced by different software or model
states as interchangeable.

## Parsed phases

For each engine the assessor extracts model-weight loading, Dynamo transform,
device graph compilation, combined compile/profile/warmup, graph capture time
and memory, and total engine initialization. Missing optional phases are
represented as null; a missing total engine initialization record invalidates
that run for cost accounting.

Repeated observations of an identical engine fingerprint are classified as:

- `cache_key_reused_without_compile`;
- `cache_key_reused_compile_reduced`;
- `cache_key_reused_compile_repeated`;
- `cache_key_missing_or_changed`.

A stable path key is therefore evidence of key reuse, not proof that expensive
device work was skipped.

## First real campaign

The first prospective medium-2K campaign contained eight isolated runs and 16
engine starts. The audit measured:

| Cost | Seconds |
| --- | ---: |
| Total engine initialization | 700.59 |
| Compile/profile/warmup | 529.19 |
| Graph capture | 119.00 |
| Initialization after the first matching fingerprint | 582.73 |

All three repeated fingerprint groups reused one vLLM cache key but still
invoked compilation and graph capture. The existing persistent cache mount is
working at the key/path layer; it does not amortize process-local ACL Graph
state or eliminate the observed Ascend compile path.

This establishes the baseline for a later prewarmed engine-pool experiment.
That experiment must preserve ABBA ordering and reset request/KV/runtime state
between trials, so reduced calibration time is not purchased with cross-run
contamination.

Artifact:
`artifacts/cis-small-proposal-medium2k-p32-propbt16k-harness-cost-20260906-r1.json`

## Independent Replication

The second prospective campaign independently reproduced the startup-cost
profile across another eight isolated runs:

| Cost | Seconds |
| --- | ---: |
| Total engine initialization | 705.82 |
| Compile/profile/warmup | 528.96 |
| Graph capture | 120.00 |
| Initialization after the first matching fingerprint | 584.93 |

Again, all three repeated fingerprint groups were classified as
`cache_key_reused_compile_repeated`, with 13 repeated compiler invocations and
13 repeated graph captures. Across both campaigns, engine initialization cost
`1406.41s`, of which `1058.15s` was compile/profile/warmup and `239.00s` was
graph capture. This replicated evidence makes an isolated prewarmed engine pool
and graph-lifecycle reuse a concrete harness optimization target.

Artifact:
`artifacts/cis-small-proposal-medium2k-p32-propbt16k-harness-cost-20260906-r2.json`

## Lifecycle Plan Derived From Round 2

`plan-engine-lifecycle` converts the measured fingerprints and startup times
into an order-preserving lifecycle schedule. For the eight-run round-2 plan:

| Strategy | Starts | Peak reserved memory | Projected startup | Eligible |
| --- | ---: | ---: | ---: | --- |
| Isolated process | 16 | `0.90` | `705.82s` | yes |
| Role-sticky | 4 | `0.90` | `150.21s` | yes |
| Fully resident | 3 | `1.26` | `119.46s` | no |

The selected role-sticky strategy keeps the base engine resident and replaces
the proposal engine only when its content-addressed configuration fingerprint
changes. It preserves the exact frozen run order and projects `555.61s`, or
`78.72%`, less engine-startup time. The fully resident strategy is rejected
because the three unique engines would exceed the frozen `0.95` aggregate
memory-reservation limit.

The plan is deliberately `validation_required`, not executable evidence. Every
measurement epoch must assert zero running requests, clear prefix KV state,
rebuild algorithm batchers and score caches, reset request IDs and metric
snapshots, reseed the workload, and synchronize the device. Fresh isolated
versus pooled ABBA pairs must then prove token-exact output equality and absence
of cross-epoch state before pooled measurements can enter formal calibration.

Artifact:
`artifacts/cis-small-proposal-medium2k-p32-propbt16k-engine-lifecycle-plan-20260906.json`
