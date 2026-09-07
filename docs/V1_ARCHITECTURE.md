# Inference Autopilot v1 Architecture

## Objective

Version 1 optimizes a fixed, exact inference-scaling algorithm under a frozen
workload and environment contract. It must find a deployment policy with less
device experimentation than an exhaustive sweep and retain enough provenance
to reproduce or reject every performance claim.

The first adapter is chang's Conditional IS small-proposal path. The optimizer
does not own vLLM token scheduling. Its optional stage-wavefront backend is a
local graph-operation admission policy: it packs ready proposal calls under
attested engine limits, while vLLM continues to schedule their tokens.

## Contracts

```text
Algorithm Adapter
  -> Inference Graph IR
  -> semantic invariants and stage-supported runtime fields

Workload + Environment
  -> immutable calibration manifests
  -> stage traces and telemetry

Search Space Compiler
  -> feasible, content-addressed deployment candidates

Graph Capture Planner + later Cost Model
  -> resource-constrained policy components

Calibration Harness
  -> paired observations and graded evidence

Interaction Repair Planner
  -> evidence-gated cross-layer diagnosis
  -> coupled follow-up calibration candidate

Policy Bundle
  -> selected static configuration, graph plan and guarded hot policy
```

Each artifact is strict and content-addressed. Unknown fields fail closed so a
new framework default cannot silently enter an old experiment cohort.

## Parameter layers

Every search knob has two independent classifications. `change_scope` states
when a value can change; `tuning_layer` states which optimizer owns it.

| Layer | Examples | Permitted change scopes | v1 treatment |
| --- | --- | --- | --- |
| `static_deployment` | max sequences, token budget, memory fraction, TP/DP | deployment, engine restart | offline candidate search |
| `graph_capture` | graph mode, capture sizes, graph memory budget | deployment, engine restart | dedicated planner |
| `hot_policy` | score priority, batch wait, stage-wavefront admission | policy boundary, per request | isolated paired studies before selector use |
| `algorithm_semantics` | candidates, rollouts, block size, correction mode | any explicit boundary | separate semantic cohorts only |

`applies_to_stages` binds a knob to the graph. When a graph is supplied to the
compiler, every binding must name a real stage and that stage must declare the
knob in `tunable_runtime_fields`.

## Conditional IS stage ownership

| Stage | Resource | Primary pressure | Initial optimizer inputs |
| --- | --- | --- | --- |
| `candidate_generate` | base model | candidate width and short decode batches | base capacity, graph shapes |
| `proposal_rollout_generate` | proposal model | `C * R` width and long suffixes | proposal capacity, graph shapes |
| `target_score` | base model | rollout-token teacher forcing | base token budget, score priority, graph shapes |
| `reward_evaluate` | CPU | completed trajectory count | telemetry only in v1 |
| `importance_reduce` | CPU/device candidate | segmented weight reductions | profile gate for a later plugin |
| `candidate_select` | CPU/device candidate | one categorical decision | retained sampler already makes this low priority |

Dependencies express data readiness, not a demand to serialize physical model
execution. The serving engines remain responsible for batching ready work.

## Optimization order

1. Validate algorithm, workload, environment and source bindings.
2. Compile hard feasibility constraints before any model execution.
3. Plan graph captures from stage-shape observations and measured costs.
4. Select a small deployment design using historical anchors and boundaries.
5. Collect paired calibration evidence and fit a stage-aware cost residual.
6. Route measured discontinuities to cross-layer repair planners instead of
   extrapolating the scalar response model through them.
7. Select one policy under throughput, latency, memory and semantic constraints.
8. Validate on held-out workload regimes before enabling runtime switching.

The first feedback rule covers scheduler capacity and graph capture coverage.
A complete paired regression outside replay noise is eligible only when runtime
telemetry also proves that realized concurrency crossed the affected engine's
capture ceiling. The repair candidate, failed source cell, active baseline and
all supporting evidence remain content-addressed.

## Non-goals for v1

- replacing MARS program admission or PIC-KV offloading;
- implementing another generic request scheduler;
- searching algorithm-changing knobs in the deployment cohort;
- treating simulation or synthetic graph profiles as performance evidence;
- claiming online adaptation before the offline selector beats the best manual
  configuration on held-out workloads.

## Milestone gates

| Gate | Required evidence |
| --- | --- |
| M1 contracts | round-trip schemas, stage-binding failures, deterministic hashes |
| M2 graph planner | exactness against exhaustive search, resource constraints, all-eager fallback |
| M3 device calibration | same-card stage profiles for eager and graph replay, memory and capture cost |
| M4 selector | fewer measurements than sweep, crash-aware feasibility, held-out prediction error |
| M5 product | reproducible policy bundle, backend export and explicit rollback configuration |

M1 is implemented. M2 has an exact offline planner plus real vLLM-Ascend
runtime-shape import and an evidence-gated capacity/graph interaction repair.
The M3 collection path flushes graph statistics from chang's library-mode
`AsyncLLM` without modifying chang's repository or vLLM. The M4 selector,
active acquisition, paired-regression guard and independent holdout assessment
contracts are implemented; promotion remains blocked until formal calibration
supplies grade-A cross-regime evidence.
