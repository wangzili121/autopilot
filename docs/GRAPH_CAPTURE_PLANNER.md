# Workload-Aware Graph Capture Planner

## Problem

CUDA and ACL graph backends expose graph modes and capture sizes, but a static
default bucket sequence cannot know the shape distribution produced by an
inference-scaling algorithm. Conditional IS also sends different work to its
base and proposal engines: short candidate generation, wide proposal rollout
and rollout-token target scoring should not be collapsed into one average
request length.

The planner selects capture sizes for one engine role and graph mode from a
stage-labelled profile. It is a policy generator, not a graph implementation.

## Input contract

Each observed `(stage_id, shape_size)` supplies:

- execution count in one representative profile window;
- measured eager latency per execution.

Each candidate capture size supplies:

- measured replay latency for every stage in the profile;
- incremental graph memory;
- one-time capture latency.

Budgets bound bucket count, total graph memory, total capture time, per-event
padding ratio and optional minimum graph hit rate. The objective can amortize
capture time over multiple workload windows and add explicit padding and memory
prices. Zero prices are valid when those quantities are hard constraints only.

## Assignment rule

For a selected sorted bucket set, an event uses the smallest bucket at least as
large as its shape. It falls back to eager execution when no such bucket exists
or its padding ratio exceeds the configured limit.

The reported runtime objective is:

```text
sum(event_count * selected_execution_latency)
+ padding_units * padding_penalty
+ total_capture_time / amortization_windows
+ graph_memory_gib * memory_penalty
```

The output separately reports runtime, objective, hit rate, padding, graph
memory and startup time so an artificial penalty cannot be mistaken for a
measured latency.

## Solver

Capture sizes and observations are ordered by shape. Selecting a bucket assigns
the still-unassigned interval above the previous bucket and through the new
bucket. The implementation uses dynamic programming over the last selected
bucket and bucket count. For each state it retains the Pareto frontier of:

- graph memory;
- capture time;
- objective value;
- captured execution count.

Dominated labels cannot lead to a better feasible continuation and are removed.
This avoids enumerating every subset while retaining exact solutions for the
additive profile model. Tests compare the result with exhaustive enumeration on
a small instance.

All-eager execution is an explicit candidate when minimum hit rate is zero. A
graph plan is therefore never selected solely because graph execution is
available.

## Commands

```bash
inference-autopilot trace-graph-workload result.json \
  --trace-id conditional-is-short-p96 \
  --output graph-trace.json

inference-autopilot audit-graph-trace graph-trace.json

inference-autopilot import-vllm-graph-metrics runner.log \
  --profile-id conditional-is-short-p96 \
  --output vllm-graph-metrics.json

inference-autopilot audit-vllm-graph-metrics vllm-graph-metrics.json

inference-autopilot diagnose-vllm-graph-runtime vllm-graph-metrics.json

inference-autopilot merge-vllm-graph-metrics \
  short-profile.json mixed-profile.json long-profile.json \
  --profile-id graph-training-corpus \
  --output merged-graph-metrics.json

inference-autopilot plan-vllm-graph-experiment vllm-graph-metrics.json \
  --campaign-id conditional-is-graph-abba \
  --blocks 1 \
  --pair-seeds 2026090401 2026090402 \
  --include-replay-control \
  --output graph-experiment-plan.json

inference-autopilot audit-vllm-graph-experiment graph-experiment-plan.json

inference-autopilot assess-vllm-graph-experiment \
  graph-experiment-plan.json artifacts/ \
  --comparison-id conditional-is-graph-abba--replay-control \
  --output graph-assessment.json

inference-autopilot evaluate-vllm-graph-policy \
  graph-experiment-plan.json holdout-profile.json \
  --policy-id trace-preserving-pruned \
  --output holdout-coverage.json

inference-autopilot plan-graph-capture \
  examples/conditional-is-base.acl-graph-profile.example.json \
  --output capture-plan.json

inference-autopilot audit-graph-capture capture-plan.json
```

The repository example is synthetic and only tests the contract and optimizer.
Real profiles must bind the exact model pair, vLLM/vLLM-Ascend commits, graph
mode, dtype, algorithm budget, workload regime and device identifier in the
surrounding calibration manifest.

`trace-graph-workload` maps base sampling, base scoring and proposal sampling to
the `candidate_generate`, `target_score` and `proposal_rollout_generate` graph
stages. It
preserves each backend call's request-group count and wall service time and
attests the source result with SHA256. Its request-group boundary summary uses
only observed endpoints, high-frequency widths and weighted quantiles.

The trace deliberately labels durations as backend-call wall service time,
including queueing. They describe workload pressure but are not accepted as
eager or graph replay latency. Those costs must come from controlled graph
calibration arms; this prevents concurrent Conditional IS calls from being
misrepresented as kernel microbenchmarks.

Request-group width is also not a vLLM graph shape. Async continuous batching
can merge multiple simultaneous API calls, so the actual graph input must be
read from vLLM's `num_unpadded_tokens` and `num_padded_tokens` statistics. The
request-group boundaries guide workload construction only; the planner never
uses them as capture sizes directly.

Training profiles may be merged only when engine roles, graph modes and
framework-configured buckets match exactly. Event counts are aggregated by
unpadded shape, padded shape and runtime mode; duplicate source profiles are
rejected. The resulting trace-preserving policy retains the union of buckets
used across the training regimes.

The promotion path adds a regime-labelled corpus above this mechanical merge.
Each source profile belongs to exactly one `train` or `holdout` regime, the same
regime identifier cannot leak across splits, and the same source log cannot be
reused under another label. The corpus and all embedded profiles are
content-addressed. `plan-vllm-robust-graph-experiment` derives buckets only from
the train split, checks every held-out graph event, and refuses to emit a run
plan when the minimum regime counts are unmet, any holdout event changes its
bucket, or the candidate is identical to the framework default.

Repeated random seeds should share one regime identifier and increase its
profile count, not its regime count. The regime declaration is content-bound
but remains diagnostic until the surrounding formal workload manifest attests
its load, context and arrival fields. The gate also turns a negative result into
a useful search decision: when the cross-regime union uses every configured
bucket, exact pruning is exhausted and the next candidate must come from
measured cost-aware remapping, another graph mode, or a different deployment
knob.

The bounded-remapping search is the next, explicitly non-exact tier. For each
engine role it treats ordered buckets as a resource-constrained shortest-path
problem. An edge assigns one interval of observed shapes to its next retained
bucket; its per-profile resources are remapped events and candidate padding.
The solver keeps only non-dominated resource labels for the same endpoint and
bucket count. This Pareto dynamic program exactly represents the additive
subset problem without materializing all `2^N` subsets, which is essential for
vLLM's 51-bucket proposal policy.

Only paths satisfying every train profile's minimum graph retention, maximum
remapped-event fraction and maximum added-padding ratio remain eligible.
Selection is lexicographic: smallest bucket count first, then lower remap and
padding. The artifact records the represented subset-space size, actual state
transitions and retained labels, plus the best feasible point for each bucket
count, so the decision can be audited rather than hidden behind one scalar
score. The selected policy must independently satisfy predeclared holdout
constraints on every holdout profile before an ABBA plan is emitted. Holdout
limits default to the train limits for backward compatibility, but can be wider
than a deliberately strict train safety envelope. Holdout data remains
acceptance-only and cannot change which subset the train search selects.

Search may be scoped to one engine with `--candidate-engine-role`. Unselected
roles remain byte-for-byte equivalent to the framework capture policy and are
still checked in every coverage record. This supports mechanism-specific
interventions such as pruning a low-hit proposal engine while leaving a
high-hit base engine unchanged; it is not permission to choose the role after
inspecting a claimed test split.

This tier does not assert that fewer captures improve steady-state throughput.
It reduces an exponential policy space to a small measured frontier. Capture
time, graph memory and end-to-end performance are learned from device runs; a
candidate is useful only if its paired effect clears the replay-control noise
envelope and its formal workload quality/SLO checks pass.

An independent holdout audit replays the candidate's smallest-fitting-bucket
assignment over graph events in the held-out profile. It reports remapped event
counts, graph fallbacks, retained graph coverage, and padding delta for each
engine. `mapping_exact` is deliberately strict: every source graph event must
keep its original padded bucket. A non-exact policy may still enter measured
calibration, but it cannot be described as trace-preserving on that holdout.

The graph-profile wrapper calls `AsyncLLM.do_log_stats()` immediately before
each chang backend shuts down and surrounds the resulting table with an engine
role delimiter. The importer then preserves exact unpadded/padded token counts,
runtime modes and frequencies. It reports both bucket utilization and a
trace-preserving set that removes only configured buckets unused by the
observed workload. This is a safe seed for calibration, not a universal claim:
different load or length regimes can reactivate a bucket.

### Phase-aware runtime diagnosis

The stock vLLM v0.18 graph table aggregates scheduler steps by token shape and
runtime mode. Under `FULL_DECODE_ONLY`, an eager event can mean at least three
different things: prompt prefill, a mixed prefill/decode batch, or a decode
batch outside the captured domain. Those mechanisms require different
interventions, so a global graph-hit rate is not a sufficient tuning signal.

The version-bound observability patch adds one immutable field to each
`CUDAGraphStat`. vLLM-Ascend labels the step from its scheduler state before
execution:

```text
decode request := num_computed_tokens >= num_prompt_tokens
all decode      -> decode
some decode     -> mixed
no decode       -> prefill
```

This is the same prompt-boundary predicate used by vLLM's speculative-decode
metadata path. It therefore handles a small final chunked-prefill block without
guessing from scheduled token count. The patch changes logging only and is
applied with strict `git apply` against both the vLLM and vLLM-Ascend source
trees in the container.

The importer accepts the new six-column table and maps legacy tables to the
explicit phase `unknown`. Per-phase audits report event count, graph hit rate,
padding and shape distributions. Diagnosis fails closed unless every event has
a phase label. For `FULL_DECODE_ONLY`, eager decode events are then split at
the configured capture ceiling:

- decode above the ceiling motivates an isolated capture-expansion versus
  admission-limit probe;
- decode within the ceiling points to a bucket or graph-compatibility gap;
- mixed steps motivate a separate stage-admission, P/D scheduling or
  chunked-prefill-policy probe;
- pure prefill remains expected eager work under this graph mode.

These are mechanism candidates, not performance claims. Event counts do not
estimate latency benefit, and the diagnosis never changes a deployment by
itself. A selected intervention still needs a paired end-to-end experiment,
replay-noise gate, quality checks and runtime-configuration attestation.

The wrapper also records role-specific backend load wall time around chang's
existing `_load_backend` boundary. This complements vLLM's graph-capture and
engine-init log counters with the deployment-visible cost. The NPU launcher
attests the config, dataset, wrapper and Python source snapshot before launch,
then appends the result and runner-log hashes. These diagnostic attestations do
not replace a formal calibration manifest, but they make startup regressions
and accidental source drift inspectable.

`PROMPT_PREFIX_TOKENS` enables deterministic token-length stress profiles in
the diagnostic launcher. The wrapper records the resulting minimum, maximum
and mean prompt lengths plus the synthetic prefix size. Padded prompts are
useful for exposing capacity and graph-shape boundaries when no long-context
dataset is installed, but they are synthetic: they cannot establish task
quality, selector generalization, or a formal end-to-end performance claim.

For long-context feasibility probes, the same launcher accepts
`BASE_MAX_NUM_BATCHED_TOKENS` and records the effective override in `run.meta`.
This lets the profiler test prompt-logprob workspace limits without silently
changing the checked-in baseline config. The launcher refuses non-empty output
directories and finalizes metadata even after a nonzero container exit,
including the classified outcome, exit codes, runner-log hash, optional result
hash and a post-run process-memory snapshot.

The experiment planner freezes the measured framework bucket list as its
baseline, derives a trace-preserving pruning candidate and adds a `no-graph`
ablation. It compares each candidate independently against the explicit
baseline in ABBA order. Adjacent A/B runs share a workload seed, while the
dataset subset stays fixed. The execution wrapper injects role-specific graph
settings into the parsed TOML in memory and binds the plan, run and policy to
the native result artifact.

`--include-replay-control` prepends an ABBA group whose two labels both execute
the exact vLLM-default policy. It measures harness and live-policy variation
before interpreting a small candidate effect. Request-local seeds fix random
streams, but asynchronous batch shapes can perturb logits enough to change
sampled trajectories. The assessor therefore reports exact token, semantic
answer and physical token-slot agreement separately. A fixed-trace cost claim
requires exact outputs; an end-to-end live-policy claim needs enough requests,
repeats and an explicit quality constraint in the formal calibration protocol.

The graph plan and shell launcher remain diagnostic because their hash does not
freeze the complete dataset, source and environment. Promote selected policies
into the formal harness with:

```bash
inference-autopilot build-vllm-graph-calibration \
  calibration-template.json graph-experiment-plan.json \
  --campaign-id conditional-is-graph-formal \
  --candidate-policy-id trace-preserving-pruned \
  --include-replay-control \
  --output graph-calibration-spec.json

inference-autopilot plan-calibration graph-calibration-spec.json \
  --output graph-calibration-plan.json
```

This produces a strong-baseline configuration with default buckets, an
identical replay control, and selected graph-policy candidates. It preserves the
template's semantic, workload, environment, objective and SLO contracts. The
formal chang adapter validates and injects each engine's graph mode and capture
sizes and binds the effective values back into the native result.

On the NPU host, `run_npu_graph_experiment.sh` can execute all runs or one
`COMPARISON_ID`. It validates the plan before launch, skips only results already
bound to the expected run id, and delegates every arm to the per-device lock and
process-memory guard in `run_npu_graph_metrics_smoke.sh`.

## Device calibration matrix

Start on one idle NPU and one fixed Conditional IS configuration:

1. collect eager shape histograms and per-stage service times;
2. capture a bounded candidate set derived from observed quantiles and boundary
   shapes, not every integer size;
3. measure replay latency, incremental memory and capture time per candidate;
4. generate the plan offline;
5. compare default buckets, planner buckets and all-eager in ABBA/BAAB order;
6. repeat for short, mixed-length and long-scoring workload regimes.

No end-to-end gain is claimed until the generated plan wins a paired comparison,
beats the replay-control noise envelope and satisfies the frozen quality
constraint. Exact-token equivalence is additionally required for fixed-trace
claims.

The current shared server requires the same privileged container mode used by
its existing vLLM deployments for driver discovery. The smoke launcher mounts
only the user's source, Autopilot, cache and artifact directories, restricts
the runtime with `ASCEND_RT_VISIBLE_DEVICES`, and refuses a target device with
an existing process above the configured memory threshold. It must never mount
another user's private directory.
