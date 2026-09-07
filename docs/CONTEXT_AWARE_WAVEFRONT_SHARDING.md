# Context-Aware Wavefront Sharding

## Motivation

The validated short-p96 stage-wavefront policy treats one synchronous
Conditional IS proposal call as an indivisible group. That is appropriate only
while the aggregate prefixes in one group fit the declared proposal token cap.
For medium2k, the synthetic context alone gives a lower bound of
`2048 * 8 candidates * 3 rollouts = 49,152` tokens per group, versus the
measured strong baseline's 12,288-token proposal budget. Current formal
preflight rejects this regime.

vLLM can chunk a long prefill internally, but it sees only independent requests
and does not know which proposal rollouts belong to one algorithm operation.
The synchronous `ContinuousBatchingBackend` can split oversized groups at
repeated-prefix boundaries, but chang's asynchronous vLLM backend declares
native continuous batching and bypasses that code path. The missing primitive
is algorithm-aware admission of smaller prefix cohorts before native enqueue.

## Semantic Unit

A Conditional IS proposal call contains contiguous runs of requests sharing
the same prefix, sampling policy, and requested rollout length. In the current
algorithm, each run normally contains the three rollout replicas for one
candidate. Every `GenerationRequest` carries its own derived seed, and the vLLM
adapter copies that seed into request-local sampling parameters.

The implemented sharding policy splits a caller group only between these runs.
It must preserve:

1. every request object, request-local seed, prefix, sampling policy, and token
   budget;
2. output order and callback indexes observed by the caller;
3. the barrier that completes the original synchronous call only after all of
   its shards finish;
4. repeated-prefix runs unless one run itself exceeds a hard limit;
5. Conditional IS scoring, reward evaluation, importance correction, and
   selection without modification.

This is a deployment-semantic transformation, not rollout pruning. It changes
when requests become visible to the engine, never which requests exist.

## Runtime Design

Each caller submission becomes a parent future plus one or more shard tickets.
Tickets from independent parents may be packed into a wave under four bounds:

- graph-eligible live sequences per wave;
- scheduler prefill tokens per wave;
- sequence and prefill-token bounds per shard, which reserve room for other
  parents instead of letting one parent consume a whole wave;
- maximum shards from one parent, to prevent head-of-line monopolization;
- oldest-parent admission age, to prevent starvation.

The dispatcher submits a bounded pipeline of packed shard waves. Aggregate
in-flight sequences consume credits from the selected scheduler/graph capacity,
and each native async completion returns one credit without waiting for its
whole wave to finish. Completion callbacks map shard-local indexes back to
parent indexes; the parent future still returns only after all positions are
populated. An error fails the parent and cancels its remaining queued shards.
Formal mode is fail-closed: unsupported request shapes or missing telemetry
reject the run.

The hot policy surface is deliberately small:

- `mode`: `off`, `auto` (indivisible parent groups), or `prefix_sharded`;
- target sequence utilization;
- collection timeout;
- maximum sequences and prefill tokens per shard;
- maximum shards per parent per wave.

Graph buckets, engine scheduler capacities, memory reservations, and model
parallelism remain restart-bound settings.

## Required Telemetry

Every wave records target, sequence-cap, prefill-cap, timeout, or shutdown as
its release reason. The sharded mode additionally records:

- parent groups and shard tickets admitted;
- shards per parent and parent completion span;
- repeated-prefix runs preserved or forcibly split;
- sequence and prefill-token utilization;
- oldest-parent wait and fairness violations;
- cancelled shards and oversized atomic runs.

Formal conversion rejects missing counters, incomplete parent barriers,
oversized atomic requests or waves, cancelled shards, fairness violations, and
wave records that exceed either the configured shard or wave bounds. The
runtime-closure assessor uses rollout runs, rather than whole parent calls, as
the effective graph-admission unit.

## Implemented Contract

`PrefixShardedStageWavefrontAdmissionBackend` provides the parent future,
prefix-run planner, callback remapping, fair wave packing, cancellation, and
telemetry. The chang adapter binds the following policy to each run manifest:

- `proposal_stage_wavefront_mode=prefix_sharded`;
- `proposal_graph_capture_ceiling` and the existing wave token budget;
- `proposal_stage_wavefront_max_shard_sequences`;
- `proposal_stage_wavefront_max_shard_prefill_tokens`;
- `proposal_stage_wavefront_max_shards_per_parent`;
- `proposal_stage_wavefront_max_inflight_waves`.

The last setting bounds dispatcher fan-out. A second independent bound is derived
from the selected graph-eligible wave size: the sum of sequences in all in-flight
waves cannot exceed that size. This lets long-context partial waves overlap inside
the native asynchronous vLLM scheduler without turning admission into an unbounded
request queue.

Runtime evidence records current and peak in-flight waves/sequences,
per-sequence versus whole-batch credit releases, sequence-credit stall
count/time, and dispatch failures. Formal result conversion rejects nonzero
terminal in-flight work, missing credits, dispatch failures, or any observed cap
violation.

The first serial implementation is intentionally retained only as diagnostic
evidence. On medium2k it limited proposal execution to six running requests and
regressed QPS by 59.20%. A wave-level 48-credit pipeline recovered that loss but
remained 2.00% below direct submission. Per-sequence credit release produced a
raw 7.06% signal, but a context-identical replay measured a 16.26% noise
envelope and rejected promotion. See `EXPERIMENT_LOG_20260907.md`.

For the intended medium2k diagnostic, a three-rollout prefix run is the atomic
unit. A shard sequence cap of `3` and token cap of `8192` preserve each run,
while a `16384`-token wave can mix two independent parents. This differs from
the previously inconclusive `12288 -> 16384` engine-budget experiment, which
did not control parent occupancy inside a wave.

This separates three outcomes that QPS alone cannot explain: the wider target
was reached, the token cap made it unreachable, or collection delay dominated.

## Validation Sequence

1. Pure contract tests use request-local pseudorandom backends to prove identical
   outputs, callback indexes, errors, and parent barriers under reordered shard
   completion.
2. A trace replay compares request IDs, seeds, prefixes, and generation lengths
   between indivisible and sharded admission; any multiset difference rejects
   the implementation before device use.
3. A short-context diagnostic must reproduce the existing indivisible policy
   when no split is required.
4. A medium2k low-load mechanism probe verifies zero oversized atomic runs and
   measures whether mixed scheduler steps, graph hit rate, and parent latency
   improve.
5. Fresh-seed same-card ABBA compares sharded versus off and indivisible modes
   with work-normalized throughput and quality non-inferiority.
6. Mixed-length and sparse-arrival holdouts gate the guarded policy table. Long
   contexts stay on the manual fallback until their own token and memory
   feasibility boundary is measured.

No sharded policy is activation-eligible merely because request-local seeds are
preserved. Device evidence must also show that reduced phase mixing outweighs
the added barriers and smaller engine waves.
