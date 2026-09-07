# Stage Wavefront Admission

## Problem

vLLM schedules the requests that have reached its engine. It does not know that
Conditional IS alternates between base candidate generation, proposal rollout
generation, base target scoring, and selection. Under 96 independent outer
workers, proposal rollout groups arrive continuously. A new prefill is therefore
commonly scheduled while older rollouts are decoding.

The phase probe measured this directly on the frozen BF16 short-p96 stack:

| Proposal event | Count | Share |
| --- | ---: | ---: |
| decode | 165 | 20.22% |
| mixed prefill/decode | 648 | 79.41% |
| pure prefill | 3 | 0.37% |

Only 80 of 816 proposal engine steps used an ACL graph. Eighty-five pure decode
steps were also above the 512-sequence capture ceiling because the scheduler
admitted as many as 768 concurrent proposal sequences.

Reducing `proposal_max_num_batched_tokens` from 131072 to 768 did not solve this
mechanism. It increased mixed steps to 96.99%, reduced graph hit rate to 2.86%,
and reduced QPS by 8.33% in a diagnostic replay. The scheduler used every step's
remaining token budget for a small new prefill, so token-budget rollback alone
made the phase interleaving worse.

## Runtime Primitive

`StageWavefrontAdmissionBackend` is an adapter-local backend wrapper. It queues
complete synchronous algorithm call groups, packs them into a bounded wave, and
submits the whole wave through one proposal-backend call. It never splits or
rewrites a caller group. Results and completion callbacks are mapped back to the
original group in order.

Waves do not overlap. New proposal prefills cannot enter while the previous
proposal wave is still decoding. Conditional IS step numbers and rollout lengths
are retained as wave-composition telemetry, but they are not barriers: different
iterations may share a wave because all represent the same proposal-generation
operation and have no dependency on one another.

Two live limits are enforced:

1. `sum(group sequence count) <= selected graph-eligible wave width`;
2. `sum(len(request.prefix)) <= proposal max_num_batched_tokens`.

The second constraint makes admission context-aware. A validated short-context
policy can admit a wider wave, while long or late-step prefixes automatically
produce a narrower wave without changing engine configuration or algorithm
semantics. A bounded wait releases sparse traffic and terminal tails.

The current runtime keeps each synchronous caller group indivisible. Formal
preflight therefore rejects a workload when
`context_tokens * candidate_count * rollout_count` alone exceeds the proposal
token cap. This lower-bound test proves that even the synthetic context portion
cannot fit one group; question and generated-prefix tokens only increase the
gap. Medium2k and long-context activation requires a separately validated
repeated-prefix group-sharding policy. It is not covered by the short-context
result below.

## Automatic Wave Planning

The offline planner combines:

- outer request concurrency;
- nominal algorithm fanout, `candidate_count * rollout_count`;
- the measured graph capture ceiling;
- scheduler `max_num_seqs`;
- a minimum full-wave utilization target.

It enumerates every feasible number of indivisible groups. Exact partitions of
the active outer workload above the utilization floor are preferred; otherwise
a balance score penalizes underfilled tails, low utilization, and excess waves.
For the measured p96 case, the inputs are 96 outer requests, 24 nominal rollout
sequences per group, a graph ceiling of 512, and scheduler capacity 768. The
selected width is 16 groups or 384 sequences: six exact nominal waves at 75% of
the graph-eligible capacity.

The structural challenger is 20 groups or 480 sequences. It requires only five
nominal waves and raises mean capacity utilization from 75% to 90%, but its tail
contains 16 of 20 groups instead of preserving an exact partition. Its balance
score is slightly higher (`0.775` versus `0.720`), while the default planner's
exact-partition rule still prefers 384. This creates one interpretable causal
comparison: whether eliminating one proposal wave is worth the larger batch and
non-exact tail. The 21-group, 504-sequence boundary has a lower balance score
because its tail falls to 12 groups, so it is held behind the 480 result rather
than included in an undirected sweep.

At runtime EOS and short candidate blocks change group width. The dispatcher
therefore packs actual groups under both sequence and prefill-token limits. The
nominal plan is a prior and hard safety boundary, not an assumption that every
wave remains rectangular.

## Initial Device Evidence

All rows below use the same NPU7, model weights, source snapshot, BF16 config,
96 requests, 96 workers, workload seed, and subset seed. They are an A-B-A
mechanism sequence, not yet a prospective randomized ABBA claim.

| Run | QPS | Elapsed | Accuracy | Proposal graph hit | Proposal mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| control A1 | 0.397744 | 241.36s | 0.40625 | 9.80% | 79.41% |
| atomic wavefront B1 | 0.454358 | 211.29s | 0.40625 | 95.97% | 2.88% |
| control A2 | 0.388218 | 247.28s | 0.36458 | 6.10% | 82.97% |
| 2D wavefront B2 | 0.416123 | 230.70s | 0.36458 | 96.05% | 2.91% |

The paired effects are +14.23% for B1/A1 and +7.19% for B2/A2, with a 10.65%
geometric-mean effect. Accuracy is exactly equal within each pair, although the
two fresh executions of the same control differ in both trajectory and aggregate
accuracy. The controls' QPS differs by 2.39%, below the earlier 5.61%
conservative replay envelope.

The speedup is not explained only by shorter stochastic trajectories. Candidate
compute volume was 1.28% and 0.89% lower in the two pairs, while realized
forward-slot throughput improved 12.77% and 6.23%. The paired geometric-mean
improvement is 9.45% for forward slots/s and 9.54% for estimated dense FLOPs/s.

The first non-atomic prototype already raised proposal graph hit rate to 96.18%
but used 30 waves, including 23 partial waves, and improved QPS by only 4.03%.
Merging each wave into one backend call reduced the run to 19 waves and exposed
the larger end-to-end effect. This isolates batching atomicity as part of the
mechanism rather than attributing the gain to graph mode alone.

The same seed does not reproduce exact sampled trajectories on the existing
asynchronous vLLM path: even the control replay changed aggregate accuracy. The
formal study must therefore use paired repeated quality statistics and preserve
compute accounting. It must not claim fixed-token-trace equivalence.

The 2D token constraint did not bind in this short-context replay: the largest
admitted wave contained 101,145 prefix tokens under the 131,072-token cap. This
validates that the added constraint does not disturb the short policy, but it is
not evidence for mixed or long contexts. Those regimes must exercise dynamic
wave narrowing and reject any indivisible group that exceeds the declared cap.

## Formal Runtime Contract

The chang pressure runner accepts wavefront policy only as a complete set of
manifest-bound deployment settings:

- `proposal_stage_wavefront_mode`: `off`, `auto`, or `prefix_sharded`;
- `proposal_graph_capture_ceiling`: the measured sequence ceiling;
- `proposal_stage_wavefront_max_wait_seconds`: bounded tail release;
- `proposal_stage_wavefront_min_utilization`: planner utilization floor.

`prefix_sharded` additionally requires explicit maximum sequences and prefill
tokens per shard, a maximum shard count per parent per wave, and a maximum
in-flight wave count. Its planner
uses one repeated-prefix rollout run as the atomic admission unit. This permits
medium-context parent calls that exceed the wave token cap without changing
request seeds, outputs, or the caller-visible completion barrier.

An `auto` run is formal-eligible only when the manifest also supplies the full
base and proposal graph modes and capture-size lists, and the declared proposal
ceiling equals the largest configured proposal capture size. For `auto`, the
nominal `candidate_count * rollout_count` group must fit both that ceiling and
`proposal_max_num_seqs`. For `prefix_sharded`, one `rollout_count` run must fit
the declared shard and wave bounds.

The native result records the full offline plan and every realized wave,
including sequence width, aggregate prefix tokens, stage composition, admission
wait, partial status, release reason, and whether an indivisible group exceeded
either limit. Release reasons distinguish target-width admission, sequence or
prefill-token capacity, collection timeout, and shutdown. Their observation
fractions make width experiments mechanistically interpretable: a nominally
wider policy that is repeatedly released by the token cap did not actually test
that sequence width.
Formal calibration exposes those counters as numeric observation metrics and
sets zero oversized waves as a hard constraint. The launch manifest also hashes
the complete plugin-side Python package and the worker verifies that snapshot
before loading either model.

## Prospective Short-p96 ABBA

The first formal study used fresh paired seeds `2026090673` and `2026090674` on
the same NPU7. It compared the unchanged proposal backend with the manifest-bound
automatic 384-sequence wavefront policy in ABBA order.

| Run | Policy | QPS | Accuracy | Proposal calls | Max call width | Waves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | off | 0.382071 | 0.44792 | 264 | 24 | 0 |
| B1 | auto-384 | 0.456084 | 0.46875 | 17 | 384 | 17 |
| B2 | auto-384 | 0.446976 | 0.44792 | 18 | 384 | 18 |
| A2 | off | 0.382919 | 0.45833 | 264 | 24 | 0 |

The paired QPS effects were `+19.37%` and `+16.73%`, for a `+18.04%`
geometric-mean effect. Total forward-slot volume changed by only `+0.78%` and
`-0.70%`; forward-slot rate improved `+20.31%` and `+15.91%`, with a `+18.09%`
geometric mean. Estimated dense-FLOP rate improved `+17.99%` geometrically.
Both quality non-inferiority checks passed.

The realized policy used 17 and 18 non-overlapping waves with no oversized
wave. Mean sequence utilization was `93.15%` and `88.02%`; maximum aggregate
prefix demand was 104,172 and 106,644 tokens, below the 131,072-token bound.
The proposal path contracted from 264 backend calls of at most 24 sequences to
17 or 18 calls of at most 384 sequences, directly confirming the intended
mechanism.

The formal assessment and runtime-closure checks pass, but there is no replay
control with the exact same fully explicit settings hash. The earlier `5.61%`
replay envelope is useful context, not a formal noise gate for this campaign.
The policy therefore advances to width and workload-regime holdouts rather than
being declared generally promotable from this one short-context study.

## Prospective Width Selection

The width holdout used fresh seeds `2026090677` and `2026090678` in ABBA order
on the first server's NPU2. It compared the selected exact-partition width of
384 sequences with the higher-utilization, five-nominal-wave challenger of 480.
Both variants used the same frozen plugin snapshot and inference-scaling source.

| Seed | 384 QPS | 480 QPS | QPS effect | Slot-rate effect | p95 effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026090677 | 0.457662 | 0.424083 | -7.34% | -3.96% | +7.43% |
| 2026090678 | 0.449949 | 0.438414 | -2.56% | -0.67% | +2.38% |

The candidate-over-baseline QPS geometric mean is `-4.98%`. Total forward-slot
volume was `+3.64%` and `+1.94%` in the candidate runs because the asynchronous
sampled trajectories were not identical; after normalization, forward-slot
rate still fell by `2.33%` geometrically and estimated dense-FLOP rate fell by
`2.35%`. Candidate p95 latency rose by `4.88%` geometrically. All four runs met
the declared absolute quality and latency constraints, but the wider policy did
not meet the performance objective.

The release telemetry explains why nominal wave count alone was misleading.
The 384 runs each used 18 waves; over their 36 combined waves, 44.44% reached
the target width, 33.33% hit sequence capacity, 22.22% timed out, and none hit
the prefill-token bound. The 480 runs used 15 and 17 waves; over 32 combined
waves, only 31.25% reached target width, 43.75% hit sequence capacity, 6.25%
hit the prefill-token bound, and 25.00% timed out. Partial-wave share increased
from 55.56% to 68.75%, while mean peak proposal KV usage rose from 6.23% to
7.73%. There were no preemptions and no oversized waves.

The selector therefore retains 384 for this short-p96 environment. The 504
boundary is not executed: it already has a worse structural tail than 480, and
480 failed in both paired directions. This is a scoped result, not a claim that
384 is optimal for other context, concurrency, fanout, model, or device regimes.
The next material admission experiment requires context-aware group sharding;
changing the 0.5-second collection timeout can affect only the timeout-released
minority here and has lower expected value than opening the mixed/long-context
regimes.

## Product Boundary

This primitive is not a replacement for vLLM continuous batching or a cluster
orchestrator. vLLM continues to own token-level scheduling, KV allocation,
prefix caching, graph execution, and sampling. The wrapper contributes the
algorithm operation boundary that vLLM cannot infer from flat requests.

It also does not replace MARS-style program admission or offloading. A future
Muyuan plugin can expose wavefront planning as one optional execution policy for
an algorithm adapter, while MARS remains responsible for broader program and
resource placement. The contract is narrow: ready calls of one graph operation
are packed under measured engine limits, with a fallback to direct submission.

## Next Optimization Surface

The current planner chooses a nominal wave width from static workload and engine
contracts. The next version should learn a small, constrained admission policy
from observed group arrivals rather than expand into an unconstrained Cartesian
sweep. Its inputs are ready groups, actual sequence counts, aggregate prefix
tokens, oldest-group age, graph ceiling, scheduler token budget, KV pressure,
and recent service time.

The policy has three decisions:

1. target wave width or utilization below the attested hard ceiling;
2. bounded collection time under low or bursty arrival rates;
3. bounded reordering of independent ready groups to reduce token-budget
   fragmentation without starving the oldest group.

For prefix-sharded mode, dispatch is also a bounded pipeline. The manifest sets a
maximum number of in-flight waves, while aggregate in-flight sequences consume
credits from the graph-eligible scheduler capacity selected by the planner. This
prevents long-context token-limited waves from serializing full decode without
allowing unbounded admission.

Offline calibration should select among a small set of interpretable policies
and emit a guarded policy table by workload regime. Online logic may choose only
among those prevalidated policies; graph sizes, memory reservations, and engine
capacity remain restart-bound. A lower-level atomic bulk-enqueue hook is a
separate follow-up for eliminating the residual mixed steps that arise while a
merged adapter call is still inserted into vLLM request by request.

## Promotion Gates

Before activation, the final implementation requires:

1. a prospective same-card ABBA or BAAB study with fresh paired seeds;
2. gain above the matching replay-control envelope;
3. frozen algorithm semantics and compute-accounting checks;
4. repeated quality non-inferiority rather than one-run accuracy equality;
5. short, mixed-length, long-context, and sparse-arrival holdouts;
6. fail-open direct submission when load cannot fill a wave within its latency
   budget;
7. runtime attestation of capture ceiling, scheduler capacity, token budget, and
   actual wave composition.
