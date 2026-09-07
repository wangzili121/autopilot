# Conditional IS Graph-Bucket Experiment Log, 2026-09-05

This record captures the first end-to-end use of the graph-profile corpus,
coverage-constrained bucket search, immutable ABBA launcher, and result
assessor on chang's Conditional IS small-proposal path. All performance data
in this file is diagnostic unless explicitly promoted by the evidence gate.

## Environment

- Host: `liteserver-cf0e-1.novalocal`
- Device: Ascend 910B3, NPU 4
- Container: `quay.io/ascend/vllm-ascend:v0.18.0`
- Base model: Qwen2.5-1.5B-Instruct
- Proposal model: Qwen2.5-0.5B-Instruct
- Algorithm path: Conditional IS small proposal
- Graph mode: `FULL_DECODE_ONLY`

Each new run records hashes for the source snapshot, config, dataset, wrapper,
runner log, and result. The candidate experiment below passed all provenance
and same-environment checks.

## Replay Noise

An identical-configuration ABBA replay used 32 requests per run. Its two QPS
ratios were `0.981508` and `0.960170`; the geometric mean was `0.970780`.
The conservative observed pair envelope is therefore about `3.983%`.

The paired requests and seeds were identical, but vLLM asynchronous scheduling
produced different token outputs and padded forward-token counts. This makes
the replay useful for estimating live-run drift, but not formal steady-state
evidence.

Artifact: `artifacts/npu4-cis-graph-replay32-20260905/assessment.json`

## Workload Profiles

The v2 corpus separates train and holdout by regime. All four profiles use the
same short GSM8K workload and vary only offered load, so context length remains
an uncovered dimension.

| Regime | Split | Requests | QPS | Base buckets used | Proposal buckets used |
| --- | --- | ---: | ---: | --- | --- |
| `short-load4` | train | 4 | 0.3056 | 1,2,4,8,16,24,32 | 1,2,8,16,24,32,40,48,56,64,72,80,88,96 |
| `short-load8` | train | 8 | 0.3663 | 1,4,8,16,24,32,40 | 1,24,32,40,48,56,64,72,80,88,96 |
| `short-load32` | holdout | 32 | 0.3395 | 1,2,4,8,16,24,32,40 | 1,4,8,16,24,32,40,48,56,64,72,80,88,96 |
| `short-load96` | holdout | 96 | 0.3525 | 1,2,4,8,16,24,32,40 | 1,2,4,8,16,24,32,40,48,56,72,80,88,96 |

The bucket union changes with load. In particular, an exact policy learned
from load 4 and load 8 removed proposal bucket 4, but both holdout profiles
used it. The exact promotion gate correctly rejected this policy with
`holdout_mapping_not_exact`.

Corpus artifact:
`artifacts/npu4-cis-graph-profile-corpus-short-load-v2/corpus.json`

Exact-gate artifact:
`artifacts/npu4-cis-graph-profile-corpus-short-load-v2/exact-promotion.json`

## Context-Length Expansion

Synthetic inert prefixes were added without changing the GSM8K question. The
medium profile succeeded with the default base token budget, while the first
long-context run exposed a concrete feasibility boundary:

| Regime | Base token budget | Result | QPS | Base graph hit | Proposal graph hit |
| --- | ---: | --- | ---: | ---: | ---: |
| about 2K, load 8 | 10,240 | success | 0.1599 | 100.00% | 100.00% |
| about 8K, load 16 | 10,240 | OOM | - | - | - |
| about 8K, load 16 | 2,048 | success | 0.0509 | 4.13% | 91.94% |
| about 8K, load 16 | 3,072 | success, repeat 1 | 0.0539 | - | - |
| about 8K, load 16 | 3,072 | success, repeat 2 | 0.0541 | - | - |
| about 8K, load 16 | 3,584 | OOM, repeat 1 | - | - | - |
| about 8K, load 16 | 3,584 | OOM, repeat 2 | - | - | - |
| about 8K, load 8 | 2,048 | success | 0.0565 | 5.83% | 93.80% |

The 10,240-token failure was not reported as ordinary KV exhaustion. In the
tested vLLM-Ascend 0.18 path, base prompt-logprob scoring reached
`compute_logprobs` and materialized a full-vocabulary FP32 `log_softmax`; the
failed temporary allocation was 4.79 GiB with 2.86 GiB free. Reducing
`base.max_num_batched_tokens` to 2,048 avoided that allocation peak, but caused
most long base prefill work to execute eagerly in 2,048-token chunks. This is
both a workload-dependent tuning boundary and direct evidence for a future
selected-token-scoring optimization on this runtime path.

Artifacts:

- `artifacts/npu4-cis-graph-profile-medium2048-load8-20260905-r2`
- `artifacts/npu4-cis-graph-profile-long8192-load16-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load16-basebt2048-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load16-basebt3072-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load16-basebt3072-r2-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load16-basebt3584-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load16-basebt3584-r2-20260905`
- `artifacts/npu4-cis-graph-profile-long8192-load8-basebt2048-20260905`

## Confirmed Ordered Feasibility Boundary

The ordered feasibility planner searched the declared domain
`[1024,1536,2048,2560,3072,3584,4096,5120,6144,8192,10240]` under a frozen
context digest, `7ab9a9b...312ed`. Success and resource exhaustion each
required two observations before they could define a boundary.

The search completed with 3,072 as the largest confirmed feasible value and
3,584 as the adjacent confirmed exhausted value. Both 3,072 runs completed all
16 requests with accuracy `0.4375`; elapsed times were `297.05 s` and
`296.01 s`. Both 3,584 runs failed at the same vLLM prompt-logprob operation,
requesting a 2.03 GiB FP32 `log_softmax` allocation with 1.91 GiB free. Higher
single-observation OOMs are retained in the evidence ledger but need no repeat
once the lower adjacent exhausted point is confirmed.

The recommendation is scoped to this exact algorithm, model pair, 8K prefix,
load 16, memory split, graph policy, runtime image and source snapshot. It is a
resource-feasibility result, not evidence that 3,072 maximizes throughput.

Artifacts:

- `artifacts/npu4-cis-long8k-base-token-feasibility-20260905/spec.json`
- `artifacts/npu4-cis-long8k-base-token-feasibility-20260905/plan.json`

## MRV2 Capability And Scoring Probe

The model runner was promoted from an implicit environment choice to an
explicit, hashed deployment setting. The stock image and a minimal diagnostic
compatibility patch were probed before admitting MRV2 into the search space.

| Runner | Graph | Workload | Result | QPS | P95 |
| --- | --- | --- | --- | ---: | ---: |
| stock MRV2 | default | short, load 4 | initialization failure | - | - |
| patched MRV2 | default | short, load 4 | first-decode failure | - | - |
| patched MRV2 | none | short, load 4 | success | 0.0465 | 84.97 s |
| MRV1 | none | short, load 4 | success | 0.1062 | 37.67 s |
| patched MRV2 | none | 8K, load 16, base budget 10,240 | success | 0.0392 | 402.59 s |
| MRV1 | none | 8K, load 16, base budget 10,240 | OOM | - | - |

The stock MRV2 failed because Ascend's `RequestState` adapter did not pass two
arguments added by the bundled vLLM version. The minimal patch fixed that
interface only. With ACL Graph enabled, execution then failed because Ascend's
MRV2 graph manager read the removed `attn_metadata` field. With graph disabled,
MRV2 completed, but short-load throughput was 56.2% below the matched MRV1 run
and P95 latency was 2.26 times higher.

The long-context pair isolates the scoring-memory effect from graph memory.
MRV1 still failed with graph disabled: an 8,480-position scoring request tried
to allocate a 4.80 GiB full-vocabulary FP32 `log_softmax` with 3.23 GiB free.
Patched MRV2 completed all 16 requests at the same 10,240 scheduler budget.
This supports extracting MRV2's exact token-reduction idea into an MRV1 scorer,
but does not support migrating the application to MRV2 on this image.

The first two failed runs were initially labelled `resource_exhausted` because
the launcher matched an informational graph warning containing the phrase
"Out of Memory." The classifier now requires a runtime-specific OOM signature,
and a regression test separates interface failures from true exhaustion. The
original metadata is retained rather than rewritten; the table above uses the
actual exceptions in the hashed logs.

Artifacts:

- `artifacts/npu4-cis-mrv2-short-load4-smoke-20260905`
- `artifacts/npu4-cis-mrv2-patched-short-load4-smoke-20260905`
- `artifacts/npu4-cis-mrv2-patched-nograph-short-load4-20260905`
- `artifacts/npu4-cis-mrv1-nograph-short-load4-20260905`
- `artifacts/npu4-cis-mrv2-patched-nograph-long8192-load16-basebt10240-20260905`
- `artifacts/npu4-cis-mrv1-nograph-long8192-load16-basebt10240-20260905`

## Consilience Scoring Scope

The newer local inference-scaling revision implements Consilience by asking an
exact scoring backend for per-position statistics. Its current Transformers
path computes full-vocabulary normalized log-probabilities and then takes the
top-k values. The vLLM backend delegates this request to the separate exact
backend rather than serving it natively.

Consilience therefore has the same broad `positions x vocabulary` intermediate
pressure as MRV1 selected-token scoring, and adds a top-k reduction. It does not
require a full probability matrix semantically. A tiled exact pass can maintain
online `logsumexp`, the selected token, a size-k top-logit set, and the entropy
moment. The new scoring planner models these statistics separately and its CPU
reference implementation matches naive full `log_softmax` numerically.

Artifact:
`artifacts/consilience-long3584-score-reduction-plan.json`

### Standalone NPU Reduction Probe

An exact PyTorch/NPU prototype then isolated scoring from model-runner effects.
It processed the real 3,584 by 151,936 shape and compared every tiled result
against native FP32 full log-softmax. With a 512-position by 32K-vocabulary
tile, selected-only scoring was 9.2% faster than native and reduced incremental
peak allocation from 4,180.01 MiB to 160.04 MiB. For selected plus Consilience
top-k mean, the same tile was 8.1% slower but reduced the peak from 4,180.03 MiB
to 160.08 MiB. Maximum absolute error was at most 2.86e-6.

The result validates the reduction contract and identifies an initial Pareto
point. It does not yet establish an end-to-end gain because model integration,
operator launch overhead, and real logits lifetimes remain unmeasured. The
streaming capability therefore remains `unknown` until an MRV1 adapter passes
the full workload gate.

Artifacts:

- `artifacts/npu4-score-reduction-p512-grid-20260905`
- `artifacts/npu4-score-reduction-p512-refine-20260905`
- `artifacts/npu4-score-reduction-p3584-grid-20260905`
- `artifacts/npu4-score-reduction-selected-p3584-20260905`

## Coverage-Constrained Search

The bounded search enumerated all non-empty subsets independently for both
roles: 255 base subsets and 32,767 proposal subsets. Each train profile was
required to retain every source graph event, remap at most 5% of graph events,
and add at most 0.5% padded token work. The selected policy was then evaluated
unchanged on both holdout regimes.

Selected buckets:

- Base: `1,4,8,16,24,32,40` (8 to 7 buckets)
- Proposal: `8,16,24,32,40,48,56,64,72,80,88,96` (15 to 12 buckets)

All profiles retained 100% of source graph events. Holdout effects were:

| Regime | Role | Remapped graph events | Added padding ratio |
| --- | --- | ---: | ---: |
| `short-load32` | base | 0.2585% | 0.0385% |
| `short-load32` | proposal | 0.4287% | 0.0253% |
| `short-load96` | base | 0.3464% | 0.0500% |
| `short-load96` | proposal | 0.3861% | 0.0224% |

Search artifact:
`artifacts/npu4-cis-graph-profile-corpus-short-load-v2/bounded-search-5pct.json`

## Cross-Context Safety Margin

The v4 corpus adds medium and long context to candidate derivation while
reserving the independently collected long-context load-8 profile for holdout.
A single 5% train/holdout threshold selected a 12-bucket proposal policy whose
long holdout remap rose to 7.41%, so the gate rejected it.

The search now supports separate, predeclared train and holdout constraints.
Using a stricter 4% train remap envelope and a 5% holdout acceptance envelope
selected:

- Base: `1,4,8,16,24,32,40` (8 to 7 buckets)
- Proposal: `2,4,8,16,24,32,40,48,56,64,72,80,88,96` (15 to 14 buckets)

All source graph events remained captured. On the frozen long-context holdout,
base remap was 1.68% with 0.130% added padding, and proposal remap was 2.47%
with 0.086% added padding. The candidate passed all three holdout regimes.
Holdout profiles only accept or reject the train-selected policy; they never
participate in subset selection.

Corpus and search artifacts:

- `artifacts/npu4-cis-graph-profile-corpus-context-v4/corpus.json`
- `artifacts/npu4-cis-graph-profile-corpus-context-v4/bounded-search-train4-holdout5.json`
- `artifacts/npu4-cis-graph-profile-corpus-context-v4/bounded-plan-train4-holdout5.json`

## Candidate ABBA

One diagnostic ABBA block compared the bounded policy with the vLLM default
at load 8. Both pairs used matching request identities, seeds, workload config,
source snapshot, config, dataset, wrapper, host, image, and NPU.

| Pair | Baseline QPS | Candidate QPS | QPS ratio | Graph capture delta | Wrapper load delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 951 | 0.3161 | 0.3240 | +2.52% | -3.0 s | -1.44 s |
| seed 952 | 0.3435 | 0.3461 | +0.75% | -3.0 s | -3.61 s |
| aggregate | - | - | +1.63% geomean | -3.0 s median | -2.53 s median |

The repeatable result is a proposal graph-capture reduction from 8 seconds to
5 seconds after removing three low-demand buckets. Base graph capture remained
at 10 seconds. Proposal graph memory changed from 0.33 GiB to 0.32 GiB in one
pair and remained 0.33 GiB in the other.

The `+1.63%` QPS geomean is smaller than the replay envelope and is confounded
by stochastic output and compute-work differences. The assessment therefore
sets `formal_claim_eligible=false` and `steady_state_comparable=false`. No
throughput or quality improvement should be claimed from this block.

Assessment artifact:
`artifacts/cis-short-load-bounded-graph-5pct-diagnostic-assessment.json`

## Formal Replay-Control Calibration

A fresh manifest-bound ABBA campaign ran four identical `40/48` scheduler
configurations on physical NPU 6. The source snapshot, model weights, dataset,
arrival trace, graph, semantic invariants and every deployment setting were
hashed before launch. All four observations passed ordering, non-overlap,
required-metric and strong-baseline checks and were promoted to
`A_formal_paired`.

| Sequence | Role label | QPS | P95 | Accuracy |
| ---: | --- | ---: | ---: | ---: |
| 0 | baseline | 0.2834 | 56.39 s | 0.5625 |
| 1 | replay candidate | 0.2740 | 58.32 s | 0.4375 |
| 2 | replay candidate | 0.2726 | 58.59 s | 0.4375 |
| 3 | baseline | 0.2678 | 59.63 s | 0.4375 |

The paired effects were `-3.32%` and `+1.79%`; their median was `-0.76%` and
the candidate/baseline geometric-mean ratio was `0.9920`. The conservative
replay-noise envelope is therefore `3.32%`. A deployment candidate must exceed
that observed effect-size floor before it can be called useful on this regime.

The proposal engine exposed queue, running-request, KV and preemption metrics;
the base engine's metrics endpoint returned an empty mapping. The adapter now
omits aggregate preemptions unless both roles are observed, preserving missing
telemetry instead of treating it as zero.

Artifact:
`artifacts/cis-small-proposal-short-p16-replay-20260905-r4/assessment.json`

## Pressure-Derived Capacity Candidate

The next formal ABBA campaign tested the observed concurrency pressure instead
of repeating a historical sweep point. It raised base/proposal scheduler
capacity from `40/48` to `128/384`. Token budgets, memory split, waits, score
priority, algorithm semantics, and both graph-capture bucket lists were frozen.
In particular, proposal graph capture still ended at 48, isolating scheduler
capacity from graph-policy expansion.

| Pair | Baseline QPS | Candidate QPS | Directional effect |
| --- | ---: | ---: | ---: |
| seed 2026090571 | 0.2656 | 0.2263 | -14.79% |
| seed 2026090572 | 0.2677 | 0.2383 | -11.00% |
| aggregate | - | - | -12.90% median |

The candidate/baseline geometric-mean ratio was `0.8708`. The effect is outside
the earlier observed `3.32%` replay envelope in magnitude, but the current
assessment deliberately does not attach that reference. The earlier replay did
not explicitly bind graph buckets in its manifest, while this campaign did;
runtime logs show the same effective buckets, but the strict gate does not infer
contract equality from logs. Proposal waiting dropped from a maximum above 320
to zero and observed running requests reached 384, but throughput fell and P95
rose from about 59 seconds to 67-71 seconds. Proposal KV peak remained below
4.8%, so KV exhaustion does not explain the regression.

This rejects queue elimination as a standalone objective. Scheduler capacity,
captured batch range, realized batch efficiency, forward-token work, and SLOs
must be modeled jointly. The negative runs are grade-A response data, not failed
experiments.

Artifact:
`artifacts/cis-small-proposal-short-p16-capacity-128-384-20260905-r1/assessment.json`

## First Offline Policy And Acquisition

The replay and capacity campaigns were merged into eight grade-A feature rows.
A constrained response model evaluated eight legal capacity pairs. It selected
the measured `40/48` fallback: its QPS lower confidence bound was `0.2545`,
versus `0.2233` for measured `128/384`. Three larger interpolated points were
also rejected by conservative latency and accuracy bounds.

The trust-region acquisition planner then excluded both exact observations and
limited the next experiment to one changed knob within normalized model
distance 0.15. Exactly one candidate remained: `40/96`. It changes only
proposal scheduler capacity and keeps proposal graph capture capped at 48. This
is an active-learning interpolation experiment, not a deployment policy. Its
generated campaign contains eight runs: a manifest-identical replay ABBA group
followed by a `40/96` ABBA group. The local group provides a compatible noise
floor without borrowing an incompletely specified older context.

Artifacts:

- `artifacts/cis-small-proposal-short-p16-capacity-policy-20260905.json`
- `artifacts/cis-small-proposal-short-p16-capacity-acquisition-round1-20260905.json`
- `artifacts/cis-small-proposal-short-p16-capacity-round1-20260905.calibration.json`

## Decision And Next Gate

The infrastructure has demonstrated a useful negative and a useful positive:

1. Exact bucket pruning learned from only low-load traces does not generalize
   and is rejected before device benchmarking.
2. Bounded remapping can reduce proposal startup cost without dropping graph
   coverage, but a train safety margin is necessary for context transfer.
3. Base token budget is a censored feasibility variable under long target
   scoring: the repeated adjacent boundary is 3,072 feasible and 3,584 OOM on
   the same fixed workload.

The pressure-derived `128/384` test is complete and harmful. The acquired
one-factor `40/96` experiment below is also complete and harmful, so the next
campaign brackets proposal capacity between 48 and 96 before widening another
knob. Later candidates must come from the confirmed feasible domain and repeat
the same boundary method across context/load regimes and memory splits. Scoring
backend and statistic requirements remain a separate search axis from scheduler
capacity.

## Active Capacity Round 1: 40/96

The acquired `40/96` candidate was run in an eight-observation campaign: four
manifest-identical replay controls followed by a candidate ABBA group. Graph
capture remained capped at proposal batch 48, so the intervention changed only
`proposal_max_num_seqs`.

| Pair | Baseline QPS | 40/96 QPS | QPS effect | P95 effect |
| --- | ---: | ---: | ---: | ---: |
| seed 2026090581 | 0.2805 | 0.2578 | -8.09% | +8.05% |
| seed 2026090582 | 0.2803 | 0.2698 | -3.72% | +4.00% |
| aggregate | - | - | -5.91% median | worse in both pairs |

The candidate/baseline QPS geomean ratio was `0.9407`. The replay-control pair
effects were `+5.11%` and `-4.21%`, giving a `5.11%` local noise envelope; the
candidate regression is outside that envelope. Proposal mean waiting fell from
`207.64` to `170.98` requests and mean running requests rose from `43.33` to
`86.81`, while average total forward-token-slots remained essentially unchanged
(`495,578` versus `495,177`). Larger active batches reduced queue pressure but
made execution slower under the frozen graph policy.

Artifact:
`artifacts/cis-small-proposal-short-p16-capacity-round1-20260905-r1/assessment.json`

## Paired-Aware Selector Correction And Round 2

After adding the two `40/96` response rows, the first response model incorrectly
ranked that measured regression above the fallback. Twelve noisier `40/48`
replicates produced a wider absolute interval, while two close `40/96`
replicates appeared overconfident. Two fail-closed corrections were added:

1. Every target inherits the largest observed within-configuration replicate
   deviation as an operational-noise floor.
2. Complete ABBA pair metadata is retained, and a candidate whose upper paired
   directional effect is negative after the declared minimum repeats receives
   `paired_objective_regression`.

The corrected policy selects `40/48`; `40/96` and `128/384` are rejected by the
paired guard. The original space then emitted `no_candidate`. A v2 bracket space
added `40/64` and `40/80`; acquisition selected `40/64` with score `0.97265`.
That selection was compiled into an eight-run replay-plus-candidate campaign.

## Active Capacity Round 2: 40/64

The bracketed `40/64` campaign completed all eight observations and passed the
formal gates. Its replay-control effects were `+2.33%` and `+3.60%`, producing a
`3.60%` local noise envelope.

| Pair | Baseline QPS | 40/64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090591 | 0.2748 | 0.2016 | -26.63% |
| seed 2026090592 | 0.2576 | 0.1969 | -23.58% |
| aggregate | - | - | -25.11% median |

The candidate/baseline geomean ratio was `0.7488`. Accuracy was unchanged in
aggregate at `0.4375`, while P95 rose from `59.73s` to `80.25s`. Proposal mean
waiting changed little (`220.46` to `218.44`), mean running requests rose from
`43.21` to `59.69`, and the candidate reached the configured maximum of 64.
The proposal graph-capture list still ended at 48. This is a sharp execution
cliff, not evidence for continuing the proposal-capacity sweep to 80.

After merging 24 grade-A rows, the paired-aware policy retains only `40/48` as
eligible. The observed capacity/graph mismatch is now handled by a dedicated,
content-addressed interaction repair planner. It requires a formal regression
outside replay noise plus measured concurrency beyond the capture ceiling. The
generated next candidate is `40/64 + graph64`, compared with the active
`40/48 + graph48` baseline and a fresh replay group. Because the practical
candidate differs from the active baseline in both capacity and graph coverage,
the campaign tests a coupled repair hypothesis rather than claiming an isolated
graph effect.

Artifacts:

- `artifacts/cis-small-proposal-short-p16-capacity-bracket-round2-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-short-p16-graph-repair64-plan-20260905.json`
- `artifacts/cis-small-proposal-short-p16-graph-repair64-20260905.calibration.json`

## Coupled Capacity/Graph Repair: 40/64 + Graph64

The repair campaign completed all eight planned observations. Its replay pair
effects were `+0.13%` and `+9.24%`, giving a conservative same-campaign noise
envelope of `9.24%`.

| Pair | Baseline QPS | 40/64 + graph64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090593 | 0.2746 | 0.3141 | +14.39% |
| seed 2026090594 | 0.2721 | 0.3102 | +14.00% |
| aggregate | - | - | +14.19% median |

The candidate/baseline QPS geomean ratio was `1.1419`, so the gain exceeded the
local replay envelope. Aggregate P95 fell from `58.50s` to `51.01s`, accuracy
remained `0.5`, mean proposal running requests rose from `43.55` to `56.73`,
and mean waiting requests fell from `215.54` to `191.98`. Graph initialization
logs reported the same rounded proposal startup cost (`7s`, `0.32 GiB`) for the
48- and 64-ceiling policies.

This campaign proves that the practical `40/64 + graph64` candidate beats the
active `40/48 + graph48` baseline. It does not by itself assign the full gain to
the extra graph bucket, because capacity and graph policy changed together.

The joint search-space contract now represents graph buckets as a validated
`integer_sequence` knob and derives scalar ceiling, coverage, and uncaptured
capacity features. A policy fit over 28 compatible grade-A rows selects the
measured `40/64 + graph64` cell, rejects `40/64 + graph48` for both paired QPS
regression and its P95 bound, and keeps `40/48 + graph64` as an unmeasured
interpolation. Active acquisition selects that missing cell as the next
one-factor experiment, completing the 2x2 interaction matrix.

Artifacts:

- `artifacts/cis-small-proposal-short-p16-graph-repair64-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-short-p16-capacity-graph-policy-20260905.json`
- `artifacts/cis-small-proposal-short-p16-capacity-graph-next-20260905.plan.json`
- `artifacts/cis-small-proposal-short-p16-graph64-factorial-20260905.calibration.json`

## Isolated Graph64 Factor: 40/48 + Graph64

The missing factorial cell was run on physical NPU2 because the original NPU6
was occupied. The host, models, source tree, container, workload, and ABBA
protocol were held fixed, but the different physical device means this is a
same-card transfer validation rather than evidence that can be merged directly
into the NPU6 policy partition.

All eight observations passed the formal gates. The replay pair effects were
`-4.54%` and `-0.09%`, giving a conservative `4.54%` local noise envelope.

| Pair | 40/48 + graph48 QPS | 40/48 + graph64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090595 | 0.2815 | 0.2799 | -0.58% |
| seed 2026090596 | 0.2650 | 0.2601 | -1.86% |
| aggregate | - | - | -1.22% median |

The candidate/baseline QPS geomean ratio was `0.9878`; quality constraints
passed, but the effect did not exceed replay noise. Graph64 is therefore not an
independent improvement at proposal capacity 48. Combined with the prior
`40/64 + graph64` result, the evidence supports a conditional interaction:
graph coverage becomes useful when scheduler capacity reaches the newly
captured execution domain, rather than graph64 being a globally preferable
static setting.

Artifact:
`artifacts/cis-small-proposal-short-p16-graph64-factorial-npu2-20260905-r1/assessment.json`

## Workload Transfer: Short P16 To P32

A new fail-closed transfer planner bound the p16 policy bundle, NPU2 calibration
template, target p32 arrival trace, explicit guard deviations, and fresh pair
seeds. It generated a replay group plus a selected-policy ABBA group. The target
request count of 32 remained inside the source policy's `[16,128]` range; actual
prompt mean was `117.34`, inside `[64,256]`. Workload ID, worker count, and the
NPU6-to-NPU2 environment identity remained explicit transfer deviations rather
than silently extending the p16 activation guard.

All eight runs passed the formal gates. Replay QPS effects were `-1.67%` and
`-4.72%`, establishing a `4.72%` local noise envelope.

| Pair | 40/48 + graph48 QPS | 40/64 + graph64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090597 | 0.3008 | 0.3293 | +9.49% |
| seed 2026090598 | 0.2903 | 0.3187 | +9.79% |
| aggregate | - | - | +9.64% median |

The candidate/baseline geomean ratio was `1.0964`, beyond replay noise.
Aggregate P95 improved from `107.94s` to `98.69s`; aggregate accuracy was
`0.5000` versus `0.4844`, with the declared quality constraint satisfied. The
proposal running mean increased from `45.07` to `59.84`; waiting mean fell from
`481.29` to `473.78`.

This is positive p32/NPU2 transfer evidence for the complete selected policy.
It remains a separate workload/environment partition and is not independent
holdout evidence or permission to broaden the p16 policy's online guard.

Artifacts:

- `artifacts/cis-small-proposal-short-p32-policy-transfer-20260905.plan.json`
- `artifacts/cis-small-proposal-short-p32-policy-transfer-20260905.calibration.json`
- `artifacts/cis-small-proposal-short-p32-policy-transfer-npu2-20260905-r1/assessment.json`

## Independent P32 Policy Holdout

The p32 response rows above were used to fit a p32-specific two-candidate
policy, so they were not reused as validation evidence. A second eight-run
campaign froze that policy and used fresh seeds `2026090599` and `2026090600`
on the same physical NPU2. The campaign again contained a manifest-identical
replay group followed by a selected-policy ABBA group.

All observations passed the formal gates. Replay effects were `-0.06%` and
`+2.69%`, establishing a `2.69%` local noise envelope.

| Pair | 40/48 + graph48 QPS | 40/64 + graph64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090599 | 0.2879 | 0.3260 | +13.22% |
| seed 2026090600 | 0.2683 | 0.3001 | +11.83% |
| aggregate | - | - | +12.53% median |

The candidate/baseline geomean ratio was `1.1252`, beyond replay noise. P95
fell from `110.70s` and `118.61s` to `98.08s` and `106.02s`, respectively.
Accuracy changed from `0.4375` to `0.46875` in the first pair and from `0.375`
to `0.4375` in the second; both quality constraints passed.

The independent holdout gate was predeclared to require two successful
candidate replicates, no observed failures, at least `3%` improvement over the
fallback, and at most `3%` regret to the measured feasible oracle. It returned
`validated`: mean holdout QPS was `0.3130` for the selected policy versus
`0.2794` for the fallback (`+12.02%`), and the selected policy was the measured
oracle with zero regret. The result validates this policy only for the current
short-context p32/NPU2 partition.

Artifacts:

- `artifacts/cis-small-proposal-short-p32-policy-holdout-npu2-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-short-p32-holdout-features-20260905.json`
- `artifacts/cis-small-proposal-short-p32-policy-holdout-assessment-20260905.json`

## Context Boundary: Short P32 To Medium2K P32

The frozen short-p32 policy was next compared with the same fallback after
raising requested context to 2048 tokens. The actual prompt mean was `2183.66`,
outside the source policy's `[64,256]` activation range. The workload ID change
was predeclared, and prompt length remained a runtime-verified deferred guard.

All eight observations passed the formal gates. Replay effects were `-2.77%`
and `+3.91%`, defining a `3.91%` local noise envelope.

| Pair | 40/48 + graph48 QPS | 40/64 + graph64 QPS | QPS effect |
| --- | ---: | ---: | ---: |
| seed 2026090601 | 0.2062 | 0.2166 | +5.04% |
| seed 2026090602 | 0.2086 | 0.2099 | +0.66% |
| aggregate | - | - | +2.85% median |

The geomean ratio was `1.0283`. Quality constraints passed, but the QPS effect
did not exceed replay noise and missed the predeclared `3%` transfer threshold.
The formal transfer assessment therefore returned `rejected` with
`source_policy_activation_eligible=false`. This is not evidence for widening the
short policy guard or creating a medium2k endpoint.

The same evidence shows why the next search changes stages. Base work accounted
for `97.47%` of estimated dense-forward FLOPs and target scoring for `89.25%` of
forward token slots, while proposal KV peaked near `1.5%` and no preemptions
occurred. The mechanism-guided planner selected a one-factor base token-budget
probe from `10240` to `12288` rather than another proposal-capacity increase.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-policy-transfer-npu2-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-medium2k-p32-policy-transfer-assessment-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-mechanism-probe-plan-20260906.json`

## Medium2K Mechanism Probe: Base Token Budget

The mechanism-guided `10240 -> 12288` base token-budget experiment completed
eight formal runs on NPU2 with seeds `2026090603` and `2026090604`. Replay QPS
effects were `-4.40%` and `-2.19%`; candidate effects were `-12.06%` and
`-0.84%`. The candidate's `-6.45%` median regression was outside the `4.40%`
noise envelope. The predeclared `+3%` gate rejected promotion while retaining
all eight grade-A rows for response fitting.

The updated medium2k selector evaluated 216 legal points but activated only the
measured fallback under its P95 and quality bounds. Auditing the next acquisition
found and fixed two general bugs: unreachable graph buckets could rank as
experiments, and the policy's human-facing candidate report limit could truncate
the acquisition surrogate. A nearer formal one-knob regression now also closes
farther candidates in the same direction. The resulting next candidate changes
only proposal `max_num_batched_tokens` from 12288 to 16384.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-basebt12k-npu2-20260906-r1/assessment.json`
- `artifacts/cis-small-proposal-medium2k-p32-policy-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-acquisition-round1-20260906.json`

## Medium2K Acquisition: Proposal Token Budget

The repaired full-space acquisition selected a one-factor proposal token-budget
increase from 12288 to 16384. Its eight-run NPU2 campaign used fresh seeds
`2026090605` and `2026090606`; all observations were grade A with no formal
issues. Replay effects were `+6.49%` and `+2.29%`. Candidate effects were
`-5.34%` and `+1.28%`, giving a `-2.03%` median inside the `6.49%` replay
envelope. The predeclared result gate returned `inconclusive`, not a speedup or
a directional-regression claim, and retained the rows for response fitting.

Refitting on 24 rows returned `no_feasible_candidate`: even the fallback's P95
estimate was `146.10s +/- 26.25s`, whose `172.35s` upper bound exceeded the
frozen `170s` SLO. No medium2k endpoint may be added to the runtime pool from
this evidence. Before another NPU round, Autopilot needs a sequential paired
effect model that distinguishes shrinkable estimator uncertainty from the
non-shrinking operational noise used for deployment SLO protection.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-npu2-20260906-r1/assessment.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-acquisition-assessment-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-policy-r3-20260906.json`

## Sequential Effect Diagnostic

The new sequential layer replayed the proposal-16K assessment using exact
same-seed replay/candidate quadruples. After directional log-drift subtraction,
seed `2026090605` had adjusted log effect `-0.11781` and seed `2026090606` had
`-0.00989`; their adjusted geomean ratio was `0.93815`.

This is deliberately recorded as `retrospective_diagnostic`, because the data
predated the confidence specification. With only two pairs, the betting-mixture
interval remained the full frozen `[-0.2, 0.2]` support. The audit returns
`diagnostic_only` and neither promotes the candidate nor closes its direction.
The next valid experimental action is to freeze prospective boundaries and a
fresh-seed pool before generating another ABBA campaign.

Artifact:

- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-sequential-effect-20260906.json`

## Prospective Sequential Replication: Proposal Token Budget

The prospective rule was frozen before collecting seeds `2026090607` through
`2026090610`. Two independent eight-run campaigns each contained a replay ABBA
group and a proposal-16K ABBA group. All 16 runs completed with grade-A quality,
no support-bound violations, and no formal assessment issues.

The first campaign's replay-adjusted log effects were `-0.05952` and `-0.04728`.
The second campaign reversed direction, with adjusted effects `+0.03759` and
`+0.02380`. Across all four pairs, the sample mean adjusted log effect was
`-0.01135`, equivalent to an adjusted geomean ratio of `0.98871` (about
`-1.13%`). This is below the frozen `+3%` useful-gain threshold, but the sign
instability also prevents a defensible directional-regression claim.

At four pairs the betting-mixture confidence sequence remains the full frozen
`[-0.2, 0.2]` support. More importantly, projecting the implemented interval to
the 20-pair budget while holding future observations at the current mean gives
`[-0.14063, +0.11792]`, which crosses neither zero nor the useful-gain boundary.
The projection is planning-only, but it shows that mechanically launching rounds
3 through 10 is unlikely to produce a decision with this method. Collection is
therefore paused while a small-sample empirical-Bernstein or predictable plug-in
confidence sequence is benchmarked against the existing betting and
finite-horizon references.

The two campaigns also expose a separate, higher-confidence infrastructure
opportunity. Each campaign started 16 engines. Round 2 spent `705.82s` in engine
initialization, including `528.96s` in compile/profile/warmup and `120.00s` in
graph capture. All three repeated fingerprint groups reused their vLLM cache key
yet repeated compilation and graph capture; the assessor counted 13 repeated
compiler invocations and 13 repeated captures. Together, rounds 1 and 2 spent
`1406.41s` initializing engines, including `1058.15s` in compile/warmup and
`239.00s` in capture. This motivates an isolated prewarmed engine-pool harness,
not another cache-directory experiment.

Artifacts:

- `examples/npu/conditional-is-medium2k-p32-propbt16k.prospective-sequential-effect.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-prospective-sequential-assessment-20260906-r1-r2.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-harness-cost-20260906-r1.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-harness-cost-20260906-r2.json`

## Sequential-Method And Lifecycle Design Audit

No third proposal-16K campaign was launched. A diagnostic comparison replayed
the four frozen adjusted effects through the formal betting method,
finite-horizon Hoeffding, and a predictable plug-in hedged-capital sequence.
The hedged method narrowed the planning-only 20-pair interval to
`[-0.06614, +0.04994]`, versus `[-0.14063, +0.11792]` for the frozen method,
while its minimum simulated simultaneous coverage over all planned looks was
`0.983`. It still crossed no decision boundary and had little simulated power
to promote a true `+0.058` log effect by 20 pairs. This rules out spending more
NPU time on an unchanged evidence design.

The independent round-2 harness evidence was then compiled into a lifecycle
plan. Exact isolated startup was `705.82s` across 16 engine starts. The selected
role-sticky schedule preserves all eight run positions, keeps the invariant
base engine resident, swaps the proposal fingerprint twice, and projects four
starts totaling `150.21s`: a `555.61s` (`78.72%`) reduction. Keeping all three
fingerprints resident would project `119.46s`, but its `1.26` aggregate memory
reservation violates the frozen `0.95` cap.

The lifecycle artifact is not yet formal-execution eligible. Fresh seeds
`2026090611` and `2026090612` are reserved for isolated-versus-pooled ABBA
validation after the reset-capable worker exists.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-sequential-design-study-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-engine-lifecycle-plan-20260906.json`
