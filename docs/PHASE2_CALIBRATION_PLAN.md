# Phase 2 Calibration Plan

## Objective

Build the smallest trustworthy dataset that can answer this question:

> Given an inference graph, model pair and workload regime, which deployment
> configuration meets quality and latency constraints while maximizing useful
> throughput?

The first selector targets `conditional_is_small_proposal`. The contracts remain
general enough for on-policy Conditional IS, SMC, MH and Best-of-N adapters.

## Product Boundary

Inference Autopilot will produce a recommendation artifact, not a second serving
or orchestration runtime:

```text
algorithm adapter -> Inference Graph IR
calibration harness -> evidence ledger
offline selector -> policy.json + vLLM/config patch
guarded controller -> optional regime switch among validated policies
```

vLLM owns token-level scheduling. MARS may own program admission/offloading, and
PIC-KV may own non-contiguous KV reuse. Autopilot models and selects their public
knobs through adapters. Its narrow stage-wavefront policy may batch ready calls
from one algorithm-graph operation before vLLM admission; it does not duplicate
cluster scheduling, offloading, KV management, or token scheduling.

The project remains standalone during development. A future Muyuan integration
should consume the same CLI and schemas instead of moving benchmark-specific
logic into this repository.

## Fixed Semantic Baseline

Every deployment comparison must keep these properties fixed:

- chang's exact `conditional_is_small_proposal` path;
- base candidates, proposal rollouts and base target scoring with importance
  correction enabled;
- the same candidate, rollout, block and total-token budget within a comparison;
- asynchronous submission, native continuous batching, APC, chunked prefill and
  all retained chang optimizations;
- the current retained MRV1 path and sampler unless that component is the named
  variable under test;
- identical model weights, prompts, seeds, arrival trace and quality evaluator.

Approximate pruning, uncorrected IS and numerically non-identical scoring stay in
separate semantic cohorts and Pareto studies.

## Search Space

Phase 2 tunes deployment variables only:

| Group | Initial knobs | Change timing |
| --- | --- | --- |
| Engine capacity | base/proposal `max_num_seqs` | deployment or engine restart |
| Token capacity | base/proposal `max_num_batched_tokens` | deployment or engine restart |
| Memory | base/proposal memory fractions | deployment or engine restart |
| Runtime policy | batch wait and base score priority | validated policy switch |
| Graph-operation admission | stage-wavefront mode, utilization and tail wait | adapter restart or validated policy boundary |
| Placement | colocated, model split or full replicas | deployment |

Seed the space with measured anchors rather than a blind Cartesian grid:

- 40/96: historical low-capacity point;
- 128/768: strongest retained short-context anchor on the frozen BF16 stack;
- 256/896: historical high-capacity candidate, retained as negative evidence
  after its direct `+1.07%` effect fell inside the matching replay envelope;
- 128/1024 and 384/768: known non-monotonic boundary points.

The graph marks candidate count, rollout count and block size as tunable but
quality sensitive. They remain frozen until the deployment-only selector passes
held-out validation.

## Workload Regimes

Calibration must cover at least:

| Regime | Required signal |
| --- | --- |
| low load | 8 and 32 simultaneous requests; latency dominates |
| saturated | 96 and 384 requests; throughput and queueing dominate |
| mixed length | short, about 2K and about 8K prompts in one arrival trace |
| shared prefix | long shared prefix with APC on; cache reuse is observable |
| long scoring | independent long prompts; score memory and base saturation matter |

After these closed-loop traces work, add Poisson and bursty arrivals at multiple
QPS levels. Request count alone is not a sufficient load feature.

## Experimental Protocol

1. Freeze source, environment, model and workload manifests before launch.
2. Compare each candidate with the strongest known manual configuration for that
   workload, not vLLM defaults or a synchronous implementation.
3. Use same-card ABBA or BAAB order with no overlapping NPU jobs. Record warmups
   separately.
4. Run at least two ordered pairs; promote to grade A only when all manifest and
   quality checks pass.
5. Preserve crashes, OOMs and SLO violations as constraint records.
6. Report confidence intervals across repetitions and seeds, not only the best
   run.

Required metrics are throughput, wall time, P50/P95/P99, accuracy, output or
selected-candidate agreement, generated/scored token counts, per-stage service
time, batch occupancy, ready/running queues, KV peak and preemptions.
Admission-policy studies additionally require planned and realized wave widths,
prefix-token volume, partial-wave fraction, admission wait, and zero unaccounted
limit violations.

## Staged Execution

### Stage 2A: Reproduce Anchors

Run 40/96, 128/768 and 256/896 under low-load, saturated and long-scoring
regimes. The goal is not a new speedup claim; it is to determine which historical
relationships survive under a frozen current stack.

Exit criterion: at least one grade-A comparison group per regime and no missing
provenance fields.

### Stage 2B: Constrained Offline Search

Use grade-B runs to seed bounds, then select new configurations based on measured
uncertainty and feasibility. The first model should be simple and inspectable:
regime classification plus a per-regime response surface or tree model. Add a
Bayesian optimizer only if it reduces required evaluations in replayed traces.

Exit criterion: on held-out workloads, the recommendation is within 5% of the
exhaustive manual-best throughput while satisfying the same P95, memory and
quality constraints.

### Stage 2C: Search-Efficiency Evaluation

Compare Autopilot with random search, the historical manual anchors and a dense
local sweep. Measure number of NPU trials and wall-clock calibration cost needed
to find a configuration within 5% of manual best.

Exit criterion: fewer trials than the dense sweep without weakening the selected
baseline or changing algorithm semantics.

### Stage 2D: Guarded Online Policies

Package a small set of validated policies for low-load, saturated and
long-scoring regimes. Switch only at safe boundaries and use hysteresis and
rollback on P95, preemption or quality alarms. Parameters that require engine
restart remain static recommendations.

Exit criterion: mixed traces improve the declared throughput/latency objective
without oscillation or new failures versus the best single static policy.

## Next Code Increment

The next implementation should add:

1. a run manifest schema containing all provenance and semantic invariants;
2. a harness wrapper that emits paired run groups directly into the ledger;
3. a deployment search-space schema with constraints and restart requirements;
4. a feature extractor for prompt length, arrival load, queueing, stage mix and
   KV pressure;
5. validation that prevents records from different semantic cohorts entering
   the same deployment-only fit.

Current status: all five items, the formal protocol gate, the first
`conditional_is_small_proposal` pressure-runner adapter, deterministic
search-space compilation, budgeted candidate selection, graph-profile corpus
search, ordered censored feasibility search, conservative response-model
selection, and trust-region acquisition are implemented. Formal short-p16
capacity experiments found non-monotonic regressions at proposal capacities 64
and 96. The capacity/graph repair planner now turns a measured domain crossing
into a bound follow-up experiment instead of letting the scalar tuner continue
the sweep. The resulting `40/64 + graph64` campaign improved median paired QPS
by `14.19%` beyond its replay envelope. Structured graph policies are now part
of the compiled search space, and the joint model has selected the final missing
`40/48 + graph64` factorial cell as a one-knob acquisition. Broader context/load
partitions and held-out policy validation remain the next evidence milestone.
The missing graph64 factorial cell has since shown no independent gain at
capacity 48, while the selected combined policy transferred from p16 to p32 at
`+9.64%` median paired QPS beyond replay noise. A content-addressed policy
transfer planner now makes workload and environment guard deviations explicit
and emits a runnable selected-versus-fallback campaign. A fresh-seed p32
holdout then validated the frozen p32 policy at `+12.53%` median paired QPS
against a `2.69%` replay envelope. A second context-length regime and guarded
routing remain outstanding. The first content-addressed runtime pool and
fail-closed request-group decision core are now implemented; mixed-trace replay,
endpoint adapters, and a second context-length policy remain outstanding.
The first medium2k transfer has since been rejected: its `+2.85%` median QPS
effect remained inside `3.91%` replay noise and crossed the source prompt guard.
A graph- and telemetry-guided bootstrap planner now diagnoses sparse workload
partitions before response-model fitting; its first real plan selected a
one-factor base token-budget probe from 10240 to 12288. A validated medium2k
policy remains outstanding. That probe formally regressed by `-6.45%`. A
full-surrogate acquisition then tested proposal token budget 16384 and returned
an inconclusive `-2.03%` median inside `6.49%` replay noise. The 24-row refit
returned `no_feasible_candidate` because the fallback's conservative P95 upper
bound was `172.35s`, above the frozen `170s` SLO. The next code increment is a
sequential paired-effect model that separates estimator uncertainty from
non-shrinking operational/SLO variability before allocating more NPU repeats.
That increment is now implemented with replay-adjusted extraction, prospective
versus retrospective modes, two time-uniform confidence methods, strict
content/context audit, and fresh-seed calibration compilation. The historical
proposal-16K result remains diagnostic only. The next evidence action is to
freeze a prospective replication spec, then collect a complete fresh-seed ABBA
block only if an idle device is available.
