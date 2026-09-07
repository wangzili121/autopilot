# Inference Autopilot

Inference Autopilot is a standalone project for describing inference-scaling
algorithms as performance graphs and turning heterogeneous benchmark artifacts
into auditable optimization evidence. It is being developed independently and
can later be integrated into Muyuan as an infra plugin.

For a Chinese project-level view of the product, verified progress, measured
effects, and upcoming milestones, see the [project progress center](project/README.md).
Paper surveys and individual readings are indexed in [papers](papers/README.md).
Existing systems, vendor tools, overlap, and remaining project boundaries are
tracked in [related work](related_work/README.md).

The project now provides the contracts and the first workload-aware policy
planner needed by an automatic tuner:

- an Inference Graph IR that separates algorithm semantics, model work, CPU
  work, data dependencies and tunable runtime fields;
- an evidence ledger that preserves source hashes and grades every observation
  before it can influence a selector.
- a constrained graph-capture planner that selects CUDA or ACL capture buckets
  from stage-labelled workload profiles and measured replay costs.
- a phase-aware vLLM/vLLM-Ascend graph probe that separates prefill, decode and
  mixed scheduler steps before proposing capture or admission changes.
- a graph- and context-aware stage wavefront adapter that automatically packs
  Conditional IS proposal calls under sequence and prefill-token limits.
- a capability-gated exact-scoring planner and NPU probe for selected-token,
  top-k confidence, and later entropy reductions.
- a conservative offline response model that emits content-addressed policy
  bundles with SLO, failure-domain, activation, and manual-fallback guards.
- a trust-region active-learning planner that chooses the next unmeasured
  experiment from policy uncertainty, optimistic improvement, distance, and
  configuration-change cost.
- an evidence-gated interaction repair planner that detects scheduler capacity
  extending beyond the measured graph domain and generates a bound ABBA test
  instead of continuing an unsafe one-dimensional sweep.
- a runtime-configuration closure attestation that binds requested settings to
  vLLM's resolved compilation ranges, graph buckets, KV capacity, and startup
  resource costs before those fields enter the response model.
- a fail-closed policy-transfer planner that turns workload or environment
  extrapolation into an explicit selected-versus-fallback ABBA experiment.
- a formal transfer assessor that distinguishes reusable target-policy evidence
  from permission to activate the source policy.
- an independent holdout gate that measures gain over the manual baseline and
  regret to the best candidate actually measured on the holdout.
- a guarded runtime policy pool that routes only to validated, configuration-
  attested prewarmed endpoints and otherwise uses the manual fallback.
- a two-level host interference guard that content-addresses CPU, memory and
  all-NPU telemetry and blocks polluted shared-host runs before assessment.
- an execution-readiness gate that turns audited retry outcomes and physical
  run durations into a pre-launch completion-probability and NPU-cost decision.
- an AIC-NPU stage cost model that fits sparse measurements inside discrete
  NPU shape buckets, emits uncertainty intervals, and abstains outside support.

It does not replace vLLM token scheduling, MARS program admission/offloading,
PIC-KV reuse, or the inference-scaling algorithm implementation. Its optional
wavefront wrapper contributes algorithm-operation boundaries that a flat engine
request stream cannot infer. Later selectors can choose among these mechanisms
through adapters without owning their broader execution policy.

## Current commands

Install the plugin in editable mode:

```bash
python3 -m pip install -e .
```

Describe chang's exact Conditional IS small-proposal path:

```bash
inference-autopilot graph \
  --candidate-count 4 \
  --rollout-count 3 \
  --block-size 16 \
  --total-length 128 \
  --output conditional-is-graph.json
```

Normalize existing inference-scaling results and inspect their evidence grade:

```bash
inference-autopilot import-results /path/to/results/profiling \
  --output evidence-ledger.json
inference-autopilot audit evidence-ledger.json
```

Turn the graded ledger into a sparse selector table and audit feature coverage:

```bash
inference-autopilot features-ledger evidence-ledger.json \
  --output selector-features.json
inference-autopilot audit-features selector-features.json
```

For a newly completed run, extract its workload, graph-demand and telemetry
features before assessment with:

```bash
inference-autopilot features-run run-manifest.json observation.json \
  --output pending-features.json
```

This row is deliberately `ungraded`; only the output ledger from
`assess-calibration` can make a record eligible for selector fitting.

Fit and query the first AIConfigurator-inspired Ascend cost layer without a
device run:

```bash
inference-autopilot fit-npu-cost-model \
  examples/npu-stage-calibration.example.json \
  --output npu-stage-cost-model.json
inference-autopilot predict-npu-graph-cost \
  npu-stage-cost-model.json \
  examples/npu-graph-cost-query.example.json \
  --output npu-graph-cost-prediction.json
```

The example is synthetic. Real profiles bind environment, model-set, and
effective-configuration hashes. See [the AIC-NPU cost model](docs/AIC_NPU_COST_MODEL.md).

Compile the bounded deployment space into immutable, runner-ready candidates:

```bash
inference-autopilot compile-space \
  examples/conditional-is.deployment-space.example.json \
  --capabilities examples/npu/vllm-ascend-0.18.capabilities.example.json \
  --output compiled-space.json
inference-autopilot audit-space compiled-space.json
inference-autopilot space-config compiled-space.json CANDIDATE_ID \
  --output candidate-configuration.json
```

The compiler expands choice/range domains deterministically, rejects candidates
that violate hard constraints, and binds every candidate and the full artifact
to stable hashes. Algorithm-changing knobs require explicit opt-in and are kept
in separate semantic cohorts. Capability-gated values such as MRV2 require a
measured environment profile; unsupported and unknown values are removed before
candidate planning. See [the search-space compiler](docs/SEARCH_SPACE_COMPILER.md).

Plan exact scoring reductions separately from scheduler capacity. This example
models the observed long-context shape and Consilience's selected-token plus
top-k statistics:

```bash
inference-autopilot plan-score-reduction \
  --reward consilience \
  --importance-correction \
  --positions 3584 \
  --vocab-size 151936 \
  --memory-budget-gib 1.91 \
  --capabilities examples/npu/vllm-ascend-0.18-scoring.capabilities.example.json \
  --output score-reduction-plan.json
```

See [the scoring reduction plan](docs/SCORING_REDUCTION_PLAN.md).

Search knobs also declare a `tuning_layer` and optional graph-stage bindings.
Validate those bindings while compiling:

```bash
inference-autopilot graph --output conditional-is-graph.json
inference-autopilot compile-space \
  examples/conditional-is.deployment-space.example.json \
  --graph conditional-is-graph.json \
  --output compiled-space.json
```

Plan graph capture buckets from a stage-labelled cost profile:

```bash
inference-autopilot plan-graph-capture \
  examples/conditional-is-base.acl-graph-profile.example.json \
  --output capture-plan.json
inference-autopilot audit-graph-capture capture-plan.json
```

The checked-in profile contains illustrative numbers for offline verification,
not measured NPU performance. See [the graph capture planner](docs/GRAPH_CAPTURE_PLANNER.md).

Extract the real stage and call-shape distribution from a Conditional IS
pressure result before designing graph calibration arms:

```bash
inference-autopilot trace-graph-workload result.json \
  --trace-id conditional-is-short-p96 \
  --output graph-trace.json
inference-autopilot audit-graph-trace graph-trace.json
```

The NPU profile launcher flushes vLLM's internal graph-shape counters at engine
shutdown. Import and audit them independently of the request-group trace:

```bash
inference-autopilot import-vllm-graph-metrics runner.log \
  --profile-id conditional-is-short-p96 \
  --output vllm-graph-metrics.json
inference-autopilot audit-vllm-graph-metrics vllm-graph-metrics.json
inference-autopilot diagnose-vllm-graph-runtime vllm-graph-metrics.json
```

The optional v0.18 instrumentation is enabled by setting
`GRAPH_PHASE_METRICS=1` with `MODEL_RUNNER=MRV1` in the NPU launcher. It patches
only observability records: model execution, scheduling and sampling semantics
are unchanged. Legacy five-column logs remain importable but fail closed for
phase diagnosis.

For Conditional IS pressure profiling, `PROPOSAL_STAGE_WAVEFRONT=auto` enables
the stage-wavefront adapter and requires an attested
`PROPOSAL_GRAPH_CAPTURE_CEILING`. The planner derives the nominal width from
outer concurrency and rollout fanout; runtime packing also enforces the proposal
token budget against actual prefix lengths. See
[stage wavefront admission](docs/STAGE_WAVEFRONT_ADMISSION.md).
The medium/long-context extension is intentionally separate because one
algorithm caller group can exceed the scheduler token budget; see
[context-aware wavefront sharding](docs/CONTEXT_AWARE_WAVEFRONT_SHARDING.md).

The same planner can be inspected without a device run:

```bash
inference-autopilot plan-stage-wavefront \
  --outer-concurrency 96 \
  --sequences-per-group 24 \
  --graph-capture-ceiling 512 \
  --scheduler-sequence-cap 768 \
  --output stage-wavefront-plan.json
```

Merge compatible profiles from several workload regimes before pruning, then
evaluate the candidate on a held-out profile:

```bash
inference-autopilot merge-vllm-graph-metrics \
  short-profile.json mixed-profile.json long-profile.json \
  --profile-id graph-training-corpus \
  --output merged-graph-metrics.json

inference-autopilot evaluate-vllm-graph-policy \
  graph-experiment-plan.json holdout-profile.json \
  --policy-id trace-preserving-pruned \
  --output holdout-coverage.json
```

For an experiment candidate, use the fail-closed corpus path instead of relying
on an informal merge. It binds each profile to one train or holdout regime,
forbids a regime from appearing in both splits, and requires independent
holdout profiles before producing an ABBA plan:

```bash
inference-autopilot build-vllm-graph-corpus \
  --corpus-id conditional-is-load-corpus \
  --train load8=load8-profile.json \
  --train load32=load32-profile.json \
  --holdout load96=load96-profile.json \
  --output graph-profile-corpus.json

inference-autopilot plan-vllm-robust-graph-experiment \
  graph-profile-corpus.json \
  --campaign-id conditional-is-robust-graph-abba \
  --minimum-train-regimes 2 \
  --minimum-holdout-regimes 1 \
  --blocks 2 \
  --pair-seeds 2026090501 2026090502 2026090503 2026090504 \
  --promotion-output graph-policy-promotion.json \
  --output graph-experiment-plan.json
```

The promotion artifact is written even when the command exits with status 2.
Rejection reasons distinguish too few regimes, holdout remapping, and the
important no-op case where the union of training traces already uses every
framework bucket. A no-op is evidence to stop exact bucket pruning, not a reason
to manufacture a performance experiment.

When exact pruning is exhausted, generate a bounded-remapping candidate with an
ordered, resource-constrained shortest-path search. A Pareto dynamic program
represents the full framework bucket-subset space without materializing its
`2^N` members, minimizes bucket count on the train split, and gates the frozen
policy on every holdout profile. Use stricter train limits as a predeclared
safety margin when deployment acceptance allows a wider envelope:

```bash
inference-autopilot plan-vllm-coverage-graph-experiment \
  graph-profile-corpus.json \
  --campaign-id conditional-is-bounded-graph-abba \
  --policy-id bounded-remap-train4-holdout5 \
  --candidate-engine-role proposal \
  --minimum-retained-graph-fraction 1.0 \
  --maximum-remapped-graph-fraction 0.04 \
  --maximum-added-padding-ratio 0.005 \
  --holdout-maximum-remapped-graph-fraction 0.05 \
  --holdout-maximum-added-padding-ratio 0.005 \
  --blocks 2 \
  --pair-seeds 2026090511 2026090512 2026090513 2026090514 \
  --search-output graph-bucket-search.json \
  --output graph-experiment-plan.json
```

The search artifact records the represented subset count, evaluated transitions,
retained non-dominated states, one best point for every feasible bucket count,
the selected train worst case, and each profile's holdout result. Omitted
`--holdout-*` options inherit the corresponding train limit. Holdout profiles
never influence subset selection; they only accept or reject the frozen
candidate. This is candidate generation only; throughput still has to beat the
replay-noise envelope in paired device experiments.

`--candidate-engine-role` scopes search to `base` or `proposal` and freezes the
other role at the framework default. Repeat the option to search both roles;
omit it for the same two-role behavior. Role scoping is useful when measured
graph hit rate and capture cost identify only one engine as the intervention
target.

Generate content-addressed ABBA runs for the explicit vLLM default,
trace-preserving pruning and a no-graph ablation:

```bash
inference-autopilot plan-vllm-graph-experiment vllm-graph-metrics.json \
  --campaign-id conditional-is-graph-abba \
  --blocks 1 \
  --pair-seeds 2026090401 2026090402 \
  --include-replay-control \
  --output graph-experiment-plan.json
```

The lightweight graph experiment is a diagnostic profiler. Assess one group
with strict source, policy, graph-log and output checks:

```bash
inference-autopilot assess-vllm-graph-experiment \
  graph-experiment-plan.json artifacts/ \
  --comparison-id conditional-is-graph-abba--replay-control \
  --output graph-assessment.json
```

For a formal run, bind measured graph policies to a calibration template that
already freezes the workload, environment, source and quality/SLO contract:

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

The chang adapter applies `base/proposal_graph_mode` and role-specific capture
sizes from each immutable run manifest. A partial or invalid graph policy fails
formal preflight.

Build a small, replayable initial design instead of running the whole space:

```bash
inference-autopilot plan-candidates \
  examples/conditional-is-short-p96.design.example.json \
  examples/conditional-is.deployment-space.example.json \
  compiled-space.json \
  --features selector-features.json \
  --output candidate-plan.json

inference-autopilot candidate-calibration-spec \
  examples/conditional-is-short-p96.calibration.example.json \
  candidate-plan.json \
  --output selected-calibration-spec.json
```

The planner selects the strong baseline first, then compatible evidence anchors,
uncovered domain/constraint boundaries, and maximin space-filling points. The
selection is bound to workload, environment, graph, evidence table, and compiled
space digests. See [the budgeted candidate planner](docs/BUDGETED_CANDIDATE_PLANNER.md).

Bracket an ordered resource-feasibility boundary before spending calibration
trials on unsafe settings:

```bash
inference-autopilot plan-ordered-feasibility \
  long8k-base-token-feasibility.json \
  --output long8k-base-token-feasibility-plan.json
```

Successful and resource-exhausted attempts are retained as source-addressed
censored evidence. The planner requests one midpoint probe at a time, supports
repeat-confirmation thresholds, and rejects contradictory monotonic evidence.
The NPU runner also records nonzero attempts with an explicit outcome, exit
codes, log digest and post-run process-memory snapshot, so failed probes remain
auditable inputs instead of disappearing from the search history.
See [the ordered feasibility planner](docs/ORDERED_FEASIBILITY_PLANNER.md).

Shared-host campaigns use
[retry-aware execution](docs/RETRY_AWARE_EXECUTION.md): a bounded telemetry
window admits each launch, environment-only failures are quarantined and retried
without changing the logical ABBA position or seed, and a hash-chained attempt
ledger must bind exactly one clean result per planned run before assessment.

Before spending another complete run, estimate whether the current host history
can finish the frozen campaign within its reliability and NPU-hour budget:

```bash
inference-autopilot assess-execution-readiness \
  examples/npu/conditional-is-medium2k-p32-off16k.execution-readiness.json \
  target-plan.json history-campaign/ \
  --output execution-readiness-assessment.json
```

The assessment is both content-addressed and source-bound. Passing it as
`EXECUTION_READINESS_ASSESSMENT` makes the campaign launcher verify the exact
target-plan digest and fail closed before NPU work. See
[the execution-readiness gate](docs/EXECUTION_READINESS.md).

Select a deployment policy only after formal calibration has produced grade-A
feature rows:

```bash
inference-autopilot select-policy \
  examples/conditional-is-short-p96.policy-selection.example.json \
  compiled-space.json selector-features.json \
  --output policy-bundle.json
inference-autopilot audit-policy policy-bundle.json
```

The selector uses conservative objective/SLO bounds, local failure evidence and
an explicit extrapolation limit. It emits a manual fallback and activation
guard even when a candidate is selected; with the current legacy ledger it
correctly exits with `insufficient_evidence`. Independent validation compares a
frozen policy with both that fallback and the measured holdout oracle. See
[the offline policy selector](docs/OFFLINE_POLICY_SELECTOR.md).

Choose the next paired experiment without manually taking the next grid point:

```bash
inference-autopilot plan-policy-experiments \
  policy-acquisition.json policy-bundle.json \
  compiled-space.json selector-features.json \
  --output policy-experiment-plan.json
inference-autopilot policy-experiment-calibration-spec \
  calibration-template.json policy-experiment-plan.json \
  --include-replay-control \
  --campaign-id capacity-graph-next-r1 \
  --pair-seeds 11 22 \
  --output selected-calibration-spec.json
inference-autopilot assess-policy-experiment \
  acquisition-assessment.json policy-experiment-plan.json \
  calibration-assessment.json --output acquisition-assessment.json
```

The acquisition planner excludes exact observations, enforces a configurable
knob and model-distance trust region, and rewards both possible improvement and
decision-relevant uncertainty. It recomputes the full surrogate candidate set
even when the policy bundle truncates candidates for reporting, rejects graph
bucket changes outside the scheduler's reachable execution domain, and does not
continue farther past a formally regressive one-knob probe. The calibration
compiler overlays selected knobs on the full baseline. Graph bucket policies
can be modeled directly as validated `integer_sequence` knobs; the selector
derives scalar ceiling and capacity-coverage features rather than comparing
variable-length lists.
`--include-replay-control` adds a manifest-identical group to estimate noise
inside the same campaign. See
[active-learning experiment acquisition](docs/ACTIVE_LEARNING_EXPERIMENTS.md).
The next evidence layer separates shrinkable treatment-effect uncertainty from
non-shrinking deployment variation; see
[the sequential paired-effect design](docs/SEQUENTIAL_EFFECT_MODEL.md).

Apply that layer to one or more formal calibration assessments, then compile
fresh-seed replication without changing the candidate or control:

```bash
inference-autopilot assess-sequential-effect \
  sequential-effect.json calibration-assessment-r1.json \
  --output sequential-effect-assessment.json
inference-autopilot audit-sequential-effect sequential-effect-assessment.json
inference-autopilot sequential-effect-calibration-spec \
  sequential-effect.json calibration-template.json \
  --campaign-id candidate-r2 --pair-seeds 101 102 \
  --output candidate-r2.calibration.json
```

The effect spec freezes exact semantic, workload, environment and deployment
hashes, a bounded log-effect range, alpha, minimum useful gain, pair budget and
fresh-seed pool. It supports a betting-mixture e-process and a finite-horizon
Hoeffding reference. Existing data can only be marked
`retrospective_diagnostic`; only a `prospective` spec frozen before new pairs
may emit `promote` or `close_direction`. Effect-bound violations invalidate the
decision instead of being clipped. Every assessment also projects its interval
at the frozen maximum pair budget under an explicit constant-future-mean
assumption. That projection is planning evidence only: it can stop wasteful
collection planning, but cannot promote or close a candidate.

Compare finite-budget methods on the frozen observations without changing the
formal source decision:

```bash
inference-autopilot compare-sequential-designs \
  sequential-design-study.json sequential-effect-assessment.json \
  --output sequential-design-study-result.json
inference-autopilot audit-sequential-design-study \
  sequential-design-study-result.json
```

The diagnostic study includes a predictable plug-in hedged capital sequence,
planned-look simulations under observed-residual and bounded-endpoint noise,
and a separate `exclude_useful_gain` outcome for search pruning. See
[the sequential design study](docs/SEQUENTIAL_DESIGN_STUDY.md).

Bootstrap a sparse workload partition from graph structure and formal runtime
pressure before a statistical response model has enough distinct configurations:

```bash
inference-autopilot plan-mechanism-probes \
  mechanism-probe.json graph.json compiled-space.json selector-features.json \
  --output mechanism-probe-plan.json
inference-autopilot mechanism-probe-calibration-spec \
  calibration-template.json mechanism-probe-plan.json \
  --include-replay-control --pair-seeds 11 22 \
  --output mechanism-probe.calibration.json
```

The planner binds compute, score/generation token slots, queue pressure, KV and
graph coverage to the stages affected by each legal knob. It emits measurements,
not policies, and fails closed without formal same-partition control evidence.
See [mechanism-guided probes](docs/MECHANISM_GUIDED_PROBES.md).

When a formal capacity probe regresses after observed concurrency crosses the
configured graph-capture ceiling, create a coupled repair experiment:

```bash
inference-autopilot plan-capacity-graph-repair \
  calibration-assessment.json calibration-spec.json \
  --repair-id proposal-graph-repair-64 \
  --output capacity-graph-repair-plan.json
inference-autopilot capacity-graph-repair-calibration-spec \
  calibration-spec.json capacity-graph-repair-plan.json \
  --campaign-id proposal-graph-repair-64-r1 \
  --pair-seeds 11 22 \
  --output repair-calibration-spec.json
```

The planner requires a complete paired regression outside the replay envelope,
candidate records bound to the frozen spec, and runtime evidence that the
affected engine crossed its graph domain. It records the one-setting repair
relative to the failed cell separately from the combined delta relative to the
active baseline. See
[capacity and graph interaction repair](docs/CAPACITY_GRAPH_INTERACTION.md).

Validate a selected policy under a different workload or environment without
silently widening its activation domain:

```bash
inference-autopilot plan-policy-transfer \
  policy-transfer-spec.json policy-bundle.json calibration-template.json \
  --output policy-transfer-plan.json
inference-autopilot audit-policy-transfer policy-transfer-plan.json
inference-autopilot policy-transfer-calibration-spec \
  policy-transfer-plan.json --output transfer-calibration-spec.json
```

The planner names every activation-guard deviation, blocks undeclared
extrapolation, binds the target arrival trace, and preserves the policy fallback
plus a replay control. See [guarded policy transfer](docs/POLICY_TRANSFER.md).
The first frozen short-p32 policy has also passed an independent fresh-seed
holdout: `+12.53%` median paired QPS against a `2.69%` replay envelope, with the
predeclared holdout assessor reporting `validated` and zero measured-oracle
regret. This validation remains scoped to the p32/NPU2 workload partition.

Compile validated policies into prewarmed runtime endpoints, then make a
content-addressed request-group decision:

```bash
inference-autopilot compile-runtime-policy-pool \
  runtime-policy-pool-spec.json \
  --policy policy-bundle.json \
  --assessment policy-holdout-assessment.json \
  --output runtime-policy-pool.json
inference-autopilot route-runtime-policy \
  runtime-policy-pool.json runtime-routing-request.json \
  --output runtime-routing-decision.json
```

The router verifies semantic and activation guards, endpoint readiness, complete
configuration hashes, failure counters, and live SLO metrics. Unknown context,
unsafe boundaries, ambiguity, and endpoint health violations fail closed to the
manual fallback. Engine-restart settings are never mutated per request. See
[guarded runtime policy routing](docs/GUARDED_RUNTIME_ROUTING.md).

Use `--strict` during import when an unsupported or malformed JSON artifact
should fail the command after the ledger has been written.

## Evidence policy

| Grade | Meaning | Allowed use |
| --- | --- | --- |
| `A_formal_paired` | Repeated, ordered comparison against an equivalent strong baseline | selector fitting and performance claims |
| `B_controlled_single` | Controlled end-to-end observation without sufficient repeats | calibration and prior construction only |
| `C_diagnostic` | Smoke, simulation or microbenchmark | diagnostics only |
| `X_excluded` | Failed, confounded or semantically incompatible run | constraints only, never throughput fitting |

The legacy importer is conservative: it never infers grade A from a filename or
from a single sweep. Unknown formats are recorded as rejections instead of being
silently coerced.

Accumulate independently assessed campaigns before feature extraction:

```bash
inference-autopilot merge-ledgers replay-assessment.json candidate-assessment.json \
  --output calibration-ledger.json
```

Both standalone ledgers and calibration assessments are accepted. Exact
duplicate records are removed. A shared record ID with different content, or
incompatible ledger producers, fails closed.

## Formal calibration

Turn a frozen experiment specification into deterministic ABBA/BAAB runs:

```bash
inference-autopilot plan-calibration \
  examples/conditional-is-short-p96.calibration.example.json \
  --output calibration-plan.json

inference-autopilot run-manifest calibration-plan.json RUN_ID \
  --output run-manifest.json

inference-autopilot assess-calibration calibration-plan.json runs/ \
  --output calibration-assessment.json

inference-autopilot assess-harness-cost calibration-plan.json campaign/ \
  --output harness-cost-assessment.json
inference-autopilot audit-harness-cost harness-cost-assessment.json

inference-autopilot attest-runtime-closure calibration-plan.json campaign/ \
  --output runtime-closure.json
inference-autopilot audit-runtime-closure runtime-closure.json

inference-autopilot enrich-runtime-features selector-features.json \
  runtime-closure.json --output runtime-enriched-features.json
```

The external workload runner must copy `run_id` and `run_manifest_sha256` into
its observation. The assessor checks manifest binding, actual execution order,
non-overlap, required metrics, repeated pairs and the strong-baseline assertion
before producing grade-A evidence.

The assessment reports paired primary-metric effects and recognizes an
identical-configuration candidate as a replay control. Its largest formal pair
variation defines a conservative noise envelope, so a tuner can distinguish a
measured candidate effect from ordinary live-policy and harness drift.

Harness cost is assessed separately from steady-state throughput. The cost
artifact binds every manifest, effective configuration, and runner log, then
groups engine startup observations by a model/software/source/configuration
fingerprint. Reusing a vLLM cache key is not reported as a cache hit when device
compilation or graph capture still repeats. See
[the harness cost model](docs/HARNESS_COST_MODEL.md).

Runtime closure is a separate fail-closed gate. It parses each engine's resolved
vLLM configuration and binds it to the run manifest, effective configuration,
and raw log. This exposes implicit graph-capacity coupling and prevents two
nominally similar candidates from entering the selector with hidden runtime
defaults. See [runtime configuration closure](docs/RUNTIME_CONFIGURATION_CLOSURE.md).

Compile measured cost evidence into an order-preserving engine lifecycle plan:

```bash
inference-autopilot plan-engine-lifecycle \
  engine-lifecycle.json calibration-plan.json harness-cost-assessment.json \
  --output engine-lifecycle-plan.json
inference-autopilot audit-engine-lifecycle engine-lifecycle-plan.json
```

The planner compares isolated, role-sticky, and fully resident strategies under
a frozen memory cap. It emits a complete content-addressed action schedule and
an explicit epoch-reset contract, but always requires isolated-versus-pooled
validation before persistent runs can become formal evidence. Round 2 projects
16 engine starts down to four and `78.72%` less startup time. See
[the lifecycle harness design](docs/ENGINE_LIFECYCLE_HARNESS.md).

Prepare one manifest-bound launch bundle for chang's real small-proposal path:

```bash
inference-autopilot prepare-run calibration-plan.json RUN_ID \
  --source-repo /path/to/inference_scaling \
  --source-config configs/gsm8k_full.toml \
  --data data/gsm8k/test.jsonl \
  --output-dir artifacts/RUN_ID \
  --require-formal
```

The command performs a no-NPU static audit and writes `run-manifest.json`,
`effective-config.json`, and `launch.json`. Execute the structured command in
`launch.json` only after `compatibility.formal_eligible` is true. The worker
records request-level tail latency, both engine batch statistics, vLLM runtime
metrics, compute counters, outputs, and source binding, then emits a standard
observation. Launches bind both chang's source snapshot and the complete
plugin-side Python implementation. A manifest may additionally select the
stage-wavefront proposal admission policy; `auto` requires an explicit graph
capture policy and records planned versus realized waves. See
[the chang runner adapter](docs/CHANG_RUNNER_ADAPTER.md) and
[stage-wavefront admission](docs/STAGE_WAVEFRONT_ADMISSION.md).

The feature table keeps deployment-time context, observed runtime telemetry and
optimization targets in separate namespaces. It also derives Conditional IS
graph pressure such as `C * R` parallel width and proposal/target-scoring token
slot upper bounds. See [workload and graph features](docs/WORKLOAD_FEATURES.md).

## Conditional IS graph

The current adapter models this dependency graph per generation step:

```text
base candidate generation (C)
  -> proposal rollout generation (C * R)
       -> base target scoring (C * R) --+
       -> CPU reward (C * R) ----------+-> IS reduction (C) -> selection (1)
```

Target scoring is omitted only for the explicitly biased, uncorrected ablation.
The graph records upper bounds because EOS may terminate candidates or rollouts
early. Dependencies describe data readiness; physical batching remains owned by
the serving engines.

## Roadmap

1. Import and grade the existing manual sweeps; identify missing workload and
   provenance fields.
2. Use the manifest-bound chang pressure runner to emit grade-A paired records
   for low-load, saturated, mixed-length and long-scoring regimes.
3. Fit a constrained offline selector over runtime capacity, token budgets,
   memory split, placement and graph-operation admission, with crashes and
   oversized waves treated as feasibility evidence.
4. Compare the selector against the best manual sweep, not framework defaults.
5. Add guarded online regime switching only after the offline selector is
   reproducibly useful.

The design is informed by recent systems work including
[AIConfigurator](https://arxiv.org/abs/2601.06288),
[SLO-Guard](https://arxiv.org/abs/2604.17627),
[SlidingServe](https://arxiv.org/abs/2606.05933),
[WAR](https://arxiv.org/abs/2607.17299), and
[MISA-T](https://arxiv.org/abs/2608.11152). These are design inputs, not claims
that their reported gains transfer to this workload.

## Paper reading records

- [Paper survey, reading index, and individual Chinese interpretations](papers/README.md)
- [2026 autotuning survey and revised roadmap](papers/2026_AUTOTUNING_SURVEY_AND_ROADMAP.md)

## Design records

- [v1 architecture and milestone gates](docs/V1_ARCHITECTURE.md)
- [Conditional IS graph-bucket experiment log, 2026-09-05](docs/EXPERIMENT_LOG_20260905.md)
- [Conditional IS optimization experiment log, 2026-09-06](docs/EXPERIMENT_LOG_20260906.md)
- [Conditional IS medium-context experiment log, 2026-09-07](docs/EXPERIMENT_LOG_20260907.md)
- [Workload-aware Graph Capture Planner](docs/GRAPH_CAPTURE_PLANNER.md)
- [Phase 1 data audit](docs/PHASE1_DATA_AUDIT.md)
- [Phase 2 calibration plan](docs/PHASE2_CALIBRATION_PLAN.md)
- [Calibration harness](docs/CALIBRATION_HARNESS.md)
- [Host interference guard](docs/HOST_INTERFERENCE_GUARD.md)
- [Chang runner adapter](docs/CHANG_RUNNER_ADAPTER.md)
- [Workload and graph features](docs/WORKLOAD_FEATURES.md)
- [Phase 3 feature-table audit](docs/PHASE3_FEATURE_AUDIT.md)
- [Search-space compiler](docs/SEARCH_SPACE_COMPILER.md)
- [Harness cost model](docs/HARNESS_COST_MODEL.md)
- [Runtime configuration closure](docs/RUNTIME_CONFIGURATION_CLOSURE.md)
- [Stage-wavefront admission](docs/STAGE_WAVEFRONT_ADMISSION.md)
- [Budgeted candidate planner](docs/BUDGETED_CANDIDATE_PLANNER.md)
- [Ordered feasibility planner](docs/ORDERED_FEASIBILITY_PLANNER.md)
- [Guarded policy transfer](docs/POLICY_TRANSFER.md)
- [Guarded runtime policy routing](docs/GUARDED_RUNTIME_ROUTING.md)
