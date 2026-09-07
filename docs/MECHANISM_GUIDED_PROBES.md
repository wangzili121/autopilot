# Mechanism-Guided Probes

## Purpose

The conservative response model cannot rank a new workload partition before it
has enough distinct formal configurations. A blind grid is wasteful at exactly
that point. The mechanism probe planner provides a content-addressed bootstrap:
it combines the algorithm Graph IR, formal same-partition telemetry, and a
compiled legal deployment space to select the next measurement.

Its output is an experiment, not a policy or a performance claim. Every selected
configuration must still pass the manifest-bound replay plus ABBA harness.

## Signals

The first implementation aggregates only grade-A control rows and diagnoses:

- base/proposal dense-forward compute share;
- base target-scoring and proposal-generation token-slot share;
- prompt share of modeled work;
- scheduler capacity utilization and waiting-to-running ratio;
- KV-cache peak, preemptions, and graph coverage gaps;
- prompt-equivalent batches admitted by each engine token budget;
- a conservative token-capacity utilization estimate from the observed maximum
  batch width and the graph-bound prompt-plus-generation length.

The Graph IR maps every changed knob back to affected stages. Candidate settings
come only from the compiled space, so algorithm-changing, unsupported, or
constraint-violating combinations cannot be invented by the planner.

## Selection Rules

The planner excludes exact observations and enforces a configurable knob-change
budget. It assigns positive mechanism evidence only when the proposed direction
matches observed pressure:

- raise `max_num_batched_tokens` only when the affected prompt/score stage is
  compute-dominant and the token budget is itself under observed pressure;
- raise `max_num_seqs` when capacity utilization and queue pressure agree;
- expand graph buckets only after an observed coverage gap;
- raise memory reservation only under KV or preemption pressure;
- reduce resource settings only under corresponding resource pressure.

Compute dominance alone is not capacity pressure. A role can perform most of
the work while its token budget remains far above every observed batch. Such a
configuration receives `no_observed_token_capacity_pressure` instead of being
expanded mechanically. Likewise, token-budget reduction is not credited under
a steady-state throughput objective unless KV pressure or preemption supplies a
performance mechanism; startup-time and memory-footprint optimization use
separate objectives.

Ambiguous batch-wait changes, unbound graph knobs, and candidates without a
positive mechanism fail closed. Distance from the control breaks ties in favor
of the nearest informative step. Resource-increasing probes retain an explicit
risk tag and must run through formal preflight.

## Commands

```bash
inference-autopilot plan-mechanism-probes \
  mechanism-probe.json graph.json compiled-space.json features.json \
  --output mechanism-probe-plan.json

inference-autopilot audit-mechanism-probe-plan mechanism-probe-plan.json

inference-autopilot mechanism-probe-calibration-spec \
  calibration-template.json mechanism-probe-plan.json \
  --include-replay-control \
  --campaign-id medium2k-base-token-r1 \
  --pair-seeds 11 22 \
  --output medium2k-base-token-r1.calibration.json

inference-autopilot assess-mechanism-probe \
  mechanism-assessment.json mechanism-probe-plan.json \
  calibration-assessment.json --output mechanism-probe-assessment.json
```

The assessment gate is frozen before candidate execution. It separately reports
`validated_improvement` and `eligible_for_response_model`: a formally complete
negative probe cannot become a speedup claim, but it remains valuable training
and constraint evidence for the next acquisition round.

## First Real Plan

The first plan used six formal fallback observations from the 2K-context p32
transfer campaign. Base-model work accounted for `97.47%` of estimated dense
forward FLOPs, target scoring for `89.25%` of forward token slots, and the prompt
for `91.92%` of modeled prompt-plus-generation work. Proposal KV peaked at only
`1.14%`, with no preemptions or graph coverage gap.

The compiled space contained 216 legal configurations spanning base/proposal
sequence capacity, token capacity, and graph buckets. The one-knob
identifiability budget rejected 207 coupled changes; exact-observation checks
removed the fallback and the already measured `40/64 + graph64` transfer. The
planner selected `base_max_num_batched_tokens=12288`, the nearest unmeasured
increase from 10240, bound to `candidate_generate` and `target_score`.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-mechanism-compiled-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-mechanism-probe-plan-20260906.json`
- `artifacts/cis-small-proposal-medium2k-p32-basebt12k-20260906.calibration.json`

## First Real Result

The eight-run NPU2 campaign completed with no formal issues. Its identical
replay effects were `-4.40%` and `-2.19%`, giving a `4.40%` local noise
envelope. Raising only base `max_num_batched_tokens` from 10240 to 12288
produced paired QPS effects of `-12.06%` and `-0.84%`; the `-6.45%` median was
negative and outside that envelope. P95 changed from `130.50s` to `148.82s` in
the first pair and from `147.24s` to `147.61s` in the second. Declared quality
constraints passed.

The frozen gate therefore returned `rejected` and
`eligible_for_response_model=true`. The result rules out this deployment point
without discarding it: active acquisition can use the formal regression to
close farther increases in the same one-knob direction and search a different
stage.

Artifacts:

- `artifacts/cis-small-proposal-medium2k-p32-basebt12k-npu2-20260906-r1/assessment.json`
- `artifacts/cis-small-proposal-medium2k-p32-basebt12k-mechanism-assessment-20260906.json`
