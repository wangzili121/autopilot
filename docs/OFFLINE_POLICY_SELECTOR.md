# Offline Policy Selector

Date: 2026-09-05

## Scope

The v1 selector chooses a deployment configuration for one fixed semantic
cohort. It is not an algorithm selector and does not silently compare exact
Conditional IS with pruning, uncorrected sampling, or another reward. Algorithm
budgets can be added later as separate cohorts with explicit quality gates.

The implementation lives in
[`policy_selection.py`](../src/inference_autopilot/policy_selection.py). Its
contracts are:

- [`policy-selection-spec.schema.json`](../schemas/policy-selection-spec.schema.json)
- [`policy-bundle.schema.json`](../schemas/policy-bundle.schema.json)
- [`policy-holdout-spec.schema.json`](../schemas/policy-holdout-spec.schema.json)
- [`policy-holdout-assessment.schema.json`](../schemas/policy-holdout-assessment.schema.json)

## Model And Decision Rule

Only grade-A rows marked `response_model_fit` can estimate throughput, latency,
quality, or resource responses. Grade-X failures can only influence the local
failure probability. Grade-B results remain useful as candidate-design priors,
but cannot become selector targets.

For each legal compiled candidate, the selector combines its deployment knobs
with the target workload and algorithm features. Replicate runs with the same
complete model feature vector are first aggregated into one support point, so a
frequently measured configuration cannot crowd other configurations out of the
nearest-neighbor set. Numeric dimensions are normalized over the evidence plus
legal candidate domain; categorical dimensions use match/mismatch distance. A
deterministic inverse-distance local model estimates every objective and SLO
target. Within-support replicate spread remains part of uncertainty. The
largest observed within-configuration deviation for each target is also used as
an empirical operational-noise floor. A two-run support point therefore cannot
report zero uncertainty merely because its two values happen to match.

The uncertainty interval contains two terms:

```text
uncertainty = multiplier * (max(local neighbor spread,
                                empirical replicate-noise floor)
                            + nearest distance * target response scale)
```

Grade-A ABBA metadata is retained in the feature table. When a measured
candidate has at least `minimum_paired_replicates` complete pairs against the
declared fallback and every directional pair effect is negative, the selector
adds `paired_objective_regression` and rejects deployment promotion. Absolute
response fitting can still interpolate unmeasured points for acquisition, but
it cannot erase a directly measured paired regression.

The decision is conservative:

- maximize objectives are ranked by their lower confidence bound;
- minimize objectives are ranked by their upper confidence bound;
- `<=` SLOs must pass on the upper bound and `>=` SLOs on the lower bound;
- candidates beyond the calibrated distance or local failure threshold are
  rejected;
- insufficient grade-A coverage emits `insufficient_evidence` instead of a
  guessed recommendation.

The model is deliberately small and inspectable for the first real data cycle.
Its value is the end-to-end contract, uncertainty and failure behavior, not a
claim that inverse-distance regression is the final surrogate. A Gaussian
process, random forest, or learned stage-cost residual can replace it behind the
same artifact boundary after enough formal data exists.

## Structured Runtime Policies

Runner settings are not all scalars. ACL/CUDA Graph capture policies are sorted
integer bucket sequences, so the deployment search-space contract supports a
validated `integer_sequence` choice domain. Internally these sequences are
frozen before hashing and emitted as runner-ready JSON arrays.

The response model never treats a whole bucket list as an ordinal number. A
shared deployment feature projection derives capture ceiling, bucket count,
capacity coverage ratio, uncaptured capacity, and whether the graph domain
covers scheduler capacity. Candidate queries, evidence matching, holdout
matching, and active acquisition all use this same projection. Consequently,
`40/64 + graph48` and `40/64 + graph64` cannot collapse into one configuration
or leak evidence across incompatible graph domains.

## Cross-Workload Use

Evidence is not partitioned by a human-written workload name. Training rows are
matched by algorithm, semantic class, graph digest, environment, exact static
features, and declared numeric ranges. This allows nearby load and context
points to inform a target that has a different workload ID.

The emitted activation guard remains specific to the target workload ID and
records all exact/ranged features. A runtime integration must use the fallback
when identity, calibrated domain, failure probability, or SLO guards do not
hold. The current milestone emits this policy; it does not yet hot-switch a
running vLLM engine.

## Commands

Compile the legal space, then select a policy from a graded feature table:

```bash
inference-autopilot select-policy \
  examples/conditional-is-short-p96.policy-selection.example.json \
  compiled-space.json selector-features.json \
  --output policy-bundle.json

inference-autopilot audit-policy policy-bundle.json
```

Blocked selections return status 2 while still writing the audit artifact.

Validate a frozen policy on an independently collected feature table:

```bash
inference-autopilot assess-policy-holdout \
  policy-holdout-spec.json policy-bundle.json \
  compiled-space.json holdout-features.json \
  --output policy-holdout-assessment.json

inference-autopilot audit-policy-holdout policy-holdout-assessment.json
```

The holdout assessor rejects train/holdout row-ID overlap. It requires a
predeclared number of successful grade-A replicates for both the selected candidate
and fallback, rejects observed failures when configured, and checks every SLO
using the worst holdout observation. It reports:

- gain over the current manual fallback;
- regret to the best SLO-feasible candidate actually measured on holdout;
- `validated`, `rejected`, or `insufficient_holdout` status.

This makes “beats the best manual configuration with fewer measurements than a
sweep” an executable acceptance rule instead of a narrative claim.

## Current Evidence Result

The 2026-09-05 legacy re-import contains 81 accepted records but no grade-A
response rows. Running the selector against it correctly returns
`insufficient_evidence` with zero evaluated candidates. The missing data cannot
be repaired by changing the surrogate model.

The current joint short-context model uses 28 compatible grade-A rows. It
selects measured `40/64 + graph64`, whose paired QPS gain over the active
`40/48 + graph48` fallback is `13.996%` to `14.389%`. It rejects the measured
`40/64 + graph48` cell using both `paired_objective_regression` and the latency
bound. The unmeasured `40/48 + graph64` cell remains an interpolation and is
sent to active acquisition rather than silently promoted. A broad
2,400-candidate NPU sweep is not part of the plan.

## Acceptance Gates

1. Formal preflight passes source, data, model, graph, environment, and semantic
   attestations.
2. Replay control establishes the noise envelope on the frozen stack.
3. Training uses only complete grade-A groups; crashes enter only feasibility.
4. Leave-one-context-out replay stays within the predeclared regret bound.
5. The frozen policy improves over the current manual fallback on independent
   holdout while satisfying latency, quality, memory, and failure constraints.
6. Only then can a backend adapter consume the policy bundle; guarded online
   switching remains a later milestone.
