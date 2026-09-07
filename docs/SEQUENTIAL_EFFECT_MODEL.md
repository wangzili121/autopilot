# Sequential Paired-Effect Model

## Why This Layer Is Needed

The medium2k experiments exposed two different uncertainties that must not be
represented by one scalar spread:

1. operational variation determines whether a deployed endpoint can satisfy an
   SLO and must not disappear merely because calibration is repeated;
2. uncertainty in a candidate's treatment effect should shrink as independent
   paired seeds accumulate and must support valid data-dependent stopping.

The current response model deliberately uses a non-shrinking replicate floor.
It is suitable for conservative SLO screening but cannot decide how many more
pairs to collect after an inconclusive acquisition.

## Replay-Adjusted Observation

Each formal campaign runs an identical replay ABBA group and a candidate ABBA
group with the same pair seeds. For a maximize objective and seed `i`, define:

```text
candidate_log_effect_i = log(candidate_i / baseline_i)
replay_log_drift_i     = log(replay_second_i / replay_first_i)
adjusted_effect_i      = candidate_log_effect_i - replay_log_drift_i
```

For a minimize objective, reverse both log ratios before subtraction. This is a
multiplicative difference-in-differences estimator. Pairing by campaign, seed,
objective direction, semantic cohort, workload, environment, and exact control
is mandatory; an unmatched row is insufficient evidence rather than an
imputed sample.

The proposal-16K campaign illustrates why adjustment matters. Raw candidate
effects were `-5.34%` and `+1.28%`, while same-seed replay drift was `+6.49%`
and `+2.29%`. The adjusted effects are negative for both seeds, but two samples
are not enough for a narrow sequential confidence bound.

## Two Independent Guards

The sequential estimator produces an anytime-valid confidence sequence for the
mean adjusted log effect. Its interval may shrink with additional independent
pairs. This interval controls experimental decisions:

- promote to independent holdout only when its lower bound exceeds the frozen
  minimum improvement;
- close a one-knob direction only when its upper bound is below zero;
- request more pairs when neither boundary is crossed and budget remains;
- stop without a claim when the pair budget is exhausted.

The deployment guard separately retains the empirical operational envelope for
P95, memory, quality, and failure risk. It does not shrink by `1/sqrt(n)`.
Passing the effect sequence never overrides an SLO violation; the medium2k
fallback currently has a conservative P95 upper bound of `172.35s` against a
frozen `170s` limit, so no endpoint can be activated.

## Statistical Basis

[SCOOT](https://arxiv.org/abs/2408.04323) motivates constrained Bayesian
optimization, known-constraint pruning, learned hidden constraints, and
parallel suggestions for LLM engine tuning. [AIConfigurator](https://arxiv.org/abs/2601.06288)
motivates a framework-independent model that decomposes inference primitives
and searches graph, KV, token-capacity, and distributed launch choices.

For data-dependent stopping, fixed-horizon error bars are insufficient.
[Semiparametric Efficient Inference in Adaptive Experiments](https://arxiv.org/abs/2311.18274)
develops time-uniform inference for adaptive experiments. The 2025
[closed-form empirical Bernstein confidence-sequence work](https://arxiv.org/abs/2512.21300)
handles bounded, potentially time-varying conditional means. It also reports
that its newest approximation becomes most competitive at much larger sample
counts than our calibration budget. The first implementation should therefore
benchmark a small-sample predictable plug-in/stitched bound against a frozen
fixed-horizon reference instead of adopting the newest formula solely because
it is newer.

Adjusted log effects must be mapped to a predeclared bounded interval for the
nonasymptotic confidence sequence. The bound is part of the experiment spec and
cannot be widened after observing a sample. A bound violation invalidates the
sequential decision and triggers a new spec; clipping would manufacture
confidence.

## Implemented Contract

`SequentialEffectSpec` now freezes the candidate, manifest-identical replay
control, semantic/workload/environment hashes, exact deployment hashes,
bounded log-effect support, alpha, minimum useful gain, pair budget, replication
batch size, and fresh-seed pool. Its `analysis_mode` is part of the digest:

- `retrospective_diagnostic` extracts and reports old data but cannot promote or
  close a direction;
- `prospective` may cross a frozen decision boundary using only experiments
  planned after the rule was fixed.

This prevents an already-observed result from being converted into a
sequential claim by choosing favorable bounds after the fact. A sample outside
the frozen support invalidates the analysis; it is never clipped.

Two dependency-free time-uniform methods are implemented and tested:

1. `betting_mixture` inverts two mixtures of nonnegative fixed-fraction betting
   e-processes. The positive mixture supplies the lower boundary and the
   negative mixture supplies the upper boundary, with alpha split between them.
2. `finite_horizon_hoeffding` allocates alpha across every look in the frozen
   maximum pair budget. It is a deliberately simple conservative reference for
   replay and numerical tests.

Both require fresh independent pair seeds, bounded observations, and a stable
conditional mean for the chosen workload partition. The current implementation
does not silently claim robustness to arbitrary workload drift; replay drift is
measured and subtracted inside each seed, while cross-regime drift remains a
separate partitioning problem.

The assessment also projects the confidence interval at the frozen maximum
pair count under the explicit planning assumption that future effects equal the
current sample mean. This projection is not a decision boundary and cannot
promote or close a candidate. It exposes an underpowered design before the
harness mechanically consumes every remaining seed.

`sequential-effect-calibration-spec` verifies all frozen hashes, preserves the
full baseline and candidate settings, inserts a manifest-identical replay
control, and accepts only complete two-pair ABBA/BAAB blocks from the frozen
seed pool. Decisions are content-addressed and re-derived during audit, so a
rehashed edit to an observation or conclusion is rejected.

## Proposal-16K Replay

The existing proposal-16K assessment was intentionally evaluated in
`retrospective_diagnostic` mode. Its replay-adjusted log effects are `-0.11781`
and `-0.00989`, with adjusted geomean ratio `0.93815`. Both observations are
negative, but at two pairs the betting interval remains the full predeclared
`[-0.2, 0.2]` support. The result is therefore `diagnostic_only`, not a formal
directional closure, and its next action is to freeze a prospective spec before
collecting any new pair.

Artifacts:

- `examples/npu/conditional-is-medium2k-p32-propbt16k.sequential-effect.json`
- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-sequential-effect-20260906.json`

## Proposal-16K Prospective Evidence

The prospective specification was frozen before four new pairs were collected
in two campaigns. The replay-adjusted log effects were `-0.05952`, `-0.04728`,
`+0.03759`, and `+0.02380`. Their mean is `-0.01135`, or an adjusted geomean
ratio of `0.98871`. The reversal between campaigns is exactly why replay
adjustment, fresh seeds, and a sequential decision rule are required; neither
campaign should be selected in isolation.

The implemented betting-mixture interval remains `[-0.2, 0.2]` after four
pairs. Under the explicitly non-decisional assumption that future observations
equal the current sample mean, its projected 20-pair interval is
`[-0.14063, +0.11792]`. Continuing the same design would therefore spend the
remaining 16 pairs without an expected boundary crossing. No third campaign is
generated until a better calibrated small-sample method is implemented and
replayed on the frozen observations.

Artifact:

- `artifacts/cis-small-proposal-medium2k-p32-propbt16k-prospective-sequential-assessment-20260906-r1-r2.json`

## Remaining Work

1. Freeze a future protocol around the validated predictable plug-in method or
   a stronger finite-horizon alternative; the completed diagnostic comparison
   cannot retroactively change the source decision.
2. Reduce paired measurement variance with state-isolated persistent epochs,
   then validate pooled execution against the isolated harness before collecting
   more NPU evidence.
3. Feed a prospective `promote`, `close_direction`, or predeclared
   `exclude_useful_gain` decision back into active
   acquisition while retaining the non-shrinking SLO envelope in policy
   selection and runtime activation.
4. Evaluate search efficiency against random search, scalar grid search, and
   the current trust-region heuristic using replayed campaign traces before
   broadening the NPU search.

This layer remains independent of `conditional_is_small_proposal`: any adapter
that emits formal replay-plus-candidate pairs can use it, while Graph IR and
semantic cohort bindings prevent evidence from incompatible inference-scaling
algorithms from being pooled.
