# Sequential Design Study

## Purpose

NPU calibration pairs are expensive. A confidence method that is formally valid
but poorly matched to a 20-pair deadline can consume the full budget without
reaching a useful decision. `compare-sequential-designs` evaluates candidate
methods before they are frozen into the next prospective protocol.

The study is always `diagnostic_only`. It cannot replace the method in the
source prospective assessment, promote a configuration, or close a search
direction. Its content-addressed output binds the source assessment, effect
contract, observed adjusted effects, simulation design, and all derived rows.

## Methods

The first study compares:

- `frozen_spec_method`, the method selected before the source evidence;
- `finite_horizon_hoeffding`, the existing conservative alpha-spending
  reference;
- `hedged_capital_predictable_plugin`, a dependency-free implementation of the
  variance-adaptive hedged capital process from Waudby-Smith and Ramdas.

The hedged implementation uses predictable variance estimates and clips each
bet against the hypothesized bounded mean. It inverts the maximum capital over
all prefixes, so the reported interval is time-uniform rather than a terminal
fixed-sample interval.

The comparison reports three distinct boundaries:

- `promote`: the lower bound exceeds the minimum useful gain;
- `close_direction`: the upper bound is below zero;
- `exclude_useful_gain`: the upper bound is below the minimum useful gain but
  not below zero.

The third outcome matters for autotuning. It can prune a candidate that is
unlikely to justify deployment without claiming the knob direction is harmful.
It is only eligible for formal use when predeclared in a future protocol.

## Simulations

Two deterministic simulation families expose different operating regimes:

- `observed_symmetric_residual` reuses the magnitudes of centered observed
  residuals with random signs, preserving each scenario mean and approximating
  the measured low-variance regime;
- `bounded_endpoints` draws only the frozen support endpoints with probabilities
  chosen to preserve the scenario mean, providing a deliberately high-variance
  stress case.

For every method, noise model, and true-effect scenario, the study records
simultaneous coverage across every planned look, decision rates, stopping pairs,
and terminal interval width. Simulation is a regression and design diagnostic,
not a proof of validity or a license for post-hoc method selection.

Recent work makes the finite-horizon emphasis especially relevant:

- [Learning to Bet for Horizon-Aware Anytime-Valid Testing](https://arxiv.org/abs/2603.19551)
  models deadline-aware betting as finite-horizon control and is scheduled for
  ICML 2026;
- [Time-sensitive anytime-valid testing](https://arxiv.org/abs/2605.06521)
  optimizes e-processes for hard deadlines and early-rejection value;
- [Gaussian-efficient testing by betting on the mean of bounded data](https://arxiv.org/abs/2608.21694)
  improves terminal bounded-mean intervals, but is not silently treated as a
  confidence sequence here;
- [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686)
  supplies the hedged predictable plug-in construction and an official
  [ConfSeq implementation](https://github.com/gostevehoward/confseq).

## Commands

```bash
inference-autopilot compare-sequential-designs \
  sequential-design-study.json sequential-effect-assessment.json \
  --output sequential-design-study-result.json
inference-autopilot audit-sequential-design-study \
  sequential-design-study-result.json
```

## Proposal-16K Result

The first real study re-derived all three methods from the four prospective,
replay-adjusted medium-2K observations. The predictable plug-in interval was
the narrowest: its current log-effect interval was `[-0.15493, +0.14841]` and
its planning-only 20-pair projection was `[-0.06614, +0.04994]`. The frozen
betting-mixture projection was `[-0.14063, +0.11792]`; finite-horizon
Hoeffding was wider at `[-0.17487, +0.15217]`.

Across 1,000 deterministic trials per scenario, the minimum simultaneous
coverage observed over every planned look was `0.983` for predictable plug-in,
`0.998` for the frozen method, and `0.999` for Hoeffding. Under the measured
low-variance residual model and a true `-0.06` log effect, predictable plug-in
excluded the `+3%` useful-gain threshold in `97.8%` of trials, stopping after
`17.3` pairs on average. The conservative references produced no decision.

This is a useful method upgrade, but not a cure for the present experiment.
Even predictable plug-in does not project a promotion or directional closure
for the observed mean, and the simulations have little power to promote a
true `+0.058` log effect by 20 pairs. The next improvement must reduce paired
variance and amortize startup across independent workload epochs; it is not
another unplanned NPU replication.

Artifact:
`artifacts/cis-small-proposal-medium2k-p32-propbt16k-sequential-design-study-20260906.json`
