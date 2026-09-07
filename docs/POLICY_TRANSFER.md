# Guarded Policy Transfer

## Purpose

A policy selected for one workload must not become a production recommendation
for another workload merely because both use the same model and serving stack.
`policy_transfer.py` turns that extrapolation into an explicit experiment. It
compares the frozen selected configuration with the frozen strong fallback in a
new workload or environment partition before any runtime router may use it.

This is separate from active acquisition. Acquisition asks which unmeasured
configuration to try inside a fitted regime. Transfer asks whether an already
selected policy survives a declared change in request load, context, prefix
reuse, dataset, or physical environment.

## Fail-Closed Contract

`plan-policy-transfer` binds three immutable inputs:

- the transfer specification;
- the selected policy bundle;
- a complete calibration template for the target runtime environment.

The planner requires matching algorithm and inference-graph digests, a strong
template baseline that contains the policy fallback settings, and a target
arrival-trace digest recomputed from workload parameters. It evaluates the
policy activation guard against the target workload and environment.

Every changed exact feature, identity, or bounded feature produces a named
guard deviation. A runnable plan is emitted only when every such deviation was
listed explicitly in `allowed_guard_deviations`; otherwise the status is
`blocked` and no calibration spec is embedded. Features known only after prompt
materialization, such as actual prompt-token mean, remain named deferred checks
and must be verified from the observations.

The output plan is content-addressed and embeds a runnable calibration with:

- the policy fallback as the strong baseline;
- a manifest-identical replay control;
- the frozen selected configuration as the transfer candidate;
- fresh ABBA pair seeds and the target workload contract.

Engine-restart settings remain engine-restart settings. The transfer planner
does not imply per-request mutation of `max_num_seqs`, memory fractions, or
graph buckets. A future online controller can route safe-boundary request groups
among prewarmed policy pools only after each policy has independent validation.

## Commands

```bash
inference-autopilot plan-policy-transfer \
  policy-transfer-spec.json policy-bundle.json calibration-template.json \
  --output policy-transfer-plan.json

inference-autopilot audit-policy-transfer policy-transfer-plan.json

inference-autopilot policy-transfer-calibration-spec \
  policy-transfer-plan.json --output transfer-calibration.json

inference-autopilot assess-policy-transfer \
  transfer-assessment-spec.json policy-transfer-plan.json \
  calibration-assessment.json --output policy-transfer-assessment.json
```

The exported calibration runs through the same manifest-bound campaign wrapper
and assessor as every other formal experiment. The transfer assessor requires a
predeclared pair count and improvement threshold, optionally requires the effect
to exceed replay noise, and verifies runtime-only guard features from the
resulting evidence ledger. Its outputs keep three claims separate:

- `eligible_for_target_policy_evidence`: the transfer result may train a policy
  for the target partition;
- `source_policy_activation_eligible`: the original activation guard still
  matches, so the source policy itself may be considered;
- `target_policy_required`: the experiment is positive but crossed an identity
  or feature guard, so a new target policy and holdout are mandatory.

## First Real Transfer

The first transfer changed the short GSM8K closed-loop workload from 16 to 32
requests/workers. It tested the p16-selected `40/64 + graph64` policy against
the `40/48 + graph48` fallback on physical NPU2. The target request count stayed
inside the source policy's declared `[16,128]` range; the changed worker count,
workload ID, and NPU6-to-NPU2 environment identity were explicit allowed
deviations. Runtime prompt mean was `117.34` tokens, satisfying the deferred
`[64,256]` guard.

All eight runs were grade A. Replay effects were `-1.67%` and `-4.72%`, defining
a `4.72%` noise envelope. Selected-policy effects were `+9.49%` and `+9.79%`
QPS, for a `+9.64%` median and `1.0964` geomean ratio. Aggregate P95 improved
from `107.94s` to `98.69s`; the accuracy constraint passed.

This validates the configuration as a p32 transfer candidate on NPU2. It does
not merge p32 rows into the p16 selector, erase physical-device identity, or
constitute independent policy holdout validation.

The formal transfer assessor reports `positive_transfer_evidence` with
`target_policy_required=true` and `source_policy_activation_eligible=false`.
The three declared source-guard deviations are preserved, while the deferred
prompt-length check passed. This machine-readable result is why the next step
fit and validated a separate p32 policy instead of widening the p16 guard.

Artifacts:

- `artifacts/cis-small-proposal-short-p32-policy-transfer-npu2-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-short-p32-policy-transfer-assessment-20260905.json`

## Independent Holdout Result

The first transfer campaign was used to fit a p32-specific policy over the two
measured coupled configurations. A separate campaign then froze that policy and
used two new workload seeds on the same NPU2. Its replay envelope was `2.69%`;
the selected policy improved paired QPS by `13.22%` and `11.83%`, or `12.53%`
at the median, while satisfying latency and accuracy constraints.

Before results were assessed, the policy holdout gate required at least two
successful replicates, no observed failures, `3%` minimum gain over fallback,
and no more than `3%` regret to the measured feasible oracle. The gate returned
`validated`, with `12.02%` mean gain and zero oracle regret. The frozen p32
policy is therefore eligible for a future guarded p32/NPU2 policy pool. It is
not evidence for medium or long context, open-loop traffic, another physical
device, or another inference-scaling algorithm.

Artifacts:

- `artifacts/cis-small-proposal-short-p32-policy-holdout-npu2-20260905-r1/assessment.json`
- `artifacts/cis-small-proposal-short-p32-policy-holdout-assessment-20260905.json`

## Rejected Medium2K Transfer

The next transfer kept p32 closed-loop load fixed and changed requested context
from short prompts to 2048 tokens. Actual prompt mean was `2183.66`, so the
deferred source guard correctly failed. The selected short policy improved
paired QPS by `5.04%` and `0.66%`, a `2.85%` median inside the same-campaign
`3.91%` replay envelope. It also missed the predeclared `3%` minimum transfer
improvement.

The formal assessment returned `rejected`, with both target evidence eligibility
and source activation disabled. The fallback remains active for medium2k. The
formal rows can still diagnose the partition and constrain future experiments,
but they cannot validate a target policy or broaden the source guard.
