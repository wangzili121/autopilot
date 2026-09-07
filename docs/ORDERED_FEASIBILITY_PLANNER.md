# Ordered Feasibility Planner

## Purpose

Large deployment spaces contain knobs whose bad settings fail before they can
produce throughput observations. Dropping those trials biases later tuning and
wastes devices by repeatedly entering the same unsafe region. The ordered
feasibility planner preserves successful and resource-exhausted trials as
censored evidence and chooses the next probe needed to bracket the boundary.

The first measured use is Conditional IS base `max_num_batched_tokens` under
8K target scoring. The mechanism also applies to ordered capacity knobs such as
`max_num_seqs`, memory reservations and compile batch limits when the operator
can justify the same monotonic resource-risk assumption.

## Contract

An `ordered-feasibility-spec` declares:

- one increasing integer domain;
- a flat, content-addressed workload and environment context;
- required success and resource-exhaustion confirmation counts;
- source-addressed observations classified as `success` or
  `resource_exhausted`.

The tuned parameter must be excluded from the context; every other field that
affects comparability belongs there. Typical fields are algorithm, model pair,
source snapshot, runtime image, device type, request load, context length,
memory split and graph policy.

The planner makes one explicit assumption: for this frozen context, increasing
the ordered knob cannot turn a confirmed resource-exhausted point back into a
resource-feasible point. It rejects mixed outcomes at one value and confirmed
non-monotone evidence instead of silently forcing a boundary.

## Probe Rule

1. Confirm pending successful points before expanding the safe boundary.
2. Confirm pending resource-exhausted points before shrinking the unsafe bound.
3. With both bounds, probe the lower midpoint of the unobserved interval.
4. With only a feasible bound, probe the domain maximum to establish a bracket.
5. With only an exhausted bound, probe the domain minimum.
6. Stop when confirmed feasible and exhausted values are adjacent, the domain
   maximum succeeds, or the domain minimum is exhausted.

The provisional recommendation is only the largest observed feasible value. It
is not a throughput optimum. Once the feasible domain is known, the ordinary
candidate planner and paired calibration still choose among safe settings.

## CLI

```bash
inference-autopilot plan-ordered-feasibility \
  long8k-base-token-feasibility.json \
  --output long8k-base-token-feasibility-plan.json

inference-autopilot audit-ordered-feasibility \
  long8k-base-token-feasibility-plan.json
```

The plan is bound to the normalized input spec and context with SHA256 digests.
It records every value state, inferred unsafe values, the provisional safe
recommendation and exactly one next probe.

## First Device Result

On Ascend 910B3 with Conditional IS small-proposal, 8K synthetic prefixes and
16 concurrent requests, two runs at base `max_num_batched_tokens=3072`
succeeded and two runs at 3,584 exhausted memory in the same prompt-logprob
FP32 `log_softmax`. The completed plan therefore reports 3,072 as the largest
confirmed feasible domain value and 3,584 as the adjacent confirmed exhausted
value. This recommendation is valid only for the plan's recorded context
digest and is not a claim of throughput optimality.

The NPU runner records failed trials as first-class evidence: `run_outcome`,
runner and tee exit codes, the runner-log digest, an optional result digest and
a post-run process-memory snapshot are written even when the container exits
nonzero.
