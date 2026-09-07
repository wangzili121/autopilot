# Guarded Runtime Policy Routing

## Scope

The runtime layer chooses among already running, independently validated
endpoints. It does not mutate vLLM engine-start parameters while a request is in
flight. Settings such as scheduler capacity, memory fraction, and graph buckets
remain attached to prewarmed endpoints; routing occurs only at a declared safe
request-group boundary.

This is deliberately narrower than a cluster orchestrator. Service discovery,
replica placement, endpoint startup, admission control, and KV offloading remain
external responsibilities. The router consumes endpoint health and emits a
deterministic endpoint decision that an adapter can use.

## Policy Pool Compilation

`compile-runtime-policy-pool` accepts a pool spec plus policy bundles and
independent holdout assessments. Compilation fails unless:

- every policy has status `selected`;
- every holdout has status `validated` and binds that exact policy digest;
- holdout selected/fallback candidates and deployment settings match the policy;
- every policy shares the declared semantic contract and manual fallback;
- every endpoint is bound to a content hash of its complete deployment settings;
- overlapping activation guards have distinct explicit priorities.

The result is an immutable policy pool. It contains no unvalidated candidate or
surrogate prediction.

## Runtime Decision

A routing request binds the pool digest, request-group context, safe-boundary
flag, and a complete state record for every endpoint. Candidate evaluation
checks:

1. algorithm, semantic cohort/class, graph, workload, and environment identity;
2. exact and ranged activation features, including observed prompt length;
3. endpoint readiness, admission state, failure count, and configuration hash;
4. declared live constraints after the startup sample threshold.

One eligible highest-priority policy endpoint yields `selected`. No eligible
endpoint, an unsafe boundary, or a health/SLO violation yields `fallback`.
Equal-priority ambiguity also falls back. If the fallback endpoint itself is not
healthy, the result is `unavailable` rather than silently selecting an invalid
optimized endpoint.

Both requests and decisions are content addressed, so a production adapter can
retain them as replayable control-plane evidence.

## Commands

```bash
inference-autopilot compile-runtime-policy-pool \
  runtime-policy-pool-spec.json \
  --policy policy-bundle.json \
  --assessment policy-holdout-assessment.json \
  --output runtime-policy-pool.json

inference-autopilot audit-runtime-policy-pool runtime-policy-pool.json

inference-autopilot route-runtime-policy \
  runtime-policy-pool.json runtime-routing-request.json \
  --output runtime-routing-decision.json

inference-autopilot audit-runtime-policy-decision runtime-routing-decision.json
```

## First Real Pool

The first pool contains the independently validated short-p32/NPU2
`40/64 + graph64` endpoint and the strong `40/48 + graph48` fallback. Its live
policy begins checking P95 and preemption deltas after 32 samples and rejects any
consecutive endpoint failure.

The supplied in-domain example selects the optimized endpoint. Changing only
observed prompt mean from `117.34` to `2048` produces
`range_feature_out_of_bounds:workload.prompt_tokens.mean` and routes to the
fallback. This is the intended behavior until a separate medium/long-context
policy passes its own transfer and holdout gates.

Artifacts:

- `artifacts/cis-small-proposal-short-p32-runtime-policy-pool-20260905.json`
- `artifacts/cis-small-proposal-short-p32-runtime-route-decision-20260905.json`
- `artifacts/cis-small-proposal-long-context-runtime-fallback-decision-20260905.json`

The next milestone is a mixed-trace replay assessor that compares this router
with the best single static policy and measures fallback rate, SLO violations,
endpoint switches, and objective value. A live service adapter comes only after
that offline control-plane evaluation passes.
