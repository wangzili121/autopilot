# Runtime Configuration Closure

## Problem

A deployment candidate is not fully described by the values supplied to vLLM.
The runtime derives compilation ranges, graph capture buckets, graph ceilings,
KV capacity, and concurrency from those values and its own version-specific
defaults. In the Ascend 0.18 stack, changing `max_num_seqs` also changes the ACL
Graph bucket set and graph memory reservation. Treating only the requested
setting as a model feature therefore creates a hidden intervention.

Runtime closure makes this derivation observable and content-addressed. It does
not replace vLLM configuration, continuous batching, or an online scheduler.
It verifies what the backend actually instantiated and supplies those facts to
the offline tuner.

## Evidence Contract

For every planned run, the attestor binds:

- the calibration plan and run manifest;
- the adapter-produced effective configuration;
- the raw runner log;
- one ordered base and proposal engine configuration.

Each engine record contains the requested sequence and token capacities plus the
resolved dtype, model length, prefix/chunked-prefill flags, compilation backend,
compile-range endpoints, graph mode, full capture-size list, capture ceiling,
available KV GiB, KV token count, modeled maximum concurrency, and graph capture
cost. File digests and the assessment digest make later mutation detectable.

The assessment fails closed when a run is missing, bindings disagree, required
runtime fields are absent, a compile range does not cover the requested token
budget, or an explicit graph policy is not honored. Implicit graph defaults are
accepted but labelled `runtime_resolved`.

## Selector Join

The feature join requires an exact match between feature-row locators and
attested run IDs. It adds scalar `runtime.base.*`, `runtime.proposal.*`, and
`runtime_closure.*` fields while preserving evidence grade and eligibility. The
closure assessment digest is embedded in every enriched row.

```bash
inference-autopilot attest-runtime-closure plan.json campaign/ \
  --output runtime-closure.json
inference-autopilot audit-runtime-closure runtime-closure.json

inference-autopilot enrich-runtime-features features.json runtime-closure.json \
  --output features-runtime-enriched.json
inference-autopilot audit-features features-runtime-enriched.json
```

## First NPU Finding

The short-context p96 historical-anchor campaign produced 12 formal runs and 24
engine closures. All compile-range ceilings matched the requested token budgets,
but all graph policies were runtime-resolved. The base engine captured through
its full requested capacity for `40`, `128`, and `256` sequences. The proposal
engine captured through `96` for the small anchor but stopped at `512` for both
the `768` and `896` sequence settings, leaving 256 and 384 scheduler slots above
the graph ceiling. Median available proposal KV memory fell from `20.61 GiB` at
`40/96` to `17.18 GiB` at `128/768` and `16.71 GiB` at `256/896`.

This does not by itself prove that the uncovered capacity is harmful. It proves
that the prior composite sweep changed scheduler capacity, compile range, graph
policy, and KV headroom together. Follow-up experiments must explicitly freeze
or factor those mechanisms before attributing the measured throughput gain.
