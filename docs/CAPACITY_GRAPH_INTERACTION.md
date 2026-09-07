# Capacity And Graph Interaction Repair

## Problem

Scheduler capacity and graph capture coverage are individually valid tuning
dimensions, but they are not independent. Raising `max_num_seqs` can make the
runtime execute batch shapes above the largest captured graph. A scalar tuner
that models capacity without this execution-domain transition can interpolate
across a performance cliff and then continue probing farther into the wrong
region.

The repair planner closes that feedback loop. It consumes a completed formal
calibration assessment plus the exact calibration spec that produced it. It is
not allowed to diagnose a graph-domain interaction from a microbenchmark,
unpaired run, prediction, or configuration label alone.

## Evidence Gate

`plan-capacity-graph-repair` requires all of the following:

- the assessment is formally complete;
- the selected effect is a complete non-control paired comparison;
- quality constraints passed and the median objective effect is negative;
- the regression magnitude exceeds the campaign's replay-noise envelope;
- candidate evidence records match the candidate settings in the frozen spec;
- scheduler capacity exceeds the role-specific graph capture ceiling;
- observed `vllm:num_requests_running` also exceeds that ceiling.

The last two conditions distinguish a configured mismatch from a realized
domain crossing. Without runtime occupancy telemetry, the planner fails closed.

## Repair Semantics

For every affected engine role, the proposed target capture size is:

```text
max(configured max_num_seqs, ceil(observed maximum running requests))
```

The target is added to the existing sorted capture list. Other deployment and
algorithm settings remain unchanged. The output separates two deltas:

- `repair_delta` compares the repaired candidate with the measured harmful
  candidate; it isolates what the repair changed.
- `production_delta` compares the repaired candidate with the active strong
  baseline; it states what the next practical ABBA candidate changes.

This distinction prevents the combined capacity-plus-graph experiment from
being misreported as an isolated graph speedup. The earlier harmful capacity
cell remains necessary evidence for interpreting the interaction.

## Artifact Contract

The plan binds the canonical SHA256 of the source assessment and calibration
spec, the assessment's plan digest, complete paired effects, supporting evidence
record IDs, observed graph-domain excess, repaired settings, and both deltas.
Tampering invalidates the plan digest.

`capacity-graph-repair-calibration-spec` accepts only the bound calibration
template. It emits a new ABBA spec against the active baseline and includes a
manifest-identical replay-control group by default. The resulting campaign is
then executed and assessed by the existing calibration harness.

## Current Conditional IS Finding

On the short GSM8K p16 workload, `40/64` with proposal graph capture capped at
48 regressed by `25.11%` median QPS versus `40/48`. Both candidate observations
reached 64 running proposal requests while the capture ceiling remained 48.
The generated repair candidate is therefore `40/64` with a proposal capture
bucket at 64, evaluated against the active `40/48 + graph48` baseline. Its two
formal pair effects were `+14.39%` and `+14.00%` QPS. The `+14.19%` median gain
exceeded the campaign's `9.24%` replay envelope, while aggregate P95 improved
from `58.50s` to `51.01s` with unchanged accuracy.

This establishes the combined repair against the production baseline, not the
isolated causal effect of graph64. The joint optimizer therefore retains all
four cells in the `capacity {48,64} x graph ceiling {48,64}` space. Its active
acquisition rule selects the missing `40/48 + graph64` cell next, because that
single graph-policy change completes the interaction matrix without repeating
the known harmful capacity/graph mismatch.

That missing cell has now been measured on physical NPU2. Its two graph64 pair
effects at fixed `40/48` capacity were `-0.58%` and `-1.86%`, for a `-1.22%`
median that remained inside the campaign's `4.54%` replay envelope. The quality
gates passed, so this is usable negative evidence: expanding graph coverage
alone is not a demonstrated speedup when runtime concurrency remains capped at
48.

The complete matrix is split across NPU6 and NPU2, so it is sufficient to guide
the next search but not to estimate a publishable cross-device interaction
coefficient. A future free-card repeat should measure all four cells on one
physical device. Until then, the optimizer treats graph ceiling as conditional
on scheduler capacity and observed occupancy, and keeps environment identity in
the policy key.
