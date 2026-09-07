# Workload And Graph Features

Date: 2026-09-04

## Purpose

The selector feature table is the boundary between experiment evidence and a
future optimizer. It answers three different questions without mixing their
data:

1. What was known before choosing a deployment configuration?
2. What pressure did the engines report while executing it?
3. What outcome should the optimizer predict or constrain?

Those become `static_features`, `telemetry_features`, and `targets`. A QPS or
P95 value can never silently become an input feature from the same trial.

The artifact contract is
[`selector-feature-table.schema.json`](../schemas/selector-feature-table.schema.json).
Rows are sparse: absent legacy values are omitted, then named in requirement
audits where they affect eligibility. No value is inferred from a filename or
replaced by a project default.

## Static Features

The extractor preserves scalar values from four namespaces:

- `workload.*`: request count, workers or arrival rate, actual prompt-token
  distribution, requested context, prefix mode and length regime;
- `algorithm.*`: candidate count, rollout count, block size, total generation
  length, exact-correction mode and other scalar semantic invariants;
- `deployment.*`: sequence capacity, token budgets, memory split, batch wait,
  score priority and future adapter settings;
- `environment.*`: hardware, software and model cohort fields needed to prevent
  accidental transfer across incompatible stacks.

For exact small-proposal Conditional IS, let `C` be candidates, `R` rollouts,
`B` block size, `T` total generation length, and `S = ceil(T / B)`. The table
derives these graph-demand upper bounds:

```text
candidate sequence submissions = C * S
proposal sequence submissions  = C * R * S
target-score submissions        = C * R * S, when exact correction is enabled
maximum parallel width          = C * R
candidate token slots           = C * T
proposal token slots            = C * R * sum(max(T - min((i + 1) * B, T), 0))
target-score token slots         = proposal token slots, when correction is enabled
```

These are structural upper bounds, not predictions of actual work. EOS and
terminal candidates can reduce execution. They are useful because two requests
with the same input context may exert very different pressure when `C`, `R`,
`B`, or exact target scoring changes.

Graph bucket lists are structured deployment values and are not inserted into
the scalar model verbatim. For each engine role, the extractor instead derives:

- capture ceiling and distinct bucket count;
- capture-ceiling to scheduler-capacity ratio;
- scheduler capacity above the capture ceiling;
- whether graph coverage reaches the configured scheduler capacity.

These features keep equal-capacity observations with different graph domains
distinct. A capacity-only policy may pin graph mode and capture ceiling in its
selection context; a joint policy can model the derived coverage values
explicitly without comparing raw variable-length lists.

The deployment search-space schema represents the underlying bucket list with
the `integer_sequence` value type. Lists must contain sorted, unique positive
integers. Compilation freezes them for stable candidate hashes and restores
JSON arrays for calibration and runner adapters; no comma-delimited surrogate
or policy-name convention is used.

## Telemetry And Targets

Telemetry currently includes queue and service time, continuous-batching
occupancy, per-engine vLLM runtime metrics, compute counters, and selected
legacy resource measurements. It is intended for saturation diagnosis,
feasibility modeling, and later online state estimation.

Targets include completed QPS, elapsed time, latency quantiles, accuracy,
preemptions, proposal KV peak and run success. The standard response-model gate
requires at least completed QPS and P95. More objective-specific gates can be
added when the search-space contract is connected to the optimizer.

## Eligibility

Feature completeness is an additional gate; it never upgrades evidence:

| Evidence | Possible feature role |
| --- | --- |
| Grade A plus complete selector context and targets | response-model fitting |
| Grade B plus sufficient context and targets | prior construction only |
| Grade C | diagnostic analysis only |
| Grade X plus sufficient boundary context | feasibility model only |
| Ungraded observation | pending diagnostic inspection only |

`performance_claim` remains tied to the evidence assessor's grade A decision.
A grade-A row with missing selector context can therefore support its paired
experimental claim while being excluded from a generalized response model.

## Commands

Extract a table from assessed or imported evidence:

```bash
PYTHONPATH=src python3 -m inference_autopilot features-ledger \
  evidence-ledger.json --output selector-features.json
```

Extract one pending row directly after execution:

```bash
PYTHONPATH=src python3 -m inference_autopilot features-run \
  run-manifest.json observation.json --output pending-features.json
```

Reloading and auditing a table recomputes both its feature catalog and audit.
Any hand-edited or stale derived section is rejected:

```bash
PYTHONPATH=src python3 -m inference_autopilot audit-features \
  selector-features.json
```

None of these commands requires an NPU. The next NPU phase uses formal runs to
populate grade-A rows across low-load, saturated, mixed-length and long-scoring
regimes.
