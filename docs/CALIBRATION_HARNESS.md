# Calibration Harness

## Purpose

The harness separates experimental protocol from workload execution. It does
not launch vLLM or own NPU scheduling. It freezes what a run is supposed to be,
lets an external inference-scaling adapter execute it, then verifies that the
returned observations are eligible for formal use.

```text
calibration spec
  -> deterministic ABBA/BAAB plan
  -> immutable manifest per run
  -> external runner
  -> run observations
  -> protocol assessment and evidence ledger
```

## 1. Freeze A Spec

Start from
`examples/conditional-is-short-p96.calibration.example.json`. Replace every
placeholder digest and environment value before a real campaign. In particular,
`strong_baseline: true` is an assertion that the baseline contains chang's
retained optimizations; it must not be copied to a weaker baseline.

The spec freezes four independent contracts:

- semantic contract: graph digest and algorithm invariants;
- workload contract: dataset, arrival trace and workload parameters;
- environment contract: devices, software revisions and model identities;
- objective: primary metric, quality/SLO constraints and required metrics.

The baseline and candidates may differ only in deployment settings for the
first exact-runtime campaign.

## 2. Generate The Plan

```bash
inference-autopilot plan-calibration calibration-spec.json \
  --output calibration-plan.json
```

One ABBA block expands each candidate to four runs. Adjacent A/B or B/A runs
share a workload seed, while the second pair uses another seed. Multiple
candidates receive independent comparison groups so one failed candidate does
not downgrade a valid group.

The plan embeds the full spec and its canonical SHA256. Editing the embedded
spec after planning invalidates the plan during loading.

## 3. Bind Every Run

Read `runs` in `sequence_index` order. Before each launch, emit its manifest:

```bash
inference-autopilot run-manifest calibration-plan.json RUN_ID \
  --output run-manifest.json
```

The manifest includes the exact configuration, workload seed, semantic and
environment contracts, objective and required metrics. Its
`run_manifest_sha256` must be passed through the external runner unchanged.

The runner writes one JSON observation per run:

```json
{
  "schema_version": "1.0",
  "run_id": "campaign--candidate--000-baseline",
  "run_manifest_sha256": "<64 lowercase hex characters>",
  "started_at_unix": 1788400000.0,
  "finished_at_unix": 1788400100.0,
  "status": "success",
  "metrics": {
    "completed_qps": 0.88,
    "latency_seconds": {"p50": 100.0, "p95": 108.0, "p99": 109.0},
    "accuracy": 0.45,
    "preemptions": 0,
    "proposal_kv_peak_fraction": 0.16
  },
  "artifact": {
    "path": "native-result.json",
    "sha256": "<64 lowercase hex characters>"
  },
  "notes": []
}
```

Use `oom`, `crash` or `cancelled` instead of inventing performance numbers for
failed runs. A failed observation may use a diagnostic `metrics` object.

The NPU launcher records target-device process snapshots plus periodic host and
all-device telemetry. A content-addressed per-run report rejects sibling-NPU
process churn, excessive CPU/I/O/run-queue or memory pressure, missing samples,
and target-device occupancy changes. A campaign report also detects state
changes between otherwise clean runs. Either gate maps to
`environment_contaminated` before an observation enters formal assessment.
This classification is separate from candidate resource exhaustion: neither
becomes a throughput sample, but only the latter may constrain the candidate's
feasible deployment region. See `HOST_INTERFERENCE_GUARD.md`.

The campaign launcher separates each immutable logical run from its physical
attempts. It admits launches only after a bounded stable window, quarantines an
environment-contaminated attempt, and retries the same manifest and seed within
a declared budget. OOM, crashes, and missing observations are never retried.
Only one clean accepted attempt is promoted to the canonical run directory and
`observations/`. A hash-chained attempt ledger must audit successfully before
the cross-run gate and formal assessment. See `RETRY_AWARE_EXECUTION.md`.

## 4. Assess The Campaign

```bash
inference-autopilot assess-calibration calibration-plan.json observations/ \
  --output calibration-assessment.json
```

Candidate-only campaigns should bind the formal replay-control assessment used
as their effect-size floor:

```bash
inference-autopilot assess-calibration calibration-plan.json observations/ \
  --replay-noise-assessment replay-control-assessment.json \
  --output calibration-assessment.json
```

The reference is accepted only when it is formally complete and contains a
complete replay-control for the same primary metric, optimization direction,
algorithm contract, workload contract, environment contract, and full baseline
settings. The output records assessment, plan, and context hashes. When local
and compatible referenced controls are available, the larger envelope is used.

The command returns zero only when every comparison group passes the formal
gate. It checks:

- all planned run IDs appear exactly once;
- observation manifest digests match;
- actual start times follow the planned ABBA/BAAB order;
- no observed runs overlap;
- all successful runs contain finite required metrics;
- the spec certifies a strong baseline.

For each group:

- complete successful groups become grade A;
- successful runs in an incomplete or disordered group remain grade B;
- failed, malformed or manifest-mismatched runs become grade X;
- an SLO or quality-constraint violation is retained on an otherwise formal
  record and tagged `constraint_violation`.

The assessment also computes the primary metric per paired seed. It records the
raw candidate/baseline ratio, a direction-normalized relative improvement, the
group geometric-mean ratio and median improvement. A candidate whose settings
exactly equal the baseline is recognized as a replay control. The maximum
absolute pair variation across formal replay controls becomes a conservative
`replay_noise_envelope`; complete quality-valid candidates are marked when their
median improvement exceeds it. This is an effect-size guard, not a substitute
for confidence intervals or enough requests and blocks.

Constraint violations do not make an observation methodologically weak. They
are high-quality evidence that a configuration is infeasible for that objective.

## Runner Integration

The `chang-pressure-v1` adapter now translates a run manifest into a structured
launch bundle and translates the native result back into a run observation. It
rejects mismatched runtime settings, workloads, semantic invariants and manifest
digests. See `docs/CHANG_RUNNER_ADAPTER.md` for the dry-run and NPU workflow.

The first formal artifact is now the short-context replay-control campaign at
`artifacts/cis-small-proposal-short-p16-replay-20260905-r4/assessment.json`.
It contributes four grade-A records and a `3.32%` replay-noise envelope. The
first candidate campaign at
`artifacts/cis-small-proposal-short-p16-capacity-128-384-20260905-r1/assessment.json`
contributes four more grade-A records. Its scheduler capacity change is a large
negative result, so it informs the response model without becoming a deployment
recommendation. The upgraded context gate does not attach the earlier envelope:
that replay used the same effective graph buckets, but did not bind them in its
manifest. The next acquired campaign includes its own explicit replay group.
Broader fitting still requires paired observations across load and context
regimes.
