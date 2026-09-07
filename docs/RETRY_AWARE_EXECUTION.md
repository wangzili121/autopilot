# Retry-Aware Calibration Execution

## Why This Exists

A calibration plan describes logical ABBA/BAAB positions. A launch on a shared
host is only one physical attempt to execute that position. Conflating the two
either aborts a whole campaign on unrelated activity or, worse, assigns a new
seed and schedule position to a retry. Both choices damage the paired design.

The retry-aware executor keeps the logical plan immutable while treating
observable environment contamination as a recoverable execution event.

## State Model

```text
logical run from frozen plan
  -> bounded host admission window
  -> prepared physical attempt
  -> in-run host telemetry and outcome classification
     -> clean success: promote to canonical run bundle
     -> environment contamination: quarantine and retry same logical run
     -> OOM/crash/missing result: retain and stop without retry
  -> audit complete attempt ledger
  -> cross-run host gate
  -> formal calibration assessment
```

The default budget permits six admission windows before each physical attempt
and two environment retries, for at most three launched attempts per logical
run. These values are explicit campaign metadata rather than hidden scheduler
behavior.

## Admission

`host_interference.py check-window` samples immediately and at a declared
interval over a bounded window. It applies the normal CPU, I/O, runnable-queue,
memory, NPU-telemetry, and sibling-process-stability checks, plus a strict
target-idle condition. A failed window launches no model and consumes no
physical-attempt index.

Admission reduces avoidable collisions; it cannot predict a foreign job that
starts after launch. In-run telemetry remains authoritative, and a contaminated
attempt is quarantined even when its admission window was clean.

## Retry Semantics

Only `environment_contaminated` is retried. Every retry reuses the exact logical
`run_id`, manifest, configuration, workload seed, pair index, and planned order.
The attempt index changes only the physical bundle path and container identity.

The following outcomes are not retried:

- model or framework resource exhaustion;
- runner crash or nonzero execution failure;
- missing or malformed observation;
- manifest, source, or environment-contract mismatch.

This distinction prevents an unsafe candidate from eventually appearing
successful merely because failures were discarded.

## Artifact Layout

```text
campaign/
  admission/<logical-run-id>/window-NNN.{jsonl,report.json}
  attempts/<logical-run-id>/attempt-NNN/   # quarantined or failed
  <logical-run-id>/                        # one accepted canonical bundle
  observations/<logical-run-id>.json       # accepted only
  admission-status.tsv
  attempt-status.tsv
  run-status.tsv
  attempt-ledger.jsonl
  attempt-ledger-assessment.json
```

Formal assessment continues to read the canonical layout. Runtime-closure,
harness-cost, and other downstream tools therefore need no retry-specific path
logic.

## Tamper-Evident Ledger

Each launched attempt appends one canonical JSON record. It binds:

- logical run and monotonically increasing attempt index;
- internal and file-level run-manifest digests;
- admission, execution metadata, host report, and optional observation hashes;
- outcome and bundle location;
- the preceding record hash.

The append operation takes an exclusive file lock, validates the complete
existing chain, writes one record, flushes it, and calls `fsync`. The final audit
rehashes every retained artifact, checks each run manifest against the frozen
plan, requires exactly one accepted attempt per logical run in plan order, and
fails before formal assessment on any discrepancy.

This is tamper-evident local provenance, not a malicious-host security boundary.
Remote signing or immutable object storage can be added later without changing
the logical/physical run contract.

## Audited Resume

Set `RESUME_CAMPAIGN=1` only for an existing, unassessed campaign. Before any
launch, the executor requires the supplied spec to be byte-identical to the
frozen copy, validates the partial hash chain and every retained artifact, and
requires accepted runs to form an exact prefix of plan order. It skips those
accepted runs and resumes the first incomplete logical run at its next attempt
index. A failed resume audit launches nothing.
