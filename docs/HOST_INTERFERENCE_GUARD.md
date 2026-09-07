# Host Interference Guard

## Problem

Paired order alone does not make a shared accelerator host stationary. A
foreign job can start or stop on another NPU, CPU or I/O pressure can change,
and a quiet target device can still share host resources with active services.
The medium2k replay control made this concrete: two pairs with identical
deployment settings differed by `1.97%` and `16.26%`. A candidate effect below
that envelope is not claimable, even when every target-NPU process check passes.

The guard prevents obvious host-state changes from entering the response model.
It complements replay controls; it does not estimate all remaining measurement
noise or retroactively explain campaigns that did not collect telemetry.

Before a physical attempt, the campaign launcher also runs a bounded stability
window with the same host checks and a stricter target-NPU-idle requirement.
Failed windows launch no model. See `RETRY_AWARE_EXECUTION.md` for admission,
quarantine, and retry semantics.

## Per-Run Telemetry

`host_interference.py monitor` samples immediately before launch, periodically
during execution, and once after termination. Each JSONL row records:

- aggregate CPU jiffies, logical CPU count, runnable processes, and load;
- total and available memory;
- every physical NPU's health, power, temperature, AI Core utilization, HBM
  use, and process identity from one `npu-smi info` call;
- sample time, collection duration, and a digest of the raw NPU response.

The sampler runs on the host and never opens a model or reserves an accelerator.
The NPU launcher defaults to a ten-second interval. Missing samples, failed NPU
queries, or a gap above the declared limit fail closed.

The per-run assessor excludes the target NPU's process lifecycle. It rejects:

- a process identity change on any sibling NPU;
- excessive host CPU, I/O wait, or runnable-queue pressure;
- insufficient available memory;
- missing target-device or NPU telemetry.

Static sibling activity is retained as a warning and covariate rather than
automatically rejected. This permits controlled work on a shared server while
making that condition visible.

## Cross-Run Gate

After every planned run succeeds, `assess-campaign` verifies all report hashes
and raw telemetry digests against the immutable plan. It then compares ordered
runs for:

- CPU, I/O wait, run-queue, and available-memory shifts;
- sibling-NPU process changes that occurred between runs;
- large changes in mean sibling-NPU AI Core utilization;
- mismatched target-NPU temperature or power state before model launch.

The resulting `campaign-host-interference-report.json` binds the plan and every
run report. A non-clean status stops the campaign before the formal calibration
assessment, so contaminated observations never become grade-A evidence.

All thresholds are explicit launcher settings. They are conservative defaults,
not universal hardware constants, and should eventually be calibrated from
repeated controls on each host class.

## Artifacts And Outcomes

Every new run bundle contains:

```text
host-telemetry.jsonl
host-interference-report.json
execution.meta
```

The campaign root additionally contains
`campaign-host-interference-report.json`. `execution.meta` records policy
values, assessor exit status, and artifact hashes.

`clean` permits normal outcome classification. `contaminated` and
`insufficient_telemetry` are mapped to `environment_contaminated`; they do not
constrain the candidate's feasible deployment region because the failure is not
attributable to that configuration. A bounded retry reuses the same immutable
logical run and seed. The rejected physical attempt remains hash-bound in the
campaign's attempt ledger and never enters `observations/`.

## Known Limits

The current guard detects observable host-state changes, not every source of
device-frequency or runtime nondeterminism. It cannot diagnose the completed r5
campaign because that campaign predates the sampler. The next exact replay must
collect these artifacts, and a low-noise claim still requires an identical
configuration replay control with enough independent pairs.

Future adapters can emit the same report contract from NVIDIA DCGM, ROCm SMI,
or cluster telemetry. The calibration and evidence layers need not depend on a
vendor-specific command format.
