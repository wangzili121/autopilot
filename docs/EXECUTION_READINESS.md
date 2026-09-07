# Execution Readiness Gate

The execution-readiness gate decides whether a frozen calibration campaign is
worth launching on the currently characterized host. It is deliberately
separate from candidate quality: a promising configuration can still be
deferred when shared-host interference makes a complete paired experiment
unlikely or too expensive.

## Evidence contract

`assess-execution-readiness` accepts one target calibration plan and one or
more retry-aware history campaign directories. Before using a physical attempt,
it:

1. audits the campaign's hash-chained attempt ledger and retained artifacts;
2. requires the same semantic, workload and environment contracts as the
   target plan;
3. deduplicates attempts by their ledger record hashes;
4. classifies only `accepted` and `environment_contaminated` attempts as host
   outcomes; and
5. treats crashes and missing observations as separate non-environment
   failures.

The generated assessment binds the target plan, target context, history plans,
history ledgers, final ledger records and every included physical-attempt
record by SHA256. Derived summaries and the launch decision are recomputed
during audit, so rewriting a conclusion and merely rehashing the JSON fails.

## Decision model

For `s` accepted attempts and `n` host-classifiable attempts, the gate computes
a one-sided Wilson lower confidence bound `p_lower`. Under the explicitly
frozen `stationary_independent_bernoulli` planning model, with `r` allowed
environment retries and `m` logical runs:

```text
P(logical run completes) = 1 - (1 - p_lower)^(r + 1)
P(campaign completes)    = P(logical run completes)^m
```

Expected attempt count is evaluated at the same lower clean-probability bound.
The NPU-hour estimate multiplies that count by the median observed physical-run
duration and a predeclared safety factor. It is a conservative planning
estimate, not a confidence bound on wall-clock duration.

The Bernoulli assumption is part of the hashed policy rather than an implicit
claim. Strong temporal dependence or a changing host population requires a new
model and new readiness ID. Immediate stable-window admission still runs before
every physical attempt; readiness does not replace that guard.

## Usage

Create and audit a decision from retry-aware history:

```bash
inference-autopilot assess-execution-readiness \
  examples/npu/conditional-is-medium2k-p32-off16k.execution-readiness.json \
  target-plan.json history-campaign/ \
  --output execution-readiness-assessment.json
inference-autopilot audit-execution-readiness \
  execution-readiness-assessment.json
```

Both commands exit with status `0` for `launch` and `2` for a valid `defer`.
Invalid or tampered evidence fails with another nonzero status.

To make the decision an executor precondition, pass the frozen report to the
campaign launcher:

```bash
EXECUTION_READINESS_ASSESSMENT=/path/to/execution-readiness-assessment.json \
  scripts/run_npu_calibration_campaign.sh
```

The launcher audits the report and verifies that its target-plan file digest
matches the generated campaign plan before any inference run starts. A valid
`defer` is recorded as `execution_readiness_deferred`; a stale or invalid report
fails closed.

## r8 result

The first live input contains five complete physical attempts: two accepted
and three quarantined for sibling-NPU process churn. The point clean rate is
`40%`; its one-sided 90% Wilson lower bound is `14.27%`. With two retries per
logical run, the model estimates a `36.99%` lower completion probability for
one run and `1.87%` for all four ABBA positions. Expected cost is `1.20`
NPU-hours at the frozen 1.15 duration safety factor, but completion reliability
fails both predeclared thresholds. The correct action is therefore
`defer_host`, not another immediate replay.
