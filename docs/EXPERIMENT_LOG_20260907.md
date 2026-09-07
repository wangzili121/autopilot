# Conditional IS Medium-Context Experiment Log, 2026-09-07

## Scope

These runs tested context-aware proposal admission on chang's frozen exact
`conditional_is_small_proposal` path. All formal campaigns used GSM8K,
medium2k synthetic context, 32 requests/workers, C8, R3, block size 48, total
length 192, Qwen2.5 1.5B/0.5B, MRV1, ACL full-decode graphs, one Ascend 910B3
NPU, and source snapshot
`4cd534ae6f5447918ff553fa9e2320457ef163e3c369af83a5dbfcc6f8dad917`.

The retained manual baseline used base `max_num_seqs=40`, proposal
`max_num_seqs=48`, base token budget 10240, proposal token budget 16384, and
the existing memory split and sampler optimizations. Admission changed only the
proposal enqueue boundary; request identities, local seeds, outputs, scoring,
importance correction, and parent completion barriers were preserved.

## Serial Sharding Diagnostic

The first prefix-sharded implementation submitted one token-limited wave and
waited for its complete decode before submitting another. One completed pair in
r2 showed why this is not viable:

| Run | QPS | P95 | Proposal max running |
| --- | ---: | ---: | ---: |
| off baseline | 0.206225 | 155.05s | 653 queued/in-flight at backend peak |
| serial prefix sharding | 0.084132 | 379.72s | 6 running |

QPS regressed by `59.20%` and P95 increased by `144.91%`. The candidate emitted
327 waves, all partial; 99.69% were released by the prefill-token bound, and
mean admission wait reached 109.01 seconds. The campaign was stopped before a
formal four-run claim. This is diagnostic evidence that long-context sharding
must overlap decode, not evidence against sharding itself.

## Bounded Wave Pipeline

r3 allowed multiple wave futures while enforcing an aggregate 48-sequence
credit limit derived from the selected scheduler/graph capacity. It retained
wave-level credit release: all sequences in a wave had to finish before that
wave returned its credits.

| Pair | QPS effect | P95 effect | Forward-slot-rate effect |
| --- | ---: | ---: | ---: |
| 1 | -1.60% | +1.90% | -0.40% |
| 2 | -2.41% | +2.22% | +0.06% |

The QPS geometric-mean effect was `-2.00%`. Candidate runs reached 15-16
concurrent waves and exactly 48 in-flight proposal sequences, with no dispatch
failure or cap violation. This recovered nearly all of the serial prototype's
loss but did not beat direct submission. Roughly 104.3 seconds per candidate
run were still spent waiting for sequence credits.

## Per-Sequence Credit Release

r4 connected admission to the native async backend's completion callback. Each
finished sequence now returns one credit immediately, while parent futures still
complete only after all original output positions are present. Both candidate
runs reported a streaming-credit fraction of 1.0, zero batch credit releases,
no dispatch failures, and a raw proposal maximum of exactly 48 requests. Credit
stall time fell to 91.7 and 92.9 seconds.

| Pair | QPS effect | P95 effect | Forward-slot-rate effect |
| --- | ---: | ---: | ---: |
| 1 | +2.22% | -2.14% | +0.88% |
| 2 | +12.13% | -10.49% | +13.62% |

The raw QPS geometric-mean effect was `+7.06%`; P95 improved `6.41%`
geometrically and quality constraints passed. The two baselines, however,
dropped from 0.205193 to 0.188534 QPS over the campaign. The pair asymmetry made
an exact replay control mandatory before any promotion.

## Exact Replay And Final Decision

r5 used baseline and candidate configurations with byte-equivalent settings,
the exact r4 algorithm, workload, environment, and baseline context, fresh
paired seeds, and ABBA order.

| Pair | Baseline QPS | Identical candidate QPS | Apparent effect |
| --- | ---: | ---: | ---: |
| 1 | 0.205889 | 0.209953 | +1.97% |
| 2 | 0.190268 | 0.221207 | +16.26% |

The conservative replay-noise envelope is therefore `16.26%`. Reassessing r4
against this context-matched reference yields:

```text
median_directional_relative_improvement = 7.18%
replay_noise_envelope                   = 16.26%
effect_exceeds_replay_noise             = false
effect_outside_replay_noise             = false
```

The per-sequence implementation remains a technically valid candidate and its
mechanism counters improved, but this experiment does **not** prove an
end-to-end speedup. It must not enter the deployable policy pool or be described
as a 7% optimization.

## Measurement-Infrastructure Response

The target-NPU before/after process guard did not fire in r5: NPU2 was empty at
both boundaries. It could not observe sibling NPU activity or host pressure.
The harness now adds periodic host/NPU telemetry, content-addressed per-run
interference reports, and a cross-run host-state gate before formal assessment.
See `HOST_INTERFERENCE_GUARD.md`.

This addition cannot retroactively explain r5. A low-noise claim still requires
a telemetry-clean identical-settings replay with enough independent pairs.

## Live Guard Validation

Two telemetry-enabled replay attempts were admitted only after NPU2 was empty.
Both stopped on the second logical run before producing a formal assessment:

| Campaign | First clean QPS | Second QPS | Apparent change | Rejection |
| --- | ---: | ---: | ---: | --- |
| r6 | 0.224359 | 0.217095 | -3.24% | NPU0/1 processes exited during run 2 |
| r7 | 0.232304 | 0.196727 | -15.32% | NPU0/1 processes started during run 2 |

For r6, CPU P95 remained near 5.8% and 5.6%; for r7 it remained near 3.4%
and 5.5%. The target card was isolated in both attempts. r7 also recorded an
initial NPU2 temperature change from 35C to 45C. The hard finding in both cases
was sibling-NPU process churn, not a throughput-based heuristic or the thermal
threshold.

These attempts validate behavior, not a new noise number: contaminated pairs
are excluded rather than folded into a larger replay envelope. They make the
16.26% r5 drift plausible as shared-host/runtime instability, but cannot prove
its historical cause because r5 predates telemetry. Immediate retries are
paused until a sufficiently long quiet window or isolated host is available.

The harness now implements retry-aware campaign execution. A bounded stability
window must pass before launch; a contaminated physical attempt remains under
`attempts/`, and the same logical ABBA position is retried without changing its
manifest or seed. OOM, crashes, and missing observations remain terminal. Every
attempt is linked into a hash-chained ledger, and exactly one clean result per
logical run must pass artifact and frozen-plan audit before formal assessment.
The implementation passed a four-run migration exercise using the real r5 plan,
manifests, and observations without modifying those source artifacts.

The first retry-aware live campaign, r8, accepted its first baseline/candidate
pair after one baseline admission rejection. Pair-one QPS was `0.211495` versus
`0.220372`, an apparent `+4.20%`. The next candidate needed two rejected windows
before launch and was then quarantined because NPU0/3 processes changed during
execution. Three subsequent windows also observed NPU0/3 process churn, so the
campaign initially stopped without formal assessment.

An audited resume validated the partial ledger, skipped both accepted plan
positions, and continued the same third logical run at `attempt-001`. That
attempt and the final allowed `attempt-002` were also quarantined. The three
complete but excluded QPS values were `0.197252`, `0.210510`, and `0.198034`;
their CPU P95 values were only 4.7%-5.7%, while NPU0/1/3 process identity changed
during execution. The final partial ledger has five valid records, two accepted
observations, three contaminated attempts, and no formal assessment. This is
executor-validation evidence, not a replay-noise estimate.

The initial stop also exposed and fixed an overly global admission budget: six
windows now apply independently before each physical attempt rather than being
consumed across the whole logical run. The resume preserved globally unique
window numbers and monotonically increasing physical-attempt indexes.

## Execution-Readiness Decision

The r8 retry ledger was then used as the first input to the pre-launch
execution-readiness gate. Of five host-classifiable physical attempts, two were
accepted and three were environment-contaminated. The point clean probability
is `0.40`; its one-sided 90% Wilson lower bound is `0.142706`. Under the frozen
stationary independent Bernoulli planning model and two retries, the lower
modeled completion probability is `0.369930` per logical run and `0.018727` for
the four-run ABBA campaign. Median physical-run duration is `363.07` seconds,
and the conservative expected cost is `1.20` NPU-hours with a 1.15 safety
factor.

The cost remains below the two-hour ceiling, but both the 0.50 clean-probability
and 0.80 campaign-completion thresholds fail. The content-addressed decision is
therefore `defer_host`. This is an execution-control result: it prevents wasted
card time and does not change the performance conclusion. The clean first pair's
apparent `+4.20%` remains insufficient for a candidate claim.

## Evidence Locations

- serial diagnostic: `artifacts/cis-small-proposal-medium2k-p32-prefix-shard3x8k-vs-off16k-npu2-20260906-r2`;
- bounded pipeline: `artifacts/cis-small-proposal-medium2k-p32-prefix-pipeline-vs-off16k-npu2-20260906-r3`;
- streaming credit: `artifacts/cis-small-proposal-medium2k-p32-prefix-stream-credit48-vs-off16k-npu2-20260907-r4`;
- noise-gated r4 decision: the preceding directory's
  `assessment-with-replay-noise.json`;
- exact replay: `artifacts/cis-small-proposal-medium2k-p32-off16k-replay-npu2-20260907-r5`;
- first live guard rejection:
  `artifacts/cis-small-proposal-medium2k-p32-off16k-telemetry-replay-npu2-20260907-r6`;
- second live guard rejection:
  `artifacts/cis-small-proposal-medium2k-p32-off16k-quiescent-replay-npu2-20260907-r7`;
- first retry-aware live campaign:
  `artifacts/cis-small-proposal-medium2k-p32-off16k-retry-aware-replay-npu2-20260907-r8`;
- r8 execution-readiness decision: the preceding directory's
  `execution-readiness-assessment.json`.

## Normal Conditional IS Multi-NPU Preflight

The project focus changed from `conditional_is_small_proposal` to normal,
same-model `conditional_is`. A read-only host audit found:

- the first server's eight NPUs were occupied by four TP2 serving instances;
- the second server's NPU4-7 were occupied by two ongoing TP2 normal
  Conditional IS runs, while NPU0-3 had no model allocation;
- the HumanEval run used Qwen3-Coder-30B-A3B-Instruct, BF16, MRV1, APC,
  chunked prefill, TP2, `C=4`, `R=3`, block 32 and maximum 512 tokens.

At the audit point that run had completed 66 of 100 problems. It is an ongoing
preliminary two-card baseline, not a result. No four-card job was launched on
NPU0-3 because simultaneous sibling jobs would contaminate the topology
comparison and risk host-level interference. The next launch waits for a clean
window and follows `docs/CONDITIONAL_IS_MULTI_NPU_PLAN.md`.
