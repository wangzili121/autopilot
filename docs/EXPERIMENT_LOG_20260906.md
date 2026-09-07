# Conditional IS Autotuning Experiment Log, 2026-09-06

## Scope

This campaign revisited the historical short-context p96 capacity anchors on a
frozen current stack, then directly compared the two high-capacity settings.
The objective was not to repeat the old sweep. It was to establish formal
same-stack evidence, expose coupled runtime defaults, and choose the next
identifiable intervention for the automatic tuner.

The algorithm was chang's exact `conditional_is_small_proposal` path with C8,
R3, block size 48, total length 192, importance correction, native Ascend
categorical sampling, MRV1, APC, chunked prefill, and asynchronous vLLM engines.
Both Qwen model trees, the GSM8K input, source snapshot, container, driver,
physical NPU, and workload contract were content-addressed.

## Host Selection

The first server had all eight devices occupied and its root filesystem was at
99%, so it was not used. The second server had NPU 5, 6, and 7 idle. All runs
used only physical NPU 7 under the manifest-bound
`privileged_visible_devices` mode. NPU 7 was empty again after each campaign;
no other device or user container was changed.

## Historical Anchors On The Frozen BF16 Stack

The 12-run campaign used one replay group plus two candidate groups, each with
two ABBA pairs. The identical `40/96` replay produced pair effects of `+3.24%`
and `-4.30%`, defining a conservative local noise envelope of `4.30%`.

| Candidate vs `40/96` | Pair 1 | Pair 2 | Geometric mean | Gate |
| --- | ---: | ---: | ---: | --- |
| `128/768`, tokens `32768/131072` | +9.08% | +6.86% | +7.97% | outside replay noise |
| `256/896`, tokens `65536/147456` | +14.11% | +6.07% | +10.02% | outside replay noise |

All quality constraints passed. Mean candidate QPS was `0.391276` for
`128/768` and `0.392155` for `256/896`. The two candidates were therefore
nearly tied in absolute throughput even though each independently beat a
different paired `40/96` baseline.

This BF16 result does not reproduce the historical FP16 claim of approximately
2x for `40/96 -> 128/768`. The historical result remains a hypothesis and
ranking prior, not a transferable effect estimate.

## Direct High-Capacity Comparison

A fresh-seed four-run ABBA campaign compared `256/896` directly with
`128/768`:

| Pair | `128/768` QPS | `256/896` QPS | Relative effect |
| --- | ---: | ---: | ---: |
| seed 2026090621 | 0.386345 | 0.389151 | +0.73% |
| seed 2026090622 | 0.389823 | 0.395336 | +1.41% |

The geometric-mean effect was `+1.07%`, with all declared quality constraints
passing. The matching `128/768` replay subsequently produced `+5.61%` and
`+3.53%` identical-configuration effects. Its conservative `5.61%` local noise
envelope contains the direct effect, so `256/896` was rejected as a meaningful
improvement on this workload. This supersedes the apparent ranking from the
independently paired anchor comparisons.

## Runtime Closure Finding

The new runtime-configuration attestor bound all requested settings to the
engine configuration that vLLM actually resolved. Across the 12 historical
anchor runs:

| Configuration | Role | Graph buckets | Capture ceiling | Scheduler capacity | Median available KV |
| --- | --- | ---: | ---: | ---: | ---: |
| `40/96` | base | 8 | 40 | 40 | 29.42 GiB |
| `40/96` | proposal | 15 | 96 | 96 | 20.61 GiB |
| `128/768` | base | 19 | 128 | 128 | 28.23 GiB |
| `128/768` | proposal | 51 | 512 | 768 | 17.18 GiB |
| `256/896` | base | 35 | 256 | 256 | 26.50 GiB |
| `256/896` | proposal | 51 | 512 | 896 | 16.71 GiB |

The compile-range ceilings matched every requested token budget, but all graph
policies were implicit runtime defaults. Increasing `proposal_max_num_seqs`
beyond 512 did not extend graph coverage beyond 512; it only increased the
uncaptured scheduler domain and reduced KV headroom. This is a measured hidden
coupling, not yet proof that the gap harms steady-state throughput.

## Harness Cost

The 12-run anchor campaign spent `1044.82s` in engine startup, including
`767.47s` in compilation/warmup and `195s` in graph capture. All six repeated
engine fingerprints reused a cache key but still repeated compilation. The
four-run direct comparison added `368.91s` of startup and `83s` of graph
capture. Startup and compilation cost must therefore enter experiment
acquisition and lifecycle planning rather than being treated as free.

## Control-Plane Failure And Recovery

The direct campaign's fourth run outlived an SSH output pipe after verbose graph
progress exhausted the consumer. The launcher recorded exit 141 and an empty
log while the already-created container continued normally. The container
produced its manifest-bound native result and observation. Its complete Docker
log was captured server-side before auto-removal; the empty original log, both
hashes, and the recovery reason are retained in `recovery.meta`.

The campaign launcher now defaults to `STREAM_RUNNER_LOG=0`: complete logs are
still written and hashed on the server, but high-volume runner output is not
streamed through the orchestration channel. Runtime closure and harness-cost
attestation both pass on the recovered four-run campaign.

## Base Token-Budget Probe

A graph-bound search space contained 42 legal token-capacity combinations around
the `128/768` anchor. Sequence capacity, memory split, waits, model runner,
sampler, and algorithm semantics were frozen. The original mechanism planner
selected the one-knob base token increase from `32768` to `49152` because the
base model accounts for `86.79%` of estimated dense-forward work.

The four-run ABBA probe produced pair effects of `+1.81%` and `-0.30%`, a
geometric mean of `+0.75%`, and passed all quality checks. It failed both the
predeclared `3%` effect gate and the `5.61%` replay-noise gate. Work-normalized
metrics also moved in the wrong direction: total forward slots/s changed by
`-0.02%` and `-0.97%`, base score slots/s by `-0.28%` and `-1.39%`, and estimated
dense FLOPs/s by `-0.04%` and `-1.06%`. The candidate is retained as negative
response-model evidence, not promoted as an optimization.

Runtime closure confirmed that `49152` was the resolved compile ceiling and did
not alter the graph domain. It reduced base available KV from `28.23 GiB` to
`27.36 GiB`. Its four engine starts cost `365.83s`, including `256.71s` of
compile/warmup and `79s` of graph capture.

## Saturation-Aware Replanning

The mechanism planner now estimates token-capacity utilization from observed
maximum batch width and graph-bound prompt-plus-generation length. The anchor's
estimated utilization is only `22.72%` for base and `5.68%` for proposal.
Increasing a token budget without measured capacity pressure no longer receives
a positive mechanism score; reducing one without KV or preemption pressure also
does not. Replanning the same 42-point space therefore returns `no_candidate`
instead of proposing another unidentifiable token increase.

Future causal rollback experiments can still explain the historical
`40/96 -> 128/768` gain one component at a time:

1. base token budget `32768 -> 10240`, with all other settings fixed;
2. proposal token budget `131072 -> 12288`, with all other settings fixed;
3. base and proposal sequence-capacity rollback probes with graph policy
   explicitly attested;
4. a graph-policy intervention only after measured internal graph shapes show
   demand above the current capture ceiling.

Each probe reuses the matching replay noise assessment, runs with fresh seeds,
and remains response-model evidence even when it is a regression. Only after
these main effects are known should the tuner spend runs on selected
interactions or extend the policy to p32 and medium-context holdouts.

## Graph-Demand Profiles And Holdout Gating

Five independent BF16 p96 profiles were collected with graph metrics enabled.
An earlier attempt failed before model execution because the read-only source
mount also covered its artifact cache; the launcher now mounts a dedicated
writable source cache and records run/subset seed overrides.

| Profile | Base graph hit | Base buckets | Proposal graph hit | Proposal buckets |
| --- | ---: | ---: | ---: | ---: |
| R2, seed 20260808 | 89.57% | 18/19 | 6.93% | 25/51 |
| R3, seed 2026090661 | 88.80% | 19/19 | 8.10% | 32/51 |
| R4, seed 2026090662 | 89.65% | 18/19 | 7.09% | 32/51 |
| R5, seed 2026090663 | 89.56% | 18/19 | 5.58% | 26/51 |
| R6, seed 2026090664 | 88.27% | 19/19 | 6.68% | 27/51 |

An exact trace-preserving policy derived from R2 failed on R3. A second policy
derived from the R2+R3 union preserved both training mappings but remapped 10 of
54 proposal graph events on unseen R4, so the promotion gate rejected it with
`holdout_mapping_not_exact`. This demonstrates that one or two stochastic
profiles are not sufficient evidence for exact bucket pruning.

The original coverage-constrained optimizer attempted all proposal subsets
directly, which is infeasible for 51 buckets (`2^51 - 1` candidates). It has
been replaced by an ordered Pareto dynamic program. On the real corpus it
represented that complete space using 17,817 evaluated transitions and 1,656
retained proposal states, completing in under one second.

Under predeclared train limits of 4% remapping and 0.5% added padding, the new
solver proposed base `19 -> 12` and proposal `51 -> 36` buckets. The frozen R4
holdout rejected it: base added padding reached `0.82%`, while proposal remapping
reached `20.37%` and added padding `0.81%`. No graph candidate from these first
two training profiles was eligible for a throughput experiment.

A stronger R2+R3+R4 within-regime sample set reduced exact pruning to only five
proposal buckets, but R5 still activated the removed bucket 80. Since base graph
hit rate remained near 90% while proposal stayed near 6%, the search interface
was extended to freeze base at its framework default and optimize proposal only.
The resulting proposal `51 -> 40` policy passed R5 as a validation profile with
`4.55%` remapping, `0.12%` added padding, and full graph-event retention.

That frozen role-specific candidate then failed the unseen R6 test profile:
proposal remapping rose to `8/53 = 15.09%` and added padding to `0.53%`, above
the predeclared `5%` and `0.5%` limits. Base remained exactly mapped. The
initially generated 16-run replay-plus-candidate ABBA plan was therefore not
executed; the independent preflight gate prevented roughly two hours of invalid
device experiments.

The final evidence audit also corrected the regime labels: R2-R4 are independent
samples of the same short-p96 workload regime, not three deployment regimes.
With all three assigned to one train regime, the formal search rejects the
candidate for both `insufficient_train_regimes` and
`holdout_constraint_violation`. The earlier plan with seed-specific regime
labels is superseded and cannot be used for a performance claim.

Static proposal bucket pruning is stopped for this p96 regime. The next graph
work must separate prefill and decode demand or measure a different graph mode,
rather than tuning thresholds against the failed test. Other deployment knobs
remain eligible only when their own pressure signal and matching replay control
support a causal experiment.

## Phase Diagnosis And Stage Wavefront

A version-bound vLLM/vLLM-Ascend probe added prefill, decode, and mixed labels to
the graph statistics without changing scheduler behavior. A p96 replay of the
same R6 workload showed that proposal graph failure was not mainly a missing
bucket problem:

| Role | Events | Graph hit | Decode | Mixed | Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 3995 | 87.96% | 3514 | 319 | 162 |
| proposal | 816 | 9.80% | 165 | 648 | 3 |

All 80 captured proposal events were decode. Of the 85 eager proposal decode
events, all exceeded the 512 capture ceiling. The larger issue was 648 mixed
events, or 79.41% of all proposal engine steps. This directly selected
algorithm-operation admission as the next mechanism.

An isolated proposal token-budget rollback from 131072 to 768 was negative. It
increased proposal mixed share to 96.99%, reduced graph hit to 2.86%, and
reduced QPS by 8.33% in one diagnostic replay. The vLLM scheduler used remaining
per-step capacity for new prefill work, so the scalar token limit could not
create decode-only windows.

The first stage-wavefront prototype released 16 ready Conditional IS rollout
groups at a time while keeping waves non-overlapping. Proposal graph hit rose to
96.18% and mixed share fell to 2.49%, but strict stage/length queues created 30
waves, 23 partial waves, and up to 163.13s admission wait. QPS improved only
4.03% over the first phase-labelled control.

The second prototype merged each wave into one backend call and allowed ready
proposal-generation groups from different Conditional IS iterations to share a
wave. The automatic planner selected 16 nominal groups, or 384 sequences, from
the 96-request outer concurrency, 24-sequence algorithm fanout, 512 graph
ceiling, and 768 scheduler capacity. It produced:

| Run | QPS | Elapsed | Accuracy | Proposal graph hit | Proposal mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| control A1 | 0.397744 | 241.36s | 0.40625 | 9.80% | 79.41% |
| atomic wavefront B1 | 0.454358 | 211.29s | 0.40625 | 95.97% | 2.88% |
| control A2 | 0.388218 | 247.28s | 0.36458 | 6.10% | 82.97% |
| 2D wavefront B2 | 0.416123 | 230.70s | 0.36458 | 96.05% | 2.91% |

The paired effects were +14.23% for B1/A1 and +7.19% for B2/A2, producing a
10.65% geometric-mean improvement. Accuracy matched exactly within each pair.
Candidate compute volume was only 1.28% and 0.89% lower, while forward-slot
throughput improved 12.77% and 6.23%; its paired geometric-mean gain was 9.45%.
The 2D dispatcher constrained both live sequence count and aggregate prefix
tokens; its largest short-context wave used 101,145 of 131,072 allowed prefix
tokens, so the token dimension did not bind in this regime. This is strong
diagnostic evidence, while the pre-registered fresh-seed ABBA study remains the
promotion gate.

## Formal Stage-Wavefront ABBA

The prospective short-p96 campaign used fresh paired seeds `2026090673` and
`2026090674` on NPU7. It compared the unchanged proposal path with the fully
manifest-bound automatic 384-sequence stage-wavefront policy in ABBA order.

| Pair | QPS effect | Forward-slot volume | Forward-slot rate | Accuracy A/B |
| --- | ---: | ---: | ---: | ---: |
| seed 2026090673 | +19.37% | +0.78% | +20.31% | 0.44792 / 0.46875 |
| seed 2026090674 | +16.73% | -0.70% | +15.91% | 0.45833 / 0.44792 |

The QPS geometric-mean effect is `+18.04%`; forward-slot rate improves
`+18.09%` geometrically and estimated dense-FLOP rate improves `+17.99%`.
Both quality checks pass. The candidate produced 17 and 18 waves, zero
oversized waves, mean sequence utilization of 93.15% and 88.02%, and maximum
aggregate prefix demand of 104,172 and 106,644 tokens under the 131,072-token
bound. The proposal backend changed from 264 calls of at most 24 sequences in
each control to 17 or 18 calls of at most 384 sequences.

`assessment-v2.json`, `runtime-closure-v2.json`, and `harness-cost.json` are all
complete. There is no replay control with the exact same explicit settings hash,
so the historical `5.61%` replay envelope is contextual evidence only. The next
predeclared experiment compares 384 against 480 sequences before mixed-length
and long-context holdouts.

The first 384-versus-480 launch attempt (`r1`) is invalid environment evidence.
NPU7 was empty at 21:10:58, but an unrelated two-device job claiming NPU6 and
NPU7 started at 21:12:01 while the first baseline was initializing. The base
engine completed graph capture; proposal initialization then found only 1.2 of
60.96 GiB free and aborted. No workload request was executed and no width
comparison was made. The remaining runs were not used. The launcher now records
before/after device-process snapshots, labels late occupancy as
`environment_contaminated`, and stops an ordered campaign after the first
non-success observation. A replacement `r2` uses fresh seeds `2026090677` and
`2026090678` and may launch only after an idle-device check. It is rebound to
the first server's NPU2 because that device is expected to become available
first; the new physical device and environment identity are part of the frozen
spec rather than an unrecorded launcher override.

An earlier NPU7 form of the replacement campaign passed a no-device formal
preflight before waiting for capacity. All four ABBA bundles were prepared with
`--require-formal`; they shared spec hash prefix `4febffef21d3`, plan hash prefix
`ee86c9cf05f2`, exact semantic class, and graph hash
`a46acdc960d12f5dab0af3515d083f1cf6780af34870e1febc01f85e86249cea`.
Their ordered roles and seeds were `384/2026090677`, `480/2026090677`,
`480/2026090678`, and `384/2026090678`. This proves readiness only, not a
performance result. Rebinding to NPU2 intentionally changes the spec and plan
hashes, so the final NPU2 bundles require their own formal preflight. Physical
execution remains gated on two consecutive idle-device checks.

The first NPU2-bound launch (`r2`) stopped during bundle preparation, before any
device process was started. The frozen source tree initially used absolute
host-side model symlinks; inside the read-only `/source` container mount those
targets were unreachable, so the `model_paths` formal check failed while the
other 27 checks passed. The model directories were replaced with same-filesystem
hard links whose weight inodes and hashes match the validated copies. A new
`r3` campaign ID preserves the failed-preflight record and reuses the seeds
because no model execution or workload request occurred.

The `r3` preparation then passed all 28 formal checks for all four bundles, but
the bundle launcher rejected the host-two `privileged_visible_devices` contract
before starting a container because host one was deliberately launched with
isolated physical-to-container device mapping. The corrected `r4` contract
declares `isolated_device_mapping`; again, no model execution or request used
the frozen seeds in `r3`.

The NPU2-bound `r4` campaign completed all four runs successfully from 22:14:34
to 22:43:28. Every run had an empty device before and after execution, identical
host-process snapshot hashes, `run_outcome=success`, and plugin snapshot
`5d80b167184c5d9ef6fc8a22f0117d7674848e32e10dc078ae1afc7b7620ca92`.
The campaign assessment is formal-complete with no issues; runtime closure is
complete across all eight engine instances with no graph-policy, compile-range,
or capacity-gap mismatches.

| Pair | 384 QPS | 480 QPS | QPS effect | Forward-slot-rate effect |
| --- | ---: | ---: | ---: | ---: |
| seed 2026090677 | 0.457662 | 0.424083 | -7.34% | -3.96% |
| seed 2026090678 | 0.449949 | 0.438414 | -2.56% | -0.67% |

The 480-over-384 QPS geometric-mean effect is `-4.98%`; work-normalized
forward-slot rate is `-2.33%` and estimated dense-FLOP rate is `-2.35%`.
Candidate p95 latency regressed `4.88%` geometrically. The wider policy reduced
proposal calls and realized waves from 18 per run to 15 and 17, but increased
partial-wave share from 55.56% to 68.75%. It was prefill-token limited once in
each run, whereas 384 was never token limited. All absolute constraints passed,
with no preemptions or oversized waves. The short-p96 selector retains 384 and
does not spend another four-run campaign on the structurally weaker 504 tail.

The isolated harness also measured 368.39 seconds of aggregate engine startup,
257.83 seconds of compile warmup, and five repeated compiler invocations across
the four state-isolated runs. These costs do not explain the paired workload
effect, but they remain evidence for the separate engine-lifecycle optimization
surface of the automatic harness.
