# Active-Learning Experiment Acquisition

## Purpose

An automatic tuner needs two different decisions:

1. which measured configuration is safe enough to deploy;
2. which unmeasured configuration is worth benchmarking next.

The offline policy selector answers the first question. The acquisition planner
in `policy_acquisition.py` answers the second without treating an uncertain
prediction as a deployment recommendation.

## Inputs And Binding

`plan-policy-experiments` consumes four immutable inputs:

- an acquisition spec;
- a selected policy bundle;
- the compiled legal search space;
- the exact feature table used to fit the policy.

The output binds the SHA256 of all four. A changed policy, candidate space, or
training row set invalidates the plan. Candidate observations are detected by
matching the full compiled deployment configuration against the policy's
response-model rows, rather than trusting a candidate label.

## Trust Region

The planner rejects candidates that:

- have already been measured exactly;
- are not SLO-feasible under the current conservative policy model;
- exceed the maximum normalized model distance;
- change more knobs than the declared control-relative budget;
- change only graph buckets that are unreachable below the engine's sequence
  capacity;
- continue farther in a numeric direction after a nearer formal paired probe
  regressed in every pair;
- have no declared optimistic path to beat the control.

The policy bundle may truncate `ranked_candidates` for human-readable output.
Acquisition refits the same frozen response model with a report limit covering
the complete compiled space, checks that selected and fallback decisions remain
unchanged, and then scores every legal candidate. Report truncation therefore
cannot silently become search-space truncation.

The knob-change limit is an experimental identifiability guard. A one-knob
round can attribute an effect to proposal concurrency; it cannot confuse that
effect with a simultaneous base-capacity or graph-policy change.

## Acquisition Score

For each surviving candidate, the planner normalizes by the fitted objective's
observed response scale and computes:

```text
score = w_improvement * optimistic_improvement
      + w_uncertainty * prediction_interval_width
      - w_distance * nearest_model_distance
      - w_change * changed_knob_count
```

For a maximize objective, optimistic improvement compares the candidate upper
bound with the control lower bound. For minimize objectives, the direction is
reversed. This is a deterministic, inspectable value-of-information heuristic;
it can later be replaced by constrained expected improvement once enough
formal data supports a probabilistic surrogate.

## Calibration Compilation

`policy-experiment-calibration-spec` overlays selected deployment settings on a
complete calibration baseline. Settings outside the compiled search space,
including a frozen graph mode or bucket policy, remain unchanged. Structured
graph bucket sequences may also be explicit knobs when the experiment targets
that layer. The generated spec therefore describes a controlled intervention
and can be expanded into the same manifest-bound ABBA protocol as a manual
candidate. The optional local replay control uses the complete baseline
settings, so omitted/default settings cannot be mistaken for a
manifest-identical noise reference.

## Current Result

The first real plan had exact `40/48` and `128/384` observations and selected
the one-factor interpolation `40/96`. Its formal campaign measured paired QPS
effects of `-8.09%` and `-3.72%`; the `-5.91%` median regression was larger than
the same-campaign `5.11%` replay envelope. The selector's paired-effect guard
therefore rejects both measured capacity increases and keeps `40/48` active.

The original legal space then correctly returned `no_candidate`: `40/96` was
already measured, while every other unmeasured point changed two knobs. A
content-addressed v2 space added only `40/64` and `40/80` to bracket the failed
interval. Re-fitting left the fallback unchanged, and acquisition selected
`40/64` as the nearer, higher-value-of-information point.

That campaign is now complete. The paired QPS effects were `-26.63%` and
`-23.58%`, a `-25.11%` median regression outside its `3.60%` replay envelope.
The candidate reached 64 running proposal requests while its graph-capture list
still ended at 48. Continuing the scalar sweep to `40/80` would therefore cross
the same unmodeled execution-domain boundary.

The capacity/graph repair campaign then measured `40/64 + graph64` at `+14.39%`
and `+14.00%` paired QPS versus `40/48 + graph48`; the `+14.19%` median exceeded
its `9.24%` replay envelope. After graph buckets became a first-class
`integer_sequence` knob, the joint policy selected this measured repaired cell
and rejected the graph48 mismatch.

One legal cell remains unobserved: `40/48 + graph64`. The acquisition planner
selects it with normalized distance `0.0808`, changes only
`proposal.capture_sizes`, and emits a replay-plus-candidate ABBA campaign. This
is the next causal factorization experiment, not a reason to widen capacity to
80 or 96.

The selected cell has now completed a formal campaign on NPU2. At fixed
`40/48`, graph64 produced paired QPS effects of `-0.58%` and `-1.86%`; its
`-1.22%` median stayed inside a `4.54%` replay envelope. The acquisition was
still valuable despite finding no speedup: it rules out a global graph64 policy
and identifies the earlier repair as a capacity/coverage interaction.

The next acquisition phase moves from filling this local matrix to workload
partitioning. The first short-p32 transfer measured `+9.64%` median paired QPS,
and a fresh-seed holdout validated its frozen p32 policy at `+12.53%` against a
`2.69%` replay envelope. The next high-value boundary is therefore context
length rather than another nearby closed-loop concurrency point. A medium- or
long-context partition should first compare the frozen selected policy with its
strong fallback; only after that transfer check should acquisition spend budget
on new token capacity, memory split, batching, or graph-bucket candidates in
that regime.

That context-boundary experiment is now complete. The short-p32 configuration
returned only `+2.85%` median QPS at a 2K prompt mean, inside a `3.91%` replay
envelope, and its transfer gate rejected promotion. Because a statistical model
has only two distinct configurations in this new partition, mechanism-guided
bootstrap now precedes ordinary acquisition. Formal telemetry identifies base
target scoring rather than proposal KV as the dominant stage and selects the
nearest unmeasured base token-budget increase. Once that probe adds another
distinct support point, the conservative response model resumes control of the
search.

The base token-budget probe is complete. Raising 10240 to 12288 produced
`-12.06%` and `-0.84%` paired QPS effects, a `-6.45%` median regression outside
the campaign's `4.40%` replay envelope. The negative point remains eligible for
response fitting, while the frozen mechanism gate rejects it for promotion.

After merging 16 formal rows, the medium2k policy conservatively retained its
fallback. The first acquisition implementation exposed two planner hazards
before another device run: a graph48 bucket above base `max_num_seqs=40` was an
unreachable no-op, and the policy bundle reported only 16 of 216 candidates.
Execution-domain filtering and full-surrogate recomputation fixed both. The
formal base-12K regression then blocks farther base token-budget increases, so
the next isolated experiment changes proposal `max_num_batched_tokens` from
12288 to 16384.

That eight-run acquisition is also complete. Replay effects were `+6.49%` and
`+2.29%`, setting a `6.49%` envelope. Proposal-16K effects were `-5.34%` and
`+1.28%`, for a `-2.03%` median inside the envelope. The frozen acquisition gate
returns `inconclusive`, keeps the rows eligible for response fitting, and
forbids promotion. After all 24 formal rows were merged, the fallback's
conservative P95 upper bound rose to `172.35s`, above the predeclared `170s`
limit, so the refitted policy correctly returned `no_feasible_candidate`.

Simple replication is not yet the next device action. The response model's
operational noise floor intentionally does not shrink with replicate count.
The next statistical layer must separate non-shrinking deployment/SLO
variability from shrinkable uncertainty in a paired treatment effect, then use
the latter for sequential replication and stopping.

That statistical layer is now implemented. It extracts only same-seed replay
and candidate quadruples, subtracts directional replay log drift, and applies a
content-addressed bounded confidence sequence. The proposal-16K history was
replayed as `retrospective_diagnostic`: adjusted log effects were `-0.11781`
and `-0.00989`, but the two-pair interval still covered the entire frozen
`[-0.2, 0.2]` range. It therefore does not close the proposal-token direction.
A new NPU replication is permitted only after a `prospective` spec freezes the
effect support, alpha, gain threshold, pair budget and fresh seeds.
