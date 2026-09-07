# Phase 1 Data Audit

Date: 2026-09-04

## Scope

The audit scanned all 149 JSON files under the legacy inference-scaling
`results/profiling` directory. The source tree was read-only; normalized output
was written outside this repository and is not committed as benchmark evidence.

Command:

```bash
PYTHONPATH=src python3 -m inference_autopilot import-results \
  /path/to/inference-scaling/results/profiling \
  --output /tmp/inference-autopilot-evidence-ledger.json
```

## Result

The importer accepted 69 source files and produced 81 records:

| Grade | Records | Allowed use |
| --- | ---: | --- |
| `A_formal_paired` | 0 | selector fitting and claims |
| `B_controlled_single` | 32 | calibration only |
| `C_diagnostic` | 48 | diagnostics only |
| `X_excluded` | 1 | feasibility constraint only |

Accepted formats:

| Format | Records | Interpretation |
| --- | ---: | --- |
| capacity sweep v1 | 7 | `max_num_seqs` and token-budget sweep |
| named capacity/load sweep v1 | 4 | 8/32-request small-versus-large comparison |
| KV pressure v1 | 4 | three APC runs plus one 8K OOM constraint |
| full end-to-end run v1 | 18 | capacity, memory, pruning and long-context runs |
| method comparison v5 | 48 | old synchronous-versus-asynchronous diagnostics |

The 18 full records are separated by semantic cohort:

- 9 exact-algorithm runtime candidates;
- 5 approximate pruning/selective-rescoring records;
- 4 numerically non-identical selected-token or tiled-scoring records.

The other 15 modern records lack enough embedded algorithm provenance to prove
semantic equivalence. They are tagged `legacy_unverified_semantics` even when
the surrounding experiment notes describe the exact small-proposal path.

## Rejections

The remaining 80 files were kept in the rejection report with content hashes
and top-level schema signatures. The largest groups are:

- 29 custom continuous-batching comparison reports;
- 18 stage-batching and output-agreement reports;
- 8 chunked pipeline microbenchmarks;
- trace event/summary files and several one-off prototype schemas.

Most are at best grade-C diagnostics under the current baseline policy. They
should receive dedicated parsers only when a concrete model feature needs them;
generic flattening would make their synthetic timings look interchangeable with
real end-to-end observations.

## What The Existing Data Can Establish

The ledger is sufficient to seed a bounded configuration space:

- capacity is load dependent; 40/96, 128/768 and 256/896 are useful anchors;
- larger capacity is not monotonic, as 128/1024, 384/768 and 256/1024 regress;
- low-load and saturated regimes choose different latency/throughput tradeoffs;
- shared-prefix APC can remove real KV pressure, while short GSM8K is not
  generally KV bound;
- failed configurations, including the historical 8K scoring OOM, must be
  modeled as constraints rather than discarded.

It is not sufficient to fit or validate a publishable automatic selector. Zero
records encode repeated, ordered comparisons against the current strong
baseline.

## Missing Fields

The next harness must record these fields directly rather than reconstructing
them from filenames or prose:

- inference-scaling commit, vLLM commit, vLLM-Ascend commit and patch set;
- device model/count, topology, driver/CANN version and competing processes;
- base/proposal model identifiers and immutable weight/config hashes;
- candidate count, rollout count, block size, total length and correction mode;
- graph mode, sampler implementation, APC, chunked prefill and MRV1/MRV2 path;
- prompt/output distributions, prefix sharing, arrival process and random seeds;
- P50/P95/P99, accuracy, output agreement, stage service time, batch occupancy,
  queue depth, KV peak, preemptions and failures;
- comparison group, run order, repetition index and warmup state.

## Decision

Do not implement Bayesian optimization or online switching from this ledger yet.
The immediate next deliverable is a calibration harness that emits grade-A
records conforming to the same ledger contract. Phase 2 defines that protocol.
