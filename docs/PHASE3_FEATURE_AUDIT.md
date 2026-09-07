# Phase 3 Feature Table Audit

Date: 2026-09-04

## Deliverable

Phase 3 adds a typed selector-table boundary over both legacy evidence and new
manifest-bound observations. It is implemented in
[`features.py`](../src/inference_autopilot/features.py), exposed through three
CLI commands, and checked by
[`selector-feature-table.schema.json`](../schemas/selector-feature-table.schema.json).

No NPU is required for feature extraction. The output is a sparse JSON table
whose feature catalog and audit are derived from its rows and revalidated when
the artifact is loaded.

## Historical Audit

The feature extractor was run over all 81 records accepted from the legacy
`results/profiling` tree:

| Role | Eligible records |
| --- | ---: |
| Response-model fitting | 0 |
| Prior construction | 18 |
| Diagnostic analysis | 48 |
| Feasibility modeling | 1 |
| Performance claims | 0 |

The 18 prior records are the full end-to-end runs and remain separated into
their exact, approximate-algorithm, and numerical-runtime cohorts. They cannot
be pooled as one algorithm. The feasibility record is the observed 8K scoring
OOM; arrival-process metadata is not required for this static memory boundary.

The table contains 122 distinct static keys, 81 telemetry keys, and 10 target
keys across the heterogeneous legacy formats. These counts describe schema
coverage, not 213 complete columns in every row.

## Missing Selector Context

The audit confirms that the old measurements cannot train the intended
generalized selector:

| Missing requirement | Records |
| --- | ---: |
| Exact-correction flag | 81 |
| Total generation length | 81 |
| Numeric or categorical context descriptor | 59 |
| Base/proposal memory fraction | 56 each |
| Base/proposal token budget | 52 each |
| Base/proposal sequence capacity | 48 each |
| Candidate count, block size | 41 each |
| Rollout count | 23 |
| Workers or arrival rate | 8 |

This is intentionally stricter than the original result summaries. In
particular, sequence capacities such as 128/768 and 256/896 are not treated as
globally ordered choices: context length, load shape, algorithm fan-out, token
budgets and memory split are all selector inputs.

## Integrity Checks

- Grade and purpose are inherited; extraction cannot upgrade evidence.
- A successful raw observation stays `ungraded` until paired assessment.
- Response targets are kept out of static and telemetry input namespaces.
- Derived graph demand is emitted only when all defining semantic values exist.
- Eligibility is recomputed from grade, targets, and missing requirements.
- A hand-edited feature catalog, audit, evidence role, or eligibility flag is
  rejected when the table is loaded.

## Next Step

The data plane is now ready for formal measurements. The deterministic
search-space compiler and budgeted initial-design planner are implemented. Real
selector fitting remains blocked on grade-A runs across context and load
regimes.
