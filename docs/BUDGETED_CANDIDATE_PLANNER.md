# Budgeted Candidate Planner

Date: 2026-09-04

## Purpose

The initial-design planner turns thousands of legal deployment candidates into
a small, auditable calibration batch. It is deliberately not presented as a
learned optimizer: the historical ledger has no grade-A response surface, so a
Bayesian or neural recommendation would currently imply more evidence than we
have.

The contracts are:

- [`candidate-design-spec.schema.json`](../schemas/candidate-design-spec.schema.json)
- [`candidate-plan.schema.json`](../schemas/candidate-plan.schema.json)
- [`candidate-configurations.schema.json`](../schemas/candidate-configurations.schema.json)

## Selection Order

The planner fills `candidate_budget` in four deterministic layers:

1. Select the declared strong baseline. Its settings must match exactly one
   compiled candidate.
2. Select A/B evidence anchors compatible with the algorithm, semantic class,
   workload, environment, and context bounds.
3. Cover unvisited numeric domain endpoints and nearest legal sum-constraint
   boundaries.
4. Fill remaining positions with normalized maximin distance over all knobs.

Numeric knob distances are normalized by their legal range. Categorical values
use match/mismatch distance. Every tie is resolved by a stable hash of the
design seed and candidate ID, so input file or evidence row order cannot change
the result.

## Historical Evidence Rules

Only explicitly allowed grade-A or grade-B rows can become anchors. C/X rows,
failed runs, algorithm mismatches, and semantic cohorts outside the design
contract are rejected.

Legacy records may partially specify deployment settings. A row must meet the
configured minimum number of known settings, and every known value must match a
legal candidate. If unspecified modern knobs leave multiple matches, the
representative nearest to the current strong baseline is used. The selected
entry retains all contributing evidence row IDs.

Context aliases are handled through explicit numeric ranges in the design spec.
For the short-P96 example, `context_tokens`, actual prompt-token mean, legacy
`prompt_tokens_mean`, and `prompt_tokens_approx` must all be at most 512 when
present. This prevents a recorded 2K/8K point from entering the short-context
anchor set merely because another context field was absent. Unknown context is
retained at lower completeness; it is never fabricated.

## Binding And Export

The plan binds:

- candidate-design spec SHA256;
- source and compiled search-space SHA256;
- feature-table SHA256 when evidence is supplied;
- algorithm ID, semantic cohort, graph SHA256, workload ID, and environment ID.

The full candidate plan has its own content digest. Reloading verifies the
digest, contiguous selection order, unique candidates, baseline-first rule,
single semantic cohort, and derived audit.

`candidate-configs` exports the selected candidates, excluding the baseline by
default. `candidate-calibration-spec` performs the stronger connection: it
verifies graph/workload/environment/baseline identity and replaces the manual
candidates in a calibration spec. Algorithm-changing selections cannot pass
this deployment-only export path.

## Short-P96 Audit

The eight-position example was run against the 2,400 legal candidates and all
81 imported historical feature rows:

| Selection source | Candidates |
| --- | ---: |
| Strong baseline | 1 |
| Compatible evidence anchors | 3 |
| Domain/constraint boundary coverage | 3 |
| Maximin space filling | 1 |

Evidence filtering reported:

| Outcome | Rows |
| --- | ---: |
| Evidence rows scanned | 81 |
| Grade/algorithm/semantic ineligible | 58 |
| Workload or context conflict | 15 |
| Compatible anchor rows | 6 |
| No legal modern candidate match | 2 |

The three-anchor budget selected three of six unique compatible anchor
candidates. Across all eight selections, 13 domain or memory-boundary tags are
covered.

The baseline is not exported as a challenger. The remaining seven candidates
produce 28 planned executions under the current four-run ABBA protocol. Reducing
`candidate_budget` reduces this first calibration batch before any NPU work is
started.

## Commands

```bash
PYTHONPATH=src python3 -m inference_autopilot plan-candidates \
  examples/conditional-is-short-p96.design.example.json \
  examples/conditional-is.deployment-space.example.json \
  compiled-space.json \
  --features selector-features.json \
  --output candidate-plan.json

PYTHONPATH=src python3 -m inference_autopilot audit-candidate-plan \
  candidate-plan.json

PYTHONPATH=src python3 -m inference_autopilot candidate-configs \
  candidate-plan.json --output candidate-configurations.json

PYTHONPATH=src python3 -m inference_autopilot candidate-calibration-spec \
  examples/conditional-is-short-p96.calibration.example.json \
  candidate-plan.json --output selected-calibration-spec.json
```

Planning, auditing, and export do not require an NPU.

The checked-in Conditional IS calibration example still contains placeholder
graph, environment, model, dataset, and arrival-trace attestations. Candidate
planning binds those values but does not certify them. They must be replaced by
frozen real digests before `prepare-run --require-formal` can pass.
