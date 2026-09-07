# Search-Space Compiler

Date: 2026-09-04

## Boundary

The compiler turns a declarative deployment space into the complete set of
legal candidate configurations. It is not an optimizer and it does not schedule
all candidates for execution.

This separation gives later optimizers a fixed universe: random search,
space-filling design, Bayesian optimization, and a learned selector must choose
from the same constrained artifact. They cannot silently invent settings or
bypass memory and compatibility rules.

The input and output contracts are:

- [`deployment-search-space.schema.json`](../schemas/deployment-search-space.schema.json)
- [`compiled-search-space.schema.json`](../schemas/compiled-search-space.schema.json)

## Input Model

Every knob defines:

- `name`: framework-independent feature name, such as `base.max_num_seqs`;
- `setting_name`: the concrete calibration/runner setting, such as
  `base_max_num_seqs`;
- a finite `choices` or numeric `range` domain;
- when changing it takes effect;
- whether it preserves or changes algorithm semantics.

Parameters that must be present but are not being searched belong in
`fixed_settings`. The Conditional IS example fixes both batch-wait values, so
every compiled deployment candidate contains all ten settings required by the
chang runner, including the selected model runner.

The first constraint forms are:

- `sum_less_equal`, used for the base/proposal memory split;
- `allowed_combinations`, used for capacity pairs supported by the bounded
  initial search.

Allowed-combination values must exist in their knob domains. Numeric sums use
decimal arithmetic to avoid rejecting a boundary because of binary floating
point. Numeric range expansion also uses decimal steps.

Knob values may declare `capability_requirements`. Compilation is fail-closed:
such a space requires a runtime capability profile, and values whose status is
`unsupported` or `unknown` are rejected before experiment planning. The
compiled artifact binds the profile SHA256, so a candidate set derived for one
vLLM/vLLM-Ascend combination cannot silently be reused for another.

## Semantic Isolation

Deployment-only compilation rejects every `changes_algorithm` knob by default.
An explicit `--allow-algorithm-changes` compiles those values into
`semantic_settings` and assigns a separate `semantic_cohort_id` to each semantic
combination. Such a candidate cannot be exported as an ordinary calibration
configuration because it also needs a matching graph digest, quality protocol,
and semantic contract.

This is the guard that prevents rollout count, pruning, approximate scoring, or
another quality-sensitive change from entering the deployment-only response
surface.

## Determinism And Integrity

- Knobs and domain values are canonically ordered before expansion.
- A configurable Cartesian-product limit fails before materializing an
  accidentally unbounded space.
- Every candidate ID binds its generic knob values, deployment settings,
  semantic settings, and algorithm ID.
- The compiled artifact binds the canonical input-space SHA256 and has its own
  content SHA256.
- Reloading recomputes candidate IDs, semantic cohorts, and the audit.

Constraint rejection counts are reported per constraint. One rejected point may
violate multiple constraints, so those per-constraint counts need not sum to the
unique rejected-candidate count.

## Conditional IS Audit

The current example expands to 19,200 raw combinations:

```text
2 model runners
* 4 base sequence capacities
* 5 proposal sequence capacities
* 3 base token budgets
* 5 proposal token budgets
* 4 base memory fractions
* 4 proposal memory fractions
* 2 score priorities
= 19,200
```

The allowed capacity pairs reject 11,520 combinations and the memory-sum rule
rejects 7,200, with overlap between those sets. The measured Ascend 0.18
capability profile rejects all 9,600 MRV2 points because the stock runner fails
initialization. The resulting environment-specific artifact contains 2,400
legal MRV1 deployment candidates in one exact algorithm cohort. A future
environment that validates MRV2 can expose 4,800 legal candidates without
changing the portable source space.

These 2,400 candidates are an enumerable design space, not an experiment plan.
The budgeted candidate planner now chooses a small initial subset based on the
strong baseline, compatible evidence anchors, boundaries, and maximin coverage.

## Commands

```bash
PYTHONPATH=src python3 -m inference_autopilot compile-space \
  examples/conditional-is.deployment-space.example.json \
  --capabilities examples/npu/vllm-ascend-0.18.capabilities.example.json \
  --max-cartesian-product 100000 \
  --output compiled-space.json

PYTHONPATH=src python3 -m inference_autopilot audit-space compiled-space.json

PYTHONPATH=src python3 -m inference_autopilot space-config \
  compiled-space.json CANDIDATE_ID \
  --output candidate-configuration.json
```

Compilation and export do not require an NPU.
