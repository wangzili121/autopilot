"""Deterministic ABBA/BAAB calibration schedule generation."""

from __future__ import annotations

from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationPlan,
    CalibrationSpec,
    PlannedRun,
    canonical_sha256,
)


_PATTERNS = {
    "ABBA": ("baseline", "candidate", "candidate", "baseline"),
    "BAAB": ("candidate", "baseline", "baseline", "candidate"),
}


def build_plan(spec: CalibrationSpec) -> CalibrationPlan:
    """Expand every candidate into independent ordered comparison groups."""

    runs: list[PlannedRun] = []
    sequence_index = 0
    for candidate in spec.candidates:
        group_id = f"{spec.campaign_id}--{candidate.configuration_id}"
        group_sequence_index = 0
        for block_index in range(spec.protocol.blocks):
            roles = _PATTERNS[spec.protocol.pattern]
            for position, role in enumerate(roles):
                pair_index = block_index * 2 + position // 2
                configuration_id = (
                    spec.baseline.configuration_id
                    if role == "baseline"
                    else candidate.configuration_id
                )
                run_id = f"{group_id}--{group_sequence_index:03d}-{role}"
                runs.append(
                    PlannedRun(
                        run_id=run_id,
                        comparison_group_id=group_id,
                        sequence_index=sequence_index,
                        group_sequence_index=group_sequence_index,
                        pair_index=pair_index,
                        block_index=block_index,
                        variant_role=role,
                        configuration_id=configuration_id,
                        workload_seed=spec.protocol.pair_seeds[pair_index],
                        expected_result_path=f"runs/{run_id}.json",
                    )
                )
                sequence_index += 1
                group_sequence_index += 1
    return CalibrationPlan(
        spec_sha256=canonical_sha256(spec.to_dict()),
        spec=spec,
        runs=tuple(runs),
    )


def build_run_manifest(plan: CalibrationPlan, run_id: str) -> dict[str, Any]:
    """Return the immutable contract a runner must bind into its observation."""

    run = next((item for item in plan.runs if item.run_id == run_id), None)
    if run is None:
        raise KeyError(f"unknown run id: {run_id}")
    configurations = {plan.spec.baseline.configuration_id: plan.spec.baseline}
    configurations.update(
        {candidate.configuration_id: candidate for candidate in plan.spec.candidates}
    )
    payload = {
        "schema_version": "1.0",
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "spec_sha256": plan.spec_sha256,
        "run": run.to_dict(),
        "semantic_contract": plan.spec.semantic_contract.to_dict(),
        "workload_contract": {
            **plan.spec.workload_contract.to_dict(),
            "workload_seed": run.workload_seed,
        },
        "environment_contract": plan.spec.environment_contract.to_dict(),
        "objective": plan.spec.objective.to_dict(),
        "required_metrics": list(plan.spec.required_metrics),
        "configuration": configurations[run.configuration_id].to_dict(),
    }
    return {**payload, "run_manifest_sha256": canonical_sha256(payload)}
