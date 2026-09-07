"""Evidence-gated repair plans for scheduler-capacity and graph-domain conflicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from math import ceil, isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    ProtocolSpec,
    canonical_json,
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)


_PLAN_KEYS = {
    "schema_version",
    "producer",
    "repair_id",
    "source",
    "comparison",
    "diagnoses",
    "repair",
    "audit",
    "capacity_graph_repair_plan_sha256",
}


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _capture_sizes(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    sizes = tuple(_positive_int(item, context) for item in value)
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError(f"{context} must be sorted and unique")
    return sizes


def _effect_is_actionable(effect: Mapping[str, Any]) -> bool:
    improvement = effect.get("median_directional_relative_improvement")
    return (
        effect.get("replay_control") is False
        and effect.get("formal_group") is True
        and effect.get("quality_constraints_satisfied") is True
        and effect.get("effect_outside_replay_noise") is True
        and isinstance(effect.get("expected_pair_count"), int)
        and effect.get("complete_pair_count") == effect.get("expected_pair_count")
        and not isinstance(improvement, bool)
        and isinstance(improvement, (int, float))
        and isfinite(float(improvement))
        and float(improvement) < 0
    )


def _record_running_maximum(record: Mapping[str, Any], engine_role: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    runtime = metrics.get("vllm_runtime_metrics")
    if not isinstance(runtime, Mapping):
        return None
    engine = runtime.get(engine_role)
    if not isinstance(engine, Mapping):
        return None
    running = engine.get("vllm:num_requests_running")
    if not isinstance(running, Mapping):
        return None
    maximum = running.get("maximum")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not isfinite(float(maximum))
        or float(maximum) < 0
    ):
        return None
    return float(maximum)


def _configuration_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {"setting": name, "baseline": baseline.get(name), "candidate": candidate.get(name)}
        for name in sorted(set(baseline) | set(candidate))
        if canonical_json({"value": baseline.get(name)})
        != canonical_json({"value": candidate.get(name)})
    ]


@dataclass(frozen=True, slots=True)
class CapacityGraphRepairPlan:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expect_keys(self.payload, _PLAN_KEYS, "capacity-graph repair plan")
        raw = dict(self.payload)
        digest = str(raw.pop("capacity_graph_repair_plan_sha256", ""))
        require_digest(digest, "capacity_graph_repair_plan_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("capacity-graph repair plan SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported capacity-graph repair plan: {raw['schema_version']}"
            )
        require_id(str(raw["repair_id"]), "repair_id")
        source = require_object(raw["source"], "capacity-graph repair source")
        expect_keys(
            source,
            {
                "assessment_content_sha256",
                "assessment_plan_sha256",
                "calibration_spec_sha256",
            },
            "capacity-graph repair source",
        )
        for name, value in source.items():
            require_digest(str(value), name)
        comparison = require_object(raw["comparison"], "capacity-graph comparison")
        expect_keys(
            comparison,
            {
                "comparison_group_id",
                "candidate_configuration_id",
                "primary_metric",
                "direction",
                "median_directional_relative_improvement",
                "replay_noise_envelope",
                "pair_effects",
            },
            "capacity-graph comparison",
        )
        diagnoses = raw["diagnoses"]
        if not isinstance(diagnoses, list) or not diagnoses:
            raise ValueError("capacity-graph repair plan requires diagnoses")
        repair = require_object(raw["repair"], "capacity-graph repair")
        expect_keys(
            repair,
            {
                "comparison_kind",
                "active_baseline_configuration",
                "source_candidate_configuration",
                "repaired_candidate_configuration",
                "repair_delta",
                "production_delta",
                "hypothesis",
            },
            "capacity-graph repair",
        )
        for name in (
            "active_baseline_configuration",
            "source_candidate_configuration",
            "repaired_candidate_configuration",
        ):
            ConfigurationSpec.from_dict(require_object(repair[name], name))
        if not isinstance(repair.get("repair_delta"), list) or not repair["repair_delta"]:
            raise ValueError("capacity-graph repair delta cannot be empty")
        if not isinstance(repair.get("production_delta"), list):
            raise ValueError("capacity-graph production delta must be an array")
        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", require_object(json.loads(canonical), "plan"))

    def to_dict(self) -> dict[str, Any]:
        return require_object(json.loads(canonical_json(self.payload)), "plan")

    def audit(self) -> dict[str, Any]:
        return {
            "repair_id": self.payload["repair_id"],
            "comparison_group_id": self.payload["comparison"]["comparison_group_id"],
            "candidate_configuration_id": self.payload["comparison"][
                "candidate_configuration_id"
            ],
            "engine_roles": [item["engine_role"] for item in self.payload["diagnoses"]],
            **dict(self.payload["audit"]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapacityGraphRepairPlan":
        return cls(raw)


def plan_capacity_graph_repair(
    assessment: Mapping[str, Any],
    calibration_spec: CalibrationSpec,
    *,
    repair_id: str,
    candidate_configuration_id: str | None = None,
) -> CapacityGraphRepairPlan:
    """Turn a measured regression beyond graph coverage into a repair experiment."""

    require_id(repair_id, "repair_id")
    assessment = require_object(assessment, "calibration assessment")
    if assessment.get("formal_complete") is not True:
        raise ValueError("capacity-graph repair requires a formally complete assessment")
    assessment_plan_sha256 = str(assessment.get("plan_sha256", ""))
    require_digest(assessment_plan_sha256, "assessment plan_sha256")
    effects = assessment.get("effects")
    if not isinstance(effects, list):
        raise ValueError("calibration assessment effects must be an array")
    actionable = [
        require_object(effect, "comparison effect")
        for effect in effects
        if isinstance(effect, Mapping)
        and _effect_is_actionable(effect)
        and (
            candidate_configuration_id is None
            or effect.get("candidate_configuration_id") == candidate_configuration_id
        )
    ]
    if not actionable:
        raise ValueError(
            "assessment has no complete non-control regression outside replay noise"
        )
    actionable.sort(
        key=lambda item: (
            float(item["median_directional_relative_improvement"]),
            str(item["comparison_group_id"]),
        )
    )
    effect = actionable[0]
    candidate_id = str(effect["candidate_configuration_id"])
    pair_effects = effect.get("pair_effects")
    if not isinstance(pair_effects, list) or not pair_effects:
        raise ValueError("comparison pair_effects must be a non-empty array")
    expected_candidate_run_ids = {
        str(pair_effect.get("candidate_run_id", ""))
        for pair_effect in pair_effects
        if isinstance(pair_effect, Mapping)
    }
    if len(expected_candidate_run_ids) != len(pair_effects) or "" in expected_candidate_run_ids:
        raise ValueError("comparison pair effects require unique candidate run IDs")
    candidates = {
        candidate.configuration_id: candidate for candidate in calibration_spec.candidates
    }
    source_candidate = candidates.get(candidate_id)
    if source_candidate is None:
        raise ValueError("regressed candidate is absent from calibration spec")

    ledger = assessment.get("ledger")
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), list):
        raise ValueError("calibration assessment has no evidence records")
    group_id = str(effect["comparison_group_id"])
    candidate_records = []
    for item in ledger["records"]:
        if not isinstance(item, Mapping):
            continue
        workload = item.get("workload")
        tags = item.get("tags")
        if (
            isinstance(workload, Mapping)
            and workload.get("comparison_group_id") == group_id
            and item.get("variant") == candidate_id
            and isinstance(tags, list)
            and "candidate" in tags
        ):
            candidate_records.append(item)
    if not candidate_records:
        raise ValueError("assessment has no candidate records for the regressed group")
    candidate_run_ids = set()
    for record in candidate_records:
        source = record.get("source")
        if not isinstance(source, Mapping) or not isinstance(source.get("locator"), str):
            raise ValueError("candidate evidence record has no source locator")
        candidate_run_ids.add(source["locator"])
    if candidate_run_ids != expected_candidate_run_ids:
        raise ValueError("candidate evidence records do not match paired effect run IDs")
    expected_settings = dict(source_candidate.settings)
    for record in candidate_records:
        configuration = record.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("candidate evidence record has no configuration")
        observed_settings = {
            name: value
            for name, value in configuration.items()
            if name != "configuration_id"
        }
        if canonical_json(observed_settings) != canonical_json(expected_settings):
            raise ValueError("candidate evidence settings do not match calibration spec")

    diagnoses = []
    repaired_settings = dict(source_candidate.settings)
    for engine_role in ("base", "proposal"):
        capacity_name = f"{engine_role}_max_num_seqs"
        capture_name = f"{engine_role}_graph_capture_sizes"
        mode_name = f"{engine_role}_graph_mode"
        capacity = source_candidate.settings.get(capacity_name)
        captures = source_candidate.settings.get(capture_name)
        mode = source_candidate.settings.get(mode_name)
        if capacity is None or captures is None or mode in {None, "NONE"}:
            continue
        capacity = _positive_int(capacity, capacity_name)
        captures = _capture_sizes(captures, capture_name)
        observed = [
            (str(record.get("record_id", "")), maximum)
            for record in candidate_records
            if (maximum := _record_running_maximum(record, engine_role)) is not None
        ]
        if not observed:
            continue
        observed_max = max(maximum for _, maximum in observed)
        capture_ceiling = max(captures)
        if capacity <= capture_ceiling or observed_max <= capture_ceiling:
            continue
        target = max(capacity, ceil(observed_max))
        proposed = tuple(sorted({*captures, target}))
        repaired_settings[capture_name] = list(proposed)
        diagnoses.append(
            {
                "engine_role": engine_role,
                "capacity_setting": capacity_name,
                "capacity": capacity,
                "graph_mode_setting": mode_name,
                "graph_mode": mode,
                "capture_setting": capture_name,
                "configured_capture_sizes": list(captures),
                "capture_ceiling": capture_ceiling,
                "observed_running_max": observed_max,
                "capacity_utilization_at_peak": observed_max / capacity,
                "capture_domain_excess": observed_max - capture_ceiling,
                "target_capture_size": target,
                "proposed_capture_sizes": list(proposed),
                "supporting_record_ids": sorted(record_id for record_id, _ in observed),
                "evidence_codes": [
                    "formal_paired_regression_outside_replay_noise",
                    "scheduler_capacity_exceeds_graph_capture_ceiling",
                    "observed_concurrency_exceeds_graph_capture_ceiling",
                ],
            }
        )
    if not diagnoses:
        raise ValueError(
            "regression is not attributable to an observed capacity/graph-domain breach"
        )

    repaired_candidate = ConfigurationSpec(
        configuration_id=f"{repair_id}-repaired",
        settings=repaired_settings,
        description=(
            "Evidence-gated graph-domain repair of "
            f"{source_candidate.configuration_id}"
        ),
    )
    baseline = calibration_spec.baseline
    repair_delta = _configuration_delta(source_candidate.settings, repaired_settings)
    production_delta = _configuration_delta(baseline.settings, repaired_settings)
    replay_noise = assessment.get("replay_noise_envelope")
    if (
        isinstance(replay_noise, bool)
        or not isinstance(replay_noise, (int, float))
        or not isfinite(float(replay_noise))
        or float(replay_noise) < 0
    ):
        raise ValueError("assessment replay noise envelope must be finite and non-negative")
    comparison = {
        "comparison_group_id": group_id,
        "candidate_configuration_id": candidate_id,
        "primary_metric": str(effect["primary_metric"]),
        "direction": str(effect["direction"]),
        "median_directional_relative_improvement": float(
            effect["median_directional_relative_improvement"]
        ),
        "replay_noise_envelope": float(replay_noise),
        "pair_effects": list(pair_effects),
    }
    repair = {
        "comparison_kind": "combined_repair_against_active_baseline",
        "active_baseline_configuration": baseline.to_dict(),
        "source_candidate_configuration": source_candidate.to_dict(),
        "repaired_candidate_configuration": repaired_candidate.to_dict(),
        "repair_delta": repair_delta,
        "production_delta": production_delta,
        "hypothesis": (
            "Extending graph capture coverage to the measured scheduler domain removes "
            "the execution cliff without changing inference semantics."
        ),
    }
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "repair_id": repair_id,
        "source": {
            "assessment_content_sha256": canonical_sha256(assessment),
            "assessment_plan_sha256": assessment_plan_sha256,
            "calibration_spec_sha256": canonical_sha256(calibration_spec.to_dict()),
        },
        "comparison": comparison,
        "diagnoses": diagnoses,
        "repair": repair,
        "audit": {
            "candidate_record_count": len(candidate_records),
            "repair_engine_count": len(diagnoses),
            "repair_changed_setting_count": len(repair_delta),
            "production_changed_setting_count": len(production_delta),
        },
    }
    return CapacityGraphRepairPlan(
        {**payload, "capacity_graph_repair_plan_sha256": canonical_sha256(payload)}
    )


def calibration_spec_from_capacity_graph_repair(
    template: CalibrationSpec,
    plan: CapacityGraphRepairPlan,
    *,
    campaign_id: str,
    pair_seeds: Sequence[int],
    include_replay_control: bool = True,
) -> CalibrationSpec:
    """Compile a repair plan into a fresh manifest-bound paired experiment."""

    require_id(campaign_id, "campaign_id")
    if canonical_sha256(template.to_dict()) != plan.payload["source"][
        "calibration_spec_sha256"
    ]:
        raise ValueError("repair plan does not match calibration template")
    seeds = tuple(pair_seeds)
    if not seeds or len(seeds) % 2:
        raise ValueError("repair calibration requires two pair seeds per block")
    if len(set(seeds)) != len(seeds):
        raise ValueError("repair calibration pair seeds must be unique")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("repair calibration pair seeds must be non-negative integers")
    repair = plan.payload["repair"]
    baseline = ConfigurationSpec.from_dict(repair["active_baseline_configuration"])
    repaired = ConfigurationSpec.from_dict(repair["repaired_candidate_configuration"])
    candidates = []
    if include_replay_control:
        candidates.append(
            ConfigurationSpec(
                configuration_id=f"{campaign_id}-replay-control",
                settings=dict(baseline.settings),
                description="Manifest-identical active-baseline replay control",
            )
        )
    candidates.append(
        ConfigurationSpec(
            configuration_id=f"{campaign_id}-candidate",
            settings=dict(repaired.settings),
            description=(
                f"Combined capacity/graph repair from {plan.payload['repair_id']}"
            ),
        )
    )
    return replace(
        template,
        campaign_id=campaign_id,
        protocol=ProtocolSpec(
            pattern="ABBA",
            blocks=len(seeds) // 2,
            pair_seeds=seeds,
        ),
        baseline=baseline,
        candidates=tuple(candidates),
    )
