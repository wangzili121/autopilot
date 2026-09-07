"""Plan state-isolated engine reuse from measured calibration startup evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationPlan,
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.harness_cost import HarnessCostAssessment


_ROLES = ("base", "proposal")
_STRATEGIES = ("isolated_process", "role_sticky", "fully_resident")
_PLAN_KEYS = {
    "schema_version",
    "producer",
    "lifecycle_id",
    "status",
    "requirements",
    "source",
    "engine_runs",
    "strategies",
    "selected_strategy",
    "actions",
    "validation_gate",
    "audit",
    "engine_lifecycle_plan_sha256",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _finite_positive(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{context} must be finite and positive")
    return parsed


@dataclass(frozen=True, slots=True)
class EngineLifecyclePlanningSpec:
    lifecycle_id: str
    maximum_resident_memory_fraction: float
    preserve_run_order: bool
    required_epoch_resets: tuple[str, ...]
    validation_pair_seeds: tuple[int, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported lifecycle planning spec: {self.schema_version}")
        require_id(self.lifecycle_id, "lifecycle_id")
        if not 0 < self.maximum_resident_memory_fraction <= 1:
            raise ValueError("maximum_resident_memory_fraction must be in (0, 1]")
        if self.preserve_run_order is not True:
            raise ValueError("lifecycle planning must preserve the formal run order")
        if (
            not self.required_epoch_resets
            or len(set(self.required_epoch_resets)) != len(self.required_epoch_resets)
            or tuple(sorted(self.required_epoch_resets)) != self.required_epoch_resets
        ):
            raise ValueError("required_epoch_resets must be sorted, non-empty, and unique")
        required = {
            "assert_no_running_requests",
            "rebuild_continuous_batchers",
            "rebuild_score_caches",
            "reset_backend_metric_snapshots",
            "reset_prefix_cache",
            "reset_request_ids",
            "reseed_workload",
            "synchronize_device",
        }
        missing = sorted(required - set(self.required_epoch_resets))
        if missing:
            raise ValueError(f"lifecycle reset contract is incomplete: {missing}")
        if (
            not self.validation_pair_seeds
            or len(self.validation_pair_seeds) % 2
            or len(set(self.validation_pair_seeds)) != len(self.validation_pair_seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.validation_pair_seeds
            )
        ):
            raise ValueError("validation_pair_seeds must be unique non-negative ABBA pairs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lifecycle_id": self.lifecycle_id,
            "maximum_resident_memory_fraction": self.maximum_resident_memory_fraction,
            "preserve_run_order": self.preserve_run_order,
            "required_epoch_resets": list(self.required_epoch_resets),
            "validation_pair_seeds": list(self.validation_pair_seeds),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EngineLifecyclePlanningSpec":
        expected = {
            "schema_version",
            "lifecycle_id",
            "maximum_resident_memory_fraction",
            "preserve_run_order",
            "required_epoch_resets",
            "validation_pair_seeds",
        }
        _expect_exact_keys(raw, expected, "engine lifecycle planning spec")
        if not isinstance(raw["required_epoch_resets"], list):
            raise ValueError("required_epoch_resets must be a list")
        if not isinstance(raw["validation_pair_seeds"], list):
            raise ValueError("validation_pair_seeds must be a list")
        if not isinstance(raw["preserve_run_order"], bool):
            raise ValueError("preserve_run_order must be boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            lifecycle_id=str(raw["lifecycle_id"]),
            maximum_resident_memory_fraction=_finite_positive(
                raw["maximum_resident_memory_fraction"],
                "maximum_resident_memory_fraction",
            ),
            preserve_run_order=raw["preserve_run_order"],
            required_epoch_resets=tuple(str(value) for value in raw["required_epoch_resets"]),
            validation_pair_seeds=tuple(raw["validation_pair_seeds"]),
        )


def _configuration_settings(plan: CalibrationPlan) -> dict[str, Mapping[str, Any]]:
    return {
        configuration.configuration_id: configuration.settings
        for configuration in (plan.spec.baseline, *plan.spec.candidates)
    }


def _engine_runs(
    plan: CalibrationPlan, assessment: HarnessCostAssessment
) -> list[dict[str, Any]]:
    payload = assessment.to_dict()
    expected_plan_sha256 = canonical_sha256(plan.to_dict())
    if payload["plan_sha256"] != expected_plan_sha256:
        raise ValueError("harness cost assessment does not match calibration plan")
    if not assessment.complete:
        raise ValueError("engine lifecycle planning requires complete harness evidence")
    settings_by_id = _configuration_settings(plan)
    planned = {run.run_id: run for run in plan.runs}
    result = []
    for raw in payload["runs"]:
        run_id = str(raw["run_id"])
        if run_id not in planned:
            raise ValueError(f"harness run is absent from plan: {run_id}")
        run = planned[run_id]
        if raw["sequence_index"] != run.sequence_index:
            raise ValueError("harness run sequence differs from the frozen plan")
        settings = settings_by_id[run.configuration_id]
        engines = []
        for engine in raw["engines"]:
            role = str(engine["role"])
            if role not in _ROLES:
                raise ValueError(f"unsupported lifecycle engine role: {role}")
            engines.append(
                {
                    "role": role,
                    "engine_fingerprint_sha256": str(
                        engine["engine_fingerprint_sha256"]
                    ),
                    "engine_init_seconds": float(engine["engine_init_seconds"]),
                    "reserved_memory_fraction": float(
                        settings[f"{role}_memory_fraction"]
                    ),
                }
            )
        result.append(
            {
                "run_id": run_id,
                "sequence_index": run.sequence_index,
                "configuration_id": run.configuration_id,
                "workload_seed": run.workload_seed,
                "engines": engines,
            }
        )
    result.sort(key=lambda item: item["sequence_index"])
    if [item["run_id"] for item in result] != [run.run_id for run in plan.runs]:
        raise ValueError("harness evidence does not cover the exact planned run order")
    return result


def _validated_engine_runs(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("engine_runs must be a non-empty list")
    result = []
    for expected_index, run in enumerate(raw):
        if not isinstance(run, Mapping):
            raise ValueError("engine run must be an object")
        _expect_exact_keys(
            run,
            {"run_id", "sequence_index", "configuration_id", "workload_seed", "engines"},
            "engine run",
        )
        require_id(str(run["run_id"]), "engine run_id")
        require_id(str(run["configuration_id"]), "engine configuration_id")
        if run["sequence_index"] != expected_index:
            raise ValueError("engine run sequence must be contiguous")
        if (
            not isinstance(run["workload_seed"], int)
            or isinstance(run["workload_seed"], bool)
            or run["workload_seed"] < 0
        ):
            raise ValueError("engine workload_seed must be a non-negative integer")
        engines_raw = run["engines"]
        if not isinstance(engines_raw, list) or len(engines_raw) != 2:
            raise ValueError("engine run must contain base and proposal engines")
        engines = []
        for expected_role, engine in zip(_ROLES, engines_raw, strict=True):
            if not isinstance(engine, Mapping):
                raise ValueError("engine lifecycle evidence must be an object")
            _expect_exact_keys(
                engine,
                {
                    "role",
                    "engine_fingerprint_sha256",
                    "engine_init_seconds",
                    "reserved_memory_fraction",
                },
                "engine lifecycle evidence",
            )
            if engine["role"] != expected_role:
                raise ValueError("engine lifecycle roles must be base then proposal")
            fingerprint = str(engine["engine_fingerprint_sha256"])
            require_digest(fingerprint, "engine_fingerprint_sha256")
            memory = _finite_positive(
                engine["reserved_memory_fraction"], "reserved_memory_fraction"
            )
            if memory > 1:
                raise ValueError("reserved_memory_fraction cannot exceed one")
            engines.append(
                {
                    "role": expected_role,
                    "engine_fingerprint_sha256": fingerprint,
                    "engine_init_seconds": _finite_positive(
                        engine["engine_init_seconds"], "engine_init_seconds"
                    ),
                    "reserved_memory_fraction": memory,
                }
            )
        result.append(
            {
                "run_id": str(run["run_id"]),
                "sequence_index": expected_index,
                "configuration_id": str(run["configuration_id"]),
                "workload_seed": run["workload_seed"],
                "engines": engines,
            }
        )
    return result


def _fingerprint_data(
    runs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    observations: dict[str, list[float]] = defaultdict(list)
    memory: dict[str, set[float]] = defaultdict(set)
    roles: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        for engine in run["engines"]:
            fingerprint = str(engine["engine_fingerprint_sha256"])
            observations[fingerprint].append(float(engine["engine_init_seconds"]))
            memory[fingerprint].add(float(engine["reserved_memory_fraction"]))
            roles[fingerprint].add(str(engine["role"]))
    if any(len(values) != 1 for values in memory.values()):
        raise ValueError("one engine fingerprint maps to multiple memory reservations")
    if any(len(values) != 1 for values in roles.values()):
        raise ValueError("one engine fingerprint maps to multiple roles")
    return (
        {fingerprint: median(values) for fingerprint, values in observations.items()},
        {fingerprint: next(iter(values)) for fingerprint, values in memory.items()},
        {fingerprint: next(iter(values)) for fingerprint, values in roles.items()},
    )


def _session_fingerprints(
    runs: Sequence[Mapping[str, Any]], strategy: str
) -> list[str]:
    if strategy == "isolated_process":
        return [
            str(engine["engine_fingerprint_sha256"])
            for run in runs
            for engine in run["engines"]
        ]
    if strategy == "fully_resident":
        return sorted(
            {
                str(engine["engine_fingerprint_sha256"])
                for run in runs
                for engine in run["engines"]
            }
        )
    sessions = []
    active: dict[str, str] = {}
    for run in runs:
        for engine in run["engines"]:
            role = str(engine["role"])
            fingerprint = str(engine["engine_fingerprint_sha256"])
            if active.get(role) != fingerprint:
                sessions.append(fingerprint)
                active[role] = fingerprint
    return sessions


def _strategies(
    spec: EngineLifecyclePlanningSpec, runs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    init_seconds, memory, roles = _fingerprint_data(runs)
    measured = sum(
        float(engine["engine_init_seconds"])
        for run in runs
        for engine in run["engines"]
    )
    per_run_peak = max(
        sum(float(engine["reserved_memory_fraction"]) for engine in run["engines"])
        for run in runs
    )
    rows = []
    for strategy in _STRATEGIES:
        sessions = _session_fingerprints(runs, strategy)
        projected = (
            measured
            if strategy == "isolated_process"
            else sum(init_seconds[fingerprint] for fingerprint in sessions)
        )
        peak = (
            sum(memory.values())
            if strategy == "fully_resident"
            else per_run_peak
        )
        blockers = []
        if peak > spec.maximum_resident_memory_fraction + 1e-12:
            blockers.append("resident_memory_fraction_exceeds_limit")
        rows.append(
            {
                "strategy": strategy,
                "eligible": not blockers,
                "blockers": blockers,
                "engine_start_count": len(sessions),
                "engine_session_fingerprints": sessions,
                "unique_engine_count": len(set(sessions)),
                "unique_engine_roles": {
                    role: sum(1 for fingerprint in set(sessions) if roles[fingerprint] == role)
                    for role in _ROLES
                },
                "peak_reserved_memory_fraction": peak,
                "measured_isolated_startup_seconds": measured,
                "projected_startup_seconds": projected,
                "projected_startup_savings_seconds": measured - projected,
                "projected_startup_savings_fraction": (measured - projected) / measured,
            }
        )
    return rows


def _actions(
    spec: EngineLifecyclePlanningSpec,
    runs: Sequence[Mapping[str, Any]],
    strategy: str,
) -> list[dict[str, Any]]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unsupported lifecycle strategy: {strategy}")
    resident_fingerprints = sorted(
        {
            str(engine["engine_fingerprint_sha256"])
            for run in runs
            for engine in run["engines"]
        }
    )
    active: dict[str, str] = {}
    actions = []
    for run in runs:
        run_actions = []
        after_actions = []
        if strategy == "fully_resident" and run["sequence_index"] == 0:
            run_actions.extend(
                {
                    "action": "start_resident_engine",
                    "fingerprint": fingerprint,
                }
                for fingerprint in resident_fingerprints
            )
        for engine in run["engines"]:
            role = str(engine["role"])
            fingerprint = str(engine["engine_fingerprint_sha256"])
            if strategy == "isolated_process":
                run_actions.append(
                    {"action": "start_engine", "role": role, "fingerprint": fingerprint}
                )
                after_actions.append(
                    {"action": "stop_engine", "role": role, "fingerprint": fingerprint}
                )
            elif strategy == "fully_resident":
                run_actions.append(
                    {
                        "action": "bind_resident_engine",
                        "role": role,
                        "fingerprint": fingerprint,
                    }
                )
            else:
                previous = active.get(role)
                if previous is None:
                    run_actions.append(
                        {
                            "action": "start_engine",
                            "role": role,
                            "fingerprint": fingerprint,
                        }
                    )
                elif previous != fingerprint:
                    run_actions.extend(
                        [
                            {
                                "action": "stop_engine",
                                "role": role,
                                "fingerprint": previous,
                            },
                            {
                                "action": "start_engine",
                                "role": role,
                                "fingerprint": fingerprint,
                            },
                        ]
                    )
                else:
                    run_actions.append(
                        {
                            "action": "reuse_engine",
                            "role": role,
                            "fingerprint": fingerprint,
                        }
                    )
                active[role] = fingerprint
        run_actions.append(
            {
                "action": "reset_epoch_state",
                "requirements": list(spec.required_epoch_resets),
            }
        )
        actions.append(
            {
                "run_id": run["run_id"],
                "sequence_index": run["sequence_index"],
                "workload_seed": run["workload_seed"],
                "actions_before_measurement": run_actions,
                "actions_after_measurement": after_actions,
            }
        )
    return actions


def _derive(
    spec: EngineLifecyclePlanningSpec, runs: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]], dict[str, Any]]:
    strategies = _strategies(spec, runs)
    eligible = [row for row in strategies if row["eligible"]]
    preference = {"role_sticky": 0, "fully_resident": 1, "isolated_process": 2}
    selected = min(
        eligible,
        key=lambda row: (
            float(row["projected_startup_seconds"]),
            preference[str(row["strategy"])],
        ),
    )
    selected_name = str(selected["strategy"])
    actions = _actions(spec, runs, selected_name)
    audit = {
        "selected_strategy": selected_name,
        "run_count": len(runs),
        "measured_engine_start_count": len(runs) * len(_ROLES),
        "planned_engine_start_count": selected["engine_start_count"],
        "measured_isolated_startup_seconds": selected[
            "measured_isolated_startup_seconds"
        ],
        "projected_startup_seconds": selected["projected_startup_seconds"],
        "projected_startup_savings_seconds": selected[
            "projected_startup_savings_seconds"
        ],
        "projected_startup_savings_fraction": selected[
            "projected_startup_savings_fraction"
        ],
        "formal_execution_eligible": False,
        "validation_required": True,
    }
    return strategies, selected_name, actions, audit


def _validation_gate(spec: EngineLifecyclePlanningSpec) -> dict[str, Any]:
    return {
        "status": "required",
        "pair_seeds": list(spec.validation_pair_seeds),
        "comparisons": ["isolated_process", "role_sticky"],
        "required_checks": [
            "exact_output_token_ids_by_seed",
            "no_running_requests_before_reset",
            "prefix_cache_reset_succeeded",
            "zero_cross_epoch_score_cache_entries",
            "metric_deltas_start_from_epoch_snapshot",
            "formal_run_order_preserved",
        ],
        "promotion_rule": (
            "all state-isolation checks pass and pooled steady-state effects remain "
            "inside a separately frozen replay-noise envelope"
        ),
    }


def _build_payload(
    spec: EngineLifecyclePlanningSpec,
    runs: Sequence[Mapping[str, Any]],
    plan_sha256: str,
    harness_sha256: str,
) -> dict[str, Any]:
    validated_runs = _validated_engine_runs(list(runs))
    require_digest(plan_sha256, "lifecycle source plan_sha256")
    require_digest(harness_sha256, "harness_cost_assessment_sha256")
    strategies, selected, actions, audit = _derive(spec, validated_runs)
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "lifecycle_id": spec.lifecycle_id,
        "status": "validation_required",
        "requirements": spec.to_dict(),
        "source": {
            "plan_sha256": plan_sha256,
            "harness_cost_assessment_sha256": harness_sha256,
        },
        "engine_runs": validated_runs,
        "strategies": strategies,
        "selected_strategy": selected,
        "actions": actions,
        "validation_gate": _validation_gate(spec),
        "audit": audit,
    }
    payload["engine_lifecycle_plan_sha256"] = canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class EngineLifecyclePlan:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _PLAN_KEYS, "engine lifecycle plan")
        raw = dict(self.payload)
        digest = str(raw.pop("engine_lifecycle_plan_sha256", ""))
        require_digest(digest, "engine_lifecycle_plan_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("engine lifecycle plan SHA256 does not match content")
        if raw["schema_version"] != "1.0" or raw["status"] != "validation_required":
            raise ValueError("unsupported engine lifecycle plan")
        spec = EngineLifecyclePlanningSpec.from_dict(raw["requirements"])
        source = raw["source"]
        if not isinstance(source, Mapping):
            raise ValueError("engine lifecycle source must be an object")
        _expect_exact_keys(
            source,
            {"plan_sha256", "harness_cost_assessment_sha256"},
            "engine lifecycle source",
        )
        expected = _build_payload(
            spec,
            raw["engine_runs"],
            str(source["plan_sha256"]),
            str(source["harness_cost_assessment_sha256"]),
        )
        if expected != dict(self.payload):
            raise ValueError("engine lifecycle plan does not match derived schedule")

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def audit(self) -> dict[str, Any]:
        return dict(self.payload["audit"])

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EngineLifecyclePlan":
        return cls(dict(raw))


def plan_engine_lifecycle(
    spec: EngineLifecyclePlanningSpec,
    plan: CalibrationPlan,
    assessment: HarnessCostAssessment,
) -> EngineLifecyclePlan:
    runs = _engine_runs(plan, assessment)
    payload = _build_payload(
        spec,
        runs,
        canonical_sha256(plan.to_dict()),
        str(assessment.to_dict()["harness_cost_assessment_sha256"]),
    )
    return EngineLifecyclePlan(payload)
