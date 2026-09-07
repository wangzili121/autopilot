"""Anytime-valid decisions for replay-adjusted paired experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from math import exp, isfinite, log, log1p
from statistics import mean
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    ProtocolSpec,
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
)


_METHODS = {"betting_mixture", "finite_horizon_hoeffding"}
_MODES = {"prospective", "retrospective_diagnostic"}
_STATUSES = {
    "promote",
    "close_direction",
    "replicate",
    "budget_exhausted",
    "seed_pool_exhausted",
    "quality_rejected",
    "invalid_evidence",
    "diagnostic_only",
}
_WORKLOAD_RUN_FIELDS = {
    "workload_id",
    "workload_seed",
    "comparison_group_id",
    "pair_index",
    "sequence_index",
}
_OBSERVATION_KEYS = {
    "source_assessment_sha256",
    "plan_sha256",
    "workload_seed",
    "replay_comparison_group_id",
    "candidate_comparison_group_id",
    "replay_baseline_value",
    "replay_second_value",
    "candidate_baseline_value",
    "candidate_value",
    "candidate_directional_log_effect",
    "replay_directional_log_drift",
    "adjusted_log_effect",
}
_ASSESSMENT_KEYS = {
    "schema_version",
    "producer",
    "effect_id",
    "status",
    "requirements",
    "source",
    "observations",
    "summary",
    "decision",
    "source_issues",
    "audit",
    "sequential_effect_assessment_sha256",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_sha256(configuration: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {name: value for name, value in configuration.items() if name != "configuration_id"}
    )


@dataclass(frozen=True, slots=True)
class SequentialEffectSpec:
    effect_id: str
    analysis_mode: str
    baseline_configuration_id: str
    replay_control_configuration_id: str
    candidate_configuration_id: str
    primary_metric: str
    direction: str
    algorithm_context_sha256: str
    workload_context_sha256: str
    environment_context_sha256: str
    baseline_settings_sha256: str
    candidate_settings_sha256: str
    lower_log_effect_bound: float
    upper_log_effect_bound: float
    confidence_alpha: float
    confidence_method: str
    betting_fractions: tuple[float, ...]
    minimum_improvement_fraction: float
    minimum_decision_pairs: int
    maximum_pair_count: int
    replication_batch_pairs: int
    replication_seed_pool: tuple[int, ...]
    require_quality_constraints: bool
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported sequential effect spec: {self.schema_version}")
        require_id(self.effect_id, "effect_id")
        for name in (
            "baseline_configuration_id",
            "replay_control_configuration_id",
            "candidate_configuration_id",
        ):
            require_id(getattr(self, name), name)
        configuration_ids = {
            self.baseline_configuration_id,
            self.replay_control_configuration_id,
            self.candidate_configuration_id,
        }
        if len(configuration_ids) != 3:
            raise ValueError("sequential effect configuration ids must be distinct")
        if self.analysis_mode not in _MODES:
            raise ValueError(f"unsupported sequential analysis mode: {self.analysis_mode}")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("sequential effect direction must be maximize or minimize")
        if not self.primary_metric:
            raise ValueError("sequential effect primary_metric cannot be empty")
        for name in (
            "algorithm_context_sha256",
            "workload_context_sha256",
            "environment_context_sha256",
            "baseline_settings_sha256",
            "candidate_settings_sha256",
        ):
            require_digest(getattr(self, name), name)
        numeric = (
            self.lower_log_effect_bound,
            self.upper_log_effect_bound,
            self.confidence_alpha,
            self.minimum_improvement_fraction,
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("sequential effect numeric requirements must be finite")
        if not self.lower_log_effect_bound < 0 < self.upper_log_effect_bound:
            raise ValueError("log-effect bounds must straddle zero")
        if not 0 < self.confidence_alpha < 1:
            raise ValueError("confidence_alpha must be between zero and one")
        if self.minimum_improvement_fraction < 0:
            raise ValueError("minimum_improvement_fraction must be non-negative")
        if self.confidence_method not in _METHODS:
            raise ValueError(f"unsupported confidence method: {self.confidence_method}")
        if (
            not self.betting_fractions
            or tuple(sorted(set(self.betting_fractions))) != self.betting_fractions
            or any(not isfinite(value) or not 0 < value < 1 for value in self.betting_fractions)
        ):
            raise ValueError("betting_fractions must be sorted unique values between 0 and 1")
        for name in (
            "minimum_decision_pairs",
            "maximum_pair_count",
            "replication_batch_pairs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_decision_pairs > self.maximum_pair_count:
            raise ValueError("minimum_decision_pairs cannot exceed maximum_pair_count")
        if self.maximum_pair_count % 2 or self.replication_batch_pairs % 2:
            raise ValueError("pair budgets must preserve two-pair ABBA/BAAB blocks")
        if self.replication_batch_pairs > self.maximum_pair_count:
            raise ValueError("replication_batch_pairs cannot exceed maximum_pair_count")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.replication_seed_pool
        ):
            raise ValueError("replication_seed_pool must contain non-negative integers")
        if len(set(self.replication_seed_pool)) != len(self.replication_seed_pool):
            raise ValueError("replication_seed_pool must contain unique seeds")
        if not isinstance(self.require_quality_constraints, bool):
            raise ValueError("require_quality_constraints must be boolean")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "effect_id": self.effect_id,
            "analysis_mode": self.analysis_mode,
            "baseline_configuration_id": self.baseline_configuration_id,
            "replay_control_configuration_id": self.replay_control_configuration_id,
            "candidate_configuration_id": self.candidate_configuration_id,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "algorithm_context_sha256": self.algorithm_context_sha256,
            "workload_context_sha256": self.workload_context_sha256,
            "environment_context_sha256": self.environment_context_sha256,
            "baseline_settings_sha256": self.baseline_settings_sha256,
            "candidate_settings_sha256": self.candidate_settings_sha256,
            "lower_log_effect_bound": self.lower_log_effect_bound,
            "upper_log_effect_bound": self.upper_log_effect_bound,
            "confidence_alpha": self.confidence_alpha,
            "confidence_method": self.confidence_method,
            "betting_fractions": list(self.betting_fractions),
            "minimum_improvement_fraction": self.minimum_improvement_fraction,
            "minimum_decision_pairs": self.minimum_decision_pairs,
            "maximum_pair_count": self.maximum_pair_count,
            "replication_batch_pairs": self.replication_batch_pairs,
            "replication_seed_pool": list(self.replication_seed_pool),
            "require_quality_constraints": self.require_quality_constraints,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequentialEffectSpec":
        expected = {
            "schema_version",
            "effect_id",
            "analysis_mode",
            "baseline_configuration_id",
            "replay_control_configuration_id",
            "candidate_configuration_id",
            "primary_metric",
            "direction",
            "algorithm_context_sha256",
            "workload_context_sha256",
            "environment_context_sha256",
            "baseline_settings_sha256",
            "candidate_settings_sha256",
            "lower_log_effect_bound",
            "upper_log_effect_bound",
            "confidence_alpha",
            "confidence_method",
            "betting_fractions",
            "minimum_improvement_fraction",
            "minimum_decision_pairs",
            "maximum_pair_count",
            "replication_batch_pairs",
            "replication_seed_pool",
            "require_quality_constraints",
        }
        _expect_exact_keys(raw, expected, "sequential effect spec")
        numeric_names = {
            "lower_log_effect_bound",
            "upper_log_effect_bound",
            "confidence_alpha",
            "minimum_improvement_fraction",
        }
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], (int, float))
            for name in numeric_names
        ):
            raise ValueError("sequential effect numeric requirements must be numeric")
        integer_names = {
            "minimum_decision_pairs",
            "maximum_pair_count",
            "replication_batch_pairs",
        }
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], int)
            for name in integer_names
        ):
            raise ValueError("sequential effect pair requirements must be integers")
        fractions = raw["betting_fractions"]
        seeds = raw["replication_seed_pool"]
        if not isinstance(fractions, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in fractions
        ):
            raise ValueError("betting_fractions must be numeric values")
        if not isinstance(seeds, list):
            raise ValueError("replication_seed_pool must be a list")
        if not isinstance(raw["require_quality_constraints"], bool):
            raise ValueError("require_quality_constraints must be boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            effect_id=str(raw["effect_id"]),
            analysis_mode=str(raw["analysis_mode"]),
            baseline_configuration_id=str(raw["baseline_configuration_id"]),
            replay_control_configuration_id=str(raw["replay_control_configuration_id"]),
            candidate_configuration_id=str(raw["candidate_configuration_id"]),
            primary_metric=str(raw["primary_metric"]),
            direction=str(raw["direction"]),
            algorithm_context_sha256=str(raw["algorithm_context_sha256"]),
            workload_context_sha256=str(raw["workload_context_sha256"]),
            environment_context_sha256=str(raw["environment_context_sha256"]),
            baseline_settings_sha256=str(raw["baseline_settings_sha256"]),
            candidate_settings_sha256=str(raw["candidate_settings_sha256"]),
            lower_log_effect_bound=float(raw["lower_log_effect_bound"]),
            upper_log_effect_bound=float(raw["upper_log_effect_bound"]),
            confidence_alpha=float(raw["confidence_alpha"]),
            confidence_method=str(raw["confidence_method"]),
            betting_fractions=tuple(float(value) for value in fractions),
            minimum_improvement_fraction=float(raw["minimum_improvement_fraction"]),
            minimum_decision_pairs=raw["minimum_decision_pairs"],
            maximum_pair_count=raw["maximum_pair_count"],
            replication_batch_pairs=raw["replication_batch_pairs"],
            replication_seed_pool=tuple(seeds),
            require_quality_constraints=raw["require_quality_constraints"],
        )


def context_bindings_from_calibration_spec(
    spec: CalibrationSpec,
    candidate_configuration_id: str,
) -> dict[str, str]:
    """Build the exact context bindings required by a sequential-effect spec."""

    candidates = {
        candidate.configuration_id: candidate for candidate in spec.candidates
    }
    if candidate_configuration_id not in candidates:
        raise ValueError("candidate is absent from calibration spec")
    semantic = spec.semantic_contract
    workload = spec.workload_contract
    return {
        "algorithm_context_sha256": canonical_sha256(
            {
                "algorithm_id": semantic.algorithm_id,
                "semantic_class": semantic.semantic_class,
                "graph_sha256": semantic.graph_sha256,
                **dict(semantic.invariants),
            }
        ),
        "workload_context_sha256": canonical_sha256(
            {
                "dataset_sha256": workload.dataset_sha256,
                "arrival_trace_sha256": workload.arrival_trace_sha256,
                **dict(workload.parameters),
            }
        ),
        "environment_context_sha256": canonical_sha256(
            spec.environment_contract.to_dict()
        ),
        "baseline_settings_sha256": canonical_sha256(dict(spec.baseline.settings)),
        "candidate_settings_sha256": canonical_sha256(
            dict(candidates[candidate_configuration_id].settings)
        ),
    }


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def _betting_log_wealth(
    normalized: Sequence[float],
    hypothesized_mean: float,
    fractions: Sequence[float],
    sign: float,
) -> float:
    components = []
    for fraction in fractions:
        bet = sign * fraction
        components.append(
            sum(log1p(bet * (value - hypothesized_mean)) for value in normalized)
        )
    return _logsumexp(components) - log(len(components))


def _betting_mixture_interval(
    values: Sequence[float], spec: SequentialEffectSpec
) -> tuple[float, float]:
    lower = spec.lower_log_effect_bound
    upper = spec.upper_log_effect_bound
    if not values:
        return lower, upper
    width = upper - lower
    normalized = [(value - lower) / width for value in values]
    threshold = log(2.0 / spec.confidence_alpha)

    def lower_wealth(value: float) -> float:
        return _betting_log_wealth(normalized, value, spec.betting_fractions, 1.0)

    def upper_wealth(value: float) -> float:
        return _betting_log_wealth(normalized, value, spec.betting_fractions, -1.0)

    normalized_lower = 0.0
    if lower_wealth(0.0) >= threshold:
        left, right = 0.0, 1.0
        for _ in range(80):
            middle = (left + right) / 2.0
            if lower_wealth(middle) >= threshold:
                left = middle
            else:
                right = middle
        normalized_lower = right

    normalized_upper = 1.0
    if upper_wealth(1.0) >= threshold:
        left, right = 0.0, 1.0
        for _ in range(80):
            middle = (left + right) / 2.0
            if upper_wealth(middle) >= threshold:
                right = middle
            else:
                left = middle
        normalized_upper = left
    return lower + width * normalized_lower, lower + width * normalized_upper


def _finite_horizon_hoeffding_interval(
    values: Sequence[float], spec: SequentialEffectSpec
) -> tuple[float, float]:
    lower = spec.lower_log_effect_bound
    upper = spec.upper_log_effect_bound
    if not values:
        return lower, upper
    per_look_alpha = spec.confidence_alpha / spec.maximum_pair_count
    half_width = (upper - lower) * (
        log(2.0 / per_look_alpha) / (2.0 * len(values))
    ) ** 0.5
    sample_mean = mean(values)
    return max(lower, sample_mean - half_width), min(upper, sample_mean + half_width)


def confidence_interval(
    values: Sequence[float], spec: SequentialEffectSpec
) -> tuple[float, float]:
    """Return a time-uniform interval under the spec's bounded-effect contract."""

    if any(
        not isfinite(value)
        or value < spec.lower_log_effect_bound
        or value > spec.upper_log_effect_bound
        for value in values
    ):
        raise ValueError("an adjusted effect is outside the predeclared bounds")
    if spec.confidence_method == "betting_mixture":
        return _betting_mixture_interval(values, spec)
    return _finite_horizon_hoeffding_interval(values, spec)


def _effect_by_role(
    effects: Sequence[Mapping[str, Any]],
    spec: SequentialEffectSpec,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, list[str]]:
    replay = [
        effect
        for effect in effects
        if effect.get("candidate_configuration_id")
        == spec.replay_control_configuration_id
        and effect.get("replay_control") is True
    ]
    candidate = [
        effect
        for effect in effects
        if effect.get("candidate_configuration_id") == spec.candidate_configuration_id
        and effect.get("replay_control") is False
    ]
    issues = []
    if len(replay) != 1:
        issues.append("source_requires_exactly_one_replay_effect")
    if len(candidate) != 1:
        issues.append("source_requires_exactly_one_candidate_effect")
    return (
        replay[0] if len(replay) == 1 else None,
        candidate[0] if len(candidate) == 1 else None,
        issues,
    )


def _validate_context(
    assessment: Mapping[str, Any],
    group_ids: set[str],
    spec: SequentialEffectSpec,
) -> list[str]:
    ledger = assessment.get("ledger")
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), list):
        return ["source_evidence_ledger_missing"]
    records = [
        record
        for record in ledger["records"]
        if isinstance(record, Mapping)
        and isinstance(record.get("workload"), Mapping)
        and record["workload"].get("comparison_group_id") in group_ids
    ]
    if not records:
        return ["source_context_records_missing"]
    issues = []
    contexts = {
        "algorithm_context_sha256": {
            canonical_sha256(dict(record["algorithm"]))
            for record in records
            if isinstance(record.get("algorithm"), Mapping)
        },
        "workload_context_sha256": {
            canonical_sha256(
                {
                    name: value
                    for name, value in record["workload"].items()
                    if name not in _WORKLOAD_RUN_FIELDS
                }
            )
            for record in records
        },
        "environment_context_sha256": {
            canonical_sha256(dict(record["environment"]))
            for record in records
            if isinstance(record.get("environment"), Mapping)
        },
    }
    for name, values in contexts.items():
        if values != {getattr(spec, name)}:
            issues.append(f"{name}_mismatch")

    configurations: dict[str, set[str]] = {}
    for record in records:
        configuration = record.get("configuration")
        if not isinstance(configuration, Mapping):
            continue
        configuration_id = configuration.get("configuration_id")
        if isinstance(configuration_id, str):
            configurations.setdefault(configuration_id, set()).add(
                _settings_sha256(configuration)
            )
    expected = {
        spec.baseline_configuration_id: spec.baseline_settings_sha256,
        spec.replay_control_configuration_id: spec.baseline_settings_sha256,
        spec.candidate_configuration_id: spec.candidate_settings_sha256,
    }
    for configuration_id, digest in expected.items():
        if configurations.get(configuration_id) != {digest}:
            issues.append(f"configuration_binding_mismatch:{configuration_id}")
    return issues


def _pair_map(effect: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = effect.get("pair_effects")
    if not isinstance(raw, list):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for pair in raw:
        if not isinstance(pair, Mapping):
            continue
        seed = pair.get("workload_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in result:
            return {}
        result[seed] = pair
    return result


def _extract_source(
    source_sha256: str,
    assessment: Mapping[str, Any],
    spec: SequentialEffectSpec,
) -> tuple[list[dict[str, Any]], list[str], int]:
    issues = []
    quality_failures = 0
    plan_sha256 = assessment.get("plan_sha256")
    if not isinstance(plan_sha256, str):
        plan_sha256 = ""
    try:
        require_digest(plan_sha256, "source plan_sha256")
    except ValueError:
        issues.append("source_plan_sha256_invalid")
    if assessment.get("formal_complete") is not True:
        issues.append("source_not_formally_complete")
    if assessment.get("issues") != []:
        issues.append("source_has_calibration_issues")
    effects_raw = assessment.get("effects")
    if not isinstance(effects_raw, list) or any(
        not isinstance(effect, Mapping) for effect in effects_raw
    ):
        return [], sorted(set([*issues, "source_effects_invalid"])), quality_failures
    replay, candidate, role_issues = _effect_by_role(effects_raw, spec)
    issues.extend(role_issues)
    if replay is None or candidate is None:
        return [], sorted(set(issues)), quality_failures
    for name, effect in (("replay", replay), ("candidate", candidate)):
        if effect.get("formal_group") is not True:
            issues.append(f"{name}_effect_not_formal")
        if effect.get("primary_metric") != spec.primary_metric:
            issues.append(f"{name}_metric_mismatch")
        if effect.get("direction") != spec.direction:
            issues.append(f"{name}_direction_mismatch")
        if effect.get("complete_pair_count") != effect.get("expected_pair_count"):
            issues.append(f"{name}_effect_incomplete")
    if (
        spec.require_quality_constraints
        and candidate.get("quality_constraints_satisfied") is not True
    ):
        quality_failures += 1
    issues.extend(
        _validate_context(
            assessment,
            {
                str(replay.get("comparison_group_id", "")),
                str(candidate.get("comparison_group_id", "")),
            },
            spec,
        )
    )
    replay_pairs = _pair_map(replay)
    candidate_pairs = _pair_map(candidate)
    if not replay_pairs or replay_pairs.keys() != candidate_pairs.keys():
        issues.append("same_seed_replay_candidate_pairs_required")
        return [], sorted(set(issues)), quality_failures

    observations = []
    sign = 1.0 if spec.direction == "maximize" else -1.0
    for seed in sorted(replay_pairs):
        replay_pair = replay_pairs[seed]
        candidate_pair = candidate_pairs[seed]
        numeric_names = {
            "replay_baseline_value": replay_pair.get("baseline_value"),
            "replay_second_value": replay_pair.get("candidate_value"),
            "candidate_baseline_value": candidate_pair.get("baseline_value"),
            "candidate_value": candidate_pair.get("candidate_value"),
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) <= 0
            for value in numeric_names.values()
        ):
            issues.append(f"nonpositive_or_invalid_metric_value:{seed}")
            continue
        values = {name: float(value) for name, value in numeric_names.items()}
        candidate_log_effect = sign * log(
            values["candidate_value"] / values["candidate_baseline_value"]
        )
        replay_log_drift = sign * log(
            values["replay_second_value"] / values["replay_baseline_value"]
        )
        observations.append(
            {
                "source_assessment_sha256": source_sha256,
                "plan_sha256": plan_sha256,
                "workload_seed": seed,
                "replay_comparison_group_id": replay["comparison_group_id"],
                "candidate_comparison_group_id": candidate["comparison_group_id"],
                **values,
                "candidate_directional_log_effect": candidate_log_effect,
                "replay_directional_log_drift": replay_log_drift,
                "adjusted_log_effect": candidate_log_effect - replay_log_drift,
            }
        )
    return observations, sorted(set(issues)), quality_failures


def _summary(
    observations: Sequence[Mapping[str, Any]],
    spec: SequentialEffectSpec,
    valid_for_interval: bool,
) -> dict[str, Any]:
    values = [float(observation["adjusted_log_effect"]) for observation in observations]
    if valid_for_interval:
        interval_lower, interval_upper = confidence_interval(values, spec)
    else:
        interval_lower, interval_upper = (
            spec.lower_log_effect_bound,
            spec.upper_log_effect_bound,
        )
    sample_mean = mean(values) if values else None
    remaining_pair_budget = max(0, spec.maximum_pair_count - len(values))
    projected_lower = None
    projected_upper = None
    projected_boundary = "insufficient_observations"
    if sample_mean is not None and valid_for_interval:
        projected_values = [
            *values,
            *([sample_mean] * remaining_pair_budget),
        ]
        projected_lower, projected_upper = confidence_interval(
            projected_values, spec
        )
        target = log1p(spec.minimum_improvement_fraction)
        if projected_lower > target:
            projected_boundary = "promote"
        elif projected_upper < 0:
            projected_boundary = "close_direction"
        else:
            projected_boundary = "no_boundary"
    return {
        "pair_count": len(values),
        "sample_mean_log_effect": sample_mean,
        "adjusted_geomean_ratio": None if sample_mean is None else exp(sample_mean),
        "observed_min_log_effect": min(values) if values else None,
        "observed_max_log_effect": max(values) if values else None,
        "confidence_method": spec.confidence_method,
        "confidence_alpha": spec.confidence_alpha,
        "interval_lower_log_effect": interval_lower,
        "interval_upper_log_effect": interval_upper,
        "interval_lower_improvement_fraction": exp(interval_lower) - 1.0,
        "interval_upper_improvement_fraction": exp(interval_upper) - 1.0,
        "minimum_improvement_log_effect": log1p(spec.minimum_improvement_fraction),
        "maximum_pair_count": spec.maximum_pair_count,
        "remaining_pair_budget": remaining_pair_budget,
        "projected_at_max_pairs_assumption": "future_effects_equal_current_sample_mean",
        "projected_interval_lower_log_effect": projected_lower,
        "projected_interval_upper_log_effect": projected_upper,
        "projected_boundary_at_current_mean": projected_boundary,
    }


def _decision(
    spec: SequentialEffectSpec,
    observations: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    source_issues: Sequence[str],
    quality_failure_count: int,
    bound_violation_count: int,
) -> tuple[str, str, list[int], list[str]]:
    pair_count = len(observations)
    used_seeds = {int(observation["workload_seed"]) for observation in observations}
    available_seeds = [
        seed for seed in spec.replication_seed_pool if seed not in used_seeds
    ]
    remaining_budget = max(0, spec.maximum_pair_count - pair_count)
    requested = min(spec.replication_batch_pairs, remaining_budget)
    if requested % 2:
        requested -= 1
    next_seeds = available_seeds[:requested]
    if len(next_seeds) % 2:
        next_seeds = next_seeds[:-1]

    if source_issues or bound_violation_count:
        reasons = list(source_issues)
        if bound_violation_count:
            reasons.append("predeclared_effect_bound_violated")
        return "invalid_evidence", "repair_evidence_or_freeze_new_spec", [], sorted(set(reasons))
    if quality_failure_count:
        return (
            "quality_rejected",
            "retain_control_and_close_candidate_configuration",
            [],
            ["quality_constraints_not_satisfied"],
        )
    if spec.analysis_mode == "retrospective_diagnostic":
        return (
            "diagnostic_only",
            "freeze_prospective_spec_before_collecting_new_pairs",
            [],
            ["post_hoc_data_cannot_support_sequential_claim"],
        )
    lower = float(summary["interval_lower_log_effect"])
    upper = float(summary["interval_upper_log_effect"])
    target = float(summary["minimum_improvement_log_effect"])
    if pair_count >= spec.minimum_decision_pairs and lower > target:
        return "promote", "run_independent_holdout", [], ["lower_bound_exceeds_minimum_gain"]
    if pair_count >= spec.minimum_decision_pairs and upper < 0:
        return (
            "close_direction",
            "retain_control_and_close_direction",
            [],
            ["upper_bound_below_zero"],
        )
    if pair_count >= spec.maximum_pair_count:
        return (
            "budget_exhausted",
            "retain_control_without_claim",
            [],
            ["maximum_pair_count_reached"],
        )
    if len(next_seeds) < requested or not next_seeds:
        return (
            "seed_pool_exhausted",
            "extend_frozen_fresh_seed_pool",
            [],
            ["fresh_seed_pool_exhausted"],
        )
    return (
        "replicate",
        "collect_fresh_replay_candidate_pairs",
        next_seeds,
        ["confidence_sequence_crossed_no_boundary"],
    )


@dataclass(frozen=True, slots=True)
class SequentialEffectAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _ASSESSMENT_KEYS, "sequential effect assessment")
        raw = dict(self.payload)
        digest = str(raw.pop("sequential_effect_assessment_sha256", ""))
        require_digest(digest, "sequential_effect_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("sequential effect assessment SHA256 does not match content")
        if raw["schema_version"] != "1.0" or raw["status"] not in _STATUSES:
            raise ValueError("unsupported sequential effect assessment")
        spec = SequentialEffectSpec.from_dict(raw["requirements"])
        if raw["effect_id"] != spec.effect_id:
            raise ValueError("effect id does not match frozen requirements")
        source = raw["source"]
        if not isinstance(source, Mapping):
            raise ValueError("sequential effect source must be an object")
        _expect_exact_keys(
            source,
            {"effect_spec_sha256", "calibration_assessments"},
            "sequential effect source",
        )
        if source["effect_spec_sha256"] != spec.sha256:
            raise ValueError("sequential effect source does not match frozen requirements")
        source_assessments = source["calibration_assessments"]
        if not isinstance(source_assessments, list):
            raise ValueError("calibration_assessments must be a list")
        for item in source_assessments:
            if not isinstance(item, Mapping):
                raise ValueError("calibration assessment reference must be an object")
            _expect_exact_keys(item, {"assessment_sha256", "plan_sha256"}, "source assessment")
            require_digest(str(item["assessment_sha256"]), "assessment_sha256")
            require_digest(str(item["plan_sha256"]), "plan_sha256")
        observations = raw["observations"]
        if not isinstance(observations, list):
            raise ValueError("sequential observations must be a list")
        seen_seeds = set()
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError("sequential observation must be an object")
            _expect_exact_keys(observation, _OBSERVATION_KEYS, "sequential observation")
            require_digest(str(observation["source_assessment_sha256"]), "source digest")
            require_digest(str(observation["plan_sha256"]), "plan digest")
            seed = observation["workload_seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("sequential observation seed is invalid")
            if seed in seen_seeds:
                raise ValueError("sequential observation seeds must be globally unique")
            seen_seeds.add(seed)
            for name in _OBSERVATION_KEYS - {
                "source_assessment_sha256",
                "plan_sha256",
                "workload_seed",
                "replay_comparison_group_id",
                "candidate_comparison_group_id",
            }:
                value = observation[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                ):
                    raise ValueError(f"sequential observation {name} must be finite")
            candidate_log = float(observation["candidate_directional_log_effect"])
            replay_log = float(observation["replay_directional_log_drift"])
            adjusted = float(observation["adjusted_log_effect"])
            if abs(adjusted - (candidate_log - replay_log)) > 1e-12:
                raise ValueError("adjusted log effect does not match component effects")
        source_issues = raw["source_issues"]
        audit = raw["audit"]
        if not isinstance(source_issues, list) or any(
            not isinstance(issue, str) for issue in source_issues
        ):
            raise ValueError("source_issues must be strings")
        if source_issues != sorted(set(source_issues)):
            raise ValueError("source_issues must be sorted and unique")
        if not isinstance(audit, Mapping):
            raise ValueError("sequential effect audit must be an object")
        _expect_exact_keys(
            audit,
            {
                "source_assessment_count",
                "pair_count",
                "source_issue_count",
                "quality_failure_count",
                "bound_violation_count",
            },
            "sequential effect audit",
        )
        expected_counts = {
            "source_assessment_count": len(source_assessments),
            "pair_count": len(observations),
            "source_issue_count": len(source_issues),
        }
        if any(audit[name] != value for name, value in expected_counts.items()):
            raise ValueError("sequential effect audit counts do not match payload")
        bound_violations = sum(
            not spec.lower_log_effect_bound
            <= float(item["adjusted_log_effect"])
            <= spec.upper_log_effect_bound
            for item in observations
        )
        if audit["bound_violation_count"] != bound_violations:
            raise ValueError("bound violation count does not match observations")
        quality_failures = audit["quality_failure_count"]
        if (
            isinstance(quality_failures, bool)
            or not isinstance(quality_failures, int)
            or quality_failures < 0
        ):
            raise ValueError("quality failure count must be a non-negative integer")
        expected_summary = _summary(
            observations,
            spec,
            not source_issues and not bound_violations,
        )
        if canonical_json(raw["summary"]) != canonical_json(expected_summary):
            raise ValueError("sequential effect summary does not match observations")
        status, action, next_seeds, reasons = _decision(
            spec,
            observations,
            expected_summary,
            source_issues,
            quality_failures,
            bound_violations,
        )
        decision = raw["decision"]
        expected_decision = {
            "action": action,
            "next_pair_seeds": next_seeds,
            "reasons": reasons,
        }
        if raw["status"] != status or canonical_json(decision) != canonical_json(expected_decision):
            raise ValueError("sequential effect decision does not match frozen rules")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        return {
            "effect_id": self.payload["effect_id"],
            "status": self.status,
            **dict(self.payload["summary"]),
            **dict(self.payload["decision"]),
            **dict(self.payload["audit"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequentialEffectAssessment":
        return cls(raw)


def assess_sequential_effect(
    spec: SequentialEffectSpec,
    calibration_assessments: Sequence[tuple[str, Mapping[str, Any]]],
) -> SequentialEffectAssessment:
    """Extract same-seed replay/candidate effects and apply frozen boundaries."""

    observations: list[dict[str, Any]] = []
    source_issues = []
    quality_failure_count = 0
    source_references = []
    seen_source_digests = set()
    for source_sha256, assessment in calibration_assessments:
        require_digest(source_sha256, "calibration assessment sha256")
        if source_sha256 in seen_source_digests:
            source_issues.append(f"duplicate_source_assessment:{source_sha256}")
            continue
        seen_source_digests.add(source_sha256)
        extracted, issues, quality_failures = _extract_source(
            source_sha256, assessment, spec
        )
        observations.extend(extracted)
        source_issues.extend(f"{source_sha256}:{issue}" for issue in issues)
        quality_failure_count += quality_failures
        plan_sha256 = assessment.get("plan_sha256")
        if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
            plan_sha256 = "0" * 64
        source_references.append(
            {"assessment_sha256": source_sha256, "plan_sha256": plan_sha256}
        )
    seed_counts: dict[int, int] = {}
    for observation in observations:
        seed = int(observation["workload_seed"])
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
    duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
    if duplicate_seeds:
        source_issues.append(
            "duplicate_workload_seeds:" + ",".join(str(seed) for seed in duplicate_seeds)
        )
        observations = []
    source_issues = sorted(set(source_issues))
    observations.sort(
        key=lambda item: (item["source_assessment_sha256"], item["workload_seed"])
    )
    bound_violation_count = sum(
        not spec.lower_log_effect_bound
        <= observation["adjusted_log_effect"]
        <= spec.upper_log_effect_bound
        for observation in observations
    )
    summary = _summary(
        observations,
        spec,
        not source_issues and not bound_violation_count,
    )
    status, action, next_seeds, reasons = _decision(
        spec,
        observations,
        summary,
        source_issues,
        quality_failure_count,
        bound_violation_count,
    )
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "effect_id": spec.effect_id,
        "status": status,
        "requirements": spec.to_dict(),
        "source": {
            "effect_spec_sha256": spec.sha256,
            "calibration_assessments": sorted(
                source_references, key=lambda item: item["assessment_sha256"]
            ),
        },
        "observations": observations,
        "summary": summary,
        "decision": {
            "action": action,
            "next_pair_seeds": next_seeds,
            "reasons": reasons,
        },
        "source_issues": source_issues,
        "audit": {
            "source_assessment_count": len(source_references),
            "pair_count": len(observations),
            "source_issue_count": len(source_issues),
            "quality_failure_count": quality_failure_count,
            "bound_violation_count": bound_violation_count,
        },
    }
    payload["sequential_effect_assessment_sha256"] = canonical_sha256(payload)
    return SequentialEffectAssessment(payload)


def assess_sequential_effect_files(
    spec: SequentialEffectSpec, paths: Sequence[str]
) -> SequentialEffectAssessment:
    sources = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, Mapping):
            raise ValueError(f"calibration assessment {path} must be an object")
        sources.append((_file_sha256(path), raw))
    return assess_sequential_effect(spec, sources)


def calibration_spec_from_sequential_effect(
    template: CalibrationSpec,
    spec: SequentialEffectSpec,
    campaign_id: str,
    pair_seeds: Sequence[int],
) -> CalibrationSpec:
    """Emit an exact replay-plus-candidate ABBA/BAAB replication spec."""

    require_id(campaign_id, "campaign_id")
    if not pair_seeds or len(pair_seeds) % 2:
        raise ValueError("sequential replication requires complete two-pair blocks")
    if len(set(pair_seeds)) != len(pair_seeds):
        raise ValueError("sequential replication seeds must be unique")
    if any(seed not in spec.replication_seed_pool for seed in pair_seeds):
        raise ValueError("sequential replication seed is outside the frozen seed pool")
    bindings = context_bindings_from_calibration_spec(
        template, spec.candidate_configuration_id
    )
    for name, expected in bindings.items():
        if getattr(spec, name) != expected:
            raise ValueError(f"calibration template {name} does not match effect spec")
    if template.baseline.configuration_id != spec.baseline_configuration_id:
        raise ValueError("calibration template baseline id does not match effect spec")
    candidates = {
        candidate.configuration_id: candidate for candidate in template.candidates
    }
    if spec.replay_control_configuration_id not in candidates:
        raise ValueError("calibration template lacks the replay control")
    if candidates[spec.replay_control_configuration_id].settings != template.baseline.settings:
        raise ValueError("replay control must be manifest-identical to baseline settings")
    candidate = candidates[spec.candidate_configuration_id]
    replay = ConfigurationSpec(
        configuration_id=spec.replay_control_configuration_id,
        settings=template.baseline.settings,
        description="Manifest-identical replay control for sequential drift adjustment",
    )
    return replace(
        template,
        campaign_id=campaign_id,
        protocol=ProtocolSpec(
            pattern=template.protocol.pattern,
            blocks=len(pair_seeds) // 2,
            pair_seeds=tuple(pair_seeds),
        ),
        candidates=(replay, candidate),
    )
