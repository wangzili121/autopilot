"""Diagnostic comparison of finite-budget sequential experiment designs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
from math import exp, isfinite, log, log1p, sqrt
import random
from statistics import mean, median
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.sequential_effect import (
    SequentialEffectAssessment,
    SequentialEffectSpec,
    confidence_interval,
)


_METHODS = {
    "frozen_spec_method",
    "finite_horizon_hoeffding",
    "hedged_capital_predictable_plugin",
}
_NOISE_MODELS = {"observed_symmetric_residual", "bounded_endpoints"}
_DECISIONS = ("promote", "close_direction", "exclude_useful_gain", "no_decision")
_STUDY_KEYS = {
    "schema_version",
    "producer",
    "study_id",
    "status",
    "requirements",
    "source",
    "observed_method_comparison",
    "simulation_results",
    "audit",
    "sequential_design_study_sha256",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class SequentialDesignStudySpec:
    study_id: str
    methods: tuple[str, ...]
    pair_looks: tuple[int, ...]
    true_log_effects: tuple[float, ...]
    noise_models: tuple[str, ...]
    simulation_trials: int
    simulation_seed: int
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported sequential design study: {self.schema_version}")
        require_id(self.study_id, "study_id")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("study methods must be non-empty and unique")
        unknown_methods = sorted(set(self.methods) - _METHODS)
        if unknown_methods:
            raise ValueError(f"unsupported study methods: {unknown_methods}")
        if (
            not self.pair_looks
            or tuple(sorted(set(self.pair_looks))) != self.pair_looks
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value % 2
                for value in self.pair_looks
            )
        ):
            raise ValueError("pair_looks must be sorted unique positive even integers")
        if not self.true_log_effects or any(
            not isfinite(value) for value in self.true_log_effects
        ):
            raise ValueError("true_log_effects must be non-empty and finite")
        if not self.noise_models or len(set(self.noise_models)) != len(self.noise_models):
            raise ValueError("noise_models must be non-empty and unique")
        unknown_noise = sorted(set(self.noise_models) - _NOISE_MODELS)
        if unknown_noise:
            raise ValueError(f"unsupported noise models: {unknown_noise}")
        if (
            isinstance(self.simulation_trials, bool)
            or not isinstance(self.simulation_trials, int)
            or not 1 <= self.simulation_trials <= 100_000
        ):
            raise ValueError("simulation_trials must be between 1 and 100000")
        if (
            isinstance(self.simulation_seed, bool)
            or not isinstance(self.simulation_seed, int)
            or self.simulation_seed < 0
        ):
            raise ValueError("simulation_seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "methods": list(self.methods),
            "pair_looks": list(self.pair_looks),
            "true_log_effects": list(self.true_log_effects),
            "noise_models": list(self.noise_models),
            "simulation_trials": self.simulation_trials,
            "simulation_seed": self.simulation_seed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequentialDesignStudySpec":
        expected = {
            "schema_version",
            "study_id",
            "methods",
            "pair_looks",
            "true_log_effects",
            "noise_models",
            "simulation_trials",
            "simulation_seed",
        }
        _expect_exact_keys(raw, expected, "sequential design study spec")
        for name in ("methods", "pair_looks", "true_log_effects", "noise_models"):
            if not isinstance(raw[name], list):
                raise ValueError(f"{name} must be a list")
        return cls(
            schema_version=str(raw["schema_version"]),
            study_id=str(raw["study_id"]),
            methods=tuple(str(value) for value in raw["methods"]),
            pair_looks=tuple(raw["pair_looks"]),
            true_log_effects=tuple(
                _finite_number(value, "true_log_effect")
                for value in raw["true_log_effects"]
            ),
            noise_models=tuple(str(value) for value in raw["noise_models"]),
            simulation_trials=raw["simulation_trials"],
            simulation_seed=raw["simulation_seed"],
        )


def _predictable_plugin_lambdas(
    normalized: Sequence[float], alpha: float
) -> tuple[float, ...]:
    prefix_sum = 0.0
    squared_residual_sum = 0.0
    result = []
    for index, value in enumerate(normalized, start=1):
        previous_variance = (0.25 + squared_residual_sum) / index
        denominator = index * log1p(index) * previous_variance
        result.append(sqrt(2.0 * log(1.0 / alpha) / denominator))
        current_mean = min((0.5 + prefix_sum + value) / (index + 1), 1.0)
        squared_residual_sum += (value - current_mean) ** 2
        prefix_sum += value
    return tuple(result)


def _hedged_log_wealth(
    normalized: Sequence[float], hypothesized_mean: float, alpha: float
) -> float:
    lambdas = _predictable_plugin_lambdas(normalized, alpha * 0.5)
    positive = 0.0
    negative = 0.0
    maximum = -float("inf")
    for value, bet in zip(normalized, lambdas, strict=True):
        positive_cap = float("inf") if hypothesized_mean == 0 else 0.5 / hypothesized_mean
        negative_cap = (
            float("inf") if hypothesized_mean == 1 else 0.5 / (1.0 - hypothesized_mean)
        )
        positive += log1p(min(bet, positive_cap) * (value - hypothesized_mean))
        negative += log1p(-min(bet, negative_cap) * (value - hypothesized_mean))
        maximum = max(maximum, log(0.5) + positive, log(0.5) + negative)
    return maximum


def _find_accepted_anchor(
    accepted: Any, preferred: float
) -> float:
    if accepted(preferred):
        return preferred
    for index in range(257):
        candidate = index / 256.0
        if accepted(candidate):
            return candidate
    raise ValueError("hedged capital confidence set is empty")


def hedged_capital_interval(
    values: Sequence[float], effect_spec: SequentialEffectSpec
) -> tuple[float, float]:
    """Invert the predictable plug-in hedged capital process on bounded effects."""

    lower = effect_spec.lower_log_effect_bound
    upper = effect_spec.upper_log_effect_bound
    if not values:
        return lower, upper
    if any(not isfinite(value) or value < lower or value > upper for value in values):
        raise ValueError("an adjusted effect is outside the predeclared bounds")
    width = upper - lower
    normalized = tuple((value - lower) / width for value in values)
    threshold = log(1.0 / effect_spec.confidence_alpha)

    def accepted(hypothesized_mean: float) -> bool:
        return _hedged_log_wealth(
            normalized, hypothesized_mean, effect_spec.confidence_alpha
        ) <= threshold

    anchor = _find_accepted_anchor(accepted, mean(normalized))
    if accepted(0.0):
        normalized_lower = 0.0
    else:
        left, right = 0.0, anchor
        for _ in range(60):
            middle = (left + right) / 2.0
            if accepted(middle):
                right = middle
            else:
                left = middle
        normalized_lower = right
    if accepted(1.0):
        normalized_upper = 1.0
    else:
        left, right = anchor, 1.0
        for _ in range(60):
            middle = (left + right) / 2.0
            if accepted(middle):
                left = middle
            else:
                right = middle
        normalized_upper = left
    return lower + width * normalized_lower, lower + width * normalized_upper


def _interval(
    method: str,
    values: Sequence[float],
    effect_spec: SequentialEffectSpec,
) -> tuple[float, float]:
    if method == "frozen_spec_method":
        return confidence_interval(values, effect_spec)
    if method == "finite_horizon_hoeffding":
        return confidence_interval(
            values, replace(effect_spec, confidence_method="finite_horizon_hoeffding")
        )
    if method == "hedged_capital_predictable_plugin":
        return hedged_capital_interval(values, effect_spec)
    raise ValueError(f"unsupported sequential design method: {method}")


def _boundary(
    interval: tuple[float, float], target: float
) -> str:
    lower, upper = interval
    if lower > target:
        return "promote"
    if upper < 0:
        return "close_direction"
    if upper < target:
        return "exclude_useful_gain"
    return "no_decision"


def _interval_payload(interval: tuple[float, float]) -> dict[str, float]:
    lower, upper = interval
    return {
        "lower_log_effect": lower,
        "upper_log_effect": upper,
        "lower_improvement_fraction": exp(lower) - 1.0,
        "upper_improvement_fraction": exp(upper) - 1.0,
        "width_log_effect": upper - lower,
    }


def _observed_comparison(
    study_spec: SequentialDesignStudySpec,
    effect_spec: SequentialEffectSpec,
    values: Sequence[float],
) -> list[dict[str, Any]]:
    target = log1p(effect_spec.minimum_improvement_fraction)
    remaining = effect_spec.maximum_pair_count - len(values)
    projected = [*values, *([mean(values)] * max(0, remaining))]
    rows = []
    for method in study_spec.methods:
        current_interval = _interval(method, values, effect_spec)
        projected_interval = _interval(method, projected, effect_spec)
        first_projected = None
        for pair_count in study_spec.pair_looks:
            if pair_count < len(values):
                continue
            hypothetical = [
                *values,
                *([mean(values)] * (pair_count - len(values))),
            ]
            boundary = _boundary(_interval(method, hypothetical, effect_spec), target)
            if boundary != "no_decision":
                first_projected = {
                    "pair_count": pair_count,
                    "boundary": boundary,
                }
                break
        rows.append(
            {
                "method": method,
                "formal_for_source": (
                    method == "frozen_spec_method"
                    and effect_spec.analysis_mode == "prospective"
                ),
                "current_interval": _interval_payload(current_interval),
                "current_boundary": _boundary(current_interval, target),
                "projected_interval": _interval_payload(projected_interval),
                "projected_boundary": _boundary(projected_interval, target),
                "first_projected_boundary": first_projected,
            }
        )
    return rows


def _trial_rng(
    seed: int, noise_model: str, true_effect: float, trial_index: int
) -> random.Random:
    material = f"{seed}:{noise_model}:{true_effect:.17g}:{trial_index}".encode()
    return random.Random(int(hashlib.sha256(material).hexdigest(), 16))


def _simulate_values(
    rng: random.Random,
    noise_model: str,
    true_effect: float,
    pair_count: int,
    lower: float,
    upper: float,
    observed_values: Sequence[float],
) -> list[float]:
    if noise_model == "bounded_endpoints":
        upper_probability = (true_effect - lower) / (upper - lower)
        return [
            upper if rng.random() < upper_probability else lower
            for _ in range(pair_count)
        ]
    observed_mean = mean(observed_values)
    residuals = [abs(value - observed_mean) for value in observed_values]
    if true_effect - max(residuals) < lower or true_effect + max(residuals) > upper:
        raise ValueError(
            "observed symmetric residual model exceeds effect bounds for a scenario"
        )
    return [
        true_effect + rng.choice(residuals) * (1.0 if rng.random() < 0.5 else -1.0)
        for _ in range(pair_count)
    ]


def _simulation_results(
    study_spec: SequentialDesignStudySpec,
    effect_spec: SequentialEffectSpec,
    observed_values: Sequence[float],
) -> list[dict[str, Any]]:
    lower = effect_spec.lower_log_effect_bound
    upper = effect_spec.upper_log_effect_bound
    target = log1p(effect_spec.minimum_improvement_fraction)
    maximum_look = study_spec.pair_looks[-1]
    results = []
    for noise_model in study_spec.noise_models:
        for true_effect in study_spec.true_log_effects:
            generated = [
                _simulate_values(
                    _trial_rng(
                        study_spec.simulation_seed,
                        noise_model,
                        true_effect,
                        trial_index,
                    ),
                    noise_model,
                    true_effect,
                    maximum_look,
                    lower,
                    upper,
                    observed_values,
                )
                for trial_index in range(study_spec.simulation_trials)
            ]
            for method in study_spec.methods:
                coverage_count = 0
                decision_counts = {decision: 0 for decision in _DECISIONS}
                stopping_pairs: list[int] = []
                terminal_widths = []
                for values in generated:
                    covers_all_looks = True
                    decision = "no_decision"
                    stop_pair = maximum_look
                    for pair_count in study_spec.pair_looks:
                        interval = _interval(method, values[:pair_count], effect_spec)
                        covers_all_looks &= interval[0] <= true_effect <= interval[1]
                        if decision == "no_decision":
                            candidate = _boundary(interval, target)
                            if candidate != "no_decision":
                                decision = candidate
                                stop_pair = pair_count
                        if pair_count == maximum_look:
                            terminal_widths.append(interval[1] - interval[0])
                    coverage_count += int(covers_all_looks)
                    decision_counts[decision] += 1
                    stopping_pairs.append(stop_pair)
                results.append(
                    {
                        "method": method,
                        "noise_model": noise_model,
                        "true_log_effect": true_effect,
                        "true_improvement_fraction": exp(true_effect) - 1.0,
                        "trial_count": study_spec.simulation_trials,
                        "simultaneous_coverage_rate": (
                            coverage_count / study_spec.simulation_trials
                        ),
                        "decision_rates": {
                            decision: decision_counts[decision]
                            / study_spec.simulation_trials
                            for decision in _DECISIONS
                        },
                        "mean_stopping_pair": mean(stopping_pairs),
                        "median_stopping_pair": median(stopping_pairs),
                        "mean_terminal_interval_width": mean(terminal_widths),
                    }
                )
    return results


def _audit_payload(
    study_spec: SequentialDesignStudySpec,
    observed: Sequence[Mapping[str, Any]],
    simulation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    projected = min(
        observed,
        key=lambda row: float(row["projected_interval"]["width_log_effect"]),
    )
    minimum_coverage = {
        method: min(
            float(row["simultaneous_coverage_rate"])
            for row in simulation
            if row["method"] == method
        )
        for method in study_spec.methods
    }
    return {
        "analysis_mode": "retrospective_method_comparison",
        "method_count": len(study_spec.methods),
        "noise_model_count": len(study_spec.noise_models),
        "scenario_count": len(study_spec.true_log_effects),
        "simulation_trials_per_scenario": study_spec.simulation_trials,
        "narrowest_projected_interval_method": projected["method"],
        "minimum_simulated_simultaneous_coverage": minimum_coverage,
        "formal_method_change_allowed": False,
    }


def _build_payload(
    study_spec: SequentialDesignStudySpec,
    effect_spec: SequentialEffectSpec,
    values: Sequence[float],
    source_assessment_sha256: str,
    source_status: str,
) -> dict[str, Any]:
    if not values:
        raise ValueError("sequential design study requires observed effects")
    if study_spec.pair_looks[0] < len(values):
        raise ValueError("pair_looks cannot precede the observed pair count")
    if study_spec.pair_looks[-1] != effect_spec.maximum_pair_count:
        raise ValueError("last pair look must equal the frozen maximum pair count")
    if any(
        value < effect_spec.lower_log_effect_bound
        or value > effect_spec.upper_log_effect_bound
        for value in study_spec.true_log_effects
    ):
        raise ValueError("true effect scenario is outside the frozen support")
    observed = _observed_comparison(study_spec, effect_spec, values)
    simulation = _simulation_results(study_spec, effect_spec, values)
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "study_id": study_spec.study_id,
        "status": "diagnostic_only",
        "requirements": study_spec.to_dict(),
        "source": {
            "sequential_effect_assessment_sha256": source_assessment_sha256,
            "sequential_effect_status": source_status,
            "effect_spec": effect_spec.to_dict(),
            "observed_adjusted_log_effects": list(values),
        },
        "observed_method_comparison": observed,
        "simulation_results": simulation,
        "audit": _audit_payload(study_spec, observed, simulation),
    }
    payload["sequential_design_study_sha256"] = canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class SequentialDesignStudy:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _STUDY_KEYS, "sequential design study")
        raw = dict(self.payload)
        digest = str(raw.pop("sequential_design_study_sha256", ""))
        require_digest(digest, "sequential_design_study_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("sequential design study SHA256 does not match content")
        if raw["schema_version"] != "1.0" or raw["status"] != "diagnostic_only":
            raise ValueError("unsupported sequential design study")
        spec = SequentialDesignStudySpec.from_dict(raw["requirements"])
        source = raw["source"]
        if not isinstance(source, Mapping):
            raise ValueError("sequential design source must be an object")
        _expect_exact_keys(
            source,
            {
                "sequential_effect_assessment_sha256",
                "sequential_effect_status",
                "effect_spec",
                "observed_adjusted_log_effects",
            },
            "sequential design source",
        )
        source_digest = str(source["sequential_effect_assessment_sha256"])
        require_digest(source_digest, "sequential_effect_assessment_sha256")
        effect_spec = SequentialEffectSpec.from_dict(source["effect_spec"])
        values_raw = source["observed_adjusted_log_effects"]
        if not isinstance(values_raw, list):
            raise ValueError("observed_adjusted_log_effects must be a list")
        values = [
            _finite_number(value, "observed_adjusted_log_effect")
            for value in values_raw
        ]
        expected = _build_payload(
            spec,
            effect_spec,
            values,
            source_digest,
            str(source["sequential_effect_status"]),
        )
        if expected != dict(self.payload):
            raise ValueError("sequential design study does not match derived results")

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def audit(self) -> dict[str, Any]:
        return dict(self.payload["audit"])

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequentialDesignStudy":
        return cls(dict(raw))


def build_sequential_design_study(
    study_spec: SequentialDesignStudySpec,
    assessment: SequentialEffectAssessment,
) -> SequentialDesignStudy:
    assessment_payload = assessment.to_dict()
    effect_spec = SequentialEffectSpec.from_dict(assessment_payload["requirements"])
    values = [
        float(observation["adjusted_log_effect"])
        for observation in assessment_payload["observations"]
    ]
    payload = _build_payload(
        study_spec,
        effect_spec,
        values,
        str(assessment_payload["sequential_effect_assessment_sha256"]),
        assessment.status,
    )
    return SequentialDesignStudy(payload)
