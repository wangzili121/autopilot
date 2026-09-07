"""Trust-region active learning over a frozen offline policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.candidate_planning import SelectionContext
from inference_autopilot.features import (
    SelectorFeatureRow,
    SelectorFeatureTable,
    deployment_features,
)
from inference_autopilot.policy_selection import (
    PolicyBundle,
    PolicySelectionSpec,
    select_policy,
)
from inference_autopilot.search_space import (
    CompiledCandidate,
    CompiledSearchSpace,
    SettingValue,
    setting_map_to_dict,
)


_PLAN_STATUSES = {"planned", "no_candidate"}
_PLAN_KEYS = {
    "schema_version",
    "producer",
    "acquisition_id",
    "status",
    "source_spec_sha256",
    "policy_bundle_sha256",
    "compiled_space_sha256",
    "feature_table_sha256",
    "selection_context",
    "objective",
    "control",
    "selections",
    "ranked_candidates",
    "audit",
    "policy_experiment_plan_sha256",
}
_ASSESSMENT_STATUSES = {
    "validated_signal",
    "rejected",
    "inconclusive",
    "insufficient_evidence",
}
_ASSESSMENT_KEYS = {
    "schema_version",
    "producer",
    "assessment_id",
    "status",
    "source",
    "requirements",
    "effect",
    "eligible_for_response_model",
    "validated_improvement",
    "next_action",
    "reasons",
    "audit",
    "policy_experiment_assessment_sha256",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _same_value(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(left) == float(right)
    return canonical_json({"value": left}) == canonical_json({"value": right})


@dataclass(frozen=True, slots=True)
class AcquisitionWeights:
    optimistic_improvement: float
    uncertainty_width: float
    normalized_distance: float
    changed_knobs: float

    def __post_init__(self) -> None:
        values = (
            self.optimistic_improvement,
            self.uncertainty_width,
            self.normalized_distance,
            self.changed_knobs,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("acquisition weights must be finite and non-negative")
        if self.optimistic_improvement == 0 and self.uncertainty_width == 0:
            raise ValueError("acquisition requires an information or improvement reward")

    def to_dict(self) -> dict[str, float]:
        return {
            "optimistic_improvement": self.optimistic_improvement,
            "uncertainty_width": self.uncertainty_width,
            "normalized_distance": self.normalized_distance,
            "changed_knobs": self.changed_knobs,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AcquisitionWeights":
        keys = {
            "optimistic_improvement",
            "uncertainty_width",
            "normalized_distance",
            "changed_knobs",
        }
        _expect_exact_keys(raw, keys, "acquisition weights")
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], (int, float))
            for name in keys
        ):
            raise ValueError("acquisition weights must be numeric")
        return cls(**{name: float(raw[name]) for name in keys})


@dataclass(frozen=True, slots=True)
class PolicyAcquisitionSpec:
    acquisition_id: str
    policy_bundle_sha256: str
    candidate_budget: int
    require_policy_eligible: bool
    exclude_exactly_observed: bool
    maximum_changed_knobs: int
    maximum_normalized_distance: float
    minimum_normalized_optimistic_improvement: float
    weights: AcquisitionWeights
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported policy acquisition spec: {self.schema_version}")
        require_id(self.acquisition_id, "acquisition_id")
        require_digest(self.policy_bundle_sha256, "policy_bundle_sha256")
        for name in ("candidate_budget", "maximum_changed_knobs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"policy acquisition {name} must be positive")
        if not 0 <= self.maximum_normalized_distance <= 1:
            raise ValueError("maximum_normalized_distance must be between zero and one")
        if (
            not isfinite(self.minimum_normalized_optimistic_improvement)
            or self.minimum_normalized_optimistic_improvement < 0
        ):
            raise ValueError(
                "minimum_normalized_optimistic_improvement must be non-negative"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "acquisition_id": self.acquisition_id,
            "policy_bundle_sha256": self.policy_bundle_sha256,
            "candidate_budget": self.candidate_budget,
            "require_policy_eligible": self.require_policy_eligible,
            "exclude_exactly_observed": self.exclude_exactly_observed,
            "maximum_changed_knobs": self.maximum_changed_knobs,
            "maximum_normalized_distance": self.maximum_normalized_distance,
            "minimum_normalized_optimistic_improvement": (
                self.minimum_normalized_optimistic_improvement
            ),
            "weights": self.weights.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyAcquisitionSpec":
        keys = {
            "schema_version",
            "acquisition_id",
            "policy_bundle_sha256",
            "candidate_budget",
            "require_policy_eligible",
            "exclude_exactly_observed",
            "maximum_changed_knobs",
            "maximum_normalized_distance",
            "minimum_normalized_optimistic_improvement",
            "weights",
        }
        _expect_exact_keys(raw, keys, "policy acquisition spec")
        if not isinstance(raw["weights"], Mapping):
            raise ValueError("policy acquisition weights must be an object")
        for name in ("require_policy_eligible", "exclude_exactly_observed"):
            if not isinstance(raw[name], bool):
                raise ValueError(f"policy acquisition {name} must be boolean")
        for name in ("candidate_budget", "maximum_changed_knobs"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"policy acquisition {name} must be an integer")
        for name in (
            "maximum_normalized_distance",
            "minimum_normalized_optimistic_improvement",
        ):
            if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
                raise ValueError(f"policy acquisition {name} must be numeric")
        return cls(
            schema_version=str(raw["schema_version"]),
            acquisition_id=str(raw["acquisition_id"]),
            policy_bundle_sha256=str(raw["policy_bundle_sha256"]),
            candidate_budget=raw["candidate_budget"],
            require_policy_eligible=raw["require_policy_eligible"],
            exclude_exactly_observed=raw["exclude_exactly_observed"],
            maximum_changed_knobs=raw["maximum_changed_knobs"],
            maximum_normalized_distance=float(raw["maximum_normalized_distance"]),
            minimum_normalized_optimistic_improvement=float(
                raw["minimum_normalized_optimistic_improvement"]
            ),
            weights=AcquisitionWeights.from_dict(raw["weights"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyExperimentPlan:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _PLAN_KEYS, "policy experiment plan")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_experiment_plan_sha256", ""))
        require_digest(digest, "policy_experiment_plan_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("policy experiment plan SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(f"unsupported policy experiment plan: {raw['schema_version']}")
        if raw["status"] not in _PLAN_STATUSES:
            raise ValueError(f"unsupported policy experiment status: {raw['status']}")
        require_id(str(raw["acquisition_id"]), "acquisition_id")
        for name in (
            "source_spec_sha256",
            "policy_bundle_sha256",
            "compiled_space_sha256",
            "feature_table_sha256",
        ):
            require_digest(str(raw[name]), name)
        SelectionContext.from_dict(raw["selection_context"])
        if not isinstance(raw["control"], Mapping):
            raise ValueError("policy experiment control must be an object")
        for name in ("selections", "ranked_candidates"):
            if not isinstance(raw[name], list) or any(
                not isinstance(item, Mapping) for item in raw[name]
            ):
                raise ValueError(f"policy experiment {name} must be objects")
        selected_ids = [str(item.get("candidate_id", "")) for item in raw["selections"]]
        if len(selected_ids) != len(set(selected_ids)) or any(not item for item in selected_ids):
            raise ValueError("policy experiment selections must be unique candidates")
        if raw["status"] == "planned" and not selected_ids:
            raise ValueError("planned policy experiment requires a selection")
        if raw["status"] == "no_candidate" and selected_ids:
            raise ValueError("no-candidate policy experiment cannot contain selections")
        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        return {
            "acquisition_id": self.payload["acquisition_id"],
            "status": self.status,
            "control_candidate_id": self.payload["control"]["candidate_id"],
            "selected_candidate_ids": [
                item["candidate_id"] for item in self.payload["selections"]
            ],
            **dict(self.payload["audit"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyExperimentPlan":
        return cls(raw)


@dataclass(frozen=True, slots=True)
class PolicyExperimentAssessmentSpec:
    assessment_id: str
    policy_experiment_plan_sha256: str
    calibration_plan_sha256: str
    candidate_configuration_id: str
    minimum_complete_pairs: int
    minimum_median_improvement_fraction: float
    require_effect_outside_replay_noise: bool
    require_quality_constraints: bool
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported policy experiment assessment spec: {self.schema_version}"
            )
        require_id(self.assessment_id, "policy experiment assessment id")
        require_digest(
            self.policy_experiment_plan_sha256,
            "policy_experiment_plan_sha256",
        )
        require_digest(self.calibration_plan_sha256, "calibration_plan_sha256")
        require_id(self.candidate_configuration_id, "candidate_configuration_id")
        if (
            isinstance(self.minimum_complete_pairs, bool)
            or not isinstance(self.minimum_complete_pairs, int)
            or self.minimum_complete_pairs <= 0
        ):
            raise ValueError("minimum_complete_pairs must be a positive integer")
        if not isfinite(self.minimum_median_improvement_fraction):
            raise ValueError("minimum_median_improvement_fraction must be finite")
        for name in (
            "require_effect_outside_replay_noise",
            "require_quality_constraints",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "policy_experiment_plan_sha256": (
                self.policy_experiment_plan_sha256
            ),
            "calibration_plan_sha256": self.calibration_plan_sha256,
            "candidate_configuration_id": self.candidate_configuration_id,
            "minimum_complete_pairs": self.minimum_complete_pairs,
            "minimum_median_improvement_fraction": (
                self.minimum_median_improvement_fraction
            ),
            "require_effect_outside_replay_noise": (
                self.require_effect_outside_replay_noise
            ),
            "require_quality_constraints": self.require_quality_constraints,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyExperimentAssessmentSpec":
        keys = {
            "schema_version",
            "assessment_id",
            "policy_experiment_plan_sha256",
            "calibration_plan_sha256",
            "candidate_configuration_id",
            "minimum_complete_pairs",
            "minimum_median_improvement_fraction",
            "require_effect_outside_replay_noise",
            "require_quality_constraints",
        }
        _expect_exact_keys(raw, keys, "policy experiment assessment spec")
        if isinstance(raw["minimum_complete_pairs"], bool) or not isinstance(
            raw["minimum_complete_pairs"], int
        ):
            raise ValueError("minimum_complete_pairs must be an integer")
        improvement = raw["minimum_median_improvement_fraction"]
        if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
            raise ValueError("minimum_median_improvement_fraction must be numeric")
        return cls(
            schema_version=str(raw["schema_version"]),
            assessment_id=str(raw["assessment_id"]),
            policy_experiment_plan_sha256=str(
                raw["policy_experiment_plan_sha256"]
            ),
            calibration_plan_sha256=str(raw["calibration_plan_sha256"]),
            candidate_configuration_id=str(raw["candidate_configuration_id"]),
            minimum_complete_pairs=raw["minimum_complete_pairs"],
            minimum_median_improvement_fraction=float(improvement),
            require_effect_outside_replay_noise=raw[
                "require_effect_outside_replay_noise"
            ],
            require_quality_constraints=raw["require_quality_constraints"],
        )


def _evaluate_policy_experiment_gate(
    requirements: PolicyExperimentAssessmentSpec,
    effect: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
) -> tuple[list[str], str, bool, bool, str]:
    reasons = []
    insufficient = False
    if audit["formal_complete"] is not True:
        reasons.append("calibration_not_formally_complete")
        insufficient = True
    if audit["calibration_issue_count"]:
        reasons.append("calibration_has_issues")
        insufficient = True
    if effect is None:
        reasons.append("selected_effect_missing")
        insufficient = True
    else:
        if effect["formal_group"] is not True:
            reasons.append("selected_effect_not_formal")
            insufficient = True
        pairs = effect["complete_pair_count"]
        if isinstance(pairs, bool) or not isinstance(pairs, int):
            reasons.append("complete_pair_count_missing")
            insufficient = True
        elif pairs < requirements.minimum_complete_pairs:
            reasons.append("minimum_complete_pairs_not_met")
            insufficient = True
        improvement = effect["median_directional_relative_improvement"]
        if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
            reasons.append("median_improvement_missing")
            insufficient = True
        elif improvement < requirements.minimum_median_improvement_fraction:
            reasons.append("minimum_acquisition_improvement_not_met")
        if (
            requirements.require_effect_outside_replay_noise
            and effect["effect_outside_replay_noise"] is not True
        ):
            reasons.append("effect_not_outside_replay_noise")
        if (
            requirements.require_quality_constraints
            and effect["quality_constraints_satisfied"] is not True
        ):
            reasons.append("quality_constraints_not_satisfied")
    reasons = sorted(set(reasons))
    inconclusive_reasons = {
        "effect_not_outside_replay_noise",
        "minimum_acquisition_improvement_not_met",
    }
    if insufficient:
        status = "insufficient_evidence"
    elif not reasons:
        status = "validated_signal"
    elif (
        "effect_not_outside_replay_noise" in reasons
        and set(reasons).issubset(inconclusive_reasons)
    ):
        status = "inconclusive"
    else:
        status = "rejected"
    eligible = (
        audit["formal_complete"] is True
        and audit["calibration_issue_count"] == 0
        and effect is not None
        and effect["formal_group"] is True
        and isinstance(effect["complete_pair_count"], int)
        and not isinstance(effect["complete_pair_count"], bool)
        and effect["complete_pair_count"] >= requirements.minimum_complete_pairs
    )
    validated = status == "validated_signal"
    next_action = {
        "validated_signal": "refit_response_model_then_run_independent_holdout",
        "rejected": "retain_control_and_refit_response_model",
        "inconclusive": "collect_additional_pairs_before_expanding_search",
        "insufficient_evidence": "do_not_fit_retry_formal_acquisition",
    }[status]
    return reasons, status, eligible, validated, next_action


@dataclass(frozen=True, slots=True)
class PolicyExperimentAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _ASSESSMENT_KEYS, "policy experiment assessment")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_experiment_assessment_sha256", ""))
        require_digest(digest, "policy_experiment_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError(
                "policy experiment assessment SHA256 does not match its content"
            )
        if raw["schema_version"] != "1.0" or raw["status"] not in _ASSESSMENT_STATUSES:
            raise ValueError("unsupported policy experiment assessment")
        requirements = PolicyExperimentAssessmentSpec.from_dict(raw["requirements"])
        source = raw["source"]
        audit = raw["audit"]
        if not isinstance(source, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("policy experiment assessment source and audit must be objects")
        source_keys = {
            "assessment_spec_sha256",
            "policy_experiment_plan_sha256",
            "calibration_plan_sha256",
            "calibration_assessment_sha256",
        }
        _expect_exact_keys(source, source_keys, "policy experiment assessment source")
        for name in source_keys:
            require_digest(str(source[name]), name)
        if raw["assessment_id"] != requirements.assessment_id:
            raise ValueError("assessment id does not match frozen requirements")
        if source["assessment_spec_sha256"] != requirements.sha256:
            raise ValueError("assessment source does not match frozen requirements")
        if (
            source["policy_experiment_plan_sha256"]
            != requirements.policy_experiment_plan_sha256
            or source["calibration_plan_sha256"]
            != requirements.calibration_plan_sha256
        ):
            raise ValueError("assessment source plan bindings do not match requirements")
        _expect_exact_keys(
            audit,
            {"calibration_issue_count", "selected_effect_count", "formal_complete"},
            "policy experiment assessment audit",
        )
        if not isinstance(audit["formal_complete"], bool) or any(
            isinstance(audit[name], bool)
            or not isinstance(audit[name], int)
            or audit[name] < 0
            for name in ("calibration_issue_count", "selected_effect_count")
        ):
            raise ValueError("policy experiment assessment audit values are invalid")
        if audit["selected_effect_count"] > 1:
            raise ValueError("policy experiment assessment has duplicate effects")
        effect = raw["effect"]
        if effect is not None:
            if not isinstance(effect, Mapping):
                raise ValueError("policy experiment assessment effect must be an object")
            effect_keys = {
                "candidate_configuration_id",
                "primary_metric",
                "direction",
                "formal_group",
                "complete_pair_count",
                "median_directional_relative_improvement",
                "candidate_over_baseline_geomean_ratio",
                "effect_outside_replay_noise",
                "quality_constraints_satisfied",
            }
            _expect_exact_keys(effect, effect_keys, "policy experiment assessment effect")
            if effect["candidate_configuration_id"] != requirements.candidate_configuration_id:
                raise ValueError("assessment effect candidate does not match requirements")
        reasons, status, eligible, validated, next_action = (
            _evaluate_policy_experiment_gate(requirements, effect, audit)
        )
        if raw["reasons"] != reasons or raw["status"] != status:
            raise ValueError("assessment conclusion does not match frozen gate and effect")
        if raw["eligible_for_response_model"] is not eligible:
            raise ValueError("response-model eligibility does not match formal evidence")
        if raw["validated_improvement"] is not validated:
            raise ValueError("validated improvement does not match assessment status")
        if raw["next_action"] != next_action:
            raise ValueError("policy experiment next action does not match status")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        return {
            "assessment_id": self.payload["assessment_id"],
            "status": self.status,
            "eligible_for_response_model": self.payload["eligible_for_response_model"],
            "validated_improvement": self.payload["validated_improvement"],
            "next_action": self.payload["next_action"],
            "reasons": list(self.payload["reasons"]),
            **dict(self.payload["audit"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyExperimentAssessment":
        return cls(raw)


def _matches_deployment(
    candidate: CompiledCandidate, row: SelectorFeatureRow
) -> bool:
    for name, expected in deployment_features(candidate.deployment_settings).items():
        actual = row.static_features.get(name)
        if actual is None or not _same_value(actual, expected):
            return False
    return True


def _changed_knobs(
    control: CompiledCandidate, candidate: CompiledCandidate
) -> list[dict[str, SettingValue | None]]:
    changes = []
    for name in sorted(set(control.knob_values) | set(candidate.knob_values)):
        left = control.knob_values.get(name)
        right = candidate.knob_values.get(name)
        if left is None or right is None or not _same_value(left, right):
            changes.append({"name": name, "control": left, "candidate": right})
    return changes


def _reachable_capture_sizes(
    candidate: CompiledCandidate, knob_name: str, value: Any
) -> tuple[int, ...] | None:
    if not knob_name.endswith(".capture_sizes") or not isinstance(value, tuple):
        return None
    engine = knob_name.removesuffix(".capture_sizes")
    capacity = candidate.knob_values.get(f"{engine}.max_num_seqs")
    if capacity is None:
        capacity = candidate.deployment_settings.get(f"{engine}_max_num_seqs")
    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
        return None
    return tuple(size for size in value if size <= int(capacity))


def _inactive_changed_knobs(
    control: CompiledCandidate,
    candidate: CompiledCandidate,
    changes: Sequence[Mapping[str, Any]],
) -> list[str]:
    inactive = []
    for change in changes:
        name = str(change["name"])
        control_sizes = _reachable_capture_sizes(control, name, change["control"])
        candidate_sizes = _reachable_capture_sizes(candidate, name, change["candidate"])
        if (
            control_sizes is not None
            and candidate_sizes is not None
            and control_sizes == candidate_sizes
        ):
            inactive.append(name)
    return inactive


def _numeric_knob_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _directional_regression_blockers(
    control: CompiledCandidate,
    candidate: CompiledCandidate,
    candidates: Sequence[CompiledCandidate],
    policy_evaluations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    changes = _changed_knobs(control, candidate)
    if len(changes) != 1:
        return []
    change = changes[0]
    control_value = _numeric_knob_value(change["control"])
    candidate_value = _numeric_knob_value(change["candidate"])
    if control_value is None or candidate_value is None:
        return []
    candidate_delta = candidate_value - control_value
    if candidate_delta == 0:
        return []

    blockers = []
    for support in candidates:
        support_changes = _changed_knobs(control, support)
        if len(support_changes) != 1 or support_changes[0]["name"] != change["name"]:
            continue
        support_value = _numeric_knob_value(support_changes[0]["candidate"])
        if support_value is None:
            continue
        support_delta = support_value - control_value
        if (
            support_delta == 0
            or support_delta * candidate_delta <= 0
            or abs(support_delta) >= abs(candidate_delta)
        ):
            continue
        evaluation = policy_evaluations.get(support.candidate_id)
        effect = evaluation.get("paired_objective_effect") if evaluation else None
        if not isinstance(effect, Mapping):
            continue
        upper = effect.get("upper_directional_relative_improvement")
        if isinstance(upper, bool) or not isinstance(upper, (int, float)) or upper >= 0:
            continue
        blockers.append(
            {
                "candidate_id": support.candidate_id,
                "knob_name": change["name"],
                "control_value": control_value,
                "probe_value": support_value,
                "candidate_value": candidate_value,
                "upper_directional_relative_improvement": float(upper),
            }
        )
    blockers.sort(
        key=lambda item: (
            abs(item["probe_value"] - control_value),
            item["candidate_id"],
        )
    )
    return blockers


def _optimistic_improvement(
    direction: str,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> float:
    if direction == "maximize":
        return float(candidate["upper"]) - float(control["lower"])
    return float(control["upper"]) - float(candidate["lower"])


def _candidate_evaluation(
    candidate: CompiledCandidate,
    policy_evaluation: Mapping[str, Any] | None,
    control: CompiledCandidate,
    control_evaluation: Mapping[str, Any],
    response_rows: Sequence[SelectorFeatureRow],
    selection_spec: PolicySelectionSpec,
    acquisition_spec: PolicyAcquisitionSpec,
    target_scale: float,
    directional_regression_blockers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = sorted(
        row.row_id for row in response_rows if _matches_deployment(candidate, row)
    )
    changes = _changed_knobs(control, candidate)
    inactive_changes = _inactive_changed_knobs(control, candidate, changes)
    rejections = []
    if candidate.candidate_id == control.candidate_id:
        rejections.append("control_candidate")
    if policy_evaluation is None:
        rejections.append("missing_policy_evaluation")
        return {
            "candidate_id": candidate.candidate_id,
            "deployment_settings": setting_map_to_dict(
                candidate.deployment_settings
            ),
            "knob_values": setting_map_to_dict(candidate.knob_values),
            "changed_knobs": changes,
            "inactive_changed_knobs": inactive_changes,
            "directional_regression_blockers": list(
                directional_regression_blockers
            ),
            "observed_response_row_ids": observed,
            "policy_eligible": False,
            "policy_rejection_reasons": ["missing_policy_evaluation"],
            "nearest_normalized_distance": 1.0,
            "failure_probability": 1.0,
            "objective_estimate": None,
            "normalized_optimistic_improvement": 0.0,
            "normalized_uncertainty_width": 0.0,
            "acquisition_score": None,
            "acquisition_eligible": False,
            "acquisition_rejection_reasons": rejections,
            "rationale_tags": [],
        }
    if acquisition_spec.require_policy_eligible and not policy_evaluation["eligible"]:
        rejections.append("policy_ineligible")
    if acquisition_spec.exclude_exactly_observed and observed:
        rejections.append("already_observed")
    if len(changes) > acquisition_spec.maximum_changed_knobs:
        rejections.append("outside_knob_trust_region")
    if inactive_changes:
        rejections.append("inactive_execution_domain_change")
    if directional_regression_blockers:
        rejections.append("beyond_regressive_directional_probe")
    distance = float(policy_evaluation["nearest_normalized_distance"])
    if distance > acquisition_spec.maximum_normalized_distance:
        rejections.append("outside_model_trust_region")

    target = selection_spec.objective.target
    estimate = policy_evaluation["target_estimates"][target]
    control_estimate = control_evaluation["target_estimates"][target]
    optimistic = _optimistic_improvement(
        selection_spec.objective.direction, estimate, control_estimate
    )
    normalized_optimistic = optimistic / target_scale
    normalized_uncertainty = (
        float(estimate["upper"]) - float(estimate["lower"])
    ) / target_scale
    if (
        normalized_optimistic
        < acquisition_spec.minimum_normalized_optimistic_improvement
    ):
        rejections.append("insufficient_optimistic_improvement")
    weights = acquisition_spec.weights
    score = (
        weights.optimistic_improvement * normalized_optimistic
        + weights.uncertainty_width * normalized_uncertainty
        - weights.normalized_distance * distance
        - weights.changed_knobs * len(changes)
    )
    tags = []
    if len(changes) == 1:
        tags.append("one_factor_from_control")
    if not observed:
        tags.append("unmeasured_exact_configuration")
    if policy_evaluation["eligible"]:
        tags.append("policy_slo_feasible")
    if normalized_optimistic > 0:
        tags.append("positive_optimistic_improvement")
    if normalized_uncertainty > 0:
        tags.append("decision_uncertainty")
    return {
        "candidate_id": candidate.candidate_id,
        "deployment_settings": setting_map_to_dict(candidate.deployment_settings),
        "knob_values": setting_map_to_dict(candidate.knob_values),
        "changed_knobs": changes,
        "inactive_changed_knobs": inactive_changes,
        "directional_regression_blockers": list(directional_regression_blockers),
        "observed_response_row_ids": observed,
        "policy_eligible": bool(policy_evaluation["eligible"]),
        "policy_rejection_reasons": list(policy_evaluation["rejection_reasons"]),
        "nearest_normalized_distance": distance,
        "failure_probability": float(policy_evaluation["failure_probability"]),
        "objective_estimate": dict(estimate),
        "normalized_optimistic_improvement": normalized_optimistic,
        "normalized_uncertainty_width": normalized_uncertainty,
        "acquisition_score": score,
        "acquisition_eligible": not rejections,
        "acquisition_rejection_reasons": sorted(rejections),
        "rationale_tags": sorted(tags),
    }


def plan_policy_experiments(
    spec: PolicyAcquisitionSpec,
    policy: PolicyBundle,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable,
) -> PolicyExperimentPlan:
    """Choose informative, low-change experiments around the guarded fallback."""

    policy_payload = policy.to_dict()
    policy_digest = policy_payload["policy_bundle_sha256"]
    if spec.policy_bundle_sha256 != policy_digest:
        raise ValueError("policy acquisition spec does not match policy bundle")
    if policy.status != "selected":
        raise ValueError("policy acquisition requires a selected policy")
    compiled_digest = compiled.to_dict()["compiled_space_sha256"]
    feature_digest = canonical_sha256(features.to_dict())
    if policy_payload["compiled_space_sha256"] != compiled_digest:
        raise ValueError("policy acquisition compiled space does not match policy")
    if policy_payload["feature_table_sha256"] != feature_digest:
        raise ValueError("policy acquisition feature table does not match policy")

    selection_spec = PolicySelectionSpec.from_dict(policy_payload["selection_spec"])
    candidates = {
        candidate.candidate_id: candidate
        for candidate in compiled.candidates
        if candidate.semantic_cohort_id
        == selection_spec.selection_context.semantic_cohort_id
    }
    control_id = str(policy_payload["fallback"]["candidate_id"])
    control = candidates.get(control_id)
    if control is None:
        raise ValueError("policy fallback is absent from compiled space")
    reported_candidate_count = len(policy_payload["ranked_candidates"])
    full_selection_spec = replace(
        selection_spec,
        model=replace(
            selection_spec.model,
            ranked_candidate_limit=max(len(candidates), reported_candidate_count),
        ),
    )
    surrogate_payload = select_policy(
        full_selection_spec, compiled, features
    ).to_dict()
    if (
        surrogate_payload["status"] != policy_payload["status"]
        or surrogate_payload["fallback"]["candidate_id"] != control_id
        or surrogate_payload["selected"]["candidate_id"]
        != policy_payload["selected"]["candidate_id"]
    ):
        raise ValueError("full surrogate evaluation changed the frozen policy decision")
    policy_evaluations = {
        str(item["candidate_id"]): item
        for item in surrogate_payload["ranked_candidates"]
    }
    control_evaluation = policy_evaluations.get(control_id)
    if control_evaluation is None:
        raise ValueError("policy fallback has no response-model evaluation")
    response_ids = set(policy_payload["training"]["response_row_ids"])
    rows_by_id = {row.row_id: row for row in features.rows}
    if not response_ids.issubset(rows_by_id):
        raise ValueError("policy response rows are absent from feature table")
    response_rows = tuple(rows_by_id[row_id] for row_id in sorted(response_ids))
    target = selection_spec.objective.target
    target_scale = float(policy_payload["response_model"]["target_scales"][target])
    if not isfinite(target_scale) or target_scale <= 0:
        raise ValueError("policy objective target scale must be positive")

    evaluations = [
        _candidate_evaluation(
            candidate,
            policy_evaluations.get(candidate.candidate_id),
            control,
            control_evaluation,
            response_rows,
            selection_spec,
            spec,
            target_scale,
            _directional_regression_blockers(
                control,
                candidate,
                tuple(candidates.values()),
                policy_evaluations,
            ),
        )
        for candidate in candidates.values()
    ]
    evaluations.sort(
        key=lambda item: (
            not item["acquisition_eligible"],
            -(
                float(item["acquisition_score"])
                if item["acquisition_score"] is not None
                else float("-inf")
            ),
            item["candidate_id"],
        )
    )
    eligible = [item for item in evaluations if item["acquisition_eligible"]]
    selections = eligible[: spec.candidate_budget]
    rejections = Counter(
        reason
        for item in evaluations
        for reason in item["acquisition_rejection_reasons"]
    )
    status = "planned" if selections else "no_candidate"
    control_payload = {
        "candidate_id": control.candidate_id,
        "deployment_settings": setting_map_to_dict(control.deployment_settings),
        "knob_values": setting_map_to_dict(control.knob_values),
        "observed_response_row_ids": sorted(
            row.row_id for row in response_rows if _matches_deployment(control, row)
        ),
    }
    audit = {
        "candidate_budget": spec.candidate_budget,
        "compiled_candidate_count": len(candidates),
        "policy_reported_candidate_count": reported_candidate_count,
        "surrogate_evaluated_candidate_count": len(policy_evaluations),
        "evaluated_candidate_count": len(evaluations),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_count": len(selections),
        "rejections_by_reason": dict(sorted(rejections.items())),
    }
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "acquisition_id": spec.acquisition_id,
        "status": status,
        "source_spec_sha256": spec.sha256,
        "policy_bundle_sha256": policy_digest,
        "compiled_space_sha256": compiled_digest,
        "feature_table_sha256": feature_digest,
        "selection_context": selection_spec.selection_context.to_dict(),
        "objective": {
            "target": target,
            "direction": selection_spec.objective.direction,
            "target_scale": target_scale,
        },
        "control": control_payload,
        "selections": selections,
        "ranked_candidates": evaluations,
        "audit": audit,
    }
    return PolicyExperimentPlan(
        {**payload, "policy_experiment_plan_sha256": canonical_sha256(payload)}
    )


def assess_policy_experiment(
    spec: PolicyExperimentAssessmentSpec,
    plan: PolicyExperimentPlan,
    calibration_assessment: Mapping[str, Any],
) -> PolicyExperimentAssessment:
    """Apply a frozen gate without discarding formal neutral or negative results."""

    if plan.status != "planned":
        raise ValueError("policy experiment assessment requires a planned acquisition")
    plan_digest = str(plan.payload["policy_experiment_plan_sha256"])
    if spec.policy_experiment_plan_sha256 != plan_digest:
        raise ValueError("policy experiment assessment spec is bound to another plan")
    selected_ids = {str(item["candidate_id"]) for item in plan.payload["selections"]}
    if spec.candidate_configuration_id not in selected_ids:
        raise ValueError("assessment candidate was not selected by the acquisition plan")
    if calibration_assessment.get("plan_sha256") != spec.calibration_plan_sha256:
        raise ValueError("calibration assessment is bound to another calibration plan")

    effects = calibration_assessment.get("effects")
    if not isinstance(effects, list):
        raise ValueError("calibration assessment effects must be an array")
    matches = [
        item
        for item in effects
        if isinstance(item, Mapping)
        and item.get("candidate_configuration_id")
        == spec.candidate_configuration_id
        and item.get("replay_control") is False
    ]
    if len(matches) > 1:
        raise ValueError("calibration assessment has duplicate acquisition effects")
    source_effect = matches[0] if matches else None
    effect = None
    if source_effect is not None:
        effect = {
            "candidate_configuration_id": spec.candidate_configuration_id,
            "primary_metric": source_effect.get("primary_metric"),
            "direction": source_effect.get("direction"),
            "formal_group": source_effect.get("formal_group"),
            "complete_pair_count": source_effect.get("complete_pair_count"),
            "median_directional_relative_improvement": source_effect.get(
                "median_directional_relative_improvement"
            ),
            "candidate_over_baseline_geomean_ratio": source_effect.get(
                "candidate_over_baseline_geomean_ratio"
            ),
            "effect_outside_replay_noise": source_effect.get(
                "effect_outside_replay_noise"
            ),
            "quality_constraints_satisfied": source_effect.get(
                "quality_constraints_satisfied"
            ),
        }
    issues = calibration_assessment.get("issues")
    audit = {
        "calibration_issue_count": len(issues) if isinstance(issues, list) else 1,
        "selected_effect_count": len(matches),
        "formal_complete": calibration_assessment.get("formal_complete") is True,
    }
    reasons, status, eligible, validated, next_action = (
        _evaluate_policy_experiment_gate(spec, effect, audit)
    )
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "assessment_id": spec.assessment_id,
        "status": status,
        "source": {
            "assessment_spec_sha256": spec.sha256,
            "policy_experiment_plan_sha256": plan_digest,
            "calibration_plan_sha256": spec.calibration_plan_sha256,
            "calibration_assessment_sha256": canonical_sha256(
                calibration_assessment
            ),
        },
        "requirements": spec.to_dict(),
        "effect": effect,
        "eligible_for_response_model": eligible,
        "validated_improvement": validated,
        "next_action": next_action,
        "reasons": reasons,
        "audit": audit,
    }
    return PolicyExperimentAssessment(
        {
            **payload,
            "policy_experiment_assessment_sha256": canonical_sha256(payload),
        }
    )


def configurations_from_policy_experiment_plan(
    plan: PolicyExperimentPlan,
) -> tuple[ConfigurationSpec, ...]:
    return tuple(
        ConfigurationSpec(
            configuration_id=str(item["candidate_id"]),
            settings=item["deployment_settings"],
            description=(
                f"Active-learning selection from {plan.payload['acquisition_id']}; "
                f"score={float(item['acquisition_score']):.6g}"
            ),
        )
        for item in plan.payload["selections"]
    )


def calibration_spec_from_policy_experiment_plan(
    spec: CalibrationSpec,
    plan: PolicyExperimentPlan,
    *,
    include_replay_control: bool = False,
    campaign_id: str | None = None,
    pair_seeds: Sequence[int] | None = None,
) -> CalibrationSpec:
    """Replace manual candidates with active-learning selections."""

    context = SelectionContext.from_dict(plan.payload["selection_context"])
    if context.algorithm_id != spec.semantic_contract.algorithm_id:
        raise ValueError("policy experiment algorithm does not match calibration spec")
    if context.graph_sha256 != spec.semantic_contract.graph_sha256:
        raise ValueError("policy experiment graph does not match calibration spec")
    if context.workload_id != spec.workload_contract.workload_id:
        raise ValueError("policy experiment workload does not match calibration spec")
    if context.environment_id != spec.environment_contract.environment_id:
        raise ValueError("policy experiment environment does not match calibration spec")
    control_settings = plan.payload["control"]["deployment_settings"]
    if any(
        name not in spec.baseline.settings
        or not _same_value(value, spec.baseline.settings[name])
        for name, value in control_settings.items()
    ):
        raise ValueError("policy experiment control does not match calibration baseline")
    if plan.status != "planned":
        raise ValueError("policy experiment plan has no selected candidates")
    configurations = []
    if include_replay_control:
        configurations.append(
            ConfigurationSpec(
                configuration_id=f"{plan.payload['acquisition_id']}-replay-control",
                settings=dict(spec.baseline.settings),
                description="Manifest-identical replay control for local noise",
            )
        )
    for item in plan.payload["selections"]:
        settings = dict(spec.baseline.settings)
        settings.update(item["deployment_settings"])
        configurations.append(
            ConfigurationSpec(
                configuration_id=str(item["candidate_id"]),
                settings=settings,
                description=(
                    f"Active-learning selection from {plan.payload['acquisition_id']}; "
                    f"score={float(item['acquisition_score']):.6g}"
                ),
            )
        )
    protocol = (
        replace(spec.protocol, pair_seeds=tuple(pair_seeds))
        if pair_seeds is not None
        else spec.protocol
    )
    return replace(
        spec,
        campaign_id=campaign_id or spec.campaign_id,
        protocol=protocol,
        candidates=tuple(configurations),
    )
