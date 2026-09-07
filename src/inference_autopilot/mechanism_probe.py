"""Graph- and telemetry-guided probes for sparse workload partitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from math import isfinite
from statistics import median
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
from inference_autopilot.ir import InferenceGraph
from inference_autopilot.search_space import (
    CompiledCandidate,
    CompiledSearchSpace,
    SettingValue,
    normalize_setting_map,
    setting_map_to_dict,
)


_PLAN_STATUSES = {"planned", "no_candidate", "insufficient_evidence"}
_PLAN_KEYS = {
    "schema_version",
    "producer",
    "probe_id",
    "status",
    "source_spec_sha256",
    "graph_sha256",
    "compiled_space_sha256",
    "feature_table_sha256",
    "selection_context",
    "control",
    "diagnostics",
    "selections",
    "ranked_candidates",
    "audit",
    "mechanism_probe_plan_sha256",
}
_ASSESSMENT_STATUSES = {"validated_signal", "rejected", "insufficient_evidence"}
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
    "mechanism_probe_assessment_sha256",
}


def _expect_exact_keys(
    raw: Mapping[str, Any], expected: set[str], context: str
) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )


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
class MechanismThresholds:
    compute_dominance_fraction: float
    capacity_utilization_fraction: float
    queue_to_running_ratio: float
    kv_pressure_fraction: float
    minimum_mechanism_score: float

    def __post_init__(self) -> None:
        fractions = (
            self.compute_dominance_fraction,
            self.capacity_utilization_fraction,
            self.kv_pressure_fraction,
        )
        if any(not isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError(
                "mechanism fraction thresholds must be between zero and one"
            )
        if not isfinite(self.queue_to_running_ratio) or self.queue_to_running_ratio < 0:
            raise ValueError("queue_to_running_ratio must be non-negative")
        if (
            not isfinite(self.minimum_mechanism_score)
            or self.minimum_mechanism_score < 0
        ):
            raise ValueError("minimum_mechanism_score must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "compute_dominance_fraction": self.compute_dominance_fraction,
            "capacity_utilization_fraction": self.capacity_utilization_fraction,
            "queue_to_running_ratio": self.queue_to_running_ratio,
            "kv_pressure_fraction": self.kv_pressure_fraction,
            "minimum_mechanism_score": self.minimum_mechanism_score,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismThresholds":
        keys = {
            "compute_dominance_fraction",
            "capacity_utilization_fraction",
            "queue_to_running_ratio",
            "kv_pressure_fraction",
            "minimum_mechanism_score",
        }
        _expect_exact_keys(raw, keys, "mechanism thresholds")
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], (int, float))
            for name in keys
        ):
            raise ValueError("mechanism thresholds must be numeric")
        return cls(**{name: float(raw[name]) for name in keys})


@dataclass(frozen=True, slots=True)
class MechanismProbeSpec:
    probe_id: str
    compiled_space_sha256: str
    selection_context: SelectionContext
    control_settings: Mapping[str, SettingValue]
    candidate_budget: int
    maximum_changed_knobs: int
    minimum_formal_control_rows: int
    exclude_exactly_observed: bool
    thresholds: MechanismThresholds
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported mechanism probe spec: {self.schema_version}")
        require_id(self.probe_id, "mechanism probe id")
        require_digest(self.compiled_space_sha256, "compiled_space_sha256")
        for name in (
            "candidate_budget",
            "maximum_changed_knobs",
            "minimum_formal_control_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"mechanism probe {name} must be a positive integer")
        if not isinstance(self.exclude_exactly_observed, bool):
            raise ValueError("exclude_exactly_observed must be boolean")
        object.__setattr__(
            self,
            "control_settings",
            normalize_setting_map(
                self.control_settings, "mechanism probe control settings"
            ),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probe_id": self.probe_id,
            "compiled_space_sha256": self.compiled_space_sha256,
            "selection_context": self.selection_context.to_dict(),
            "control_settings": setting_map_to_dict(self.control_settings),
            "candidate_budget": self.candidate_budget,
            "maximum_changed_knobs": self.maximum_changed_knobs,
            "minimum_formal_control_rows": self.minimum_formal_control_rows,
            "exclude_exactly_observed": self.exclude_exactly_observed,
            "thresholds": self.thresholds.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismProbeSpec":
        keys = {
            "schema_version",
            "probe_id",
            "compiled_space_sha256",
            "selection_context",
            "control_settings",
            "candidate_budget",
            "maximum_changed_knobs",
            "minimum_formal_control_rows",
            "exclude_exactly_observed",
            "thresholds",
        }
        _expect_exact_keys(raw, keys, "mechanism probe spec")
        object_keys = {"selection_context", "control_settings", "thresholds"}
        if any(not isinstance(raw[name], Mapping) for name in object_keys):
            raise ValueError("mechanism probe nested fields must be objects")
        for name in (
            "candidate_budget",
            "maximum_changed_knobs",
            "minimum_formal_control_rows",
        ):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"mechanism probe {name} must be an integer")
        return cls(
            schema_version=str(raw["schema_version"]),
            probe_id=str(raw["probe_id"]),
            compiled_space_sha256=str(raw["compiled_space_sha256"]),
            selection_context=SelectionContext.from_dict(raw["selection_context"]),
            control_settings=dict(raw["control_settings"]),
            candidate_budget=raw["candidate_budget"],
            maximum_changed_knobs=raw["maximum_changed_knobs"],
            minimum_formal_control_rows=raw["minimum_formal_control_rows"],
            exclude_exactly_observed=raw["exclude_exactly_observed"],
            thresholds=MechanismThresholds.from_dict(raw["thresholds"]),
        )


@dataclass(frozen=True, slots=True)
class MechanismProbePlan:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _PLAN_KEYS, "mechanism probe plan")
        raw = dict(self.payload)
        digest = str(raw.pop("mechanism_probe_plan_sha256", ""))
        require_digest(digest, "mechanism_probe_plan_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("mechanism probe plan SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported mechanism probe plan: {raw['schema_version']}"
            )
        if raw["status"] not in _PLAN_STATUSES:
            raise ValueError(f"unsupported mechanism probe status: {raw['status']}")
        require_id(str(raw["probe_id"]), "mechanism probe id")
        for name in (
            "source_spec_sha256",
            "graph_sha256",
            "compiled_space_sha256",
            "feature_table_sha256",
        ):
            require_digest(str(raw[name]), name)
        SelectionContext.from_dict(raw["selection_context"])
        for name in ("control", "diagnostics", "audit"):
            if not isinstance(raw[name], Mapping):
                raise ValueError(f"mechanism probe {name} must be an object")
        for name in ("selections", "ranked_candidates"):
            if not isinstance(raw[name], list) or any(
                not isinstance(item, Mapping) for item in raw[name]
            ):
                raise ValueError(f"mechanism probe {name} must contain objects")
        selected_ids = [str(item.get("candidate_id", "")) for item in raw["selections"]]
        if any(not value for value in selected_ids) or len(selected_ids) != len(
            set(selected_ids)
        ):
            raise ValueError("mechanism probe selections must be named and unique")
        if raw["status"] == "planned" and not selected_ids:
            raise ValueError("planned mechanism probe requires a selection")
        if raw["status"] != "planned" and selected_ids:
            raise ValueError("blocked mechanism probe cannot contain selections")
        if raw["status"] == "insufficient_evidence" and raw["ranked_candidates"]:
            raise ValueError("insufficient evidence cannot rank mechanism candidates")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        return {
            "probe_id": self.payload["probe_id"],
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismProbePlan":
        return cls(raw)


@dataclass(frozen=True, slots=True)
class MechanismProbeAssessmentSpec:
    assessment_id: str
    mechanism_probe_plan_sha256: str
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
                f"unsupported mechanism probe assessment spec: {self.schema_version}"
            )
        require_id(self.assessment_id, "mechanism probe assessment id")
        require_digest(self.mechanism_probe_plan_sha256, "mechanism_probe_plan_sha256")
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
            "mechanism_probe_plan_sha256": self.mechanism_probe_plan_sha256,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismProbeAssessmentSpec":
        keys = {
            "schema_version",
            "assessment_id",
            "mechanism_probe_plan_sha256",
            "calibration_plan_sha256",
            "candidate_configuration_id",
            "minimum_complete_pairs",
            "minimum_median_improvement_fraction",
            "require_effect_outside_replay_noise",
            "require_quality_constraints",
        }
        _expect_exact_keys(raw, keys, "mechanism probe assessment spec")
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
            mechanism_probe_plan_sha256=str(raw["mechanism_probe_plan_sha256"]),
            calibration_plan_sha256=str(raw["calibration_plan_sha256"]),
            candidate_configuration_id=str(raw["candidate_configuration_id"]),
            minimum_complete_pairs=raw["minimum_complete_pairs"],
            minimum_median_improvement_fraction=float(improvement),
            require_effect_outside_replay_noise=raw[
                "require_effect_outside_replay_noise"
            ],
            require_quality_constraints=raw["require_quality_constraints"],
        )


@dataclass(frozen=True, slots=True)
class MechanismProbeAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _ASSESSMENT_KEYS, "mechanism probe assessment")
        raw = dict(self.payload)
        digest = str(raw.pop("mechanism_probe_assessment_sha256", ""))
        require_digest(digest, "mechanism_probe_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError(
                "mechanism probe assessment SHA256 does not match its content"
            )
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported mechanism probe assessment: {raw['schema_version']}"
            )
        if raw["status"] not in _ASSESSMENT_STATUSES:
            raise ValueError(f"unsupported mechanism probe assessment: {raw['status']}")
        requirements = MechanismProbeAssessmentSpec.from_dict(raw["requirements"])
        if not isinstance(raw["source"], Mapping) or not isinstance(
            raw["audit"], Mapping
        ):
            raise ValueError(
                "mechanism probe assessment source and audit must be objects"
            )
        if raw["effect"] is not None and not isinstance(raw["effect"], Mapping):
            raise ValueError(
                "mechanism probe assessment effect must be null or an object"
            )
        if not isinstance(raw["reasons"], list) or any(
            not isinstance(reason, str) for reason in raw["reasons"]
        ):
            raise ValueError("mechanism probe assessment reasons must be strings")
        source = raw["source"]
        source_keys = {
            "assessment_spec_sha256",
            "mechanism_probe_plan_sha256",
            "calibration_plan_sha256",
            "calibration_assessment_sha256",
        }
        _expect_exact_keys(source, source_keys, "mechanism probe assessment source")
        for name in source_keys:
            require_digest(str(source[name]), name)
        if raw["assessment_id"] != requirements.assessment_id:
            raise ValueError("assessment id does not match frozen requirements")
        if source["assessment_spec_sha256"] != requirements.sha256:
            raise ValueError("assessment source does not match frozen requirements")
        if (
            source["mechanism_probe_plan_sha256"]
            != requirements.mechanism_probe_plan_sha256
            or source["calibration_plan_sha256"] != requirements.calibration_plan_sha256
        ):
            raise ValueError(
                "assessment source plan bindings do not match requirements"
            )
        if any(
            not isinstance(raw[name], bool)
            for name in ("eligible_for_response_model", "validated_improvement")
        ):
            raise ValueError(
                "mechanism probe assessment eligibility flags must be boolean"
            )

        audit = raw["audit"]
        _expect_exact_keys(
            audit,
            {"calibration_issue_count", "selected_effect_count", "formal_complete"},
            "mechanism probe assessment audit",
        )
        if not isinstance(audit["formal_complete"], bool) or any(
            isinstance(audit[name], bool)
            or not isinstance(audit[name], int)
            or audit[name] < 0
            for name in ("calibration_issue_count", "selected_effect_count")
        ):
            raise ValueError("mechanism probe assessment audit values are invalid")
        if audit["selected_effect_count"] > 1:
            raise ValueError(
                "mechanism probe assessment cannot contain duplicate effects"
            )

        reasons = []
        insufficient = False
        if not audit["formal_complete"]:
            reasons.append("calibration_not_formally_complete")
            insufficient = True
        if audit["calibration_issue_count"]:
            reasons.append("calibration_has_issues")
            insufficient = True
        effect = raw["effect"]
        if effect is None:
            reasons.append("selected_effect_missing")
            insufficient = True
        else:
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
            _expect_exact_keys(effect, effect_keys, "mechanism probe assessment effect")
            if (
                effect["candidate_configuration_id"]
                != requirements.candidate_configuration_id
            ):
                raise ValueError(
                    "assessment effect candidate does not match requirements"
                )
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
            if isinstance(improvement, bool) or not isinstance(
                improvement, (int, float)
            ):
                reasons.append("median_improvement_missing")
                insufficient = True
            elif improvement < requirements.minimum_median_improvement_fraction:
                reasons.append("minimum_probe_improvement_not_met")
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
        expected_reasons = sorted(set(reasons))
        if raw["reasons"] != expected_reasons:
            raise ValueError("assessment reasons do not match frozen gate and effect")
        expected_status = (
            "insufficient_evidence"
            if insufficient
            else "rejected" if reasons else "validated_signal"
        )
        if raw["status"] != expected_status:
            raise ValueError("assessment status does not match frozen gate and effect")
        expected_eligible = (
            audit["formal_complete"]
            and audit["calibration_issue_count"] == 0
            and effect is not None
            and effect["formal_group"] is True
            and isinstance(effect["complete_pair_count"], int)
            and not isinstance(effect["complete_pair_count"], bool)
            and effect["complete_pair_count"] >= requirements.minimum_complete_pairs
        )
        if raw["eligible_for_response_model"] is not expected_eligible:
            raise ValueError(
                "response-model eligibility does not match formal evidence"
            )
        validated = expected_status == "validated_signal"
        if raw["validated_improvement"] is not validated:
            raise ValueError("validated improvement does not match assessment status")
        if validated and not raw["eligible_for_response_model"]:
            raise ValueError("validated signal must be eligible for response modeling")
        expected_action = {
            "validated_signal": "refit_response_model_with_validated_probe",
            "rejected": "retain_control_and_refit_response_model",
            "insufficient_evidence": "do_not_fit_retry_formal_probe",
        }[raw["status"]]
        if raw["next_action"] != expected_action:
            raise ValueError("mechanism probe next action does not match status")
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismProbeAssessment":
        return cls(raw)


def _matches_context(row: SelectorFeatureRow, context: SelectionContext) -> bool:
    if row.cohort.get("algorithm_id") != context.algorithm_id:
        return False
    if (
        row.cohort.get("semantic_class")
        not in context.accepted_evidence_semantic_classes
    ):
        return False
    if row.cohort.get("graph_sha256") != context.graph_sha256:
        return False
    if row.cohort.get("workload_id") != context.workload_id:
        return False
    if row.cohort.get("environment_id") != context.environment_id:
        return False
    for name, expected in context.static_features.items():
        actual = row.static_features.get(name)
        if actual is None or not _same_value(actual, expected):
            return False
    for name, bounds in context.static_feature_ranges.items():
        actual = row.static_features.get(name)
        if actual is None or not bounds.contains(actual):
            return False
    return True


def _matches_deployment(candidate: CompiledCandidate, row: SelectorFeatureRow) -> bool:
    return all(
        name in row.static_features and _same_value(row.static_features[name], expected)
        for name, expected in deployment_features(candidate.deployment_settings).items()
    )


def _resolve_control(
    candidates: Sequence[CompiledCandidate], settings: Mapping[str, SettingValue]
) -> CompiledCandidate:
    matches = [
        candidate
        for candidate in candidates
        if all(
            name in candidate.deployment_settings
            and _same_value(candidate.deployment_settings[name], value)
            for name, value in settings.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"control settings matched {len(matches)} compiled candidates")
    return matches[0]


def _numeric(values: Sequence[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    ]


def _median_feature(rows: Sequence[SelectorFeatureRow], name: str) -> float | None:
    values = _numeric([row.telemetry_features.get(name) for row in rows])
    if not values:
        values = _numeric([row.static_features.get(name) for row in rows])
    if not values:
        values = _numeric([row.targets.get(name) for row in rows])
    return median(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return 0.0
    return max(0.0, numerator / denominator)


def _parameter_default(graph: InferenceGraph, name: str, default: float) -> float:
    value = next(
        (parameter.default for parameter in graph.parameters if parameter.name == name),
        default,
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _diagnostics(
    rows: Sequence[SelectorFeatureRow],
    control: CompiledCandidate,
    graph: InferenceGraph,
    thresholds: MechanismThresholds,
) -> dict[str, Any]:
    base_flops = _median_feature(
        rows, "telemetry.compute.base_backend.estimated_dense_forward_flops"
    )
    proposal_flops = _median_feature(
        rows, "telemetry.compute.proposal_backend.estimated_dense_forward_flops"
    )
    total_flops = (base_flops or 0.0) + (proposal_flops or 0.0)
    base_compute_share = _ratio(base_flops, total_flops)
    proposal_compute_share = _ratio(proposal_flops, total_flops)

    total_slots = _median_feature(rows, "telemetry.compute.total_forward_token_slots")
    base_score_slots = _median_feature(
        rows, "telemetry.compute.base_backend.score_forward_token_slots"
    )
    proposal_generation_slots = _median_feature(
        rows, "telemetry.compute.proposal_backend.generation_forward_token_slots"
    )
    base_score_slot_share = _ratio(base_score_slots, total_slots)
    proposal_generation_slot_share = _ratio(proposal_generation_slots, total_slots)

    prompt_tokens = _median_feature(rows, "workload.prompt_tokens.mean") or 0.0
    generated_tokens = _parameter_default(graph, "total_length", 0.0)
    prompt_work_fraction = _ratio(prompt_tokens, prompt_tokens + generated_tokens)
    maximum_sequence_tokens = prompt_tokens + generated_tokens

    role_metrics: dict[str, dict[str, float | None]] = {}
    for role in ("base", "proposal"):
        max_num_seqs = control.deployment_settings.get(f"{role}_max_num_seqs")
        token_budget = control.deployment_settings.get(f"{role}_max_num_batched_tokens")
        running = _median_feature(
            rows, f"telemetry.runtime.{role}.vllm:num_requests_running.mean"
        )
        waiting = _median_feature(
            rows, f"telemetry.runtime.{role}.vllm:num_requests_waiting.mean"
        )
        batch_values = [
            _median_feature(rows, f"telemetry.batching.{role}.maximum_sample_batch"),
            _median_feature(rows, f"telemetry.batching.{role}.maximum_score_batch"),
        ]
        maximum_batch = max((value or 0.0) for value in batch_values)
        configured_capacity = (
            float(max_num_seqs)
            if isinstance(max_num_seqs, (int, float))
            and not isinstance(max_num_seqs, bool)
            else 0.0
        )
        capacity_utilization = max(
            _ratio(running, configured_capacity),
            _ratio(maximum_batch, configured_capacity),
        )
        configured_token_budget = (
            float(token_budget)
            if isinstance(token_budget, (int, float))
            and not isinstance(token_budget, bool)
            else 0.0
        )
        estimated_token_capacity_utilization = min(
            _ratio(maximum_batch * maximum_sequence_tokens, configured_token_budget),
            1.0,
        )
        queue_ratio = _ratio(waiting, running)
        kv_peak = _median_feature(
            rows, f"telemetry.runtime.{role}.vllm:kv_cache_usage_perc.maximum"
        )
        if kv_peak is None:
            kv_peak = _median_feature(rows, f"resource.{role}_kv_peak_fraction")
        preemptions = _median_feature(
            rows, f"telemetry.runtime.{role}.vllm:num_preemptions.maximum"
        )
        captures = control.deployment_settings.get(f"{role}_graph_capture_sizes")
        ceiling = (
            float(max(captures)) if isinstance(captures, tuple) and captures else None
        )
        runtime_resolved_policy = ceiling is None
        if ceiling is None:
            ceiling = _median_feature(
                rows, f"runtime.{role}.graph_capture_ceiling"
            )
        uncaptured_capacity = (
            _median_feature(
                rows, f"runtime.{role}.uncovered_graph_capacity"
            )
            if runtime_resolved_policy
            else None
        )
        if uncaptured_capacity is None and ceiling is not None:
            uncaptured_capacity = max(configured_capacity - ceiling, 0.0)
        role_metrics[role] = {
            "compute_share": (
                base_compute_share if role == "base" else proposal_compute_share
            ),
            "capacity_utilization": min(capacity_utilization, 1.0),
            "estimated_token_capacity_utilization": (
                estimated_token_capacity_utilization
            ),
            "queue_to_running_ratio": queue_ratio,
            "kv_peak_fraction": kv_peak,
            "preemptions_max": preemptions,
            "graph_capture_ceiling": ceiling,
            "graph_uncaptured_capacity": uncaptured_capacity,
            "token_batch_equivalents": (
                _ratio(float(token_budget), prompt_tokens)
                if isinstance(token_budget, (int, float))
                and not isinstance(token_budget, bool)
                else None
            ),
        }

    tags = []
    dominant_role = max(
        ("base", "proposal"),
        key=lambda role: float(role_metrics[role]["compute_share"] or 0.0),
    )
    if float(role_metrics[dominant_role]["compute_share"] or 0.0) >= (
        thresholds.compute_dominance_fraction
    ):
        tags.append(f"{dominant_role}_compute_dominant")
    if base_score_slot_share >= thresholds.compute_dominance_fraction:
        tags.append("base_score_token_dominant")
    if prompt_work_fraction >= 0.5:
        tags.append("prompt_dominated_workload")
    for role, metrics in role_metrics.items():
        if (
            float(metrics["capacity_utilization"] or 0.0)
            >= thresholds.capacity_utilization_fraction
            and float(metrics["queue_to_running_ratio"] or 0.0)
            >= thresholds.queue_to_running_ratio
        ):
            tags.append(f"{role}_scheduler_queue_pressure")
        if (
            float(metrics["kv_peak_fraction"] or 0.0) >= thresholds.kv_pressure_fraction
            or float(metrics["preemptions_max"] or 0.0) > 0
        ):
            tags.append(f"{role}_kv_pressure")
        elif metrics["kv_peak_fraction"] is not None:
            tags.append(f"{role}_kv_headroom")
        if (
            float(metrics["estimated_token_capacity_utilization"] or 0.0)
            >= thresholds.capacity_utilization_fraction
        ):
            tags.append(f"{role}_token_capacity_pressure")
        if float(metrics["graph_uncaptured_capacity"] or 0.0) > 0:
            tags.append(f"{role}_graph_coverage_gap")
    return {
        "formal_control_row_count": len(rows),
        "formal_control_row_ids": sorted(row.row_id for row in rows),
        "prompt_tokens_mean": prompt_tokens,
        "prompt_work_fraction": prompt_work_fraction,
        "base_score_token_slot_share": base_score_slot_share,
        "proposal_generation_token_slot_share": proposal_generation_slot_share,
        "dominant_compute_role": dominant_role,
        "roles": role_metrics,
        "pressure_tags": sorted(tags),
    }


def _changed_knobs(
    control: CompiledCandidate, candidate: CompiledCandidate
) -> list[dict[str, Any]]:
    changes = []
    for name in sorted(set(control.knob_values) | set(candidate.knob_values)):
        before = control.knob_values.get(name)
        after = candidate.knob_values.get(name)
        if before is None or after is None or not _same_value(before, after):
            changes.append(
                {
                    "name": name,
                    "control": (list(before) if isinstance(before, tuple) else before),
                    "candidate": list(after) if isinstance(after, tuple) else after,
                }
            )
    return changes


def _knob_distance(
    name: str,
    before: Any,
    after: Any,
    candidates: Sequence[CompiledCandidate],
) -> float:
    values = _numeric([candidate.knob_values.get(name) for candidate in candidates])
    if (
        values
        and isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        span = max(values) - min(values)
        return 0.0 if span == 0 else abs(float(after) - float(before)) / span
    return 1.0


def _stage_bindings(graph: InferenceGraph, knob_name: str) -> list[str]:
    return sorted(
        stage.stage_id
        for stage in graph.stages
        if knob_name in stage.tunable_runtime_fields
    )


def _role_signal(diagnostics: Mapping[str, Any], role: str, name: str) -> float:
    value = diagnostics["roles"][role].get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mechanism_contribution(
    change: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> tuple[float, str | None, list[str]]:
    name = str(change["name"])
    role = name.split(".", 1)[0]
    before = change["control"]
    after = change["candidate"]
    if role not in {"base", "proposal"}:
        return 0.0, None, ["unmodeled_knob"]
    compute = _role_signal(diagnostics, role, "compute_share")
    capacity = _role_signal(diagnostics, role, "capacity_utilization")
    queue = _role_signal(diagnostics, role, "queue_to_running_ratio")
    queue_signal = queue / (1.0 + queue)
    kv = _role_signal(diagnostics, role, "kv_peak_fraction")
    preemptions = _role_signal(diagnostics, role, "preemptions_max")
    resource_signal = max(kv, min(preemptions, 1.0))
    prompt_fraction = float(diagnostics["prompt_work_fraction"])
    score_share = float(diagnostics["base_score_token_slot_share"])

    if name.endswith("max_num_batched_tokens") and isinstance(after, (int, float)):
        if isinstance(before, (int, float)) and float(after) > float(before):
            if f"{role}_token_capacity_pressure" not in diagnostics["pressure_tags"]:
                return (
                    0.0,
                    "token_capacity_headroom_no_increase",
                    ["no_observed_token_capacity_pressure"],
                )
            stage_work = max(prompt_fraction, score_share if role == "base" else 0.0)
            return (
                compute * stage_work,
                "increase_token_batch_capacity",
                ["resource_risk_requires_formal_probe"],
            )
        if f"{role}_kv_pressure" not in diagnostics["pressure_tags"]:
            return (
                0.0,
                "token_resource_headroom_no_reduction",
                ["no_observed_resource_pressure"],
            )
        return resource_signal, "reduce_token_batch_resource_pressure", []
    if name.endswith("max_num_seqs") and isinstance(after, (int, float)):
        if isinstance(before, (int, float)) and float(after) > float(before):
            return capacity * queue_signal, "increase_scheduler_capacity", []
        return resource_signal, "reduce_scheduler_resource_pressure", []
    if name.endswith("capture_sizes"):
        if isinstance(before, list) and isinstance(after, list) and before and after:
            expansion = max(after) > max(before)
            uncovered = _role_signal(diagnostics, role, "graph_uncaptured_capacity")
            if expansion and uncovered > 0:
                return (
                    min(uncovered / max(max(after), 1), 1.0),
                    "expand_graph_coverage",
                    [],
                )
        return 0.0, "graph_policy_without_observed_coverage_gap", []
    if name.endswith("memory_fraction") and isinstance(after, (int, float)):
        if isinstance(before, (int, float)) and float(after) > float(before):
            return resource_signal, "increase_kv_reservation", []
        return (
            max(0.0, 1.0 - kv) * 0.1,
            "release_unused_kv_reservation",
            ["coupled_memory_sum_must_remain_feasible"],
        )
    if name.endswith("batch_wait_seconds"):
        return 0.0, "batch_wait_direction_requires_arrival_trace_evidence", []
    return 0.0, None, ["unmodeled_knob"]


def _evaluate_candidate(
    candidate: CompiledCandidate,
    control: CompiledCandidate,
    candidates: Sequence[CompiledCandidate],
    context_rows: Sequence[SelectorFeatureRow],
    diagnostics: Mapping[str, Any],
    graph: InferenceGraph,
    spec: MechanismProbeSpec,
) -> dict[str, Any]:
    changes = _changed_knobs(control, candidate)
    observed = sorted(
        row.row_id for row in context_rows if _matches_deployment(candidate, row)
    )
    rejections = []
    if candidate.candidate_id == control.candidate_id:
        rejections.append("control_candidate")
    if spec.exclude_exactly_observed and observed:
        rejections.append("already_observed")
    if len(changes) > spec.maximum_changed_knobs:
        rejections.append("outside_knob_identifiability_budget")

    signals = []
    risks: set[str] = set()
    contribution = 0.0
    distances = []
    bindings: dict[str, list[str]] = {}
    for change in changes:
        name = str(change["name"])
        score, mechanism, change_risks = _mechanism_contribution(change, diagnostics)
        contribution += score
        risks.update(change_risks)
        distances.append(
            _knob_distance(
                name,
                change["control"],
                change["candidate"],
                candidates,
            )
        )
        bindings[name] = _stage_bindings(graph, name)
        if mechanism is not None:
            signals.append(
                {
                    "knob": name,
                    "mechanism": mechanism,
                    "contribution": score,
                }
            )
    distance = sum(distances) / len(distances) if distances else 0.0
    score = contribution - 0.1 * distance - 0.03 * max(len(changes) - 1, 0)
    if contribution <= 0 or score < spec.thresholds.minimum_mechanism_score:
        rejections.append("no_positive_mechanism")
    if any(not stage_ids for stage_ids in bindings.values()):
        rejections.append("missing_graph_stage_binding")
    return {
        "candidate_id": candidate.candidate_id,
        "deployment_settings": setting_map_to_dict(candidate.deployment_settings),
        "knob_values": setting_map_to_dict(candidate.knob_values),
        "changed_knobs": changes,
        "stage_bindings": bindings,
        "observed_response_row_ids": observed,
        "mechanism_signals": signals,
        "mechanism_score": score,
        "normalized_distance_from_control": distance,
        "risk_tags": sorted(risks),
        "eligible": not rejections,
        "rejection_reasons": sorted(rejections),
    }


def _build_payload(
    *,
    spec: MechanismProbeSpec,
    graph_sha256: str,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable,
    control: CompiledCandidate,
    status: str,
    diagnostics: Mapping[str, Any],
    selections: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
    context_row_count: int,
) -> MechanismProbePlan:
    rejections = Counter(
        reason for item in ranked for reason in item["rejection_reasons"]
    )
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "probe_id": spec.probe_id,
        "status": status,
        "source_spec_sha256": spec.sha256,
        "graph_sha256": graph_sha256,
        "compiled_space_sha256": compiled.to_dict()["compiled_space_sha256"],
        "feature_table_sha256": canonical_sha256(features.to_dict()),
        "selection_context": spec.selection_context.to_dict(),
        "control": {
            "candidate_id": control.candidate_id,
            "deployment_settings": setting_map_to_dict(control.deployment_settings),
            "knob_values": setting_map_to_dict(control.knob_values),
        },
        "diagnostics": dict(diagnostics),
        "selections": list(selections),
        "ranked_candidates": list(ranked),
        "audit": {
            "candidate_budget": spec.candidate_budget,
            "compiled_candidate_count": len(compiled.candidates),
            "context_row_count": context_row_count,
            "formal_control_row_count": diagnostics.get("formal_control_row_count", 0),
            "evaluated_candidate_count": len(ranked),
            "eligible_candidate_count": sum(bool(item["eligible"]) for item in ranked),
            "selected_candidate_count": len(selections),
            "rejections_by_reason": dict(sorted(rejections.items())),
        },
    }
    return MechanismProbePlan(
        {**payload, "mechanism_probe_plan_sha256": canonical_sha256(payload)}
    )


def plan_mechanism_probes(
    spec: MechanismProbeSpec,
    graph: InferenceGraph,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable,
) -> MechanismProbePlan:
    """Rank legal, unmeasured probes using graph bindings and runtime pressure."""

    compiled_digest = compiled.to_dict()["compiled_space_sha256"]
    if spec.compiled_space_sha256 != compiled_digest:
        raise ValueError("mechanism probe spec does not match compiled search space")
    if graph.algorithm_id != compiled.algorithm_id:
        raise ValueError("mechanism probe graph does not match compiled algorithm")
    graph_sha256 = canonical_sha256(graph.to_dict())
    if spec.selection_context.graph_sha256 != graph_sha256:
        raise ValueError("mechanism probe context does not match graph content")
    if spec.selection_context.algorithm_id != graph.algorithm_id:
        raise ValueError("mechanism probe context does not match graph algorithm")
    candidates = tuple(
        candidate
        for candidate in compiled.candidates
        if candidate.semantic_cohort_id == spec.selection_context.semantic_cohort_id
    )
    if not candidates:
        raise ValueError("compiled space has no candidate in the requested cohort")
    control = _resolve_control(candidates, spec.control_settings)
    context_rows = tuple(
        row for row in features.rows if _matches_context(row, spec.selection_context)
    )
    control_rows = tuple(
        row
        for row in context_rows
        if row.evidence.get("grade") == "A_formal_paired"
        and row.eligibility.response_model_fit
        and row.targets.get("run.success") is True
        and _matches_deployment(control, row)
    )
    if len(control_rows) < spec.minimum_formal_control_rows:
        diagnostics = {
            "formal_control_row_count": len(control_rows),
            "formal_control_row_ids": sorted(row.row_id for row in control_rows),
            "required_formal_control_rows": spec.minimum_formal_control_rows,
            "pressure_tags": [],
        }
        return _build_payload(
            spec=spec,
            graph_sha256=graph_sha256,
            compiled=compiled,
            features=features,
            control=control,
            status="insufficient_evidence",
            diagnostics=diagnostics,
            selections=(),
            ranked=(),
            context_row_count=len(context_rows),
        )
    diagnostics = _diagnostics(control_rows, control, graph, spec.thresholds)
    evaluations = [
        _evaluate_candidate(
            candidate,
            control,
            candidates,
            context_rows,
            diagnostics,
            graph,
            spec,
        )
        for candidate in candidates
    ]
    evaluations.sort(
        key=lambda item: (
            not item["eligible"],
            -float(item["mechanism_score"]),
            float(item["normalized_distance_from_control"]),
            item["candidate_id"],
        )
    )
    eligible = [item for item in evaluations if item["eligible"]]
    selections = eligible[: spec.candidate_budget]
    return _build_payload(
        spec=spec,
        graph_sha256=graph_sha256,
        compiled=compiled,
        features=features,
        control=control,
        status="planned" if selections else "no_candidate",
        diagnostics=diagnostics,
        selections=selections,
        ranked=evaluations,
        context_row_count=len(context_rows),
    )


def assess_mechanism_probe(
    spec: MechanismProbeAssessmentSpec,
    plan: MechanismProbePlan,
    calibration_assessment: Mapping[str, Any],
) -> MechanismProbeAssessment:
    """Apply a frozen improvement gate while retaining formal negative evidence."""

    if plan.status != "planned":
        raise ValueError("mechanism probe assessment requires a planned probe")
    plan_digest = str(plan.payload["mechanism_probe_plan_sha256"])
    if spec.mechanism_probe_plan_sha256 != plan_digest:
        raise ValueError("mechanism probe assessment spec is bound to another plan")
    selected_ids = {str(item["candidate_id"]) for item in plan.payload["selections"]}
    if spec.candidate_configuration_id not in selected_ids:
        raise ValueError("assessment candidate was not selected by the mechanism plan")
    if calibration_assessment.get("plan_sha256") != spec.calibration_plan_sha256:
        raise ValueError("calibration assessment is bound to another calibration plan")

    effects = calibration_assessment.get("effects")
    if not isinstance(effects, list):
        raise ValueError("calibration assessment effects must be an array")
    matches = [
        item
        for item in effects
        if isinstance(item, Mapping)
        and item.get("candidate_configuration_id") == spec.candidate_configuration_id
        and item.get("replay_control") is False
    ]
    if len(matches) > 1:
        raise ValueError("calibration assessment has duplicate probe effects")
    effect = matches[0] if matches else None
    issues = calibration_assessment.get("issues")
    reasons = []
    insufficient = False
    formal_complete = calibration_assessment.get("formal_complete") is True
    if not formal_complete:
        reasons.append("calibration_not_formally_complete")
        insufficient = True
    if not isinstance(issues, list) or issues:
        reasons.append("calibration_has_issues")
        insufficient = True
    if effect is None:
        reasons.append("selected_effect_missing")
        insufficient = True
    if effect is not None:
        if effect.get("formal_group") is not True:
            reasons.append("selected_effect_not_formal")
            insufficient = True
        pairs = effect.get("complete_pair_count")
        if isinstance(pairs, bool) or not isinstance(pairs, int):
            reasons.append("complete_pair_count_missing")
            insufficient = True
        elif pairs < spec.minimum_complete_pairs:
            reasons.append("minimum_complete_pairs_not_met")
            insufficient = True
        improvement = effect.get("median_directional_relative_improvement")
        if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
            reasons.append("median_improvement_missing")
            insufficient = True
        elif float(improvement) < spec.minimum_median_improvement_fraction:
            reasons.append("minimum_probe_improvement_not_met")
        if (
            spec.require_effect_outside_replay_noise
            and effect.get("effect_outside_replay_noise") is not True
        ):
            reasons.append("effect_not_outside_replay_noise")
        if (
            spec.require_quality_constraints
            and effect.get("quality_constraints_satisfied") is not True
        ):
            reasons.append("quality_constraints_not_satisfied")

    evidence_complete = (
        formal_complete
        and isinstance(issues, list)
        and not issues
        and effect is not None
        and effect.get("formal_group") is True
        and isinstance(effect.get("complete_pair_count"), int)
        and not isinstance(effect.get("complete_pair_count"), bool)
        and int(effect["complete_pair_count"]) >= spec.minimum_complete_pairs
    )
    status = (
        "insufficient_evidence"
        if insufficient
        else "rejected" if reasons else "validated_signal"
    )
    next_action = {
        "validated_signal": "refit_response_model_with_validated_probe",
        "rejected": "retain_control_and_refit_response_model",
        "insufficient_evidence": "do_not_fit_retry_formal_probe",
    }[status]
    effect_summary = None
    if effect is not None:
        effect_summary = {
            "candidate_configuration_id": spec.candidate_configuration_id,
            "primary_metric": effect.get("primary_metric"),
            "direction": effect.get("direction"),
            "formal_group": effect.get("formal_group"),
            "complete_pair_count": effect.get("complete_pair_count"),
            "median_directional_relative_improvement": effect.get(
                "median_directional_relative_improvement"
            ),
            "candidate_over_baseline_geomean_ratio": effect.get(
                "candidate_over_baseline_geomean_ratio"
            ),
            "effect_outside_replay_noise": effect.get("effect_outside_replay_noise"),
            "quality_constraints_satisfied": effect.get(
                "quality_constraints_satisfied"
            ),
        }
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "assessment_id": spec.assessment_id,
        "status": status,
        "source": {
            "assessment_spec_sha256": spec.sha256,
            "mechanism_probe_plan_sha256": plan_digest,
            "calibration_plan_sha256": spec.calibration_plan_sha256,
            "calibration_assessment_sha256": canonical_sha256(calibration_assessment),
        },
        "requirements": spec.to_dict(),
        "effect": effect_summary,
        "eligible_for_response_model": evidence_complete,
        "validated_improvement": status == "validated_signal",
        "next_action": next_action,
        "reasons": sorted(set(reasons)),
        "audit": {
            "calibration_issue_count": (len(issues) if isinstance(issues, list) else 0),
            "selected_effect_count": len(matches),
            "formal_complete": formal_complete,
        },
    }
    return MechanismProbeAssessment(
        {**payload, "mechanism_probe_assessment_sha256": canonical_sha256(payload)}
    )


def calibration_spec_from_mechanism_probe_plan(
    spec: CalibrationSpec,
    plan: MechanismProbePlan,
    *,
    include_replay_control: bool = False,
    campaign_id: str | None = None,
    pair_seeds: Sequence[int] | None = None,
) -> CalibrationSpec:
    """Compile selected mechanism probes into the formal paired harness."""

    context = SelectionContext.from_dict(plan.payload["selection_context"])
    if context.algorithm_id != spec.semantic_contract.algorithm_id:
        raise ValueError("mechanism probe algorithm does not match calibration spec")
    if context.graph_sha256 != spec.semantic_contract.graph_sha256:
        raise ValueError("mechanism probe graph does not match calibration spec")
    if context.workload_id != spec.workload_contract.workload_id:
        raise ValueError("mechanism probe workload does not match calibration spec")
    if context.environment_id != spec.environment_contract.environment_id:
        raise ValueError("mechanism probe environment does not match calibration spec")
    control_settings = plan.payload["control"]["deployment_settings"]
    if any(
        name not in spec.baseline.settings
        or not _same_value(value, spec.baseline.settings[name])
        for name, value in control_settings.items()
    ):
        raise ValueError("mechanism probe control does not match calibration baseline")
    if plan.status != "planned":
        raise ValueError("mechanism probe plan has no selected candidates")
    configurations = []
    if include_replay_control:
        configurations.append(
            ConfigurationSpec(
                configuration_id=f"{plan.payload['probe_id']}-replay-control",
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
                    f"Mechanism-guided probe from {plan.payload['probe_id']}; "
                    f"score={float(item['mechanism_score']):.6g}"
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
