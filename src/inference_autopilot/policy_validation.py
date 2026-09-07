"""Independent holdout validation for generated policy bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.features import (
    SelectorFeatureRow,
    SelectorFeatureTable,
    deployment_features,
)
from inference_autopilot.policy_selection import (
    PolicyBundle,
    ResponseConstraint,
    ResponseObjective,
)
from inference_autopilot.search_space import (
    CompiledCandidate,
    CompiledSearchSpace,
    setting_map_to_dict,
)


_ASSESSMENT_STATUSES = {"validated", "rejected", "insufficient_holdout"}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True, slots=True)
class PolicyHoldoutSpec:
    assessment_id: str
    policy_bundle_sha256: str
    minimum_successful_replicates: int
    maximum_regret_fraction: float
    minimum_improvement_over_fallback_fraction: float
    reject_observed_failures: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported policy holdout spec: {self.schema_version}")
        require_id(self.assessment_id, "policy holdout assessment_id")
        require_digest(self.policy_bundle_sha256, "policy_bundle_sha256")
        if (
            isinstance(self.minimum_successful_replicates, bool)
            or not isinstance(self.minimum_successful_replicates, int)
            or self.minimum_successful_replicates <= 0
        ):
            raise ValueError("minimum successful replicates must be positive")
        if (
            not isfinite(self.maximum_regret_fraction)
            or self.maximum_regret_fraction < 0
        ):
            raise ValueError("maximum regret fraction must be non-negative")
        if not isfinite(self.minimum_improvement_over_fallback_fraction):
            raise ValueError("minimum fallback improvement must be finite")
        if not isinstance(self.reject_observed_failures, bool):
            raise ValueError("reject_observed_failures must be boolean")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "policy_bundle_sha256": self.policy_bundle_sha256,
            "minimum_successful_replicates": self.minimum_successful_replicates,
            "maximum_regret_fraction": self.maximum_regret_fraction,
            "minimum_improvement_over_fallback_fraction": (
                self.minimum_improvement_over_fallback_fraction
            ),
            "reject_observed_failures": self.reject_observed_failures,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyHoldoutSpec":
        keys = {
            "schema_version",
            "assessment_id",
            "policy_bundle_sha256",
            "minimum_successful_replicates",
            "maximum_regret_fraction",
            "minimum_improvement_over_fallback_fraction",
            "reject_observed_failures",
        }
        _expect_exact_keys(raw, keys, "policy holdout spec")
        if (
            isinstance(raw["minimum_successful_replicates"], bool)
            or not isinstance(raw["minimum_successful_replicates"], int)
        ):
            raise ValueError("minimum successful replicates must be an integer")
        for name in (
            "maximum_regret_fraction",
            "minimum_improvement_over_fallback_fraction",
        ):
            if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
                raise ValueError(f"policy holdout {name} must be numeric")
        if not isinstance(raw["reject_observed_failures"], bool):
            raise ValueError("reject_observed_failures must be boolean")
        return cls(
            assessment_id=str(raw["assessment_id"]),
            policy_bundle_sha256=str(raw["policy_bundle_sha256"]),
            minimum_successful_replicates=raw["minimum_successful_replicates"],
            maximum_regret_fraction=float(raw["maximum_regret_fraction"]),
            minimum_improvement_over_fallback_fraction=float(
                raw["minimum_improvement_over_fallback_fraction"]
            ),
            reject_observed_failures=raw["reject_observed_failures"],
            schema_version=str(raw["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyHoldoutAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "producer",
            "assessment_id",
            "status",
            "holdout_spec_sha256",
            "policy_bundle_sha256",
            "compiled_space_sha256",
            "holdout_feature_table_sha256",
            "requirements",
            "evaluated_candidates",
            "selected",
            "fallback",
            "measured_oracle",
            "comparison",
            "reasons",
            "audit",
            "policy_holdout_assessment_sha256",
        }
        _expect_exact_keys(self.payload, expected, "policy holdout assessment")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_holdout_assessment_sha256", ""))
        require_digest(digest, "policy_holdout_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("policy holdout assessment SHA256 does not match")
        if raw["schema_version"] != "1.0":
            raise ValueError(f"unsupported policy holdout assessment: {raw['schema_version']}")
        if raw["status"] not in _ASSESSMENT_STATUSES:
            raise ValueError(f"unsupported policy holdout status: {raw['status']}")
        require_id(str(raw["assessment_id"]), "policy holdout assessment_id")
        for name in (
            "holdout_spec_sha256",
            "policy_bundle_sha256",
            "compiled_space_sha256",
            "holdout_feature_table_sha256",
        ):
            require_digest(str(raw[name]), name)
        if not isinstance(raw["evaluated_candidates"], list):
            raise ValueError("evaluated holdout candidates must be a list")
        if not isinstance(raw["reasons"], list) or any(
            not isinstance(reason, str) or not reason for reason in raw["reasons"]
        ):
            raise ValueError("policy holdout reasons must be non-empty strings")
        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        return {
            "assessment_id": self.payload["assessment_id"],
            "status": self.status,
            "reasons": list(self.payload["reasons"]),
            **dict(self.payload["audit"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyHoldoutAssessment":
        return cls(raw)


def _same_value(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(left) == float(right)
    return canonical_json({"value": left}) == canonical_json({"value": right})


def _matches_holdout_context(row: SelectorFeatureRow, policy: PolicyBundle) -> bool:
    selection = policy.payload["selection_spec"]
    context = selection["selection_context"]
    if row.cohort.get("algorithm_id") != context["algorithm_id"]:
        return False
    if row.cohort.get("semantic_class") not in context[
        "accepted_evidence_semantic_classes"
    ]:
        return False
    for name in ("graph_sha256", "workload_id", "environment_id"):
        if row.cohort.get(name) != context[name]:
            return False
    for name, expected in context["static_features"].items():
        actual = row.static_features.get(name)
        if actual is None or not _same_value(actual, expected):
            return False
    for name, bounds in context["static_feature_ranges"].items():
        actual = row.static_features.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not float(bounds["minimum"]) <= float(actual) <= float(bounds["maximum"])
        ):
            return False
    return True


def _matches_candidate(row: SelectorFeatureRow, candidate: CompiledCandidate) -> bool:
    expected = deployment_features(candidate.deployment_settings)
    return all(
        name in row.static_features
        and _same_value(row.static_features[name], value)
        for name, value in expected.items()
    )


def _constraint_result(
    constraint: ResponseConstraint, values: Sequence[float]
) -> dict[str, Any]:
    observed = max(values) if constraint.operator == "<=" else min(values)
    passed = (
        observed <= constraint.value
        if constraint.operator == "<="
        else observed >= constraint.value
    )
    return {**constraint.to_dict(), "observed_worst": observed, "passed": passed}


def _candidate_holdout_evaluation(
    candidate: CompiledCandidate,
    rows: Sequence[SelectorFeatureRow],
    objective: ResponseObjective,
    spec: PolicyHoldoutSpec,
) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row.eligibility.response_model_fit
        and row.targets.get("run.success") is True
        and all(target in row.targets for target in objective.targets)
    ]
    failures = [
        row
        for row in rows
        if row.eligibility.feasibility_model
        and row.targets.get("run.success") is False
    ]
    target_values = {
        target: [float(row.targets[target]) for row in successful]
        for target in objective.targets
    }
    constraints = [
        _constraint_result(constraint, target_values[constraint.target])
        for constraint in objective.constraints
    ] if successful else []
    sufficient = len(successful) >= spec.minimum_successful_replicates
    no_rejected_failure = not spec.reject_observed_failures or not failures
    eligible = sufficient and no_rejected_failure and all(
        result["passed"] for result in constraints
    )
    return {
        "candidate_id": candidate.candidate_id,
        "deployment_settings": setting_map_to_dict(candidate.deployment_settings),
        "successful_row_ids": sorted(row.row_id for row in successful),
        "failure_row_ids": sorted(row.row_id for row in failures),
        "successful_replicates": len(successful),
        "observed_failures": len(failures),
        "target_means": {
            target: sum(values) / len(values)
            for target, values in target_values.items()
            if values
        },
        "constraint_results": constraints,
        "eligible": eligible,
        "ineligibility_reasons": [
            reason
            for condition, reason in (
                (not sufficient, "insufficient_successful_replicates"),
                (not no_rejected_failure, "observed_failure"),
                (
                    bool(constraints) and not all(item["passed"] for item in constraints),
                    "slo_constraint",
                ),
            )
            if condition
        ],
    }


def _objective_sort_key(
    evaluation: Mapping[str, Any], objective: ResponseObjective
) -> tuple[float, str]:
    value = float(evaluation["target_means"][objective.target])
    return (-value if objective.direction == "maximize" else value, evaluation["candidate_id"])


def _relative_gain(selected: float, reference: float, direction: str) -> float:
    denominator = max(abs(reference), 1e-12)
    if direction == "maximize":
        return (selected - reference) / denominator
    return (reference - selected) / denominator


def assess_policy_holdout(
    spec: PolicyHoldoutSpec,
    policy: PolicyBundle,
    compiled: CompiledSearchSpace,
    holdout: SelectorFeatureTable,
) -> PolicyHoldoutAssessment:
    """Compare a frozen policy with its baseline and measured holdout oracle."""

    policy_digest = str(policy.payload["policy_bundle_sha256"])
    if spec.policy_bundle_sha256 != policy_digest:
        raise ValueError("policy holdout spec does not match policy bundle")
    compiled_digest = compiled.to_dict()["compiled_space_sha256"]
    if compiled_digest != policy.payload["compiled_space_sha256"]:
        raise ValueError("policy holdout compiled space does not match policy bundle")

    training_ids = set(policy.payload["training"]["response_row_ids"])
    training_ids.update(policy.payload["training"]["feasibility_row_ids"])
    holdout_ids = {row.row_id for row in holdout.rows}
    overlap = sorted(training_ids & holdout_ids)
    if overlap:
        raise ValueError(f"policy holdout leaks training rows: {overlap}")

    objective = ResponseObjective.from_dict(
        policy.payload["selection_spec"]["objective"]
    )
    cohort_id = policy.payload["selection_spec"]["selection_context"][
        "semantic_cohort_id"
    ]
    candidates = tuple(
        candidate
        for candidate in compiled.candidates
        if candidate.semantic_cohort_id == cohort_id
    )
    context_rows = [
        row for row in holdout.rows if _matches_holdout_context(row, policy)
    ]
    unmatched_rows = len(holdout.rows) - len(context_rows)
    rows_by_candidate: dict[str, list[SelectorFeatureRow]] = defaultdict(list)
    ambiguous_rows = 0
    for row in context_rows:
        matches = [
            candidate for candidate in candidates if _matches_candidate(row, candidate)
        ]
        if len(matches) != 1:
            ambiguous_rows += 1
            continue
        rows_by_candidate[matches[0].candidate_id].append(row)

    evaluations = [
        _candidate_holdout_evaluation(
            candidate,
            rows_by_candidate.get(candidate.candidate_id, ()),
            objective,
            spec,
        )
        for candidate in candidates
        if candidate.candidate_id in rows_by_candidate
    ]
    evaluations.sort(key=lambda item: item["candidate_id"])
    by_id = {item["candidate_id"]: item for item in evaluations}
    selected_raw = policy.payload["selected"]
    selected_id = None if selected_raw is None else selected_raw["candidate_id"]
    fallback_id = policy.payload["fallback"]["candidate_id"]
    selected = by_id.get(selected_id) if selected_id is not None else None
    fallback = by_id.get(fallback_id)
    eligible = [item for item in evaluations if item["eligible"]]
    oracle = (
        min(eligible, key=lambda item: _objective_sort_key(item, objective))
        if eligible
        else None
    )

    reasons: list[str] = []
    if policy.status != "selected":
        reasons.append("policy_not_selected")
    if selected is None or selected["successful_replicates"] < spec.minimum_successful_replicates:
        reasons.append("selected_candidate_missing_holdout_replicates")
    elif not selected["eligible"]:
        reasons.append("selected_candidate_failed_holdout_constraints")
    if fallback is None or fallback["successful_replicates"] < spec.minimum_successful_replicates:
        reasons.append("fallback_candidate_missing_holdout_replicates")
    elif not fallback["eligible"]:
        reasons.append("fallback_candidate_failed_holdout_constraints")
    if oracle is None:
        reasons.append("no_holdout_oracle")

    comparison: dict[str, Any] | None = None
    enough = not any(
        reason in reasons
        for reason in (
            "policy_not_selected",
            "selected_candidate_missing_holdout_replicates",
            "fallback_candidate_missing_holdout_replicates",
            "no_holdout_oracle",
        )
    )
    if enough and selected is not None and fallback is not None and oracle is not None:
        selected_value = float(selected["target_means"][objective.target])
        fallback_value = float(fallback["target_means"][objective.target])
        oracle_value = float(oracle["target_means"][objective.target])
        improvement = _relative_gain(
            selected_value, fallback_value, objective.direction
        )
        regret = max(
            0.0,
            -_relative_gain(selected_value, oracle_value, objective.direction),
        )
        comparison = {
            "objective_target": objective.target,
            "direction": objective.direction,
            "selected_value": selected_value,
            "fallback_value": fallback_value,
            "measured_oracle_value": oracle_value,
            "improvement_over_fallback_fraction": improvement,
            "regret_to_measured_oracle_fraction": regret,
        }
        if improvement < spec.minimum_improvement_over_fallback_fraction:
            reasons.append("minimum_fallback_improvement_not_met")
        if regret > spec.maximum_regret_fraction:
            reasons.append("maximum_oracle_regret_exceeded")

    insufficient_reasons = {
        "policy_not_selected",
        "selected_candidate_missing_holdout_replicates",
        "fallback_candidate_missing_holdout_replicates",
        "no_holdout_oracle",
    }
    if any(reason in insufficient_reasons for reason in reasons):
        status = "insufficient_holdout"
    elif reasons:
        status = "rejected"
    else:
        status = "validated"

    audit = {
        "holdout_row_count": len(holdout.rows),
        "context_matched_row_count": len(context_rows),
        "context_unmatched_row_count": unmatched_rows,
        "ambiguous_or_unmatched_candidate_row_count": ambiguous_rows,
        "evaluated_candidate_count": len(evaluations),
        "eligible_candidate_count": len(eligible),
        "by_evidence_grade": dict(
            sorted(Counter(str(row.evidence["grade"]) for row in context_rows).items())
        ),
    }
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "assessment_id": spec.assessment_id,
        "status": status,
        "holdout_spec_sha256": spec.sha256,
        "policy_bundle_sha256": policy_digest,
        "compiled_space_sha256": compiled_digest,
        "holdout_feature_table_sha256": canonical_sha256(holdout.to_dict()),
        "requirements": spec.to_dict(),
        "evaluated_candidates": evaluations,
        "selected": selected,
        "fallback": fallback,
        "measured_oracle": oracle,
        "comparison": comparison,
        "reasons": reasons,
        "audit": audit,
    }
    return PolicyHoldoutAssessment(
        {
            **payload,
            "policy_holdout_assessment_sha256": canonical_sha256(payload),
        }
    )
