"""Formal interpretation of a completed policy-transfer calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from math import isfinite
from typing import Any

from inference_autopilot.calibration import build_plan
from inference_autopilot.calibration.models import (
    CalibrationSpec,
    canonical_json,
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.evidence import EvidenceLedger
from inference_autopilot.features import features_from_ledger
from inference_autopilot.policy_transfer import PolicyTransferPlan


_STATUSES = {"positive_transfer_evidence", "rejected", "insufficient_evidence"}
_ASSESSMENT_KEYS = {
    "schema_version",
    "producer",
    "assessment_id",
    "status",
    "source",
    "requirements",
    "effect",
    "guard_assessment",
    "eligible_for_target_policy_evidence",
    "source_policy_activation_eligible",
    "target_policy_required",
    "reasons",
    "audit",
    "policy_transfer_assessment_sha256",
}


@dataclass(frozen=True, slots=True)
class PolicyTransferAssessmentSpec:
    assessment_id: str
    policy_transfer_plan_sha256: str
    minimum_complete_pairs: int
    minimum_median_improvement_fraction: float
    require_effect_outside_replay_noise: bool = True
    require_quality_constraints: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported policy transfer assessment spec: {self.schema_version}"
            )
        require_id(self.assessment_id, "policy transfer assessment_id")
        require_digest(self.policy_transfer_plan_sha256, "policy_transfer_plan_sha256")
        if (
            isinstance(self.minimum_complete_pairs, bool)
            or not isinstance(self.minimum_complete_pairs, int)
            or self.minimum_complete_pairs <= 0
        ):
            raise ValueError("minimum complete transfer pairs must be positive")
        if not isfinite(self.minimum_median_improvement_fraction):
            raise ValueError("minimum transfer improvement must be finite")
        if not isinstance(self.require_effect_outside_replay_noise, bool):
            raise ValueError("require_effect_outside_replay_noise must be boolean")
        if not isinstance(self.require_quality_constraints, bool):
            raise ValueError("require_quality_constraints must be boolean")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "policy_transfer_plan_sha256": self.policy_transfer_plan_sha256,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyTransferAssessmentSpec":
        expect_keys(
            raw,
            {
                "schema_version",
                "assessment_id",
                "policy_transfer_plan_sha256",
                "minimum_complete_pairs",
                "minimum_median_improvement_fraction",
                "require_effect_outside_replay_noise",
                "require_quality_constraints",
            },
            "policy transfer assessment spec",
        )
        pairs = raw["minimum_complete_pairs"]
        improvement = raw["minimum_median_improvement_fraction"]
        if isinstance(pairs, bool) or not isinstance(pairs, int):
            raise ValueError("minimum_complete_pairs must be an integer")
        if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
            raise ValueError("minimum_median_improvement_fraction must be numeric")
        for name in (
            "require_effect_outside_replay_noise",
            "require_quality_constraints",
        ):
            if not isinstance(raw[name], bool):
                raise ValueError(f"{name} must be boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            assessment_id=str(raw["assessment_id"]),
            policy_transfer_plan_sha256=str(raw["policy_transfer_plan_sha256"]),
            minimum_complete_pairs=pairs,
            minimum_median_improvement_fraction=float(improvement),
            require_effect_outside_replay_noise=raw[
                "require_effect_outside_replay_noise"
            ],
            require_quality_constraints=raw["require_quality_constraints"],
        )


@dataclass(frozen=True, slots=True)
class PolicyTransferAssessment:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expect_keys(self.payload, _ASSESSMENT_KEYS, "policy transfer assessment")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_transfer_assessment_sha256", ""))
        require_digest(digest, "policy_transfer_assessment_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("policy transfer assessment SHA256 does not match")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported policy transfer assessment: {raw['schema_version']}"
            )
        if raw["status"] not in _STATUSES:
            raise ValueError(f"unsupported policy transfer status: {raw['status']}")
        require_id(str(raw["assessment_id"]), "policy transfer assessment_id")
        source = require_object(raw["source"], "policy transfer assessment source")
        expect_keys(
            source,
            {
                "assessment_spec_sha256",
                "policy_transfer_plan_sha256",
                "calibration_plan_sha256",
                "calibration_assessment_sha256",
            },
            "policy transfer assessment source",
        )
        for name, value in source.items():
            require_digest(str(value), name)
        for name in (
            "eligible_for_target_policy_evidence",
            "source_policy_activation_eligible",
            "target_policy_required",
        ):
            if not isinstance(raw[name], bool):
                raise ValueError(f"policy transfer {name} must be boolean")
        if raw["source_policy_activation_eligible"] and raw["target_policy_required"]:
            raise ValueError("source activation and target-policy requirement conflict")
        positive = raw["status"] == "positive_transfer_evidence"
        if raw["eligible_for_target_policy_evidence"] != positive:
            raise ValueError("transfer evidence eligibility does not match status")
        if positive != (
            raw["source_policy_activation_eligible"] or raw["target_policy_required"]
        ):
            raise ValueError("positive transfer must name its activation disposition")
        reasons = raw["reasons"]
        if (
            not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or reasons != sorted(set(reasons))
        ):
            raise ValueError("policy transfer reasons must be sorted and unique")
        require_object(raw["requirements"], "policy transfer requirements")
        require_object(raw["guard_assessment"], "policy transfer guard assessment")
        audit = require_object(raw["audit"], "policy transfer audit")
        expect_keys(
            audit,
            {
                "declared_guard_deviation_count",
                "deferred_guard_check_count",
                "deferred_guard_violation_count",
                "calibration_issue_count",
            },
            "policy transfer audit",
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in audit.values()
        ):
            raise ValueError("policy transfer audit values must be non-negative counts")
        if raw["effect"] is not None:
            require_object(raw["effect"], "policy transfer effect")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    def audit(self) -> dict[str, Any]:
        return {
            "assessment_id": self.payload["assessment_id"],
            "status": self.status,
            "eligible_for_target_policy_evidence": self.payload[
                "eligible_for_target_policy_evidence"
            ],
            "source_policy_activation_eligible": self.payload[
                "source_policy_activation_eligible"
            ],
            "target_policy_required": self.payload["target_policy_required"],
            "reasons": list(self.payload["reasons"]),
            **dict(self.payload["audit"]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyTransferAssessment":
        return cls(raw)


def _same_settings(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_json(left) == canonical_json(right)


def _selected_configuration_id(plan: PolicyTransferPlan) -> str:
    calibration = CalibrationSpec.from_dict(plan.payload["calibration_spec"])
    selected_settings = require_object(
        plan.payload["policy"]["selected_deployment_settings"],
        "selected transfer settings",
    )
    matches = [
        candidate.configuration_id
        for candidate in calibration.candidates
        if _same_settings(candidate.settings, selected_settings)
        and not _same_settings(candidate.settings, calibration.baseline.settings)
    ]
    if len(matches) != 1:
        raise ValueError("transfer plan must contain one distinct selected candidate")
    return matches[0]


def _deferred_guard_assessment(
    plan: PolicyTransferPlan,
    calibration_assessment: Mapping[str, Any],
    candidate_configuration_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    guard = plan.payload["guard_evaluation"]
    deferred = guard["deferred_runtime_checks"]
    if not deferred:
        return [], []
    ledger = EvidenceLedger.from_dict(
        require_object(calibration_assessment.get("ledger"), "calibration ledger")
    )
    rows = [
        row
        for row in features_from_ledger(ledger).rows
        if row.variant == candidate_configuration_id
    ]
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for declared in deferred:
        field = str(declared["field"])
        values = [
            row.static_features[field] for row in rows if field in row.static_features
        ]
        if not values or len(values) != len(rows):
            checks.append(
                {
                    "field": field,
                    "status": "missing",
                    "observed_values": [],
                    "within_source_guard": False,
                }
            )
            missing.append(field)
            continue
        unique_values = sorted(set(values), key=lambda value: canonical_json(value))
        if "minimum" in declared and "maximum" in declared:
            numeric = all(
                not isinstance(value, bool) and isinstance(value, (int, float))
                for value in unique_values
            )
            within = numeric and all(
                float(declared["minimum"]) <= float(value) <= float(declared["maximum"])
                for value in unique_values
            )
        elif "expected" in declared:
            within = all(
                canonical_json(value) == canonical_json(declared["expected"])
                for value in unique_values
            )
        else:
            within = False
            missing.append(field)
        checks.append(
            {
                "field": field,
                "status": "observed",
                "observed_values": unique_values,
                "within_source_guard": within,
            }
        )
    return checks, missing


def assess_policy_transfer(
    spec: PolicyTransferAssessmentSpec,
    plan: PolicyTransferPlan,
    calibration_assessment: Mapping[str, Any],
) -> PolicyTransferAssessment:
    """Separate positive transfer evidence from source-policy activation rights."""

    if plan.status != "planned":
        raise ValueError("policy transfer assessment requires a planned transfer")
    plan_digest = str(plan.payload["policy_transfer_plan_sha256"])
    if spec.policy_transfer_plan_sha256 != plan_digest:
        raise ValueError("policy transfer assessment spec is bound to another plan")
    calibration = CalibrationSpec.from_dict(plan.payload["calibration_spec"])
    expected_plan = build_plan(calibration)
    calibration_plan_sha256 = canonical_sha256(expected_plan.to_dict())
    if calibration_assessment.get("plan_sha256") != calibration_plan_sha256:
        raise ValueError("calibration assessment is bound to another plan")
    candidate_id = _selected_configuration_id(plan)
    effects = calibration_assessment.get("effects")
    if not isinstance(effects, list):
        raise ValueError("calibration assessment effects must be an array")
    selected_effects = [
        require_object(effect, "selected transfer effect")
        for effect in effects
        if isinstance(effect, Mapping)
        and effect.get("candidate_configuration_id") == candidate_id
        and effect.get("replay_control") is False
    ]
    if len(selected_effects) > 1:
        raise ValueError("calibration assessment has duplicate selected effects")
    effect = selected_effects[0] if selected_effects else None

    deferred_checks, missing_deferred = _deferred_guard_assessment(
        plan, calibration_assessment, candidate_id
    )
    declared_deviations = list(plan.payload["guard_evaluation"]["deviations"])
    source_guard_match = not declared_deviations and all(
        item["within_source_guard"] for item in deferred_checks
    )
    reasons: list[str] = []
    insufficient = False
    if calibration_assessment.get("formal_complete") is not True:
        reasons.append("calibration_not_formally_complete")
        insufficient = True
    issues = calibration_assessment.get("issues")
    if not isinstance(issues, list) or issues:
        reasons.append("calibration_has_issues")
        insufficient = True
    if effect is None:
        reasons.append("selected_effect_missing")
        insufficient = True
    if missing_deferred:
        reasons.extend(
            f"deferred_feature_missing:{field}" for field in missing_deferred
        )
        insufficient = True
    if effect is not None:
        if effect.get("formal_group") is not True:
            reasons.append("selected_effect_not_formal")
            insufficient = True
        if effect.get("complete_pair_count", 0) < spec.minimum_complete_pairs:
            reasons.append("minimum_complete_pairs_not_met")
            insufficient = True
        improvement = effect.get("median_directional_relative_improvement")
        if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
            reasons.append("median_improvement_missing")
            insufficient = True
        elif float(improvement) < spec.minimum_median_improvement_fraction:
            reasons.append("minimum_transfer_improvement_not_met")
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

    if insufficient:
        status = "insufficient_evidence"
    elif reasons:
        status = "rejected"
    else:
        status = "positive_transfer_evidence"
    eligible = status == "positive_transfer_evidence"
    source_activation = eligible and source_guard_match
    target_policy_required = eligible and not source_guard_match
    effect_summary = None
    if effect is not None:
        effect_summary = {
            "candidate_configuration_id": candidate_id,
            "primary_metric": effect.get("primary_metric"),
            "direction": effect.get("direction"),
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
    assessment_digest = canonical_sha256(calibration_assessment)
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "assessment_id": spec.assessment_id,
        "status": status,
        "source": {
            "assessment_spec_sha256": spec.sha256,
            "policy_transfer_plan_sha256": plan_digest,
            "calibration_plan_sha256": calibration_plan_sha256,
            "calibration_assessment_sha256": assessment_digest,
        },
        "requirements": spec.to_dict(),
        "effect": effect_summary,
        "guard_assessment": {
            "declared_deviations": declared_deviations,
            "deferred_checks": deferred_checks,
            "source_guard_match": source_guard_match,
        },
        "eligible_for_target_policy_evidence": eligible,
        "source_policy_activation_eligible": source_activation,
        "target_policy_required": target_policy_required,
        "reasons": sorted(set(reasons)),
        "audit": {
            "declared_guard_deviation_count": len(declared_deviations),
            "deferred_guard_check_count": len(deferred_checks),
            "deferred_guard_violation_count": sum(
                1 for item in deferred_checks if not item["within_source_guard"]
            ),
            "calibration_issue_count": len(issues) if isinstance(issues, list) else 0,
        },
    }
    return PolicyTransferAssessment(
        {**payload, "policy_transfer_assessment_sha256": canonical_sha256(payload)}
    )
