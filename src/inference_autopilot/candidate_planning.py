"""Budgeted, deterministic initial designs over a compiled search space."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Any

from inference_autopilot.calibration.models import (
    ConfigurationSpec,
    CalibrationSpec,
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
from inference_autopilot.search_space import (
    CompiledCandidate,
    CompiledSearchSpace,
    DeploymentSearchSpace,
    Scalar,
    SettingValue,
    normalize_setting_map,
    setting_map_to_dict,
)


_EVIDENCE_GRADES = {
    "A_formal_paired",
    "B_controlled_single",
    "C_diagnostic",
    "X_excluded",
}
_SELECTION_REASONS = {
    "baseline",
    "evidence_anchor",
    "constraint_boundary",
    "domain_boundary",
    "space_filling",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and isfinite(value)


def _scalar_map(raw: Mapping[str, Any], context: str) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or not _is_scalar(value):
            raise ValueError(f"{context} must contain named finite scalars")
        values[key] = value
    return dict(sorted(values.items()))


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
class NumericRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum, bool)
            or isinstance(self.maximum, bool)
            or not isinstance(self.minimum, (int, float))
            or not isinstance(self.maximum, (int, float))
            or not isfinite(float(self.minimum))
            or not isfinite(float(self.maximum))
            or self.minimum > self.maximum
        ):
            raise ValueError("static feature range must contain ordered finite numbers")

    def contains(self, value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and self.minimum <= float(value) <= self.maximum
        )

    def to_dict(self) -> dict[str, float]:
        return {"minimum": float(self.minimum), "maximum": float(self.maximum)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NumericRange":
        _expect_exact_keys(raw, {"minimum", "maximum"}, "static feature range")
        return cls(raw["minimum"], raw["maximum"])


@dataclass(frozen=True, slots=True)
class SelectionContext:
    algorithm_id: str
    semantic_cohort_id: str
    graph_sha256: str
    accepted_evidence_semantic_classes: tuple[str, ...]
    workload_id: str
    environment_id: str
    static_features: Mapping[str, Scalar]
    static_feature_ranges: Mapping[str, NumericRange]

    def __post_init__(self) -> None:
        require_id(self.algorithm_id, "selection context algorithm_id")
        require_id(self.semantic_cohort_id, "selection context semantic_cohort_id")
        require_digest(self.graph_sha256, "selection context graph_sha256")
        require_id(self.workload_id, "selection context workload_id")
        require_id(self.environment_id, "selection context environment_id")
        if (
            not self.accepted_evidence_semantic_classes
            or any(not value for value in self.accepted_evidence_semantic_classes)
            or tuple(sorted(set(self.accepted_evidence_semantic_classes)))
            != self.accepted_evidence_semantic_classes
        ):
            raise ValueError("accepted evidence semantic classes must be sorted and unique")
        _scalar_map(self.static_features, "selection context static features")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(bounds, NumericRange)
            for name, bounds in self.static_feature_ranges.items()
        ):
            raise ValueError("selection context feature ranges are invalid")
        overlap = sorted(set(self.static_features) & set(self.static_feature_ranges))
        if overlap:
            raise ValueError(f"exact and ranged context features overlap: {overlap}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "semantic_cohort_id": self.semantic_cohort_id,
            "graph_sha256": self.graph_sha256,
            "accepted_evidence_semantic_classes": list(
                self.accepted_evidence_semantic_classes
            ),
            "workload_id": self.workload_id,
            "environment_id": self.environment_id,
            "static_features": dict(sorted(self.static_features.items())),
            "static_feature_ranges": {
                name: bounds.to_dict()
                for name, bounds in sorted(self.static_feature_ranges.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectionContext":
        keys = {
            "algorithm_id",
            "semantic_cohort_id",
            "graph_sha256",
            "accepted_evidence_semantic_classes",
            "workload_id",
            "environment_id",
            "static_features",
            "static_feature_ranges",
        }
        _expect_exact_keys(raw, keys, "selection context")
        semantics = raw["accepted_evidence_semantic_classes"]
        static = raw["static_features"]
        ranges = raw["static_feature_ranges"]
        if not isinstance(semantics, list) or any(
            not isinstance(value, str) for value in semantics
        ):
            raise ValueError("accepted evidence semantic classes must be strings")
        if not isinstance(static, Mapping):
            raise ValueError("selection context static_features must be an object")
        if not isinstance(ranges, Mapping) or any(
            not isinstance(bounds, Mapping) for bounds in ranges.values()
        ):
            raise ValueError("selection context static_feature_ranges must be an object")
        return cls(
            algorithm_id=str(raw["algorithm_id"]),
            semantic_cohort_id=str(raw["semantic_cohort_id"]),
            graph_sha256=str(raw["graph_sha256"]),
            accepted_evidence_semantic_classes=tuple(semantics),
            workload_id=str(raw["workload_id"]),
            environment_id=str(raw["environment_id"]),
            static_features=_scalar_map(static, "selection context static features"),
            static_feature_ranges={
                str(name): NumericRange.from_dict(bounds)
                for name, bounds in ranges.items()
            },
        )


@dataclass(frozen=True, slots=True)
class EvidenceAnchorPolicy:
    grades: tuple[str, ...]
    minimum_matched_deployment_settings: int
    maximum_anchor_candidates: int

    def __post_init__(self) -> None:
        if (
            not self.grades
            or any(grade not in _EVIDENCE_GRADES for grade in self.grades)
            or tuple(sorted(set(self.grades))) != self.grades
        ):
            raise ValueError("evidence anchor grades must be supported, sorted and unique")
        if any(grade in {"C_diagnostic", "X_excluded"} for grade in self.grades):
            raise ValueError("diagnostic and excluded evidence cannot seed performance anchors")
        if (
            isinstance(self.minimum_matched_deployment_settings, bool)
            or not isinstance(self.minimum_matched_deployment_settings, int)
            or self.minimum_matched_deployment_settings <= 0
        ):
            raise ValueError("minimum matched deployment settings must be positive")
        if (
            isinstance(self.maximum_anchor_candidates, bool)
            or not isinstance(self.maximum_anchor_candidates, int)
            or self.maximum_anchor_candidates < 0
        ):
            raise ValueError("maximum anchor candidates must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grades": list(self.grades),
            "minimum_matched_deployment_settings": (
                self.minimum_matched_deployment_settings
            ),
            "maximum_anchor_candidates": self.maximum_anchor_candidates,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceAnchorPolicy":
        keys = {
            "grades",
            "minimum_matched_deployment_settings",
            "maximum_anchor_candidates",
        }
        _expect_exact_keys(raw, keys, "evidence anchor policy")
        grades = raw["grades"]
        if not isinstance(grades, list) or any(
            not isinstance(grade, str) for grade in grades
        ):
            raise ValueError("evidence anchor grades must be strings")
        for name in (
            "minimum_matched_deployment_settings",
            "maximum_anchor_candidates",
        ):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"evidence anchor policy {name} must be an integer")
        return cls(
            grades=tuple(grades),
            minimum_matched_deployment_settings=raw[
                "minimum_matched_deployment_settings"
            ],
            maximum_anchor_candidates=raw["maximum_anchor_candidates"],
        )


@dataclass(frozen=True, slots=True)
class CandidateDesignSpec:
    design_id: str
    compiled_space_sha256: str
    candidate_budget: int
    seed: int
    selection_context: SelectionContext
    baseline_settings: Mapping[str, SettingValue]
    evidence_policy: EvidenceAnchorPolicy
    maximum_boundary_candidates: int
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_settings",
            normalize_setting_map(
                self.baseline_settings, "candidate design baseline settings"
            ),
        )
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported candidate design spec: {self.schema_version}")
        require_id(self.design_id, "design_id")
        require_digest(self.compiled_space_sha256, "compiled_space_sha256")
        for name in ("candidate_budget", "seed", "maximum_boundary_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"candidate design {name} must be a non-negative integer")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.maximum_boundary_candidates > self.candidate_budget:
            raise ValueError("boundary candidate budget cannot exceed candidate budget")
        if self.evidence_policy.maximum_anchor_candidates > self.candidate_budget:
            raise ValueError("evidence anchor budget cannot exceed candidate budget")
        if not self.baseline_settings:
            raise ValueError("candidate design requires baseline settings")
        normalize_setting_map(
            self.baseline_settings, "candidate design baseline settings"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "compiled_space_sha256": self.compiled_space_sha256,
            "candidate_budget": self.candidate_budget,
            "seed": self.seed,
            "selection_context": self.selection_context.to_dict(),
            "baseline_settings": setting_map_to_dict(self.baseline_settings),
            "evidence_policy": self.evidence_policy.to_dict(),
            "maximum_boundary_candidates": self.maximum_boundary_candidates,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateDesignSpec":
        keys = {
            "schema_version",
            "design_id",
            "compiled_space_sha256",
            "candidate_budget",
            "seed",
            "selection_context",
            "baseline_settings",
            "evidence_policy",
            "maximum_boundary_candidates",
        }
        _expect_exact_keys(raw, keys, "candidate design spec")
        context = raw["selection_context"]
        baseline = raw["baseline_settings"]
        policy = raw["evidence_policy"]
        if not all(isinstance(value, Mapping) for value in (context, baseline, policy)):
            raise ValueError("candidate design nested fields must be objects")
        for name in ("candidate_budget", "seed", "maximum_boundary_candidates"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"candidate design {name} must be an integer")
        return cls(
            schema_version=str(raw["schema_version"]),
            design_id=str(raw["design_id"]),
            compiled_space_sha256=str(raw["compiled_space_sha256"]),
            candidate_budget=raw["candidate_budget"],
            seed=raw["seed"],
            selection_context=SelectionContext.from_dict(context),
            baseline_settings=normalize_setting_map(
                baseline, "candidate design baseline settings"
            ),
            evidence_policy=EvidenceAnchorPolicy.from_dict(policy),
            maximum_boundary_candidates=raw["maximum_boundary_candidates"],
        )


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selection_index: int
    candidate_id: str
    semantic_cohort_id: str
    reason: str
    evidence_row_ids: tuple[str, ...]
    boundary_tags: tuple[str, ...]
    minimum_normalized_distance: float
    knob_values: Mapping[str, SettingValue]
    deployment_settings: Mapping[str, SettingValue]
    semantic_settings: Mapping[str, SettingValue]

    def __post_init__(self) -> None:
        for name in ("knob_values", "deployment_settings", "semantic_settings"):
            values = getattr(self, name)
            if isinstance(values, Mapping):
                object.__setattr__(
                    self,
                    name,
                    normalize_setting_map(values, f"selected candidate {name}"),
                )
        if (
            isinstance(self.selection_index, bool)
            or not isinstance(self.selection_index, int)
            or self.selection_index < 0
        ):
            raise ValueError("selection index must be a non-negative integer")
        require_id(self.candidate_id, "selected candidate_id")
        require_id(self.semantic_cohort_id, "selected semantic_cohort_id")
        if self.reason not in _SELECTION_REASONS:
            raise ValueError(f"unsupported candidate selection reason: {self.reason}")
        for name, values in (
            ("evidence row ids", self.evidence_row_ids),
            ("boundary tags", self.boundary_tags),
        ):
            if tuple(sorted(set(values))) != values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"selected candidate {name} must be sorted and unique")
        if (
            isinstance(self.minimum_normalized_distance, bool)
            or not isinstance(self.minimum_normalized_distance, (int, float))
            or not isfinite(float(self.minimum_normalized_distance))
            or not 0.0 <= self.minimum_normalized_distance <= 1.0
        ):
            raise ValueError("minimum normalized distance must be between zero and one")
        for name in ("knob_values", "deployment_settings", "semantic_settings"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"selected candidate {name} must be an object")
            normalize_setting_map(values, f"selected candidate {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_index": self.selection_index,
            "candidate_id": self.candidate_id,
            "semantic_cohort_id": self.semantic_cohort_id,
            "reason": self.reason,
            "evidence_row_ids": list(self.evidence_row_ids),
            "boundary_tags": list(self.boundary_tags),
            "minimum_normalized_distance": self.minimum_normalized_distance,
            "knob_values": setting_map_to_dict(self.knob_values),
            "deployment_settings": setting_map_to_dict(
                self.deployment_settings
            ),
            "semantic_settings": setting_map_to_dict(self.semantic_settings),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateSelection":
        keys = {
            "selection_index",
            "candidate_id",
            "semantic_cohort_id",
            "reason",
            "evidence_row_ids",
            "boundary_tags",
            "minimum_normalized_distance",
            "knob_values",
            "deployment_settings",
            "semantic_settings",
        }
        _expect_exact_keys(raw, keys, "candidate selection")
        list_names = ("evidence_row_ids", "boundary_tags")
        map_names = ("knob_values", "deployment_settings", "semantic_settings")
        if any(not isinstance(raw[name], list) for name in list_names):
            raise ValueError("selected candidate references must be lists")
        if any(not isinstance(raw[name], Mapping) for name in map_names):
            raise ValueError("selected candidate settings must be objects")
        return cls(
            selection_index=raw["selection_index"],
            candidate_id=str(raw["candidate_id"]),
            semantic_cohort_id=str(raw["semantic_cohort_id"]),
            reason=str(raw["reason"]),
            evidence_row_ids=tuple(str(value) for value in raw["evidence_row_ids"]),
            boundary_tags=tuple(str(value) for value in raw["boundary_tags"]),
            minimum_normalized_distance=raw["minimum_normalized_distance"],
            knob_values=normalize_setting_map(
                raw["knob_values"], "selected knob values"
            ),
            deployment_settings=normalize_setting_map(
                raw["deployment_settings"], "selected deployment settings"
            ),
            semantic_settings=normalize_setting_map(
                raw["semantic_settings"], "selected semantic settings"
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    design_id: str
    design_spec_sha256: str
    source_space_sha256: str
    compiled_space_sha256: str
    evidence_table_sha256: str | None
    candidate_budget: int
    available_candidate_count: int
    selection_context: SelectionContext
    selections: tuple[CandidateSelection, ...]
    evidence_audit: Mapping[str, int]
    boundary_tags_covered: tuple[str, ...]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported candidate plan: {self.schema_version}")
        require_id(self.design_id, "candidate plan design_id")
        for name in (
            "design_spec_sha256",
            "source_space_sha256",
            "compiled_space_sha256",
        ):
            require_digest(getattr(self, name), name)
        if self.evidence_table_sha256 is not None:
            require_digest(self.evidence_table_sha256, "evidence_table_sha256")
        if not self.producer:
            raise ValueError("candidate plan producer cannot be empty")
        for name in ("candidate_budget", "available_candidate_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"candidate plan {name} must be a positive integer")
        if len(self.selections) != self.candidate_budget:
            raise ValueError("candidate plan must fill its declared budget")
        if self.candidate_budget > self.available_candidate_count:
            raise ValueError("candidate budget exceeds available candidates")
        indexes = [selection.selection_index for selection in self.selections]
        if indexes != list(range(len(self.selections))):
            raise ValueError("candidate selection indexes must be contiguous")
        candidate_ids = [selection.candidate_id for selection in self.selections]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate plan selections must be unique")
        if not self.selections or self.selections[0].reason != "baseline":
            raise ValueError("candidate plan must select the strong baseline first")
        if any(
            selection.semantic_cohort_id
            != self.selection_context.semantic_cohort_id
            for selection in self.selections
        ):
            raise ValueError("candidate plan mixes semantic cohorts")
        if any(
            not isinstance(name, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in self.evidence_audit.items()
        ):
            raise ValueError("candidate plan evidence audit must contain counts")
        if tuple(sorted(set(self.boundary_tags_covered))) != self.boundary_tags_covered:
            raise ValueError("covered boundary tags must be sorted and unique")
        if any(
            not isinstance(tag, str) or not tag for tag in self.boundary_tags_covered
        ):
            raise ValueError("covered boundary tags must be non-empty strings")

    def audit(self) -> dict[str, Any]:
        reasons = Counter(selection.reason for selection in self.selections)
        return {
            "candidate_budget": self.candidate_budget,
            "selected_candidate_count": len(self.selections),
            "available_candidate_count": self.available_candidate_count,
            "unselected_candidate_count": (
                self.available_candidate_count - len(self.selections)
            ),
            "by_selection_reason": dict(sorted(reasons.items())),
            "evidence": dict(sorted(self.evidence_audit.items())),
            "boundary_tags_covered": list(self.boundary_tags_covered),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "design_id": self.design_id,
            "design_spec_sha256": self.design_spec_sha256,
            "source_space_sha256": self.source_space_sha256,
            "compiled_space_sha256": self.compiled_space_sha256,
            "evidence_table_sha256": self.evidence_table_sha256,
            "selection_context": self.selection_context.to_dict(),
            "selections": [selection.to_dict() for selection in self.selections],
            "audit": self.audit(),
        }
        return {**payload, "candidate_plan_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidatePlan":
        keys = {
            "schema_version",
            "producer",
            "design_id",
            "design_spec_sha256",
            "source_space_sha256",
            "compiled_space_sha256",
            "evidence_table_sha256",
            "selection_context",
            "selections",
            "audit",
            "candidate_plan_sha256",
        }
        _expect_exact_keys(raw, keys, "candidate plan")
        payload = dict(raw)
        digest = str(payload.pop("candidate_plan_sha256", ""))
        if digest != canonical_sha256(payload):
            raise ValueError("candidate plan SHA256 does not match its content")
        context = raw["selection_context"]
        selections = raw["selections"]
        audit = raw["audit"]
        if not isinstance(context, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("candidate plan context and audit must be objects")
        if not isinstance(selections, list) or any(
            not isinstance(selection, Mapping) for selection in selections
        ):
            raise ValueError("candidate plan selections must be objects")
        expected_audit_keys = {
            "candidate_budget",
            "selected_candidate_count",
            "available_candidate_count",
            "unselected_candidate_count",
            "by_selection_reason",
            "evidence",
            "boundary_tags_covered",
        }
        _expect_exact_keys(audit, expected_audit_keys, "candidate plan audit")
        evidence_audit = audit["evidence"]
        boundary_tags = audit["boundary_tags_covered"]
        if not isinstance(evidence_audit, Mapping) or not isinstance(
            boundary_tags, list
        ):
            raise ValueError("candidate plan audit fields have invalid types")
        plan = cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            design_id=str(raw["design_id"]),
            design_spec_sha256=str(raw["design_spec_sha256"]),
            source_space_sha256=str(raw["source_space_sha256"]),
            compiled_space_sha256=str(raw["compiled_space_sha256"]),
            evidence_table_sha256=(
                None
                if raw["evidence_table_sha256"] is None
                else str(raw["evidence_table_sha256"])
            ),
            candidate_budget=audit["candidate_budget"],
            available_candidate_count=audit["available_candidate_count"],
            selection_context=SelectionContext.from_dict(context),
            selections=tuple(
                CandidateSelection.from_dict(selection) for selection in selections
            ),
            evidence_audit=dict(evidence_audit),
            boundary_tags_covered=tuple(str(tag) for tag in boundary_tags),
        )
        if raw["audit"] != plan.audit():
            raise ValueError("candidate plan audit does not match selections")
        return plan


def _distance_dimensions(
    candidates: Sequence[CompiledCandidate],
) -> dict[str, tuple[str, float, float]]:
    names = sorted({name for candidate in candidates for name in candidate.knob_values})
    dimensions: dict[str, tuple[str, float, float]] = {}
    for name in names:
        values = [candidate.knob_values[name] for candidate in candidates]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            minimum = min(float(value) for value in values)
            maximum = max(float(value) for value in values)
            dimensions[name] = ("numeric", minimum, maximum)
        else:
            dimensions[name] = ("categorical", 0.0, 1.0)
    return dimensions


def _distance(
    left: CompiledCandidate,
    right: CompiledCandidate,
    dimensions: Mapping[str, tuple[str, float, float]],
) -> float:
    components: list[float] = []
    for name, (kind, minimum, maximum) in dimensions.items():
        left_value = left.knob_values[name]
        right_value = right.knob_values[name]
        if kind == "numeric":
            span = maximum - minimum
            component = (
                0.0 if span == 0 else abs(float(left_value) - float(right_value)) / span
            )
        else:
            component = 0.0 if _same_value(left_value, right_value) else 1.0
        components.append(component * component)
    return sqrt(sum(components) / len(components)) if components else 0.0


def _minimum_distance(
    candidate: CompiledCandidate,
    selected: Sequence[CompiledCandidate],
    dimensions: Mapping[str, tuple[str, float, float]],
) -> float:
    if not selected:
        return 1.0
    return min(_distance(candidate, other, dimensions) for other in selected)


def _tie_hash(seed: int, candidate_id: str) -> str:
    return canonical_sha256({"seed": seed, "candidate_id": candidate_id})


def _context_conflicts(row: SelectorFeatureRow, context: SelectionContext) -> bool:
    row_workload = row.cohort.get("workload_id")
    row_environment = row.cohort.get("environment_id")
    if row_workload not in {None, context.workload_id}:
        return True
    if row_environment not in {None, context.environment_id}:
        return True
    for name, expected in context.static_features.items():
        actual = row.static_features.get(name)
        if actual is not None and not _same_value(actual, expected):
            return True
    for name, bounds in context.static_feature_ranges.items():
        actual = row.static_features.get(name)
        if actual is not None and not bounds.contains(actual):
            return True
    return False


def _observed_deployment_features(
    row: SelectorFeatureRow, feature_names: set[str]
) -> dict[str, Scalar]:
    observed: dict[str, Scalar] = {}
    for feature_name in sorted(feature_names):
        value = row.static_features.get(feature_name)
        if value is not None:
            observed[feature_name] = value
    return observed


def _candidate_matches_features(
    candidate: CompiledCandidate, observed: Mapping[str, Scalar]
) -> bool:
    expected = deployment_features(candidate.deployment_settings)
    return all(
        name in expected and _same_value(expected[name], value)
        for name, value in observed.items()
    )


def _evidence_anchor_candidates(
    features: SelectorFeatureTable | None,
    candidates: Sequence[CompiledCandidate],
    baseline: CompiledCandidate,
    design: CandidateDesignSpec,
    dimensions: Mapping[str, tuple[str, float, float]],
) -> tuple[list[tuple[CompiledCandidate, tuple[str, ...]]], dict[str, int]]:
    counts: Counter[str] = Counter(
        {
            "rows_seen": 0,
            "grade_or_semantic_ineligible": 0,
            "context_conflict": 0,
            "insufficient_deployment_settings": 0,
            "no_candidate_match": 0,
            "eligible_anchor_rows": 0,
            "unique_anchor_candidates": 0,
        }
    )
    if features is None:
        return [], dict(counts)
    feature_names = {
        name
        for candidate in candidates
        for name in deployment_features(candidate.deployment_settings)
    }
    proposals: dict[str, dict[str, Any]] = {}
    for row in features.rows:
        counts["rows_seen"] += 1
        grade = str(row.evidence.get("grade", ""))
        semantic_class = str(row.cohort.get("semantic_class", ""))
        if (
            grade not in design.evidence_policy.grades
            or row.cohort.get("algorithm_id") != design.selection_context.algorithm_id
            or semantic_class
            not in design.selection_context.accepted_evidence_semantic_classes
            or row.targets.get("run.success") is False
            or "performance.completed_qps" not in row.targets
        ):
            counts["grade_or_semantic_ineligible"] += 1
            continue
        if _context_conflicts(row, design.selection_context):
            counts["context_conflict"] += 1
            continue
        observed = _observed_deployment_features(row, feature_names)
        if (
            len(observed)
            < design.evidence_policy.minimum_matched_deployment_settings
        ):
            counts["insufficient_deployment_settings"] += 1
            continue
        matches = [
            candidate
            for candidate in candidates
            if _candidate_matches_features(candidate, observed)
        ]
        if not matches:
            counts["no_candidate_match"] += 1
            continue
        counts["eligible_anchor_rows"] += 1
        representative = min(
            matches,
            key=lambda candidate: (
                _distance(candidate, baseline, dimensions),
                _tie_hash(design.seed, candidate.candidate_id),
                candidate.candidate_id,
            ),
        )
        grade_rank = 2 if grade == "A_formal_paired" else 1
        context_matches = sum(
            row.static_features.get(name) is not None
            for name in design.selection_context.static_features
        )
        context_matches += sum(
            row.static_features.get(name) is not None
            for name in design.selection_context.static_feature_ranges
        )
        score = (grade_rank, context_matches, len(observed))
        existing = proposals.setdefault(
            representative.candidate_id,
            {"candidate": representative, "row_ids": set(), "score": score},
        )
        existing["row_ids"].add(row.row_id)
        existing["score"] = max(existing["score"], score)
    counts["unique_anchor_candidates"] = len(proposals)
    ordered = sorted(
        proposals.values(),
        key=lambda item: (
            tuple(-value for value in item["score"]),
            _tie_hash(design.seed, item["candidate"].candidate_id),
            item["candidate"].candidate_id,
        ),
    )
    return [
        (item["candidate"], tuple(sorted(item["row_ids"]))) for item in ordered
    ], dict(counts)


def _boundary_tags(
    space: DeploymentSearchSpace, candidates: Sequence[CompiledCandidate]
) -> dict[str, set[str]]:
    tags: dict[str, set[str]] = defaultdict(set)
    for knob in space.knobs:
        values = knob.values()
        if (
            len(values) <= 2
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
        ):
            continue
        minimum = min(float(value) for value in values)
        maximum = max(float(value) for value in values)
        for candidate in candidates:
            actual = float(candidate.knob_values[knob.name])
            if actual == minimum:
                tags[candidate.candidate_id].add(f"domain_min:{knob.name}")
            if actual == maximum:
                tags[candidate.candidate_id].add(f"domain_max:{knob.name}")
    for constraint in space.constraints:
        if constraint.kind != "sum_less_equal":
            continue
        assert constraint.value is not None
        slacks = {
            candidate.candidate_id: float(constraint.value)
            - sum(float(candidate.knob_values[name]) for name in constraint.parameters)
            for candidate in candidates
        }
        minimum_slack = min(slacks.values())
        for candidate_id, slack in slacks.items():
            if abs(slack - minimum_slack) <= 1e-12:
                tags[candidate_id].add(f"constraint_near:{constraint.label}")
    return tags


def _choose_farthest(
    pool: Sequence[CompiledCandidate],
    selected: Sequence[CompiledCandidate],
    dimensions: Mapping[str, tuple[str, float, float]],
    seed: int,
    tags: Mapping[str, set[str]] | None = None,
    covered_tags: set[str] | None = None,
) -> tuple[CompiledCandidate, float]:
    tags = tags or {}
    covered_tags = covered_tags or set()
    ranked = sorted(
        pool,
        key=lambda candidate: (
            -len(tags.get(candidate.candidate_id, set()) - covered_tags),
            -_minimum_distance(candidate, selected, dimensions),
            _tie_hash(seed, candidate.candidate_id),
            candidate.candidate_id,
        ),
    )
    winner = ranked[0]
    return winner, _minimum_distance(winner, selected, dimensions)


def _selection(
    index: int,
    candidate: CompiledCandidate,
    reason: str,
    selected_before: Sequence[CompiledCandidate],
    dimensions: Mapping[str, tuple[str, float, float]],
    evidence_row_ids: tuple[str, ...] = (),
    boundary_tags: tuple[str, ...] = (),
) -> CandidateSelection:
    distance = (
        0.0
        if reason == "baseline"
        else _minimum_distance(candidate, selected_before, dimensions)
    )
    return CandidateSelection(
        selection_index=index,
        candidate_id=candidate.candidate_id,
        semantic_cohort_id=candidate.semantic_cohort_id,
        reason=reason,
        evidence_row_ids=tuple(sorted(set(evidence_row_ids))),
        boundary_tags=tuple(sorted(set(boundary_tags))),
        minimum_normalized_distance=distance,
        knob_values=candidate.knob_values,
        deployment_settings=candidate.deployment_settings,
        semantic_settings=candidate.semantic_settings,
    )


def build_candidate_plan(
    design: CandidateDesignSpec,
    space: DeploymentSearchSpace,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable | None = None,
) -> CandidatePlan:
    """Select baseline, evidence anchors, boundaries, then maximin coverage."""

    compiled_payload = compiled.to_dict()
    compiled_digest = str(compiled_payload["compiled_space_sha256"])
    if design.compiled_space_sha256 != compiled_digest:
        raise ValueError("candidate design is not bound to this compiled search space")
    source_digest = canonical_sha256(space.to_dict())
    if compiled.source_space_sha256 != source_digest:
        raise ValueError("compiled search space is not bound to the source space")
    context = design.selection_context
    if context.algorithm_id != compiled.algorithm_id:
        raise ValueError("selection context algorithm does not match compiled space")
    candidates = [
        candidate
        for candidate in compiled.candidates
        if candidate.semantic_cohort_id == context.semantic_cohort_id
    ]
    if not candidates:
        raise ValueError("selection context semantic cohort has no compiled candidates")
    if design.candidate_budget > len(candidates):
        raise ValueError("candidate budget exceeds the selected semantic cohort")
    baseline_matches = [
        candidate
        for candidate in candidates
        if canonical_json(dict(candidate.deployment_settings))
        == canonical_json(dict(design.baseline_settings))
    ]
    if len(baseline_matches) != 1:
        raise ValueError(
            "baseline settings must match exactly one candidate in the selected cohort"
        )
    baseline = baseline_matches[0]
    dimensions = _distance_dimensions(candidates)
    anchors, evidence_audit = _evidence_anchor_candidates(
        features, candidates, baseline, design, dimensions
    )
    anchor_rows_by_candidate = {
        candidate.candidate_id: row_ids for candidate, row_ids in anchors
    }

    selected: list[CompiledCandidate] = [baseline]
    selections = [
        _selection(
            0,
            baseline,
            "baseline",
            (),
            dimensions,
            anchor_rows_by_candidate.get(baseline.candidate_id, ()),
        )
    ]
    selected_ids = {baseline.candidate_id}
    anchors_added = 0
    for candidate, row_ids in anchors:
        if (
            len(selected) >= design.candidate_budget
            or anchors_added >= design.evidence_policy.maximum_anchor_candidates
        ):
            break
        if candidate.candidate_id in selected_ids:
            continue
        selections.append(
            _selection(
                len(selections),
                candidate,
                "evidence_anchor",
                selected,
                dimensions,
                row_ids,
            )
        )
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
        anchors_added += 1

    tags = _boundary_tags(space, candidates)
    covered_tags: set[str] = set().union(
        *(tags.get(candidate.candidate_id, set()) for candidate in selected)
    )
    all_boundary_tags = set().union(*tags.values()) if tags else set()
    boundary_added = 0
    while (
        len(selected) < design.candidate_budget
        and boundary_added < design.maximum_boundary_candidates
        and covered_tags != all_boundary_tags
    ):
        pool = [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in selected_ids
            and tags.get(candidate.candidate_id)
        ]
        if not pool:
            break
        candidate, distance = _choose_farthest(
            pool,
            selected,
            dimensions,
            design.seed,
            tags,
            covered_tags,
        )
        candidate_tags = tags[candidate.candidate_id]
        reason = (
            "constraint_boundary"
            if any(tag.startswith("constraint_near:") for tag in candidate_tags)
            else "domain_boundary"
        )
        selections.append(
            CandidateSelection(
                selection_index=len(selections),
                candidate_id=candidate.candidate_id,
                semantic_cohort_id=candidate.semantic_cohort_id,
                reason=reason,
                evidence_row_ids=anchor_rows_by_candidate.get(
                    candidate.candidate_id, ()
                ),
                boundary_tags=tuple(sorted(candidate_tags)),
                minimum_normalized_distance=distance,
                knob_values=candidate.knob_values,
                deployment_settings=candidate.deployment_settings,
                semantic_settings=candidate.semantic_settings,
            )
        )
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
        covered_tags.update(candidate_tags)
        boundary_added += 1

    while len(selected) < design.candidate_budget:
        pool = [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in selected_ids
        ]
        candidate, distance = _choose_farthest(
            pool, selected, dimensions, design.seed
        )
        selections.append(
            CandidateSelection(
                selection_index=len(selections),
                candidate_id=candidate.candidate_id,
                semantic_cohort_id=candidate.semantic_cohort_id,
                reason="space_filling",
                evidence_row_ids=anchor_rows_by_candidate.get(
                    candidate.candidate_id, ()
                ),
                boundary_tags=tuple(sorted(tags.get(candidate.candidate_id, set()))),
                minimum_normalized_distance=distance,
                knob_values=candidate.knob_values,
                deployment_settings=candidate.deployment_settings,
                semantic_settings=candidate.semantic_settings,
            )
        )
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)

    evidence_audit = {
        **evidence_audit,
        "selected_anchor_candidates": anchors_added,
    }
    evidence_digest = (
        None if features is None else canonical_sha256(features.to_dict())
    )
    return CandidatePlan(
        design_id=design.design_id,
        design_spec_sha256=canonical_sha256(design.to_dict()),
        source_space_sha256=source_digest,
        compiled_space_sha256=compiled_digest,
        evidence_table_sha256=evidence_digest,
        candidate_budget=design.candidate_budget,
        available_candidate_count=len(candidates),
        selection_context=context,
        selections=tuple(selections),
        evidence_audit=evidence_audit,
        boundary_tags_covered=tuple(sorted(covered_tags)),
    )


def configurations_from_plan(
    plan: CandidatePlan, *, include_baseline: bool = False
) -> tuple[ConfigurationSpec, ...]:
    """Export selected deployment-only candidates for calibration planning."""

    selections = plan.selections if include_baseline else plan.selections[1:]
    configurations: list[ConfigurationSpec] = []
    for selection in selections:
        if selection.semantic_settings:
            raise ValueError(
                "algorithm-changing selections require a separate semantic contract"
            )
        configurations.append(
            ConfigurationSpec(
                configuration_id=selection.candidate_id,
                settings=selection.deployment_settings,
                description=(
                    f"Candidate design {plan.design_id}: {selection.reason}"
                ),
            )
        )
    return tuple(configurations)


def calibration_spec_from_candidate_plan(
    spec: CalibrationSpec, plan: CandidatePlan
) -> CalibrationSpec:
    """Replace a calibration spec's manual candidates with a budgeted design."""

    context = plan.selection_context
    if context.algorithm_id != spec.semantic_contract.algorithm_id:
        raise ValueError("candidate plan algorithm does not match calibration spec")
    if context.graph_sha256 != spec.semantic_contract.graph_sha256:
        raise ValueError("candidate plan graph does not match calibration spec")
    if context.workload_id != spec.workload_contract.workload_id:
        raise ValueError("candidate plan workload does not match calibration spec")
    if context.environment_id != spec.environment_contract.environment_id:
        raise ValueError("candidate plan environment does not match calibration spec")
    baseline = plan.selections[0]
    if canonical_json(dict(baseline.deployment_settings)) != canonical_json(
        dict(spec.baseline.settings)
    ):
        raise ValueError("candidate plan baseline does not match calibration spec")
    for name, expected in context.static_features.items():
        if not name.startswith("workload."):
            continue
        parameter = name.removeprefix("workload.")
        if "." in parameter or parameter not in spec.workload_contract.parameters:
            continue
        actual = spec.workload_contract.parameters[parameter]
        if _is_scalar(actual) and not _same_value(actual, expected):
            raise ValueError(
                f"candidate plan workload feature {name} does not match calibration spec"
            )
    return replace(spec, candidates=configurations_from_plan(plan))
