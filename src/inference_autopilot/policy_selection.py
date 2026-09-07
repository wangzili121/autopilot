"""Conservative, failure-aware offline policy selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite, sqrt
from typing import Any

from inference_autopilot.calibration.models import (
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
from inference_autopilot.search_space import (
    CompiledCandidate,
    CompiledSearchSpace,
    Scalar,
    SettingValue,
    normalize_setting_map,
    setting_map_to_dict,
)


_POLICY_STATUSES = {"selected", "insufficient_evidence", "no_feasible_candidate"}
_POLICY_BUNDLE_KEYS = {
    "schema_version",
    "producer",
    "policy_id",
    "status",
    "selection_spec_sha256",
    "compiled_space_sha256",
    "feature_table_sha256",
    "selection_spec",
    "response_model",
    "training",
    "activation_guard",
    "selected",
    "fallback",
    "ranked_candidates",
    "audit",
    "policy_bundle_sha256",
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
    if any(
        not isinstance(key, str) or not key or not _is_scalar(value)
        for key, value in raw.items()
    ):
        raise ValueError(f"{context} must contain named finite scalars")
    canonical_json(raw)
    return dict(sorted(raw.items()))


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
class ResponseConstraint:
    target: str
    operator: str
    value: float

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("response constraint target cannot be empty")
        if self.operator not in {"<=", ">="}:
            raise ValueError("response constraint operator must be <= or >=")
        if isinstance(self.value, bool) or not isfinite(self.value):
            raise ValueError("response constraint value must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "operator": self.operator, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResponseConstraint":
        _expect_exact_keys(raw, {"target", "operator", "value"}, "response constraint")
        value = raw["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("response constraint value must be numeric")
        return cls(str(raw["target"]), str(raw["operator"]), float(value))


@dataclass(frozen=True, slots=True)
class ResponseObjective:
    target: str
    direction: str
    constraints: tuple[ResponseConstraint, ...]

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("response objective target cannot be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("response objective direction must be maximize or minimize")
        targets = [constraint.target for constraint in self.constraints]
        if len(targets) != len(set(targets)) or self.target in targets:
            raise ValueError("response objective and constraint targets must be unique")

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.target, *(constraint.target for constraint in self.constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "direction": self.direction,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResponseObjective":
        _expect_exact_keys(raw, {"target", "direction", "constraints"}, "response objective")
        constraints = raw["constraints"]
        if not isinstance(constraints, list) or any(
            not isinstance(item, Mapping) for item in constraints
        ):
            raise ValueError("response objective constraints must be objects")
        return cls(
            str(raw["target"]),
            str(raw["direction"]),
            tuple(ResponseConstraint.from_dict(item) for item in constraints),
        )


@dataclass(frozen=True, slots=True)
class LocalResponseModelSpec:
    feature_names: tuple[str, ...]
    neighbors: int
    distance_power: float
    maximum_normalized_distance: float
    uncertainty_multiplier: float
    maximum_failure_probability: float
    minimum_response_rows: int
    minimum_distinct_configurations: int
    minimum_paired_replicates: int
    ranked_candidate_limit: int

    def __post_init__(self) -> None:
        if (
            not self.feature_names
            or tuple(sorted(set(self.feature_names))) != self.feature_names
            or any(not name for name in self.feature_names)
        ):
            raise ValueError("response model feature names must be sorted and unique")
        for name in (
            "neighbors",
            "minimum_response_rows",
            "minimum_distinct_configurations",
            "minimum_paired_replicates",
            "ranked_candidate_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"response model {name} must be a positive integer")
        if not isfinite(self.distance_power) or self.distance_power <= 0:
            raise ValueError("response model distance_power must be positive")
        if not 0 <= self.maximum_normalized_distance <= 1:
            raise ValueError("maximum_normalized_distance must be between zero and one")
        if not isfinite(self.uncertainty_multiplier) or self.uncertainty_multiplier < 0:
            raise ValueError("uncertainty_multiplier must be non-negative")
        if not 0 <= self.maximum_failure_probability <= 1:
            raise ValueError("maximum_failure_probability must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "neighbors": self.neighbors,
            "distance_power": self.distance_power,
            "maximum_normalized_distance": self.maximum_normalized_distance,
            "uncertainty_multiplier": self.uncertainty_multiplier,
            "maximum_failure_probability": self.maximum_failure_probability,
            "minimum_response_rows": self.minimum_response_rows,
            "minimum_distinct_configurations": self.minimum_distinct_configurations,
            "minimum_paired_replicates": self.minimum_paired_replicates,
            "ranked_candidate_limit": self.ranked_candidate_limit,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LocalResponseModelSpec":
        keys = {
            "feature_names",
            "neighbors",
            "distance_power",
            "maximum_normalized_distance",
            "uncertainty_multiplier",
            "maximum_failure_probability",
            "minimum_response_rows",
            "minimum_distinct_configurations",
            "minimum_paired_replicates",
            "ranked_candidate_limit",
        }
        _expect_exact_keys(raw, keys, "local response model")
        features = raw["feature_names"]
        if not isinstance(features, list) or any(not isinstance(item, str) for item in features):
            raise ValueError("response model feature_names must be strings")
        integer_names = (
            "neighbors",
            "minimum_response_rows",
            "minimum_distinct_configurations",
            "minimum_paired_replicates",
            "ranked_candidate_limit",
        )
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], int)
            for name in integer_names
        ):
            raise ValueError("response model count settings must be integers")
        numeric_names = (
            "distance_power",
            "maximum_normalized_distance",
            "uncertainty_multiplier",
            "maximum_failure_probability",
        )
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], (int, float))
            for name in numeric_names
        ):
            raise ValueError("response model numeric settings must be numbers")
        return cls(
            feature_names=tuple(features),
            neighbors=raw["neighbors"],
            distance_power=float(raw["distance_power"]),
            maximum_normalized_distance=float(raw["maximum_normalized_distance"]),
            uncertainty_multiplier=float(raw["uncertainty_multiplier"]),
            maximum_failure_probability=float(raw["maximum_failure_probability"]),
            minimum_response_rows=raw["minimum_response_rows"],
            minimum_distinct_configurations=raw["minimum_distinct_configurations"],
            minimum_paired_replicates=raw["minimum_paired_replicates"],
            ranked_candidate_limit=raw["ranked_candidate_limit"],
        )


@dataclass(frozen=True, slots=True)
class PolicySelectionSpec:
    policy_id: str
    compiled_space_sha256: str
    selection_context: SelectionContext
    query_static_features: Mapping[str, Scalar]
    baseline_settings: Mapping[str, SettingValue]
    objective: ResponseObjective
    model: LocalResponseModelSpec
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_settings",
            normalize_setting_map(
                self.baseline_settings, "policy baseline settings"
            ),
        )
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported policy selection spec: {self.schema_version}")
        require_id(self.policy_id, "policy_id")
        require_digest(self.compiled_space_sha256, "compiled_space_sha256")
        _scalar_map(self.query_static_features, "policy query static features")
        normalize_setting_map(self.baseline_settings, "policy baseline settings")
        if not self.baseline_settings:
            raise ValueError("policy selection requires baseline settings")
        for feature in self.model.feature_names:
            if feature.startswith("deployment."):
                continue
            if feature not in self.query_static_features:
                raise ValueError(f"query value missing for response feature {feature}")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "compiled_space_sha256": self.compiled_space_sha256,
            "selection_context": self.selection_context.to_dict(),
            "query_static_features": dict(sorted(self.query_static_features.items())),
            "baseline_settings": setting_map_to_dict(self.baseline_settings),
            "objective": self.objective.to_dict(),
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicySelectionSpec":
        keys = {
            "schema_version",
            "policy_id",
            "compiled_space_sha256",
            "selection_context",
            "query_static_features",
            "baseline_settings",
            "objective",
            "model",
        }
        _expect_exact_keys(raw, keys, "policy selection spec")
        object_names = (
            "selection_context",
            "query_static_features",
            "baseline_settings",
            "objective",
            "model",
        )
        if any(not isinstance(raw[name], Mapping) for name in object_names):
            raise ValueError("policy selection nested fields must be objects")
        return cls(
            policy_id=str(raw["policy_id"]),
            compiled_space_sha256=str(raw["compiled_space_sha256"]),
            selection_context=SelectionContext.from_dict(raw["selection_context"]),
            query_static_features=_scalar_map(
                raw["query_static_features"], "policy query static features"
            ),
            baseline_settings=normalize_setting_map(
                raw["baseline_settings"], "policy baseline settings"
            ),
            objective=ResponseObjective.from_dict(raw["objective"]),
            model=LocalResponseModelSpec.from_dict(raw["model"]),
            schema_version=str(raw["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """Content-addressed deployment decision with explicit rollback guards."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _expect_exact_keys(self.payload, _POLICY_BUNDLE_KEYS, "policy bundle")
        raw = dict(self.payload)
        digest = str(raw.pop("policy_bundle_sha256", ""))
        require_digest(digest, "policy_bundle_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("policy bundle SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(f"unsupported policy bundle: {raw['schema_version']}")
        if not isinstance(raw["producer"], str) or not raw["producer"]:
            raise ValueError("policy bundle producer cannot be empty")
        require_id(str(raw["policy_id"]), "policy bundle policy_id")
        if raw["status"] not in _POLICY_STATUSES:
            raise ValueError(f"unsupported policy status: {raw['status']}")
        for name in (
            "selection_spec_sha256",
            "compiled_space_sha256",
            "feature_table_sha256",
        ):
            require_digest(str(raw[name]), name)

        object_names = (
            "selection_spec",
            "response_model",
            "training",
            "activation_guard",
            "fallback",
            "audit",
        )
        if any(not isinstance(raw[name], Mapping) for name in object_names):
            raise ValueError("policy bundle object fields must be objects")
        selection_spec = PolicySelectionSpec.from_dict(raw["selection_spec"])
        if selection_spec.sha256 != raw["selection_spec_sha256"]:
            raise ValueError("policy bundle selection spec SHA256 does not match")
        if selection_spec.policy_id != raw["policy_id"]:
            raise ValueError("policy bundle id does not match selection spec")
        if selection_spec.compiled_space_sha256 != raw["compiled_space_sha256"]:
            raise ValueError("policy bundle compiled space does not match selection spec")

        selected = raw["selected"]
        ranked = raw["ranked_candidates"]
        if selected is not None and not isinstance(selected, Mapping):
            raise ValueError("policy bundle selected candidate must be an object or null")
        if not isinstance(ranked, list) or any(
            not isinstance(item, Mapping) for item in ranked
        ):
            raise ValueError("policy bundle ranked candidates must be objects")
        if raw["status"] == "selected":
            if selected is None or not selected.get("eligible"):
                raise ValueError("selected policy requires an eligible candidate")
            if not ranked or ranked[0].get("candidate_id") != selected.get("candidate_id"):
                raise ValueError("selected policy must lead the reported ranking")
        elif selected is not None:
            raise ValueError("blocked policy cannot contain a selected candidate")
        if raw["status"] == "insufficient_evidence" and ranked:
            raise ValueError("insufficient-evidence policy cannot rank candidates")

        fallback = raw["fallback"]
        _expect_exact_keys(
            fallback,
            {"candidate_id", "deployment_settings", "requires_engine_restart"},
            "policy fallback",
        )
        require_id(str(fallback["candidate_id"]), "fallback candidate_id")
        if not isinstance(fallback["deployment_settings"], Mapping):
            raise ValueError("fallback deployment settings must be an object")
        normalize_setting_map(
            fallback["deployment_settings"], "fallback deployment settings"
        )
        if not isinstance(fallback["requires_engine_restart"], bool):
            raise ValueError("fallback restart flag must be boolean")
        if any(
            setting not in fallback["deployment_settings"]
            or not _same_value(fallback["deployment_settings"][setting], value)
            for setting, value in selection_spec.baseline_settings.items()
        ):
            raise ValueError("policy fallback does not match declared baseline")

        response_model = raw["response_model"]
        _expect_exact_keys(
            response_model,
            {
                "kind",
                "feature_dimensions",
                "target_scales",
                "replicate_noise_floors",
            },
            "policy response model",
        )
        if (
            response_model["kind"]
            != "normalized_inverse_distance_support_points_replicate_floor"
        ):
            raise ValueError("policy bundle uses an unsupported response model")
        if not isinstance(response_model["feature_dimensions"], Mapping):
            raise ValueError("policy response dimensions must be an object")
        for name in ("target_scales", "replicate_noise_floors"):
            values = response_model[name]
            if not isinstance(values, Mapping) or any(
                not isinstance(target, str)
                or not target
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0
                for target, value in values.items()
            ):
                raise ValueError(f"policy response {name} must be non-negative numbers")

        training = raw["training"]
        _expect_exact_keys(
            training,
            {"response_row_ids", "feasibility_row_ids", "exclusions"},
            "policy training audit",
        )
        for name in ("response_row_ids", "feasibility_row_ids"):
            values = training[name]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or values != sorted(set(values))
            ):
                raise ValueError(f"policy training {name} must be sorted and unique")
        exclusions = training["exclusions"]
        if not isinstance(exclusions, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for name, count in exclusions.items()
        ):
            raise ValueError("policy training exclusions must be non-negative counts")

        guard = raw["activation_guard"]
        guard_keys = {
            "algorithm_id",
            "semantic_cohort_id",
            "graph_sha256",
            "workload_id",
            "environment_id",
            "exact_static_features",
            "static_feature_ranges",
            "maximum_normalized_distance",
            "maximum_failure_probability",
            "slo_constraints",
            "on_violation",
        }
        _expect_exact_keys(guard, guard_keys, "policy activation guard")
        context = selection_spec.selection_context
        expected_identity = {
            "algorithm_id": context.algorithm_id,
            "semantic_cohort_id": context.semantic_cohort_id,
            "graph_sha256": context.graph_sha256,
            "workload_id": context.workload_id,
            "environment_id": context.environment_id,
        }
        if any(guard[name] != value for name, value in expected_identity.items()):
            raise ValueError("policy activation guard identity does not match selection spec")
        expected_ranges = {
            name: bounds.to_dict()
            for name, bounds in sorted(context.static_feature_ranges.items())
        }
        if guard["exact_static_features"] != dict(
            sorted(context.static_features.items())
        ) or guard["static_feature_ranges"] != expected_ranges:
            raise ValueError("policy activation domain does not match selection spec")
        if (
            guard["maximum_normalized_distance"]
            != selection_spec.model.maximum_normalized_distance
            or guard["maximum_failure_probability"]
            != selection_spec.model.maximum_failure_probability
            or guard["slo_constraints"]
            != [
                constraint.to_dict()
                for constraint in selection_spec.objective.constraints
            ]
        ):
            raise ValueError("policy activation thresholds do not match selection spec")
        if guard["on_violation"] != "fallback":
            raise ValueError("policy activation guard must fail closed to fallback")

        candidate_ids = [str(item.get("candidate_id", "")) for item in ranked]
        if len(candidate_ids) != len(set(candidate_ids)) or any(
            not candidate_id for candidate_id in candidate_ids
        ):
            raise ValueError("policy ranked candidate ids must be unique")

        audit = raw["audit"]
        audit_keys = {
            "compiled_candidate_count",
            "evaluated_candidate_count",
            "eligible_candidate_count",
            "reported_candidate_count",
            "rejections_by_reason",
        }
        _expect_exact_keys(audit, audit_keys, "policy audit")
        counts = [
            audit["compiled_candidate_count"],
            audit["evaluated_candidate_count"],
            audit["eligible_candidate_count"],
            audit["reported_candidate_count"],
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("policy audit counts must be non-negative integers")
        if audit["reported_candidate_count"] != len(ranked):
            raise ValueError("policy reported candidate audit does not match ranking")
        if audit["eligible_candidate_count"] > audit["evaluated_candidate_count"]:
            raise ValueError("eligible policy count exceeds evaluated count")

        canonical = canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def audit(self) -> dict[str, Any]:
        selected = self.payload["selected"]
        return {
            "policy_id": self.payload["policy_id"],
            "status": self.status,
            "selected_candidate_id": (
                None if selected is None else selected["candidate_id"]
            ),
            "fallback_candidate_id": self.payload["fallback"]["candidate_id"],
            **dict(self.payload["audit"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyBundle":
        return cls(raw)


def _row_matches_context(row: SelectorFeatureRow, context: SelectionContext) -> bool:
    if row.cohort.get("algorithm_id") != context.algorithm_id:
        return False
    if row.cohort.get("semantic_class") not in context.accepted_evidence_semantic_classes:
        return False
    if row.cohort.get("graph_sha256") != context.graph_sha256:
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


def _candidate_query(
    candidate: CompiledCandidate, spec: PolicySelectionSpec
) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    candidate_features = deployment_features(candidate.deployment_settings)
    for feature in spec.model.feature_names:
        if feature.startswith("deployment."):
            if feature not in candidate_features:
                raise ValueError(f"compiled candidate omits response feature {feature}")
            values[feature] = candidate_features[feature]
        else:
            values[feature] = spec.query_static_features[feature]
    return values


def _configuration_key(row: SelectorFeatureRow, features: Sequence[str]) -> str:
    deployment = {
        name: row.static_features[name]
        for name in features
        if name.startswith("deployment.") and name in row.static_features
    }
    return canonical_sha256(deployment)


def _resolve_baseline(
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
        raise ValueError(f"baseline settings matched {len(matches)} compiled candidates")
    return matches[0]


def _dimensions(
    candidates: Sequence[CompiledCandidate],
    rows: Sequence[SelectorFeatureRow],
    spec: PolicySelectionSpec,
) -> dict[str, dict[str, Any]]:
    dimensions: dict[str, dict[str, Any]] = {}
    for feature in spec.model.feature_names:
        values: list[Scalar] = [row.static_features[feature] for row in rows]
        values.extend(_candidate_query(candidate, spec)[feature] for candidate in candidates)
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            minimum = min(float(value) for value in values)
            maximum = max(float(value) for value in values)
            dimensions[feature] = {
                "kind": "numeric",
                "minimum": minimum,
                "maximum": maximum,
            }
        else:
            dimensions[feature] = {"kind": "categorical"}
    return dimensions


def _distance(
    query: Mapping[str, Scalar],
    row: SelectorFeatureRow,
    dimensions: Mapping[str, Mapping[str, Any]],
) -> float:
    components = []
    for feature, dimension in dimensions.items():
        left = query[feature]
        right = row.static_features[feature]
        if dimension["kind"] == "numeric":
            span = float(dimension["maximum"]) - float(dimension["minimum"])
            component = 0.0 if span == 0 else abs(float(left) - float(right)) / span
        else:
            component = 0.0 if _same_value(left, right) else 1.0
        components.append(component * component)
    return sqrt(sum(components) / len(components))


def _support_points(
    rows: Sequence[SelectorFeatureRow], feature_names: Sequence[str]
) -> tuple[tuple[SelectorFeatureRow, ...], ...]:
    grouped: dict[str, list[SelectorFeatureRow]] = {}
    for row in rows:
        key = canonical_json(
            {name: row.static_features[name] for name in feature_names}
        )
        grouped.setdefault(key, []).append(row)
    return tuple(
        tuple(sorted(group, key=lambda row: row.row_id))
        for _key, group in sorted(grouped.items())
    )


def _nearest(
    query: Mapping[str, Scalar],
    support_points: Sequence[tuple[SelectorFeatureRow, ...]],
    dimensions: Mapping[str, Mapping[str, Any]],
    count: int,
) -> list[tuple[float, tuple[SelectorFeatureRow, ...]]]:
    ordered = sorted(
        (
            (_distance(query, support[0], dimensions), support)
            for support in support_points
        ),
        key=lambda item: (item[0], item[1][0].row_id),
    )
    exact = [item for item in ordered if item[0] == 0]
    return exact if exact else ordered[:count]


def _weights(
    neighbors: Sequence[tuple[float, tuple[SelectorFeatureRow, ...]]],
    power: float,
) -> list[float]:
    if not neighbors:
        return []
    if neighbors[0][0] == 0:
        return [1.0] * len(neighbors)
    return [1.0 / (distance**power) for distance, _row in neighbors]


def _replicate_noise_floors(
    support_points: Sequence[tuple[SelectorFeatureRow, ...]],
    objective: ResponseObjective,
) -> dict[str, float]:
    """Retain the largest observed within-configuration deviation per target.

    A support point with only two close observations must not appear more stable
    than a well-replicated control.  The floor models run-to-run operational
    variation, so it intentionally does not shrink with replicate count.
    """

    floors: dict[str, float] = {}
    for target in objective.targets:
        deviations = []
        for support in support_points:
            if len(support) < 2:
                continue
            values = [float(row.targets[target]) for row in support]
            mean = sum(values) / len(values)
            deviations.append(max(abs(value - mean) for value in values))
        floors[target] = max(deviations, default=0.0)
    return floors


def _estimate_targets(
    neighbors: Sequence[tuple[float, tuple[SelectorFeatureRow, ...]]],
    objective: ResponseObjective,
    target_scales: Mapping[str, float],
    replicate_noise_floors: Mapping[str, float],
    model: LocalResponseModelSpec,
) -> dict[str, dict[str, Any]]:
    weights = _weights(neighbors, model.distance_power)
    total_weight = sum(weights)
    estimates: dict[str, dict[str, Any]] = {}
    nearest_distance = neighbors[0][0]
    for target in objective.targets:
        support_values = [
            [float(row.targets[target]) for row in support]
            for _distance_value, support in neighbors
        ]
        support_means = [sum(values) / len(values) for values in support_values]
        mean = sum(
            weight * value
            for weight, value in zip(weights, support_means, strict=True)
        ) / total_weight
        if nearest_distance == 0 and len(neighbors) == 1:
            local_spread = max(
                abs(value - mean) for value in support_values[0]
            )
        else:
            within_spread = sum(
                weight
                * max(abs(value - support_mean) for value in values)
                for weight, support_mean, values in zip(
                    weights, support_means, support_values, strict=True
                )
            ) / total_weight
            between_spread = sqrt(
                sum(
                    weight * (support_mean - mean) ** 2
                    for weight, support_mean in zip(
                        weights, support_means, strict=True
                    )
                )
                / total_weight
            )
            local_spread = within_spread + between_spread
        uncertainty = model.uncertainty_multiplier * (
            max(local_spread, replicate_noise_floors[target])
            + nearest_distance * target_scales[target]
        )
        estimates[target] = {
            "mean": mean,
            "lower": mean - uncertainty,
            "upper": mean + uncertainty,
            "uncertainty": uncertainty,
        }
    return estimates


def _row_matches_candidate(
    row: SelectorFeatureRow, candidate: CompiledCandidate
) -> bool:
    expected = deployment_features(candidate.deployment_settings)
    return all(
        feature in row.static_features
        and _same_value(row.static_features[feature], value)
        for feature, value in expected.items()
    )


def _paired_objective_effects(
    rows: Sequence[SelectorFeatureRow],
    candidates: Sequence[CompiledCandidate],
    baseline: CompiledCandidate,
    objective: ResponseObjective,
) -> dict[str, dict[str, Any]]:
    """Recover directional effects from formal paired feature rows.

    Absolute response fitting remains useful for interpolation, but deployment
    promotion must not erase the ABBA comparison that made a row grade A.
    Incomplete or ambiguous pairs are ignored rather than inferred.
    """

    grouped: dict[tuple[str, int], list[SelectorFeatureRow]] = {}
    for row in rows:
        group_id = row.static_features.get("workload.comparison_group_id")
        pair_index = row.static_features.get("workload.pair_index")
        if (
            not isinstance(group_id, str)
            or not group_id
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index < 0
        ):
            continue
        grouped.setdefault((group_id, pair_index), []).append(row)

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for (group_id, pair_index), pair_rows in sorted(grouped.items()):
        baseline_rows = [row for row in pair_rows if "baseline" in row.tags]
        candidate_rows = [row for row in pair_rows if "candidate" in row.tags]
        if len(baseline_rows) != 1 or len(candidate_rows) != 1:
            continue
        baseline_row = baseline_rows[0]
        candidate_row = candidate_rows[0]
        if not _row_matches_candidate(baseline_row, baseline):
            continue
        matches = [
            candidate
            for candidate in candidates
            if _row_matches_candidate(candidate_row, candidate)
        ]
        if len(matches) != 1 or matches[0].candidate_id == baseline.candidate_id:
            continue
        candidate = matches[0]
        baseline_value = baseline_row.targets.get(objective.target)
        candidate_value = candidate_row.targets.get(objective.target)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) <= 0
            for value in (baseline_value, candidate_value)
        ):
            continue
        baseline_number = float(baseline_value)
        candidate_number = float(candidate_value)
        if objective.direction == "maximize":
            effect = candidate_number / baseline_number - 1.0
        else:
            effect = baseline_number / candidate_number - 1.0
        by_candidate.setdefault(candidate.candidate_id, []).append(
            {
                "comparison_group_id": group_id,
                "pair_index": pair_index,
                "baseline_row_id": baseline_row.row_id,
                "candidate_row_id": candidate_row.row_id,
                "directional_relative_improvement": effect,
            }
        )

    effects: dict[str, dict[str, Any]] = {}
    for candidate_id, pairs in sorted(by_candidate.items()):
        ordered = sorted(
            pairs,
            key=lambda item: (
                item["comparison_group_id"],
                item["pair_index"],
                item["candidate_row_id"],
            ),
        )
        values = [float(item["directional_relative_improvement"]) for item in ordered]
        effects[candidate_id] = {
            "pair_count": len(ordered),
            "mean_directional_relative_improvement": sum(values) / len(values),
            "lower_directional_relative_improvement": min(values),
            "upper_directional_relative_improvement": max(values),
            "pairs": ordered,
        }
    return effects


def _failure_probability(
    neighbors: Sequence[tuple[float, tuple[SelectorFeatureRow, ...]]],
    power: float,
) -> float:
    weights = _weights(neighbors, power)
    total = sum(weights)
    return sum(
        weight
        * (
            sum(row.targets.get("run.success") is not True for row in support)
            / len(support)
        )
        for weight, (_distance_value, support) in zip(
            weights, neighbors, strict=True
        )
    ) / total


def _candidate_evaluation(
    candidate: CompiledCandidate,
    spec: PolicySelectionSpec,
    response_supports: Sequence[tuple[SelectorFeatureRow, ...]],
    feasibility_supports: Sequence[tuple[SelectorFeatureRow, ...]],
    dimensions: Mapping[str, Mapping[str, Any]],
    target_scales: Mapping[str, float],
    replicate_noise_floors: Mapping[str, float],
    paired_effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    query = _candidate_query(candidate, spec)
    response_neighbors = _nearest(
        query, response_supports, dimensions, spec.model.neighbors
    )
    feasibility_neighbors = _nearest(
        query, feasibility_supports, dimensions, spec.model.neighbors
    )
    nearest_distance = response_neighbors[0][0]
    target_estimates = _estimate_targets(
        response_neighbors,
        spec.objective,
        target_scales,
        replicate_noise_floors,
        spec.model,
    )
    failure_probability = _failure_probability(
        feasibility_neighbors, spec.model.distance_power
    )
    rejections = []
    if nearest_distance > spec.model.maximum_normalized_distance:
        rejections.append("extrapolation_distance")
    if failure_probability > spec.model.maximum_failure_probability:
        rejections.append("failure_probability")
    paired_effect = paired_effects.get(candidate.candidate_id)
    if (
        paired_effect is not None
        and paired_effect["pair_count"] >= spec.model.minimum_paired_replicates
        and paired_effect["upper_directional_relative_improvement"] < 0
    ):
        rejections.append("paired_objective_regression")
    for constraint in spec.objective.constraints:
        estimate = target_estimates[constraint.target]
        if constraint.operator == "<=" and estimate["upper"] > constraint.value:
            rejections.append(f"constraint:{constraint.target}")
        if constraint.operator == ">=" and estimate["lower"] < constraint.value:
            rejections.append(f"constraint:{constraint.target}")
    return {
        "candidate_id": candidate.candidate_id,
        "nearest_normalized_distance": nearest_distance,
        "failure_probability": failure_probability,
        "target_estimates": target_estimates,
        "paired_objective_effect": paired_effect,
        "response_neighbor_row_ids": sorted(
            row.row_id
            for _distance_value, support in response_neighbors
            for row in support
        ),
        "feasibility_neighbor_row_ids": sorted(
            row.row_id
            for _distance_value, support in feasibility_neighbors
            for row in support
        ),
        "eligible": not rejections,
        "rejection_reasons": sorted(rejections),
        "deployment_settings": setting_map_to_dict(candidate.deployment_settings),
        "requires_engine_restart": candidate.requires_engine_restart,
    }


def _rank_key(evaluation: Mapping[str, Any], objective: ResponseObjective) -> tuple[Any, ...]:
    estimate = evaluation["target_estimates"][objective.target]
    if objective.direction == "maximize":
        objective_key = (-estimate["lower"], -estimate["mean"])
    else:
        objective_key = (estimate["upper"], estimate["mean"])
    return (
        not evaluation["eligible"],
        *objective_key,
        evaluation["failure_probability"],
        evaluation["nearest_normalized_distance"],
        evaluation["candidate_id"],
    )


def _policy_payload(
    *,
    spec: PolicySelectionSpec,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable,
    baseline: CompiledCandidate,
    status: str,
    selected: Mapping[str, Any] | None,
    ranked: Sequence[Mapping[str, Any]],
    dimensions: Mapping[str, Mapping[str, Any]],
    response_rows: Sequence[SelectorFeatureRow],
    feasibility_rows: Sequence[SelectorFeatureRow],
    exclusions: Mapping[str, int],
    rejection_counts: Mapping[str, int],
    target_scales: Mapping[str, float],
    replicate_noise_floors: Mapping[str, float],
    evaluated_candidate_count: int,
    eligible_candidate_count: int,
) -> PolicyBundle:
    if status not in _POLICY_STATUSES:
        raise ValueError(f"unsupported policy status: {status}")
    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "policy_id": spec.policy_id,
        "status": status,
        "selection_spec_sha256": spec.sha256,
        "compiled_space_sha256": compiled.to_dict()["compiled_space_sha256"],
        "feature_table_sha256": canonical_sha256(features.to_dict()),
        "selection_spec": spec.to_dict(),
        "response_model": {
            "kind": "normalized_inverse_distance_support_points_replicate_floor",
            "feature_dimensions": dict(sorted(dimensions.items())),
            "target_scales": dict(sorted(target_scales.items())),
            "replicate_noise_floors": dict(sorted(replicate_noise_floors.items())),
        },
        "training": {
            "response_row_ids": sorted(row.row_id for row in response_rows),
            "feasibility_row_ids": sorted(row.row_id for row in feasibility_rows),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "activation_guard": {
            "algorithm_id": spec.selection_context.algorithm_id,
            "semantic_cohort_id": spec.selection_context.semantic_cohort_id,
            "graph_sha256": spec.selection_context.graph_sha256,
            "workload_id": spec.selection_context.workload_id,
            "environment_id": spec.selection_context.environment_id,
            "exact_static_features": dict(
                sorted(spec.selection_context.static_features.items())
            ),
            "static_feature_ranges": {
                name: bounds.to_dict()
                for name, bounds in sorted(
                    spec.selection_context.static_feature_ranges.items()
                )
            },
            "maximum_normalized_distance": (
                spec.model.maximum_normalized_distance
            ),
            "maximum_failure_probability": (
                spec.model.maximum_failure_probability
            ),
            "slo_constraints": [
                constraint.to_dict() for constraint in spec.objective.constraints
            ],
            "on_violation": "fallback",
        },
        "selected": selected,
        "fallback": {
            "candidate_id": baseline.candidate_id,
            "deployment_settings": setting_map_to_dict(
                baseline.deployment_settings
            ),
            "requires_engine_restart": baseline.requires_engine_restart,
        },
        "ranked_candidates": list(ranked),
        "audit": {
            "compiled_candidate_count": len(
                [
                    candidate
                    for candidate in compiled.candidates
                    if candidate.semantic_cohort_id
                    == spec.selection_context.semantic_cohort_id
                ]
            ),
            "evaluated_candidate_count": evaluated_candidate_count,
            "eligible_candidate_count": eligible_candidate_count,
            "reported_candidate_count": len(ranked),
            "rejections_by_reason": dict(sorted(rejection_counts.items())),
        },
    }
    return PolicyBundle(
        {**payload, "policy_bundle_sha256": canonical_sha256(payload)}
    )


def select_policy(
    spec: PolicySelectionSpec,
    compiled: CompiledSearchSpace,
    features: SelectorFeatureTable,
) -> PolicyBundle:
    """Select a conservative policy or emit an auditable blocked bundle."""

    compiled_digest = compiled.to_dict()["compiled_space_sha256"]
    if spec.compiled_space_sha256 != compiled_digest:
        raise ValueError("policy selection spec does not match compiled search space")
    if spec.selection_context.algorithm_id != compiled.algorithm_id:
        raise ValueError("policy selection algorithm does not match compiled space")
    candidates = tuple(
        candidate
        for candidate in compiled.candidates
        if candidate.semantic_cohort_id == spec.selection_context.semantic_cohort_id
    )
    if not candidates:
        raise ValueError("compiled space has no candidate in the requested semantic cohort")
    baseline = _resolve_baseline(candidates, spec.baseline_settings)

    required_targets = set(spec.objective.targets)
    context_rows = []
    exclusions: Counter[str] = Counter()
    for row in features.rows:
        if not _row_matches_context(row, spec.selection_context):
            exclusions["context_mismatch"] += 1
            continue
        if any(feature not in row.static_features for feature in spec.model.feature_names):
            exclusions["missing_model_feature"] += 1
            continue
        context_rows.append(row)
    response_rows = tuple(
        row
        for row in context_rows
        if row.eligibility.response_model_fit
        and row.targets.get("run.success") is True
        and required_targets.issubset(row.targets)
        and all(
            isinstance(row.targets[target], (int, float))
            and not isinstance(row.targets[target], bool)
            for target in required_targets
        )
    )
    feasibility_rows = tuple(
        row
        for row in context_rows
        if (
            row.targets.get("run.success") is True
            or row.targets.get("run.success") is False
        )
        and (row.eligibility.response_model_fit or row.eligibility.feasibility_model)
    )
    exclusions["context_rows_without_response_eligibility"] += (
        len(context_rows) - len(response_rows)
    )
    distinct = {
        _configuration_key(row, spec.model.feature_names) for row in response_rows
    }
    evidence_sufficient = (
        len(response_rows) >= spec.model.minimum_response_rows
        and len(distinct) >= spec.model.minimum_distinct_configurations
        and bool(feasibility_rows)
    )
    dimensions = (
        _dimensions(candidates, tuple(context_rows), spec)
        if context_rows
        else {}
    )
    if not evidence_sufficient:
        return _policy_payload(
            spec=spec,
            compiled=compiled,
            features=features,
            baseline=baseline,
            status="insufficient_evidence",
            selected=None,
            ranked=(),
            dimensions=dimensions,
            response_rows=response_rows,
            feasibility_rows=feasibility_rows,
            exclusions=exclusions,
            rejection_counts={},
            target_scales={},
            replicate_noise_floors={},
            evaluated_candidate_count=0,
            eligible_candidate_count=0,
        )

    target_scales = {}
    for target in spec.objective.targets:
        values = [float(row.targets[target]) for row in response_rows]
        span = max(values) - min(values)
        target_scales[target] = max(span, abs(sum(values) / len(values)) * 0.05, 1e-12)
    response_supports = _support_points(response_rows, spec.model.feature_names)
    feasibility_supports = _support_points(
        feasibility_rows, spec.model.feature_names
    )
    replicate_noise_floors = _replicate_noise_floors(
        response_supports, spec.objective
    )
    paired_effects = _paired_objective_effects(
        response_rows, candidates, baseline, spec.objective
    )
    evaluations = [
        _candidate_evaluation(
            candidate,
            spec,
            response_supports,
            feasibility_supports,
            dimensions,
            target_scales,
            replicate_noise_floors,
            paired_effects,
        )
        for candidate in candidates
    ]
    evaluations.sort(key=lambda item: _rank_key(item, spec.objective))
    eligible = [evaluation for evaluation in evaluations if evaluation["eligible"]]
    status = "selected" if eligible else "no_feasible_candidate"
    selected = eligible[0] if eligible else None
    reported = evaluations[: spec.model.ranked_candidate_limit]
    if not any(item["candidate_id"] == baseline.candidate_id for item in reported):
        baseline_evaluation = next(
            item for item in evaluations if item["candidate_id"] == baseline.candidate_id
        )
        reported.append(baseline_evaluation)
    rejection_counts: Counter[str] = Counter(
        reason for evaluation in evaluations for reason in evaluation["rejection_reasons"]
    )
    bundle = _policy_payload(
        spec=spec,
        compiled=compiled,
        features=features,
        baseline=baseline,
        status=status,
        selected=selected,
        ranked=reported,
        dimensions=dimensions,
        response_rows=response_rows,
        feasibility_rows=feasibility_rows,
        exclusions=exclusions,
        rejection_counts=rejection_counts,
        target_scales=target_scales,
        replicate_noise_floors=replicate_noise_floors,
        evaluated_candidate_count=len(evaluations),
        eligible_candidate_count=len(eligible),
    )
    return bundle
