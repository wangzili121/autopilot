"""Content-addressed routing among independently validated policy endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_json,
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.features import deployment_features
from inference_autopilot.policy_selection import PolicyBundle, ResponseConstraint
from inference_autopilot.policy_validation import PolicyHoldoutAssessment
from inference_autopilot.search_space import normalize_setting_map, setting_map_to_dict


_POOL_KEYS = {
    "schema_version",
    "producer",
    "pool_id",
    "semantic_contract",
    "source",
    "fallback_endpoint",
    "policy_endpoints",
    "routing_policy",
    "overlap_audit",
    "runtime_policy_pool_sha256",
}
_DECISION_KEYS = {
    "schema_version",
    "producer",
    "decision_id",
    "status",
    "runtime_policy_pool_sha256",
    "routing_request_sha256",
    "selected_endpoint_id",
    "selected_policy_bundle_sha256",
    "selected_configuration_sha256",
    "fallback_used",
    "reasons",
    "endpoint_evaluations",
    "runtime_policy_decision_sha256",
}


def _same_value(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(left) == float(right)
    return canonical_json({"value": left}) == canonical_json({"value": right})


def _finite_scalar_map(raw: Mapping[str, Any], context: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{context}.{name} must be a scalar")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{context}.{name} must be finite")
        values[name] = value
    return dict(sorted(values.items()))


def _configuration_sha256(settings: Mapping[str, Any]) -> str:
    normalized = normalize_setting_map(settings, "runtime endpoint settings")
    return canonical_sha256(setting_map_to_dict(normalized))


@dataclass(frozen=True, slots=True)
class RuntimePolicyEndpointSpec:
    endpoint_id: str
    policy_bundle_sha256: str
    policy_holdout_assessment_sha256: str
    priority: int

    def __post_init__(self) -> None:
        require_id(self.endpoint_id, "runtime policy endpoint_id")
        require_digest(self.policy_bundle_sha256, "policy_bundle_sha256")
        require_digest(
            self.policy_holdout_assessment_sha256,
            "policy_holdout_assessment_sha256",
        )
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 0
        ):
            raise ValueError("runtime policy endpoint priority must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "policy_bundle_sha256": self.policy_bundle_sha256,
            "policy_holdout_assessment_sha256": (self.policy_holdout_assessment_sha256),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimePolicyEndpointSpec":
        expect_keys(
            raw,
            {
                "endpoint_id",
                "policy_bundle_sha256",
                "policy_holdout_assessment_sha256",
                "priority",
            },
            "runtime policy endpoint spec",
        )
        priority = raw["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("runtime policy endpoint priority must be an integer")
        return cls(
            endpoint_id=str(raw["endpoint_id"]),
            policy_bundle_sha256=str(raw["policy_bundle_sha256"]),
            policy_holdout_assessment_sha256=str(
                raw["policy_holdout_assessment_sha256"]
            ),
            priority=priority,
        )


@dataclass(frozen=True, slots=True)
class RuntimePolicyPoolSpec:
    pool_id: str
    algorithm_id: str
    semantic_cohort_id: str
    graph_sha256: str
    accepted_semantic_classes: tuple[str, ...]
    fallback_endpoint_id: str
    fallback_deployment_settings: Mapping[str, Any]
    policy_endpoints: tuple[RuntimePolicyEndpointSpec, ...]
    require_safe_boundary: bool
    minimum_live_metric_samples: int
    maximum_consecutive_failures: int
    live_constraints: tuple[ResponseConstraint, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported runtime policy pool spec: {self.schema_version}"
            )
        require_id(self.pool_id, "runtime policy pool_id")
        require_id(self.algorithm_id, "runtime policy algorithm_id")
        require_id(self.semantic_cohort_id, "runtime policy semantic_cohort_id")
        require_digest(self.graph_sha256, "runtime policy graph_sha256")
        require_id(self.fallback_endpoint_id, "runtime fallback endpoint_id")
        if (
            not self.accepted_semantic_classes
            or tuple(sorted(set(self.accepted_semantic_classes)))
            != self.accepted_semantic_classes
            or any(not item for item in self.accepted_semantic_classes)
        ):
            raise ValueError("accepted semantic classes must be sorted and unique")
        normalized = normalize_setting_map(
            self.fallback_deployment_settings,
            "runtime fallback deployment settings",
        )
        if not normalized:
            raise ValueError("runtime fallback deployment settings cannot be empty")
        object.__setattr__(self, "fallback_deployment_settings", normalized)
        endpoint_ids = [endpoint.endpoint_id for endpoint in self.policy_endpoints]
        if not endpoint_ids or endpoint_ids != sorted(set(endpoint_ids)):
            raise ValueError("runtime policy endpoints must be sorted and unique")
        if self.fallback_endpoint_id in endpoint_ids:
            raise ValueError("runtime fallback endpoint must be distinct")
        if not isinstance(self.require_safe_boundary, bool):
            raise ValueError("require_safe_boundary must be boolean")
        for name, value in (
            ("minimum_live_metric_samples", self.minimum_live_metric_samples),
            ("maximum_consecutive_failures", self.maximum_consecutive_failures),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        targets = [constraint.target for constraint in self.live_constraints]
        if targets != sorted(set(targets)):
            raise ValueError("runtime live constraints must have sorted unique targets")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "semantic_contract": {
                "algorithm_id": self.algorithm_id,
                "semantic_cohort_id": self.semantic_cohort_id,
                "graph_sha256": self.graph_sha256,
                "accepted_semantic_classes": list(self.accepted_semantic_classes),
            },
            "fallback_endpoint": {
                "endpoint_id": self.fallback_endpoint_id,
                "deployment_settings": setting_map_to_dict(
                    self.fallback_deployment_settings
                ),
            },
            "policy_endpoints": [
                endpoint.to_dict() for endpoint in self.policy_endpoints
            ],
            "routing_policy": {
                "require_safe_boundary": self.require_safe_boundary,
                "minimum_live_metric_samples": self.minimum_live_metric_samples,
                "maximum_consecutive_failures": self.maximum_consecutive_failures,
                "live_constraints": [
                    constraint.to_dict() for constraint in self.live_constraints
                ],
                "on_no_match": "fallback",
                "on_ambiguous_match": "fallback",
                "on_health_violation": "fallback",
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimePolicyPoolSpec":
        expect_keys(
            raw,
            {
                "schema_version",
                "pool_id",
                "semantic_contract",
                "fallback_endpoint",
                "policy_endpoints",
                "routing_policy",
            },
            "runtime policy pool spec",
        )
        semantic = require_object(raw["semantic_contract"], "runtime semantic contract")
        expect_keys(
            semantic,
            {
                "algorithm_id",
                "semantic_cohort_id",
                "graph_sha256",
                "accepted_semantic_classes",
            },
            "runtime semantic contract",
        )
        classes = semantic["accepted_semantic_classes"]
        if not isinstance(classes, list) or any(
            not isinstance(item, str) for item in classes
        ):
            raise ValueError("accepted_semantic_classes must be strings")
        fallback = require_object(raw["fallback_endpoint"], "runtime fallback endpoint")
        expect_keys(
            fallback,
            {"endpoint_id", "deployment_settings"},
            "runtime fallback endpoint",
        )
        endpoints = raw["policy_endpoints"]
        if not isinstance(endpoints, list) or any(
            not isinstance(item, Mapping) for item in endpoints
        ):
            raise ValueError("runtime policy endpoints must be objects")
        routing = require_object(raw["routing_policy"], "runtime routing policy")
        expect_keys(
            routing,
            {
                "require_safe_boundary",
                "minimum_live_metric_samples",
                "maximum_consecutive_failures",
                "live_constraints",
                "on_no_match",
                "on_ambiguous_match",
                "on_health_violation",
            },
            "runtime routing policy",
        )
        if any(
            routing[name] != "fallback"
            for name in (
                "on_no_match",
                "on_ambiguous_match",
                "on_health_violation",
            )
        ):
            raise ValueError("runtime policy violations must fail closed to fallback")
        constraints = routing["live_constraints"]
        if not isinstance(constraints, list) or any(
            not isinstance(item, Mapping) for item in constraints
        ):
            raise ValueError("runtime live constraints must be objects")
        for name in ("minimum_live_metric_samples", "maximum_consecutive_failures"):
            if isinstance(routing[name], bool) or not isinstance(routing[name], int):
                raise ValueError(f"runtime routing {name} must be an integer")
        if not isinstance(routing["require_safe_boundary"], bool):
            raise ValueError("runtime require_safe_boundary must be boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            pool_id=str(raw["pool_id"]),
            algorithm_id=str(semantic["algorithm_id"]),
            semantic_cohort_id=str(semantic["semantic_cohort_id"]),
            graph_sha256=str(semantic["graph_sha256"]),
            accepted_semantic_classes=tuple(classes),
            fallback_endpoint_id=str(fallback["endpoint_id"]),
            fallback_deployment_settings=require_object(
                fallback["deployment_settings"], "runtime fallback settings"
            ),
            policy_endpoints=tuple(
                RuntimePolicyEndpointSpec.from_dict(item) for item in endpoints
            ),
            require_safe_boundary=routing["require_safe_boundary"],
            minimum_live_metric_samples=routing["minimum_live_metric_samples"],
            maximum_consecutive_failures=routing["maximum_consecutive_failures"],
            live_constraints=tuple(
                ResponseConstraint.from_dict(item) for item in constraints
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimePolicyPool:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expect_keys(self.payload, _POOL_KEYS, "runtime policy pool")
        raw = dict(self.payload)
        digest = str(raw.pop("runtime_policy_pool_sha256", ""))
        require_digest(digest, "runtime_policy_pool_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("runtime policy pool SHA256 does not match its content")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported runtime policy pool: {raw['schema_version']}"
            )
        require_id(str(raw["pool_id"]), "runtime policy pool_id")
        semantic = require_object(raw["semantic_contract"], "runtime semantic contract")
        for name in ("algorithm_id", "semantic_cohort_id"):
            require_id(str(semantic.get(name, "")), f"runtime {name}")
        require_digest(str(semantic.get("graph_sha256", "")), "runtime graph_sha256")
        classes = semantic.get("accepted_semantic_classes")
        if (
            not isinstance(classes, list)
            or classes != sorted(set(classes))
            or not classes
            or any(not isinstance(item, str) or not item for item in classes)
        ):
            raise ValueError("runtime semantic classes must be sorted and unique")
        source = require_object(raw["source"], "runtime policy pool source")
        expect_keys(
            source,
            {"pool_spec_sha256", "validated_bindings"},
            "runtime policy pool source",
        )
        require_digest(str(source.get("pool_spec_sha256", "")), "pool_spec_sha256")
        bindings = source.get("validated_bindings")
        endpoints = raw["policy_endpoints"]
        if not isinstance(bindings, list) or not isinstance(endpoints, list):
            raise ValueError(
                "runtime policy pool bindings and endpoints must be arrays"
            )
        endpoint_ids = [str(item.get("endpoint_id", "")) for item in endpoints]
        if endpoint_ids != sorted(set(endpoint_ids)) or not endpoint_ids:
            raise ValueError(
                "runtime policy pool endpoint ids must be sorted and unique"
            )
        fallback = require_object(raw["fallback_endpoint"], "runtime fallback endpoint")
        expect_keys(
            fallback,
            {
                "endpoint_id",
                "deployment_settings",
                "configuration_sha256",
                "requires_engine_restart",
            },
            "runtime fallback endpoint",
        )
        require_id(str(fallback.get("endpoint_id", "")), "runtime fallback endpoint_id")
        if fallback["endpoint_id"] in endpoint_ids:
            raise ValueError("runtime fallback endpoint must be distinct")
        if not isinstance(fallback["requires_engine_restart"], bool):
            raise ValueError("runtime fallback restart flag must be boolean")
        endpoint_by_id: dict[str, Mapping[str, Any]] = {}
        for endpoint in endpoints:
            expect_keys(
                endpoint,
                {
                    "endpoint_id",
                    "priority",
                    "policy_id",
                    "policy_bundle_sha256",
                    "policy_holdout_assessment_sha256",
                    "selected_candidate_id",
                    "deployment_settings",
                    "configuration_sha256",
                    "requires_engine_restart",
                    "activation_guard",
                    "validation",
                },
                "runtime policy endpoint",
            )
            for name in ("endpoint_id", "policy_id", "selected_candidate_id"):
                require_id(str(endpoint[name]), f"runtime endpoint {name}")
            for name in (
                "policy_bundle_sha256",
                "policy_holdout_assessment_sha256",
            ):
                require_digest(str(endpoint[name]), f"runtime endpoint {name}")
            if (
                isinstance(endpoint["priority"], bool)
                or not isinstance(endpoint["priority"], int)
                or endpoint["priority"] < 0
            ):
                raise ValueError("runtime endpoint priority must be non-negative")
            if not isinstance(endpoint["requires_engine_restart"], bool):
                raise ValueError("runtime endpoint restart flag must be boolean")
            require_object(endpoint["activation_guard"], "runtime activation guard")
            validation = require_object(
                endpoint["validation"], "runtime endpoint validation"
            )
            expect_keys(
                validation,
                {
                    "assessment_id",
                    "improvement_over_fallback_fraction",
                    "regret_to_measured_oracle_fraction",
                    "successful_selected_replicates",
                },
                "runtime endpoint validation",
            )
            require_id(str(validation["assessment_id"]), "runtime assessment_id")
            for name in (
                "improvement_over_fallback_fraction",
                "regret_to_measured_oracle_fraction",
            ):
                value = validation[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(float(value))
                ):
                    raise ValueError(f"runtime validation {name} must be finite")
            replicates = validation["successful_selected_replicates"]
            if (
                isinstance(replicates, bool)
                or not isinstance(replicates, int)
                or replicates <= 0
            ):
                raise ValueError("runtime validation replicates must be positive")
            endpoint_by_id[str(endpoint["endpoint_id"])] = endpoint
        for endpoint in [fallback, *endpoints]:
            settings = require_object(
                endpoint.get("deployment_settings"), "runtime endpoint settings"
            )
            expected = _configuration_sha256(settings)
            require_digest(
                str(endpoint.get("configuration_sha256", "")),
                "configuration_sha256",
            )
            if endpoint["configuration_sha256"] != expected:
                raise ValueError("runtime endpoint configuration SHA256 does not match")
        binding_ids: list[str] = []
        for binding in bindings:
            binding = require_object(binding, "runtime validated binding")
            expect_keys(
                binding,
                {
                    "endpoint_id",
                    "policy_bundle_sha256",
                    "policy_holdout_assessment_sha256",
                },
                "runtime validated binding",
            )
            endpoint_id = str(binding["endpoint_id"])
            require_id(endpoint_id, "runtime binding endpoint_id")
            binding_ids.append(endpoint_id)
            endpoint = endpoint_by_id.get(endpoint_id)
            if endpoint is None or any(
                binding[name] != endpoint[name]
                for name in (
                    "policy_bundle_sha256",
                    "policy_holdout_assessment_sha256",
                )
            ):
                raise ValueError("runtime validated binding does not match endpoint")
        if binding_ids != endpoint_ids:
            raise ValueError("runtime validated bindings must match policy endpoints")
        routing = require_object(raw["routing_policy"], "runtime routing policy")
        expect_keys(
            routing,
            {
                "require_safe_boundary",
                "minimum_live_metric_samples",
                "maximum_consecutive_failures",
                "live_constraints",
                "on_no_match",
                "on_ambiguous_match",
                "on_health_violation",
            },
            "runtime routing policy",
        )
        if not isinstance(routing["require_safe_boundary"], bool):
            raise ValueError("runtime safe-boundary policy must be boolean")
        for name in ("minimum_live_metric_samples", "maximum_consecutive_failures"):
            value = routing[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"runtime routing {name} must be non-negative")
        if any(
            routing[name] != "fallback"
            for name in (
                "on_no_match",
                "on_ambiguous_match",
                "on_health_violation",
            )
        ):
            raise ValueError("runtime routing violations must use fallback")
        constraints = routing["live_constraints"]
        if not isinstance(constraints, list) or any(
            not isinstance(item, Mapping) for item in constraints
        ):
            raise ValueError("runtime live constraints must be objects")
        parsed_constraints = [
            ResponseConstraint.from_dict(item) for item in constraints
        ]
        targets = [constraint.target for constraint in parsed_constraints]
        if targets != sorted(set(targets)):
            raise ValueError(
                "runtime live constraint targets must be sorted and unique"
            )
        overlap = require_object(raw["overlap_audit"], "runtime overlap audit")
        expect_keys(
            overlap,
            {"checked_pair_count", "overlapping_pair_count", "overlapping_pairs"},
            "runtime overlap audit",
        )
        pairs = overlap["overlapping_pairs"]
        if not isinstance(pairs, list) or overlap["overlapping_pair_count"] != len(
            pairs
        ):
            raise ValueError("runtime overlap audit count does not match pairs")
        expected_pair_count = len(endpoints) * (len(endpoints) - 1) // 2
        if overlap["checked_pair_count"] != expected_pair_count:
            raise ValueError(
                "runtime overlap checked-pair count does not match endpoints"
            )
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def sha256(self) -> str:
        return str(self.payload["runtime_policy_pool_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    def audit(self) -> dict[str, Any]:
        return {
            "pool_id": self.payload["pool_id"],
            "policy_endpoint_count": len(self.payload["policy_endpoints"]),
            "fallback_endpoint_id": self.payload["fallback_endpoint"]["endpoint_id"],
            **dict(self.payload["overlap_audit"]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimePolicyPool":
        return cls(raw)


def _guard_constraints(guard: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    constraints: dict[str, tuple[str, Any]] = {}
    for name, value in guard["exact_static_features"].items():
        constraints[name] = ("exact", value)
    for name, bounds in guard["static_feature_ranges"].items():
        constraints[name] = ("range", bounds)
    return constraints


def _guards_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for name in (
        "algorithm_id",
        "semantic_cohort_id",
        "graph_sha256",
        "workload_id",
        "environment_id",
    ):
        if left[name] != right[name]:
            return False
    left_constraints = _guard_constraints(left)
    right_constraints = _guard_constraints(right)
    for name in set(left_constraints) & set(right_constraints):
        left_kind, left_value = left_constraints[name]
        right_kind, right_value = right_constraints[name]
        if left_kind == right_kind == "exact":
            if not _same_value(left_value, right_value):
                return False
        elif left_kind == "range" and right_kind == "range":
            if float(left_value["maximum"]) < float(right_value["minimum"]) or float(
                right_value["maximum"]
            ) < float(left_value["minimum"]):
                return False
        else:
            exact = left_value if left_kind == "exact" else right_value
            bounds = left_value if left_kind == "range" else right_value
            if (
                isinstance(exact, bool)
                or not isinstance(exact, (int, float))
                or not float(bounds["minimum"])
                <= float(exact)
                <= float(bounds["maximum"])
            ):
                return False
    return True


def compile_runtime_policy_pool(
    spec: RuntimePolicyPoolSpec,
    policies: Sequence[PolicyBundle],
    assessments: Sequence[PolicyHoldoutAssessment],
) -> RuntimePolicyPool:
    """Bind validated policies to prewarmed endpoints and reject ambiguity."""

    policies_by_digest: dict[str, PolicyBundle] = {}
    for policy in policies:
        digest = str(policy.payload["policy_bundle_sha256"])
        if digest in policies_by_digest:
            raise ValueError(f"duplicate runtime policy bundle: {digest}")
        policies_by_digest[digest] = policy
    assessments_by_digest: dict[str, PolicyHoldoutAssessment] = {}
    for assessment in assessments:
        digest = str(assessment.payload["policy_holdout_assessment_sha256"])
        if digest in assessments_by_digest:
            raise ValueError(f"duplicate runtime holdout assessment: {digest}")
        assessments_by_digest[digest] = assessment

    required_policies = {item.policy_bundle_sha256 for item in spec.policy_endpoints}
    required_assessments = {
        item.policy_holdout_assessment_sha256 for item in spec.policy_endpoints
    }
    if set(policies_by_digest) != required_policies:
        raise ValueError("runtime policy artifacts do not exactly match pool spec")
    if set(assessments_by_digest) != required_assessments:
        raise ValueError("runtime holdout artifacts do not exactly match pool spec")

    fallback_restart_flags = {
        bool(policy.payload["fallback"]["requires_engine_restart"])
        for policy in policies
    }
    if len(fallback_restart_flags) != 1:
        raise ValueError("runtime policies disagree on fallback restart semantics")

    fallback_settings = setting_map_to_dict(spec.fallback_deployment_settings)
    endpoints: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for binding in spec.policy_endpoints:
        policy = policies_by_digest[binding.policy_bundle_sha256]
        assessment = assessments_by_digest[binding.policy_holdout_assessment_sha256]
        if policy.status != "selected":
            raise ValueError(
                f"runtime endpoint {binding.endpoint_id} policy is not selected"
            )
        if assessment.status != "validated":
            raise ValueError(
                f"runtime endpoint {binding.endpoint_id} lacks validated holdout"
            )
        if assessment.payload["policy_bundle_sha256"] != binding.policy_bundle_sha256:
            raise ValueError("runtime holdout assessment is bound to another policy")
        guard = require_object(policy.payload["activation_guard"], "activation guard")
        semantic = policy.payload["selection_spec"]["selection_context"]
        expected_semantic = {
            "algorithm_id": spec.algorithm_id,
            "semantic_cohort_id": spec.semantic_cohort_id,
            "graph_sha256": spec.graph_sha256,
        }
        if any(guard[name] != value for name, value in expected_semantic.items()):
            raise ValueError("runtime policy does not match pool semantic contract")
        if tuple(semantic["accepted_evidence_semantic_classes"]) != (
            spec.accepted_semantic_classes
        ):
            raise ValueError("runtime policy semantic classes do not match pool spec")
        fallback = require_object(policy.payload["fallback"], "policy fallback")
        if canonical_json(fallback["deployment_settings"]) != canonical_json(
            fallback_settings
        ):
            raise ValueError("runtime policies do not share the declared fallback")

        selected = require_object(policy.payload["selected"], "selected policy")
        holdout_selected = require_object(
            assessment.payload["selected"], "holdout selected candidate"
        )
        holdout_fallback = require_object(
            assessment.payload["fallback"], "holdout fallback candidate"
        )
        if selected["candidate_id"] != holdout_selected.get("candidate_id"):
            raise ValueError("runtime holdout selected candidate does not match policy")
        if fallback["candidate_id"] != holdout_fallback.get("candidate_id"):
            raise ValueError("runtime holdout fallback candidate does not match policy")
        if not holdout_selected.get("eligible") or not holdout_fallback.get("eligible"):
            raise ValueError("runtime holdout candidate is not constraint eligible")
        if canonical_json(selected["deployment_settings"]) != canonical_json(
            holdout_selected.get("deployment_settings")
        ):
            raise ValueError("runtime holdout selected settings do not match policy")
        comparison = require_object(
            assessment.payload["comparison"], "runtime holdout comparison"
        )
        selected_settings = require_object(
            selected["deployment_settings"], "selected deployment settings"
        )
        endpoint = {
            "endpoint_id": binding.endpoint_id,
            "priority": binding.priority,
            "policy_id": policy.payload["policy_id"],
            "policy_bundle_sha256": binding.policy_bundle_sha256,
            "policy_holdout_assessment_sha256": (
                binding.policy_holdout_assessment_sha256
            ),
            "selected_candidate_id": selected["candidate_id"],
            "deployment_settings": selected_settings,
            "configuration_sha256": _configuration_sha256(selected_settings),
            "requires_engine_restart": bool(selected["requires_engine_restart"]),
            "activation_guard": guard,
            "validation": {
                "assessment_id": assessment.payload["assessment_id"],
                "improvement_over_fallback_fraction": comparison[
                    "improvement_over_fallback_fraction"
                ],
                "regret_to_measured_oracle_fraction": comparison[
                    "regret_to_measured_oracle_fraction"
                ],
                "successful_selected_replicates": holdout_selected[
                    "successful_replicates"
                ],
            },
        }
        endpoints.append(endpoint)
        sources.append(
            {
                "endpoint_id": binding.endpoint_id,
                "policy_bundle_sha256": binding.policy_bundle_sha256,
                "policy_holdout_assessment_sha256": (
                    binding.policy_holdout_assessment_sha256
                ),
            }
        )

    endpoints.sort(key=lambda item: item["endpoint_id"])
    sources.sort(key=lambda item: item["endpoint_id"])
    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(endpoints):
        for right in endpoints[index + 1 :]:
            if not _guards_overlap(left["activation_guard"], right["activation_guard"]):
                continue
            if left["priority"] == right["priority"]:
                raise ValueError(
                    "overlapping runtime policy guards require distinct priorities: "
                    f"{left['endpoint_id']}, {right['endpoint_id']}"
                )
            winner = left if left["priority"] > right["priority"] else right
            overlaps.append(
                {
                    "left_endpoint_id": left["endpoint_id"],
                    "right_endpoint_id": right["endpoint_id"],
                    "resolution": "higher_priority",
                    "winner_endpoint_id": winner["endpoint_id"],
                }
            )

    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "pool_id": spec.pool_id,
        "semantic_contract": {
            "algorithm_id": spec.algorithm_id,
            "semantic_cohort_id": spec.semantic_cohort_id,
            "graph_sha256": spec.graph_sha256,
            "accepted_semantic_classes": list(spec.accepted_semantic_classes),
        },
        "source": {
            "pool_spec_sha256": spec.sha256,
            "validated_bindings": sources,
        },
        "fallback_endpoint": {
            "endpoint_id": spec.fallback_endpoint_id,
            "deployment_settings": fallback_settings,
            "configuration_sha256": _configuration_sha256(fallback_settings),
            "requires_engine_restart": bool(next(iter(fallback_restart_flags))),
        },
        "policy_endpoints": endpoints,
        "routing_policy": spec.to_dict()["routing_policy"],
        "overlap_audit": {
            "checked_pair_count": len(endpoints) * (len(endpoints) - 1) // 2,
            "overlapping_pair_count": len(overlaps),
            "overlapping_pairs": overlaps,
        },
    }
    return RuntimePolicyPool(
        {**payload, "runtime_policy_pool_sha256": canonical_sha256(payload)}
    )


@dataclass(frozen=True, slots=True)
class RuntimeRoutingRequest:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "decision_id",
            "runtime_policy_pool_sha256",
            "safe_boundary",
            "context",
            "endpoint_states",
            "routing_request_sha256",
        }
        expect_keys(self.payload, expected, "runtime routing request")
        raw = dict(self.payload)
        digest = str(raw.pop("routing_request_sha256", ""))
        require_digest(digest, "routing_request_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("runtime routing request SHA256 does not match")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported runtime routing request: {raw['schema_version']}"
            )
        require_id(str(raw["decision_id"]), "runtime decision_id")
        require_digest(
            str(raw["runtime_policy_pool_sha256"]),
            "runtime_policy_pool_sha256",
        )
        if not isinstance(raw["safe_boundary"], bool):
            raise ValueError("runtime safe_boundary must be boolean")
        context = require_object(raw["context"], "runtime routing context")
        expect_keys(
            context,
            {
                "algorithm_id",
                "semantic_cohort_id",
                "semantic_class",
                "graph_sha256",
                "workload_id",
                "environment_id",
                "static_features",
            },
            "runtime routing context",
        )
        for name in (
            "algorithm_id",
            "semantic_cohort_id",
            "semantic_class",
            "workload_id",
            "environment_id",
        ):
            require_id(str(context[name]), f"runtime context {name}")
        require_digest(str(context["graph_sha256"]), "runtime context graph_sha256")
        static = _finite_scalar_map(
            require_object(context["static_features"], "runtime static features"),
            "runtime static features",
        )
        if any(name.startswith("deployment.") for name in static):
            raise ValueError("runtime request cannot attest deployment features")
        states = raw["endpoint_states"]
        if not isinstance(states, list) or any(
            not isinstance(item, Mapping) for item in states
        ):
            raise ValueError("runtime endpoint states must be objects")
        endpoint_ids: list[str] = []
        for state in states:
            expect_keys(
                state,
                {
                    "endpoint_id",
                    "configuration_sha256",
                    "ready",
                    "accepting_requests",
                    "consecutive_failures",
                    "live_metric_samples",
                    "live_metrics",
                },
                "runtime endpoint state",
            )
            endpoint_id = str(state["endpoint_id"])
            require_id(endpoint_id, "runtime state endpoint_id")
            endpoint_ids.append(endpoint_id)
            require_digest(
                str(state["configuration_sha256"]), "runtime state configuration_sha256"
            )
            if not isinstance(state["ready"], bool) or not isinstance(
                state["accepting_requests"], bool
            ):
                raise ValueError("runtime endpoint readiness flags must be boolean")
            for name in ("consecutive_failures", "live_metric_samples"):
                value = state[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"runtime endpoint {name} must be non-negative")
            _finite_scalar_map(
                require_object(state["live_metrics"], "runtime live metrics"),
                "runtime live metrics",
            )
        if endpoint_ids != sorted(set(endpoint_ids)):
            raise ValueError("runtime endpoint states must be sorted and unique")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def sha256(self) -> str:
        return str(self.payload["routing_request_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeRoutingRequest":
        return cls(raw)


def build_runtime_routing_request(
    pool: RuntimePolicyPool,
    *,
    decision_id: str,
    safe_boundary: bool,
    context: Mapping[str, Any],
    endpoint_states: Sequence[Mapping[str, Any]],
) -> RuntimeRoutingRequest:
    """Create a request artifact bound to one immutable policy pool."""

    payload = {
        "schema_version": "1.0",
        "decision_id": decision_id,
        "runtime_policy_pool_sha256": pool.sha256,
        "safe_boundary": safe_boundary,
        "context": dict(context),
        "endpoint_states": sorted(
            (dict(state) for state in endpoint_states),
            key=lambda state: str(state.get("endpoint_id", "")),
        ),
    }
    return RuntimeRoutingRequest(
        {**payload, "routing_request_sha256": canonical_sha256(payload)}
    )


@dataclass(frozen=True, slots=True)
class RuntimePolicyDecision:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expect_keys(self.payload, _DECISION_KEYS, "runtime policy decision")
        raw = dict(self.payload)
        digest = str(raw.pop("runtime_policy_decision_sha256", ""))
        require_digest(digest, "runtime_policy_decision_sha256")
        if digest != canonical_sha256(raw):
            raise ValueError("runtime policy decision SHA256 does not match")
        if raw["schema_version"] != "1.0":
            raise ValueError(
                f"unsupported runtime policy decision: {raw['schema_version']}"
            )
        if raw["status"] not in {"selected", "fallback", "unavailable"}:
            raise ValueError(f"unsupported runtime policy status: {raw['status']}")
        require_id(str(raw["decision_id"]), "runtime decision_id")
        for name in ("runtime_policy_pool_sha256", "routing_request_sha256"):
            require_digest(str(raw[name]), name)
        if not isinstance(raw["fallback_used"], bool):
            raise ValueError("runtime fallback_used must be boolean")
        if (
            not isinstance(raw["reasons"], list)
            or any(
                not isinstance(reason, str) or not reason for reason in raw["reasons"]
            )
            or raw["reasons"] != sorted(set(raw["reasons"]))
        ):
            raise ValueError("runtime decision reasons must be sorted and unique")
        evaluations = raw["endpoint_evaluations"]
        if not isinstance(evaluations, list) or any(
            not isinstance(item, Mapping) for item in evaluations
        ):
            raise ValueError("runtime endpoint evaluations must be an array")
        evaluation_ids: list[str] = []
        for evaluation in evaluations:
            expect_keys(
                evaluation,
                {"endpoint_id", "priority", "eligible", "reasons"},
                "runtime endpoint evaluation",
            )
            endpoint_id = str(evaluation["endpoint_id"])
            require_id(endpoint_id, "runtime evaluation endpoint_id")
            evaluation_ids.append(endpoint_id)
            if (
                isinstance(evaluation["priority"], bool)
                or not isinstance(evaluation["priority"], int)
                or evaluation["priority"] < 0
                or not isinstance(evaluation["eligible"], bool)
            ):
                raise ValueError("runtime endpoint evaluation fields are invalid")
            reasons = evaluation["reasons"]
            if (
                not isinstance(reasons, list)
                or any(not isinstance(reason, str) or not reason for reason in reasons)
                or reasons != sorted(set(reasons))
            ):
                raise ValueError("runtime endpoint reasons must be sorted and unique")
        if evaluation_ids != sorted(set(evaluation_ids)):
            raise ValueError("runtime endpoint evaluations must be sorted and unique")
        if raw["status"] == "unavailable":
            if (
                any(
                    raw[name] is not None
                    for name in (
                        "selected_endpoint_id",
                        "selected_policy_bundle_sha256",
                        "selected_configuration_sha256",
                    )
                )
                or raw["fallback_used"]
            ):
                raise ValueError("unavailable runtime decision cannot select endpoint")
        else:
            require_id(str(raw["selected_endpoint_id"]), "selected endpoint_id")
            require_digest(
                str(raw["selected_configuration_sha256"]),
                "selected_configuration_sha256",
            )
        if raw["status"] == "selected":
            require_digest(
                str(raw["selected_policy_bundle_sha256"]),
                "selected_policy_bundle_sha256",
            )
            if raw["fallback_used"]:
                raise ValueError("selected runtime policy cannot mark fallback used")
        elif raw["selected_policy_bundle_sha256"] is not None:
            raise ValueError("fallback runtime decision cannot name a policy bundle")
        elif raw["status"] == "fallback" and not raw["fallback_used"]:
            raise ValueError("fallback runtime decision must mark fallback used")
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    def audit(self) -> dict[str, Any]:
        return {
            "decision_id": self.payload["decision_id"],
            "status": self.status,
            "selected_endpoint_id": self.payload["selected_endpoint_id"],
            "fallback_used": self.payload["fallback_used"],
            "reasons": list(self.payload["reasons"]),
            "eligible_policy_endpoint_count": sum(
                1 for item in self.payload["endpoint_evaluations"] if item["eligible"]
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimePolicyDecision":
        return cls(raw)


def _constraint_passes(constraint: Mapping[str, Any], value: float) -> bool:
    if constraint["operator"] == "<=":
        return value <= float(constraint["value"])
    return value >= float(constraint["value"])


def _endpoint_health_reasons(
    endpoint: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    routing: Mapping[str, Any],
    *,
    apply_live_constraints: bool,
) -> list[str]:
    if state is None:
        return ["endpoint_state_missing"]
    reasons: list[str] = []
    if state["configuration_sha256"] != endpoint["configuration_sha256"]:
        reasons.append("endpoint_configuration_mismatch")
    if not state["ready"]:
        reasons.append("endpoint_not_ready")
    if not state["accepting_requests"]:
        reasons.append("endpoint_not_accepting_requests")
    if state["consecutive_failures"] > routing["maximum_consecutive_failures"]:
        reasons.append("endpoint_failure_threshold_exceeded")
    if (
        apply_live_constraints
        and state["live_metric_samples"] >= routing["minimum_live_metric_samples"]
    ):
        for constraint in routing["live_constraints"]:
            target = constraint["target"]
            value = state["live_metrics"].get(target)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"live_metric_missing:{target}")
            elif not _constraint_passes(constraint, float(value)):
                reasons.append(f"live_constraint_violation:{target}")
    return sorted(set(reasons))


def _guard_reasons(
    pool: RuntimePolicyPool,
    endpoint: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[str]:
    semantic = pool.payload["semantic_contract"]
    reasons: list[str] = []
    for name in ("algorithm_id", "semantic_cohort_id", "graph_sha256"):
        if context[name] != semantic[name]:
            reasons.append(f"semantic_mismatch:{name}")
    if context["semantic_class"] not in semantic["accepted_semantic_classes"]:
        reasons.append("semantic_class_not_accepted")
    guard = endpoint["activation_guard"]
    for name in ("workload_id", "environment_id"):
        if context[name] != guard[name]:
            reasons.append(f"identity_mismatch:{name}")
    features = deployment_features(endpoint["deployment_settings"])
    features.update(context["static_features"])
    for name, expected in guard["exact_static_features"].items():
        if name not in features:
            reasons.append(f"exact_feature_missing:{name}")
        elif not _same_value(features[name], expected):
            reasons.append(f"exact_feature_mismatch:{name}")
    for name, bounds in guard["static_feature_ranges"].items():
        value = features.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"range_feature_missing:{name}")
        elif not float(bounds["minimum"]) <= float(value) <= float(bounds["maximum"]):
            reasons.append(f"range_feature_out_of_bounds:{name}")
    return sorted(set(reasons))


def route_runtime_policy(
    pool: RuntimePolicyPool, request: RuntimeRoutingRequest
) -> RuntimePolicyDecision:
    """Choose one validated endpoint or fail closed to the declared fallback."""

    if request.payload["runtime_policy_pool_sha256"] != pool.sha256:
        raise ValueError("runtime routing request is bound to another policy pool")
    expected_ids = {
        pool.payload["fallback_endpoint"]["endpoint_id"],
        *(item["endpoint_id"] for item in pool.payload["policy_endpoints"]),
    }
    states = {item["endpoint_id"]: item for item in request.payload["endpoint_states"]}
    if set(states) != expected_ids:
        raise ValueError("runtime endpoint state set does not match policy pool")
    routing = pool.payload["routing_policy"]
    context = request.payload["context"]
    evaluations: list[dict[str, Any]] = []
    eligible: list[Mapping[str, Any]] = []
    safe_boundary = (
        request.payload["safe_boundary"] or not routing["require_safe_boundary"]
    )
    for endpoint in pool.payload["policy_endpoints"]:
        reasons = [] if safe_boundary else ["unsafe_policy_boundary"]
        reasons.extend(_guard_reasons(pool, endpoint, context))
        reasons.extend(
            _endpoint_health_reasons(
                endpoint,
                states[endpoint["endpoint_id"]],
                routing,
                apply_live_constraints=True,
            )
        )
        reasons = sorted(set(reasons))
        evaluation = {
            "endpoint_id": endpoint["endpoint_id"],
            "priority": endpoint["priority"],
            "eligible": not reasons,
            "reasons": reasons,
        }
        evaluations.append(evaluation)
        if not reasons:
            eligible.append(endpoint)

    decision_reasons: list[str] = []
    selected: Mapping[str, Any] | None = None
    if eligible:
        maximum_priority = max(endpoint["priority"] for endpoint in eligible)
        winners = [
            endpoint
            for endpoint in eligible
            if endpoint["priority"] == maximum_priority
        ]
        if len(winners) == 1:
            selected = winners[0]
        else:
            decision_reasons.append("ambiguous_policy_match")
    else:
        decision_reasons.append("no_eligible_policy_endpoint")

    if selected is not None:
        status = "selected"
        selected_endpoint_id = selected["endpoint_id"]
        selected_policy_digest = selected["policy_bundle_sha256"]
        selected_configuration_digest = selected["configuration_sha256"]
        fallback_used = False
    else:
        fallback = pool.payload["fallback_endpoint"]
        fallback_reasons = _endpoint_health_reasons(
            fallback,
            states[fallback["endpoint_id"]],
            routing,
            apply_live_constraints=False,
        )
        if fallback_reasons:
            status = "unavailable"
            selected_endpoint_id = None
            selected_configuration_digest = None
            fallback_used = False
            decision_reasons.extend(f"fallback_{reason}" for reason in fallback_reasons)
        else:
            status = "fallback"
            selected_endpoint_id = fallback["endpoint_id"]
            selected_configuration_digest = fallback["configuration_sha256"]
            fallback_used = True
        selected_policy_digest = None

    payload = {
        "schema_version": "1.0",
        "producer": "inference-autopilot/0.1.0",
        "decision_id": request.payload["decision_id"],
        "status": status,
        "runtime_policy_pool_sha256": pool.sha256,
        "routing_request_sha256": request.sha256,
        "selected_endpoint_id": selected_endpoint_id,
        "selected_policy_bundle_sha256": selected_policy_digest,
        "selected_configuration_sha256": selected_configuration_digest,
        "fallback_used": fallback_used,
        "reasons": sorted(set(decision_reasons)),
        "endpoint_evaluations": evaluations,
    }
    return RuntimePolicyDecision(
        {**payload, "runtime_policy_decision_sha256": canonical_sha256(payload)}
    )
