"""Censored boundary search for ordered, restart-scoped capacity knobs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.evidence import SourceArtifact


_OUTCOMES = {"success", "resource_exhausted"}
_STATES = {
    "unobserved",
    "pending_success_confirmation",
    "pending_resource_exhausted_confirmation",
    "confirmed_feasible",
    "confirmed_resource_exhausted",
    "mixed",
}
_REJECTION_REASONS = {
    "mixed_outcomes_at_value",
    "non_monotone_resource_boundary",
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _positive_int(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _scalar_map(raw: Mapping[str, Any], context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} keys must be non-empty strings")
        if isinstance(value, bool) or isinstance(value, (str, int)):
            result[name] = value
        elif isinstance(value, float) and isfinite(value):
            result[name] = value
        else:
            raise ValueError(f"{context} must contain finite scalar values")
    canonical_json(result)
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class OrderedProbeObservation:
    attempt_id: str
    value: int
    outcome: str
    source: SourceArtifact
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_id(self.attempt_id, "ordered probe attempt_id")
        _positive_int(self.value, "ordered probe value")
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unsupported ordered probe outcome: {self.outcome}")
        _scalar_map(self.details, "ordered probe details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "value": self.value,
            "outcome": self.outcome,
            "source": self.source.to_dict(),
            "details": _scalar_map(self.details, "ordered probe details"),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OrderedProbeObservation":
        _expect_exact_keys(
            raw,
            {"attempt_id", "value", "outcome", "source", "details"},
            "ordered probe observation",
        )
        if not isinstance(raw["source"], Mapping):
            raise ValueError("ordered probe source must be an object")
        if not isinstance(raw["details"], Mapping):
            raise ValueError("ordered probe details must be an object")
        return cls(
            attempt_id=str(raw["attempt_id"]),
            value=raw["value"],
            outcome=str(raw["outcome"]),
            source=SourceArtifact.from_dict(raw["source"]),
            details=_scalar_map(raw["details"], "ordered probe details"),
        )


@dataclass(frozen=True, slots=True)
class OrderedFeasibilitySpec:
    probe_id: str
    parameter_name: str
    ordered_values: tuple[int, ...]
    context: Mapping[str, Any]
    required_success_observations: int
    required_resource_exhausted_observations: int
    observations: tuple[OrderedProbeObservation, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported ordered feasibility spec: {self.schema_version}"
            )
        require_id(self.probe_id, "ordered feasibility probe_id")
        require_id(self.parameter_name, "ordered feasibility parameter_name")
        if (
            not self.ordered_values
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.ordered_values
            )
            or tuple(sorted(set(self.ordered_values))) != self.ordered_values
        ):
            raise ValueError("ordered feasibility values must be positive and increasing")
        if not self.context:
            raise ValueError("ordered feasibility context cannot be empty")
        _scalar_map(self.context, "ordered feasibility context")
        _positive_int(
            self.required_success_observations,
            "required_success_observations",
        )
        _positive_int(
            self.required_resource_exhausted_observations,
            "required_resource_exhausted_observations",
        )
        attempt_ids = [observation.attempt_id for observation in self.observations]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("ordered probe attempt ids must be unique")
        outside = sorted(
            {
                observation.value
                for observation in self.observations
                if observation.value not in self.ordered_values
            }
        )
        if outside:
            raise ValueError(f"ordered probe observations are outside the domain: {outside}")

    @property
    def context_sha256(self) -> str:
        return canonical_sha256(_scalar_map(self.context, "ordered feasibility context"))

    @property
    def normalized_observations(self) -> tuple[OrderedProbeObservation, ...]:
        return tuple(
            sorted(
                self.observations,
                key=lambda observation: (observation.value, observation.attempt_id),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probe_id": self.probe_id,
            "parameter_name": self.parameter_name,
            "ordered_values": list(self.ordered_values),
            "context": _scalar_map(self.context, "ordered feasibility context"),
            "required_success_observations": self.required_success_observations,
            "required_resource_exhausted_observations": (
                self.required_resource_exhausted_observations
            ),
            "observations": [
                observation.to_dict()
                for observation in self.normalized_observations
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OrderedFeasibilitySpec":
        _expect_exact_keys(
            raw,
            {
                "schema_version",
                "probe_id",
                "parameter_name",
                "ordered_values",
                "context",
                "required_success_observations",
                "required_resource_exhausted_observations",
                "observations",
            },
            "ordered feasibility spec",
        )
        values = raw["ordered_values"]
        observations = raw["observations"]
        if not isinstance(values, list):
            raise ValueError("ordered feasibility values must be a list")
        if not isinstance(raw["context"], Mapping):
            raise ValueError("ordered feasibility context must be an object")
        if not isinstance(observations, list) or any(
            not isinstance(observation, Mapping) for observation in observations
        ):
            raise ValueError("ordered feasibility observations must be objects")
        return cls(
            schema_version=str(raw["schema_version"]),
            probe_id=str(raw["probe_id"]),
            parameter_name=str(raw["parameter_name"]),
            ordered_values=tuple(values),
            context=_scalar_map(raw["context"], "ordered feasibility context"),
            required_success_observations=raw["required_success_observations"],
            required_resource_exhausted_observations=raw[
                "required_resource_exhausted_observations"
            ],
            observations=tuple(
                OrderedProbeObservation.from_dict(observation)
                for observation in observations
            ),
        )


@dataclass(frozen=True, slots=True)
class OrderedProbeValueState:
    value: int
    success_observations: int
    resource_exhausted_observations: int
    state: str

    def __post_init__(self) -> None:
        _positive_int(self.value, "ordered probe state value")
        for name in ("success_observations", "resource_exhausted_observations"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.state not in _STATES:
            raise ValueError(f"unsupported ordered probe state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "success_observations": self.success_observations,
            "resource_exhausted_observations": (
                self.resource_exhausted_observations
            ),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OrderedProbeValueState":
        _expect_exact_keys(
            raw,
            {
                "value",
                "success_observations",
                "resource_exhausted_observations",
                "state",
            },
            "ordered probe value state",
        )
        return cls(
            value=raw["value"],
            success_observations=raw["success_observations"],
            resource_exhausted_observations=raw[
                "resource_exhausted_observations"
            ],
            state=str(raw["state"]),
        )


@dataclass(frozen=True, slots=True)
class OrderedFeasibilityPlan:
    probe_id: str
    parameter_name: str
    source_spec_sha256: str
    context: Mapping[str, Any]
    context_sha256: str
    ordered_values: tuple[int, ...]
    required_success_observations: int
    required_resource_exhausted_observations: int
    observations: tuple[OrderedProbeObservation, ...]
    value_states: tuple[OrderedProbeValueState, ...]
    confirmed_feasible_values: tuple[int, ...]
    confirmed_resource_exhausted_values: tuple[int, ...]
    inferred_resource_exhausted_values: tuple[int, ...]
    provisional_recommendation: int | None
    next_probe_value: int | None
    search_complete: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported ordered feasibility plan: {self.schema_version}"
            )
        require_id(self.probe_id, "ordered feasibility plan probe_id")
        require_id(self.parameter_name, "ordered feasibility plan parameter_name")
        require_digest(self.source_spec_sha256, "source_spec_sha256")
        require_digest(self.context_sha256, "context_sha256")
        if not self.producer:
            raise ValueError("ordered feasibility producer cannot be empty")
        spec = OrderedFeasibilitySpec(
            probe_id=self.probe_id,
            parameter_name=self.parameter_name,
            ordered_values=self.ordered_values,
            context=self.context,
            required_success_observations=self.required_success_observations,
            required_resource_exhausted_observations=(
                self.required_resource_exhausted_observations
            ),
            observations=self.observations,
        )
        if canonical_sha256(spec.to_dict()) != self.source_spec_sha256:
            raise ValueError("ordered feasibility plan is not bound to its source spec")
        if spec.context_sha256 != self.context_sha256:
            raise ValueError("ordered feasibility plan context SHA256 is invalid")
        if tuple(state.value for state in self.value_states) != self.ordered_values:
            raise ValueError("ordered feasibility states must cover the ordered domain")
        domain = set(self.ordered_values)
        for name in (
            "confirmed_feasible_values",
            "confirmed_resource_exhausted_values",
            "inferred_resource_exhausted_values",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or not set(values) <= domain:
                raise ValueError(f"{name} must be sorted, unique and inside the domain")
        for name in ("provisional_recommendation", "next_probe_value"):
            value = getattr(self, name)
            if value is not None and value not in domain:
                raise ValueError(f"{name} must be null or inside the ordered domain")
        if any(reason not in _REJECTION_REASONS for reason in self.rejection_reasons):
            raise ValueError("ordered feasibility plan has an unsupported rejection")
        if tuple(sorted(set(self.rejection_reasons))) != self.rejection_reasons:
            raise ValueError("ordered feasibility rejection reasons must be sorted")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("ordered feasibility eligibility is inconsistent")
        if not isinstance(self.search_complete, bool) or not isinstance(
            self.eligible, bool
        ):
            raise ValueError("ordered feasibility status flags must be booleans")
        if self.search_complete and self.next_probe_value is not None:
            raise ValueError("a completed feasibility search cannot request another probe")

    def audit(self) -> dict[str, Any]:
        state_counts = Counter(state.state for state in self.value_states)
        return {
            "probe_id": self.probe_id,
            "parameter_name": self.parameter_name,
            "context_sha256": self.context_sha256,
            "observation_count": len(self.observations),
            "states": dict(sorted(state_counts.items())),
            "provisional_recommendation": self.provisional_recommendation,
            "next_probe_value": self.next_probe_value,
            "search_complete": self.search_complete,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "probe_id": self.probe_id,
            "parameter_name": self.parameter_name,
            "source_spec_sha256": self.source_spec_sha256,
            "context": _scalar_map(self.context, "ordered feasibility context"),
            "context_sha256": self.context_sha256,
            "ordered_values": list(self.ordered_values),
            "required_success_observations": self.required_success_observations,
            "required_resource_exhausted_observations": (
                self.required_resource_exhausted_observations
            ),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "value_states": [state.to_dict() for state in self.value_states],
            "confirmed_feasible_values": list(self.confirmed_feasible_values),
            "confirmed_resource_exhausted_values": list(
                self.confirmed_resource_exhausted_values
            ),
            "inferred_resource_exhausted_values": list(
                self.inferred_resource_exhausted_values
            ),
            "provisional_recommendation": self.provisional_recommendation,
            "next_probe_value": self.next_probe_value,
            "search_complete": self.search_complete,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "audit": self.audit(),
        }
        return {
            **payload,
            "ordered_feasibility_plan_sha256": canonical_sha256(payload),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OrderedFeasibilityPlan":
        keys = {
            "schema_version",
            "producer",
            "probe_id",
            "parameter_name",
            "source_spec_sha256",
            "context",
            "context_sha256",
            "ordered_values",
            "required_success_observations",
            "required_resource_exhausted_observations",
            "observations",
            "value_states",
            "confirmed_feasible_values",
            "confirmed_resource_exhausted_values",
            "inferred_resource_exhausted_values",
            "provisional_recommendation",
            "next_probe_value",
            "search_complete",
            "eligible",
            "rejection_reasons",
            "audit",
            "ordered_feasibility_plan_sha256",
        }
        _expect_exact_keys(raw, keys, "ordered feasibility plan")
        payload = dict(raw)
        digest = str(payload.pop("ordered_feasibility_plan_sha256", ""))
        if digest != canonical_sha256(payload):
            raise ValueError("ordered feasibility plan SHA256 does not match its content")
        list_fields = (
            "ordered_values",
            "observations",
            "value_states",
            "confirmed_feasible_values",
            "confirmed_resource_exhausted_values",
            "inferred_resource_exhausted_values",
            "rejection_reasons",
        )
        if any(not isinstance(raw[name], list) for name in list_fields):
            raise ValueError("ordered feasibility plan list fields must be lists")
        if not isinstance(raw["context"], Mapping) or not isinstance(
            raw["audit"], Mapping
        ):
            raise ValueError("ordered feasibility plan object fields are invalid")
        plan = cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            probe_id=str(raw["probe_id"]),
            parameter_name=str(raw["parameter_name"]),
            source_spec_sha256=str(raw["source_spec_sha256"]),
            context=_scalar_map(raw["context"], "ordered feasibility context"),
            context_sha256=str(raw["context_sha256"]),
            ordered_values=tuple(raw["ordered_values"]),
            required_success_observations=raw["required_success_observations"],
            required_resource_exhausted_observations=raw[
                "required_resource_exhausted_observations"
            ],
            observations=tuple(
                OrderedProbeObservation.from_dict(observation)
                for observation in raw["observations"]
            ),
            value_states=tuple(
                OrderedProbeValueState.from_dict(state)
                for state in raw["value_states"]
            ),
            confirmed_feasible_values=tuple(raw["confirmed_feasible_values"]),
            confirmed_resource_exhausted_values=tuple(
                raw["confirmed_resource_exhausted_values"]
            ),
            inferred_resource_exhausted_values=tuple(
                raw["inferred_resource_exhausted_values"]
            ),
            provisional_recommendation=raw["provisional_recommendation"],
            next_probe_value=raw["next_probe_value"],
            search_complete=raw["search_complete"],
            eligible=raw["eligible"],
            rejection_reasons=tuple(str(reason) for reason in raw["rejection_reasons"]),
        )
        if plan.audit() != raw["audit"]:
            raise ValueError("ordered feasibility plan audit does not match its state")
        return plan


def _state_for_value(
    value: int,
    observations: Sequence[OrderedProbeObservation],
    spec: OrderedFeasibilitySpec,
) -> OrderedProbeValueState:
    outcomes = Counter(
        observation.outcome
        for observation in observations
        if observation.value == value
    )
    successes = outcomes["success"]
    exhausted = outcomes["resource_exhausted"]
    if successes and exhausted:
        state = "mixed"
    elif successes >= spec.required_success_observations:
        state = "confirmed_feasible"
    elif successes:
        state = "pending_success_confirmation"
    elif exhausted >= spec.required_resource_exhausted_observations:
        state = "confirmed_resource_exhausted"
    elif exhausted:
        state = "pending_resource_exhausted_confirmation"
    else:
        state = "unobserved"
    return OrderedProbeValueState(value, successes, exhausted, state)


def _middle(values: Sequence[int]) -> int:
    return values[(len(values) - 1) // 2]


def plan_ordered_feasibility(
    spec: OrderedFeasibilitySpec,
) -> OrderedFeasibilityPlan:
    """Bracket a monotone OOM boundary without treating failed trials as missing."""

    observations = spec.normalized_observations
    states = tuple(
        _state_for_value(value, observations, spec) for value in spec.ordered_values
    )
    feasible = tuple(
        state.value for state in states if state.state == "confirmed_feasible"
    )
    exhausted = tuple(
        state.value
        for state in states
        if state.state == "confirmed_resource_exhausted"
    )
    reasons: list[str] = []
    if any(state.state == "mixed" for state in states):
        reasons.append("mixed_outcomes_at_value")
    if feasible and exhausted and max(feasible) > min(exhausted):
        reasons.append("non_monotone_resource_boundary")

    inferred_exhausted = (
        tuple(
            value
            for value in spec.ordered_values
            if value > min(exhausted) and value not in exhausted
        )
        if exhausted
        else ()
    )
    recommendation = max(feasible) if feasible else None
    next_probe: int | None = None
    complete = False
    if not reasons:
        lower = max(feasible) if feasible else None
        upper = min(exhausted) if exhausted else None
        pending_success = [
            state.value
            for state in states
            if state.state == "pending_success_confirmation"
            and (lower is None or state.value > lower)
            and (upper is None or state.value < upper)
        ]
        pending_exhausted = [
            state.value
            for state in states
            if state.state == "pending_resource_exhausted_confirmation"
            and (lower is None or state.value > lower)
            and (upper is None or state.value < upper)
        ]
        if pending_success:
            next_probe = max(pending_success)
        elif pending_exhausted:
            next_probe = min(pending_exhausted)
        elif feasible and exhausted:
            assert lower is not None and upper is not None
            between = [
                value for value in spec.ordered_values if lower < value < upper
            ]
            if between:
                next_probe = _middle(between)
            else:
                complete = True
        elif feasible:
            above = [
                value for value in spec.ordered_values if value > max(feasible)
            ]
            if above:
                next_probe = max(above)
            else:
                complete = True
        elif exhausted:
            below = [
                value for value in spec.ordered_values if value < min(exhausted)
            ]
            if below:
                next_probe = min(below)
            else:
                complete = True
        else:
            next_probe = _middle(spec.ordered_values)

    return OrderedFeasibilityPlan(
        probe_id=spec.probe_id,
        parameter_name=spec.parameter_name,
        source_spec_sha256=canonical_sha256(spec.to_dict()),
        context=spec.context,
        context_sha256=spec.context_sha256,
        ordered_values=spec.ordered_values,
        required_success_observations=spec.required_success_observations,
        required_resource_exhausted_observations=(
            spec.required_resource_exhausted_observations
        ),
        observations=observations,
        value_states=states,
        confirmed_feasible_values=feasible,
        confirmed_resource_exhausted_values=exhausted,
        inferred_resource_exhausted_values=inferred_exhausted,
        provisional_recommendation=recommendation,
        next_probe_value=next_probe,
        search_complete=complete,
        eligible=not reasons,
        rejection_reasons=tuple(sorted(reasons)),
    )
