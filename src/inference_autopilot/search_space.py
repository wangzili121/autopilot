"""Deterministic compilation of constrained deployment search spaces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from math import isfinite, prod
from typing import Any

from inference_autopilot.calibration.models import (
    ConfigurationSpec,
    canonical_json,
    canonical_sha256,
    require_digest,
    require_id,
)
from inference_autopilot.ir import InferenceGraph


Scalar = bool | int | float | str
IntegerSequence = tuple[int, ...]
SettingValue = Scalar | IntegerSequence

_VALUE_TYPES = {"integer", "number", "boolean", "string", "integer_sequence"}
_CHANGE_SCOPES = {"deployment", "engine_restart", "policy_boundary", "per_request"}
_SEMANTIC_EFFECTS = {"preserves_algorithm", "changes_algorithm"}
_TUNING_LAYERS = {
    "static_deployment",
    "graph_capture",
    "hot_policy",
    "algorithm_semantics",
}
_LAYER_CHANGE_SCOPES = {
    "static_deployment": {"deployment", "engine_restart"},
    "graph_capture": {"deployment", "engine_restart"},
    "hot_policy": {"policy_boundary", "per_request"},
    "algorithm_semantics": _CHANGE_SCOPES,
}
_CAPABILITY_STATUSES = {"supported", "unsupported", "unknown"}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and isfinite(value)


def _is_setting_value(value: Any) -> bool:
    if _is_scalar(value):
        return True
    return (
        isinstance(value, (list, tuple))
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and list(value) == sorted(set(value))
    )


def normalize_setting_value(value: Any, context: str) -> SettingValue:
    """Validate and freeze a runner setting for stable hashes and comparisons."""

    if _is_scalar(value):
        return value
    if _is_setting_value(value):
        return tuple(value)
    raise ValueError(
        f"{context} must be a finite scalar or a sorted unique sequence of "
        "positive integers"
    )


def setting_value_to_json(value: SettingValue) -> Scalar | list[int]:
    return list(value) if isinstance(value, tuple) else value


def setting_map_to_dict(
    values: Mapping[str, SettingValue],
) -> dict[str, Scalar | list[int]]:
    return {
        name: setting_value_to_json(value)
        for name, value in sorted(values.items())
    }


def normalize_setting_map(
    raw: Mapping[str, Any], context: str
) -> dict[str, SettingValue]:
    normalized: dict[str, SettingValue] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} keys must be non-empty strings")
        normalized[name] = normalize_setting_value(value, f"{context}.{name}")
    canonical_json(setting_map_to_dict(normalized))
    return dict(sorted(normalized.items()))


def _validate_value(value: Any, value_type: str, context: str) -> SettingValue:
    if value_type == "integer_sequence":
        normalized = normalize_setting_value(value, context)
        if not isinstance(normalized, tuple):
            raise ValueError(f"{context} must contain integer_sequence values")
        return normalized
    valid = {
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
        ),
        "boolean": isinstance(value, bool),
        "string": isinstance(value, str),
    }.get(value_type, False)
    if not valid:
        raise ValueError(f"{context} must contain {value_type} values")
    if value_type == "string" and not value:
        raise ValueError(f"{context} strings cannot be empty")
    return value


def _decimal(value: int | float) -> Decimal:
    return Decimal(str(value))


def _canonical_value_key(value: SettingValue) -> str:
    normalized = normalize_setting_value(value, "setting value")
    value_type = "integer_sequence" if isinstance(normalized, tuple) else type(normalized).__name__
    return canonical_json(
        {"type": value_type, "value": setting_value_to_json(normalized)}
    )


@dataclass(frozen=True, slots=True)
class DomainSpec:
    kind: str
    values: tuple[SettingValue, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            tuple(
                normalize_setting_value(value, "domain value")
                for value in self.values
            ),
        )
        if self.kind not in {"choices", "range"}:
            raise ValueError(f"unsupported domain kind: {self.kind}")
        if self.kind == "choices":
            if not self.values:
                raise ValueError("choice domain cannot be empty")
            if any(value is not None for value in (self.minimum, self.maximum, self.step)):
                raise ValueError("choice domain cannot define range fields")
            keys = [_canonical_value_key(value) for value in self.values]
            if len(keys) != len(set(keys)):
                raise ValueError("choice domain values must be unique")
        else:
            if self.values:
                raise ValueError("range domain cannot define choices")
            numeric = (self.minimum, self.maximum, self.step)
            if any(
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in numeric
            ):
                raise ValueError("range domain fields must be finite numbers")
            assert self.minimum is not None
            assert self.maximum is not None
            assert self.step is not None
            if self.minimum > self.maximum:
                raise ValueError("range minimum cannot exceed maximum")
            if self.step <= 0:
                raise ValueError("range step must be positive")

    def expand(self, value_type: str) -> tuple[SettingValue, ...]:
        if value_type not in _VALUE_TYPES:
            raise ValueError(f"unsupported knob value type: {value_type}")
        if self.kind == "choices":
            values = tuple(
                _validate_value(value, value_type, "choice domain")
                for value in self.values
            )
            return tuple(sorted(values, key=_canonical_value_key))
        if value_type not in {"integer", "number"}:
            raise ValueError("range domains require an integer or number knob")
        assert self.minimum is not None
        assert self.maximum is not None
        assert self.step is not None
        if value_type == "integer" and any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.minimum, self.maximum, self.step)
        ):
            raise ValueError("integer range fields must be integers")
        minimum = _decimal(self.minimum)
        maximum = _decimal(self.maximum)
        step = _decimal(self.step)
        count = int((maximum - minimum) // step) + 1
        expanded: list[SettingValue] = []
        for index in range(count):
            value = minimum + index * step
            expanded.append(int(value) if value_type == "integer" else float(value))
        return tuple(expanded)

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "choices":
            return {
                "kind": self.kind,
                "values": [setting_value_to_json(value) for value in self.values],
            }
        return {
            "kind": self.kind,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DomainSpec":
        kind = str(raw.get("kind", ""))
        if kind == "choices":
            _expect_exact_keys(raw, {"kind", "values"}, "choice domain")
            values = raw["values"]
            if not isinstance(values, list) or any(
                not _is_setting_value(value) for value in values
            ):
                raise ValueError(
                    "choice domain values must be finite scalars or integer sequences"
                )
            return cls(
                kind=kind,
                values=tuple(
                    normalize_setting_value(value, "choice domain")
                    for value in values
                ),
            )
        if kind == "range":
            keys = {"kind", "minimum", "maximum", "step"}
            _expect_exact_keys(raw, keys, "range domain")
            return cls(
                kind=kind,
                minimum=raw["minimum"],
                maximum=raw["maximum"],
                step=raw["step"],
            )
        raise ValueError(f"unsupported domain kind: {kind}")


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    status: str
    reason: str
    evidence_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _CAPABILITY_STATUSES:
            raise ValueError(f"unsupported capability status: {self.status}")
        if not self.reason:
            raise ValueError("capability reason cannot be empty")
        if len(set(self.evidence_sha256)) != len(self.evidence_sha256):
            raise ValueError("capability evidence digests must be unique")
        for digest in self.evidence_sha256:
            require_digest(digest, "capability evidence SHA256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "evidence_sha256": list(self.evidence_sha256),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapabilityRecord":
        _expect_exact_keys(
            raw, {"status", "reason", "evidence_sha256"}, "capability record"
        )
        evidence = raw["evidence_sha256"]
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) for item in evidence
        ):
            raise ValueError("capability evidence_sha256 must be a list of strings")
        return cls(str(raw["status"]), str(raw["reason"]), tuple(evidence))


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityProfile:
    profile_id: str
    environment_id: str
    capabilities: Mapping[str, CapabilityRecord]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported runtime capability profile: {self.schema_version}"
            )
        require_id(self.profile_id, "capability profile_id")
        require_id(self.environment_id, "capability environment_id")
        if not self.capabilities:
            raise ValueError("runtime capability profile cannot be empty")
        for capability_id, record in self.capabilities.items():
            require_id(capability_id, "capability id")
            if not isinstance(record, CapabilityRecord):
                raise ValueError("runtime capabilities must contain capability records")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def status_for(self, capability_id: str) -> str:
        record = self.capabilities.get(capability_id)
        return "unknown" if record is None else record.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "environment_id": self.environment_id,
            "capabilities": {
                name: record.to_dict()
                for name, record in sorted(self.capabilities.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeCapabilityProfile":
        _expect_exact_keys(
            raw,
            {"schema_version", "profile_id", "environment_id", "capabilities"},
            "runtime capability profile",
        )
        capabilities = raw["capabilities"]
        if not isinstance(capabilities, Mapping) or any(
            not isinstance(item, Mapping) for item in capabilities.values()
        ):
            raise ValueError("capabilities must be an object of capability records")
        return cls(
            profile_id=str(raw["profile_id"]),
            environment_id=str(raw["environment_id"]),
            capabilities={
                str(name): CapabilityRecord.from_dict(item)
                for name, item in capabilities.items()
            },
            schema_version=str(raw["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    capability_id: str
    values: tuple[SettingValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            tuple(
                normalize_setting_value(value, "capability requirement")
                for value in self.values
            ),
        )
        require_id(self.capability_id, "required capability id")
        if not self.values or any(
            not _is_setting_value(value) for value in self.values
        ):
            raise ValueError(
                "capability requirement values must be finite settings"
            )
        keys = [_canonical_value_key(value) for value in self.values]
        if len(keys) != len(set(keys)):
            raise ValueError("capability requirement values must be unique")

    def applies(self, value: SettingValue) -> bool:
        key = _canonical_value_key(value)
        return key in {_canonical_value_key(item) for item in self.values}

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "values": [setting_value_to_json(value) for value in self.values],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapabilityRequirement":
        _expect_exact_keys(
            raw, {"capability_id", "values"}, "capability requirement"
        )
        values = raw["values"]
        if not isinstance(values, list):
            raise ValueError("capability requirement values must be a list")
        return cls(
            str(raw["capability_id"]),
            tuple(
                normalize_setting_value(value, "capability requirement")
                for value in values
            ),
        )


@dataclass(frozen=True, slots=True)
class KnobSpec:
    name: str
    setting_name: str
    value_type: str
    domain: DomainSpec
    change_scope: str
    semantic_effect: str
    description: str
    tuning_layer: str = "static_deployment"
    applies_to_stages: tuple[str, ...] = ()
    capability_requirements: tuple[CapabilityRequirement, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.name, "knob name")
        require_id(self.setting_name, "knob setting_name")
        if self.value_type not in _VALUE_TYPES:
            raise ValueError(f"unsupported knob value type: {self.value_type}")
        if self.change_scope not in _CHANGE_SCOPES:
            raise ValueError(f"unsupported knob change scope: {self.change_scope}")
        if self.semantic_effect not in _SEMANTIC_EFFECTS:
            raise ValueError(f"unsupported knob semantic effect: {self.semantic_effect}")
        if self.tuning_layer not in _TUNING_LAYERS:
            raise ValueError(f"unsupported knob tuning layer: {self.tuning_layer}")
        if self.change_scope not in _LAYER_CHANGE_SCOPES[self.tuning_layer]:
            raise ValueError(
                f"knob layer {self.tuning_layer} cannot use change scope "
                f"{self.change_scope}"
            )
        if (
            self.semantic_effect == "changes_algorithm"
            and self.tuning_layer != "algorithm_semantics"
        ):
            raise ValueError("algorithm-changing knobs must use algorithm_semantics layer")
        if len(set(self.applies_to_stages)) != len(self.applies_to_stages):
            raise ValueError("knob applies_to_stages must be unique")
        for stage_id in self.applies_to_stages:
            require_id(stage_id, "knob stage id")
        capability_ids = [
            requirement.capability_id for requirement in self.capability_requirements
        ]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("knob capability requirements must use unique capability ids")
        allowed = {_canonical_value_key(value) for value in self.values()}
        for requirement in self.capability_requirements:
            for value in requirement.values:
                _validate_value(value, self.value_type, "capability requirement")
                if _canonical_value_key(value) not in allowed:
                    raise ValueError(
                        f"capability requirement value {value!r} is outside knob domain"
                    )
        if not self.description:
            raise ValueError("knob description cannot be empty")
        self.domain.expand(self.value_type)

    def values(self) -> tuple[SettingValue, ...]:
        return self.domain.expand(self.value_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "setting_name": self.setting_name,
            "value_type": self.value_type,
            "domain": self.domain.to_dict(),
            "change_scope": self.change_scope,
            "semantic_effect": self.semantic_effect,
            "tuning_layer": self.tuning_layer,
            "applies_to_stages": list(self.applies_to_stages),
            "capability_requirements": [
                requirement.to_dict()
                for requirement in self.capability_requirements
            ],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KnobSpec":
        required_keys = {
            "name",
            "setting_name",
            "value_type",
            "domain",
            "change_scope",
            "semantic_effect",
            "description",
        }
        optional_keys = {
            "tuning_layer",
            "applies_to_stages",
            "capability_requirements",
        }
        missing = sorted(required_keys - set(raw))
        unknown = sorted(set(raw) - required_keys - optional_keys)
        if missing or unknown:
            raise ValueError(
                f"knob keys mismatch; missing={missing}, unknown={unknown}"
            )
        if not isinstance(raw["domain"], Mapping):
            raise ValueError("knob domain must be an object")
        applies_to_stages = raw.get("applies_to_stages", [])
        if not isinstance(applies_to_stages, list) or any(
            not isinstance(stage_id, str) for stage_id in applies_to_stages
        ):
            raise ValueError("knob applies_to_stages must be a list of strings")
        requirements = raw.get("capability_requirements", [])
        if not isinstance(requirements, list) or any(
            not isinstance(item, Mapping) for item in requirements
        ):
            raise ValueError("knob capability_requirements must be a list of objects")
        default_layer = (
            "algorithm_semantics"
            if raw["semantic_effect"] == "changes_algorithm"
            else "static_deployment"
        )
        return cls(
            name=str(raw["name"]),
            setting_name=str(raw["setting_name"]),
            value_type=str(raw["value_type"]),
            domain=DomainSpec.from_dict(raw["domain"]),
            change_scope=str(raw["change_scope"]),
            semantic_effect=str(raw["semantic_effect"]),
            description=str(raw["description"]),
            tuning_layer=str(raw.get("tuning_layer", default_layer)),
            applies_to_stages=tuple(applies_to_stages),
            capability_requirements=tuple(
                CapabilityRequirement.from_dict(item) for item in requirements
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    kind: str
    parameters: tuple[str, ...]
    value: float | None = None
    values: tuple[tuple[SettingValue, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            tuple(
                tuple(
                    normalize_setting_value(value, "constraint value")
                    for value in row
                )
                for row in self.values
            ),
        )
        if (
            len(self.parameters) < 2
            or any(
                not isinstance(parameter, str) or not parameter
                for parameter in self.parameters
            )
            or len(set(self.parameters)) != len(self.parameters)
        ):
            raise ValueError("constraint parameters must contain unique knob names")
        if self.kind == "sum_less_equal":
            if (
                self.value is None
                or isinstance(self.value, bool)
                or not isfinite(self.value)
            ):
                raise ValueError("sum constraint value must be finite")
            if self.values:
                raise ValueError("sum constraint cannot define allowed values")
        elif self.kind == "allowed_combinations":
            if self.value is not None:
                raise ValueError("combination constraint cannot define a sum value")
            if not self.values:
                raise ValueError("combination constraint values cannot be empty")
            if any(len(row) != len(self.parameters) for row in self.values):
                raise ValueError("allowed combination width must match its parameters")
            keys = [canonical_json(list(row)) for row in self.values]
            if len(keys) != len(set(keys)):
                raise ValueError("allowed combinations must be unique")
        else:
            raise ValueError(f"unsupported constraint kind: {self.kind}")

    @property
    def label(self) -> str:
        return f"{self.kind}:{','.join(self.parameters)}"

    def accepts(self, values: Mapping[str, SettingValue]) -> bool:
        if self.kind == "sum_less_equal":
            assert self.value is not None
            total = sum(_decimal(values[name]) for name in self.parameters)
            return total <= _decimal(self.value)
        selected = tuple(values[name] for name in self.parameters)
        return canonical_json(list(selected)) in {
            canonical_json(list(row)) for row in self.values
        }

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "sum_less_equal":
            return {
                "kind": self.kind,
                "parameters": list(self.parameters),
                "value": self.value,
            }
        return {
            "kind": self.kind,
            "parameters": list(self.parameters),
            "values": [
                [setting_value_to_json(value) for value in row]
                for row in self.values
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConstraintSpec":
        kind = str(raw.get("kind", ""))
        if kind == "sum_less_equal":
            keys = {"kind", "parameters", "value"}
            _expect_exact_keys(raw, keys, "sum constraint")
        elif kind == "allowed_combinations":
            keys = {"kind", "parameters", "values"}
            _expect_exact_keys(raw, keys, "combination constraint")
        else:
            raise ValueError(f"unsupported constraint kind: {kind}")
        parameters = raw["parameters"]
        if not isinstance(parameters, list) or any(
            not isinstance(parameter, str) for parameter in parameters
        ):
            raise ValueError("constraint parameters must be strings")
        if kind == "sum_less_equal":
            value = raw["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sum constraint value must be numeric")
            return cls(kind, tuple(parameters), float(value))
        rows = raw["values"]
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("allowed combinations must be a list of lists")
        if any(any(not _is_setting_value(value) for value in row) for row in rows):
            raise ValueError("allowed combinations must contain finite settings")
        return cls(
            kind,
            tuple(parameters),
            values=tuple(
                tuple(
                    normalize_setting_value(value, "allowed combination")
                    for value in row
                )
                for row in rows
            ),
        )


@dataclass(frozen=True, slots=True)
class DeploymentSearchSpace:
    space_id: str
    algorithm_id: str
    fixed_settings: Mapping[str, SettingValue]
    knobs: tuple[KnobSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_settings",
            normalize_setting_map(self.fixed_settings, "fixed settings"),
        )
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported deployment search space: {self.schema_version}")
        require_id(self.space_id, "space_id")
        require_id(self.algorithm_id, "algorithm_id")
        if not self.knobs:
            raise ValueError("deployment search space requires at least one knob")
        knob_names = [knob.name for knob in self.knobs]
        setting_names = [knob.setting_name for knob in self.knobs]
        if len(knob_names) != len(set(knob_names)):
            raise ValueError("deployment knob names must be unique")
        if len(setting_names) != len(set(setting_names)):
            raise ValueError("deployment knob setting names must be unique")
        if any(not isinstance(name, str) or not name for name in self.fixed_settings):
            raise ValueError("fixed setting names must be non-empty strings")
        for name in self.fixed_settings:
            require_id(name, "fixed setting name")
        if any(
            not _is_setting_value(value) for value in self.fixed_settings.values()
        ):
            raise ValueError("fixed settings must contain finite settings")
        overlap = sorted(set(setting_names) & set(self.fixed_settings))
        if overlap:
            raise ValueError(f"fixed settings overlap knob settings: {overlap}")
        knobs_by_name = {knob.name: knob for knob in self.knobs}
        for constraint in self.constraints:
            unknown = sorted(set(constraint.parameters) - set(knobs_by_name))
            if unknown:
                raise ValueError(f"constraint references unknown knobs: {unknown}")
            if constraint.kind == "sum_less_equal":
                nonnumeric = [
                    name
                    for name in constraint.parameters
                    if knobs_by_name[name].value_type not in {"integer", "number"}
                ]
                if nonnumeric:
                    raise ValueError(f"sum constraint uses nonnumeric knobs: {nonnumeric}")
            else:
                for row in constraint.values:
                    for name, value in zip(constraint.parameters, row, strict=True):
                        knob = knobs_by_name[name]
                        _validate_value(value, knob.value_type, constraint.label)
                        allowed = {_canonical_value_key(item) for item in knob.values()}
                        if _canonical_value_key(value) not in allowed:
                            raise ValueError(
                                f"constraint value {value!r} is outside domain for {name}"
                            )
        labels = [constraint.label for constraint in self.constraints]
        if len(labels) != len(set(labels)):
            raise ValueError("constraint labels must be unique")
        canonical_json(setting_map_to_dict(self.fixed_settings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "space_id": self.space_id,
            "algorithm_id": self.algorithm_id,
            "fixed_settings": setting_map_to_dict(self.fixed_settings),
            "knobs": [knob.to_dict() for knob in self.knobs],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    def tuning_layer_audit(self) -> dict[str, Any]:
        """Summarize the executable layers and stage bindings in this space."""

        by_layer = Counter(knob.tuning_layer for knob in self.knobs)
        by_stage = Counter(
            stage_id for knob in self.knobs for stage_id in knob.applies_to_stages
        )
        unbound = sorted(knob.name for knob in self.knobs if not knob.applies_to_stages)
        return {
            "by_tuning_layer": dict(sorted(by_layer.items())),
            "by_stage": dict(sorted(by_stage.items())),
            "unbound_knobs": unbound,
        }

    def validate_against_graph(self, graph: InferenceGraph) -> None:
        """Reject stage bindings that the algorithm graph cannot execute."""

        if self.algorithm_id != graph.algorithm_id:
            raise ValueError(
                f"search space algorithm {self.algorithm_id} does not match graph "
                f"{graph.algorithm_id}"
            )
        stages = {stage.stage_id: stage for stage in graph.stages}
        for knob in self.knobs:
            if knob.tuning_layer == "graph_capture" and not knob.applies_to_stages:
                raise ValueError(f"graph capture knob {knob.name} requires stage bindings")
            unknown = sorted(set(knob.applies_to_stages) - set(stages))
            if unknown:
                raise ValueError(
                    f"knob {knob.name} references unknown graph stages: {unknown}"
                )
            unsupported = [
                stage_id
                for stage_id in knob.applies_to_stages
                if knob.name not in stages[stage_id].tunable_runtime_fields
            ]
            if unsupported:
                raise ValueError(
                    f"knob {knob.name} is not declared tunable by stages: "
                    f"{sorted(unsupported)}"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeploymentSearchSpace":
        keys = {
            "schema_version",
            "space_id",
            "algorithm_id",
            "fixed_settings",
            "knobs",
            "constraints",
        }
        _expect_exact_keys(raw, keys, "deployment search space")
        fixed = raw["fixed_settings"]
        knobs = raw["knobs"]
        constraints = raw["constraints"]
        if not isinstance(fixed, Mapping):
            raise ValueError("fixed_settings must be an object")
        if not isinstance(knobs, list) or not isinstance(constraints, list):
            raise ValueError("knobs and constraints must be lists")
        if any(not isinstance(item, Mapping) for item in (*knobs, *constraints)):
            raise ValueError("knobs and constraints must contain objects")
        return cls(
            schema_version=str(raw["schema_version"]),
            space_id=str(raw["space_id"]),
            algorithm_id=str(raw["algorithm_id"]),
            fixed_settings=normalize_setting_map(fixed, "fixed settings"),
            knobs=tuple(KnobSpec.from_dict(item) for item in knobs),
            constraints=tuple(ConstraintSpec.from_dict(item) for item in constraints),
        )


def _candidate_id(
    space_id: str,
    algorithm_id: str,
    knob_values: Mapping[str, SettingValue],
    deployment_settings: Mapping[str, SettingValue],
    semantic_settings: Mapping[str, SettingValue],
) -> str:
    digest = canonical_sha256(
        {
            "algorithm_id": algorithm_id,
            "knob_values": setting_map_to_dict(knob_values),
            "deployment_settings": setting_map_to_dict(deployment_settings),
            "semantic_settings": setting_map_to_dict(semantic_settings),
        }
    )
    return f"{space_id}--{digest[:16]}"


def _semantic_cohort_id(
    algorithm_id: str, semantic_settings: Mapping[str, SettingValue]
) -> str:
    if not semantic_settings:
        return algorithm_id
    digest = canonical_sha256(
        {
            "algorithm_id": algorithm_id,
            "semantic_settings": setting_map_to_dict(semantic_settings),
        }
    )
    return f"{algorithm_id}--semantic-{digest[:12]}"


@dataclass(frozen=True, slots=True)
class CompiledCandidate:
    candidate_id: str
    semantic_cohort_id: str
    knob_values: Mapping[str, SettingValue]
    deployment_settings: Mapping[str, SettingValue]
    semantic_settings: Mapping[str, SettingValue]
    change_scopes: tuple[str, ...]
    tuning_layers: tuple[str, ...]
    requires_engine_restart: bool

    def __post_init__(self) -> None:
        for name in ("knob_values", "deployment_settings", "semantic_settings"):
            values = getattr(self, name)
            if isinstance(values, Mapping):
                object.__setattr__(
                    self,
                    name,
                    normalize_setting_map(values, f"compiled candidate {name}"),
                )
        require_id(self.candidate_id, "candidate_id")
        require_id(self.semantic_cohort_id, "semantic_cohort_id")
        for name in ("knob_values", "deployment_settings", "semantic_settings"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str) or not key or not _is_setting_value(value)
                for key, value in values.items()
            ):
                raise ValueError(
                    f"compiled candidate {name} must contain finite settings"
                )
        if not self.knob_values or not self.deployment_settings:
            raise ValueError("compiled candidate must contain knobs and deployment settings")
        if tuple(sorted(set(self.change_scopes))) != self.change_scopes:
            raise ValueError("candidate change scopes must be sorted and unique")
        if any(scope not in _CHANGE_SCOPES for scope in self.change_scopes):
            raise ValueError("candidate contains an unsupported change scope")
        if tuple(sorted(set(self.tuning_layers))) != self.tuning_layers:
            raise ValueError("candidate tuning layers must be sorted and unique")
        if any(layer not in _TUNING_LAYERS for layer in self.tuning_layers):
            raise ValueError("candidate contains an unsupported tuning layer")
        if not isinstance(self.requires_engine_restart, bool):
            raise ValueError("candidate restart flag must be boolean")
        if self.requires_engine_restart != ("engine_restart" in self.change_scopes):
            raise ValueError("candidate restart flag does not match change scopes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_cohort_id": self.semantic_cohort_id,
            "knob_values": setting_map_to_dict(self.knob_values),
            "deployment_settings": setting_map_to_dict(self.deployment_settings),
            "semantic_settings": setting_map_to_dict(self.semantic_settings),
            "change_scopes": list(self.change_scopes),
            "tuning_layers": list(self.tuning_layers),
            "requires_engine_restart": self.requires_engine_restart,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CompiledCandidate":
        keys = {
            "candidate_id",
            "semantic_cohort_id",
            "knob_values",
            "deployment_settings",
            "semantic_settings",
            "change_scopes",
            "tuning_layers",
            "requires_engine_restart",
        }
        _expect_exact_keys(raw, keys, "compiled candidate")
        mappings = ("knob_values", "deployment_settings", "semantic_settings")
        if any(not isinstance(raw[name], Mapping) for name in mappings):
            raise ValueError("compiled candidate setting fields must be objects")
        scopes = raw["change_scopes"]
        layers = raw["tuning_layers"]
        if not isinstance(scopes, list) or any(not isinstance(item, str) for item in scopes):
            raise ValueError("compiled candidate change_scopes must be strings")
        if not isinstance(layers, list) or any(not isinstance(item, str) for item in layers):
            raise ValueError("compiled candidate tuning_layers must be strings")
        if not isinstance(raw["requires_engine_restart"], bool):
            raise ValueError("compiled candidate restart flag must be boolean")
        return cls(
            candidate_id=str(raw["candidate_id"]),
            semantic_cohort_id=str(raw["semantic_cohort_id"]),
            knob_values=normalize_setting_map(raw["knob_values"], "knob values"),
            deployment_settings=normalize_setting_map(
                raw["deployment_settings"], "deployment settings"
            ),
            semantic_settings=normalize_setting_map(
                raw["semantic_settings"], "semantic settings"
            ),
            change_scopes=tuple(scopes),
            tuning_layers=tuple(layers),
            requires_engine_restart=raw["requires_engine_restart"],
        )


@dataclass(frozen=True, slots=True)
class CompiledSearchSpace:
    space_id: str
    algorithm_id: str
    source_space_sha256: str
    capability_profile_sha256: str | None
    allow_algorithm_changes: bool
    max_cartesian_product: int
    cartesian_product_size: int
    rejected_candidate_count: int
    rejections_by_constraint: Mapping[str, int]
    candidates: tuple[CompiledCandidate, ...]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported compiled search space: {self.schema_version}")
        require_id(self.space_id, "space_id")
        require_id(self.algorithm_id, "algorithm_id")
        require_digest(self.source_space_sha256, "source_space_sha256")
        if self.capability_profile_sha256 is not None:
            require_digest(
                self.capability_profile_sha256, "capability_profile_sha256"
            )
        if not self.producer:
            raise ValueError("compiled search producer cannot be empty")
        if not isinstance(self.allow_algorithm_changes, bool):
            raise ValueError("allow_algorithm_changes must be boolean")
        if (
            isinstance(self.max_cartesian_product, bool)
            or not isinstance(self.max_cartesian_product, int)
            or self.max_cartesian_product <= 0
        ):
            raise ValueError("max_cartesian_product must be a positive integer")
        if (
            isinstance(self.cartesian_product_size, bool)
            or not isinstance(self.cartesian_product_size, int)
            or self.cartesian_product_size <= 0
        ):
            raise ValueError("cartesian_product_size must be a positive integer")
        if (
            isinstance(self.rejected_candidate_count, bool)
            or not isinstance(self.rejected_candidate_count, int)
            or self.rejected_candidate_count < 0
        ):
            raise ValueError("rejected_candidate_count must be a non-negative integer")
        if self.cartesian_product_size > self.max_cartesian_product:
            raise ValueError("compiled search space exceeds its Cartesian product limit")
        expected_rejections = self.cartesian_product_size - len(self.candidates)
        if self.rejected_candidate_count != expected_rejections:
            raise ValueError("rejected candidate count is inconsistent")
        if any(
            not isinstance(name, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for name, count in self.rejections_by_constraint.items()
        ):
            raise ValueError("constraint rejection counts must be non-negative integers")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("compiled candidate ids must be sorted and unique")
        for candidate in self.candidates:
            expected_id = _candidate_id(
                self.space_id,
                self.algorithm_id,
                candidate.knob_values,
                candidate.deployment_settings,
                candidate.semantic_settings,
            )
            if candidate.candidate_id != expected_id:
                raise ValueError("compiled candidate id does not match its settings")
            expected_cohort = _semantic_cohort_id(
                self.algorithm_id, candidate.semantic_settings
            )
            if candidate.semantic_cohort_id != expected_cohort:
                raise ValueError("compiled semantic cohort does not match settings")

    def audit(self) -> dict[str, Any]:
        cohorts = Counter(candidate.semantic_cohort_id for candidate in self.candidates)
        scopes = Counter(
            scope for candidate in self.candidates for scope in candidate.change_scopes
        )
        layers = Counter(
            layer for candidate in self.candidates for layer in candidate.tuning_layers
        )
        return {
            "cartesian_product_size": self.cartesian_product_size,
            "candidate_count": len(self.candidates),
            "rejected_candidate_count": self.rejected_candidate_count,
            "rejections_by_constraint": dict(sorted(self.rejections_by_constraint.items())),
            "semantic_cohort_count": len(cohorts),
            "by_semantic_cohort": dict(sorted(cohorts.items())),
            "by_change_scope": dict(sorted(scopes.items())),
            "by_tuning_layer": dict(sorted(layers.items())),
            "requires_engine_restart_count": sum(
                candidate.requires_engine_restart for candidate in self.candidates
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "space_id": self.space_id,
            "algorithm_id": self.algorithm_id,
            "source_space_sha256": self.source_space_sha256,
            "capability_profile_sha256": self.capability_profile_sha256,
            "options": {
                "allow_algorithm_changes": self.allow_algorithm_changes,
                "max_cartesian_product": self.max_cartesian_product,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "audit": self.audit(),
        }
        return {**payload, "compiled_space_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CompiledSearchSpace":
        required_keys = {
            "schema_version",
            "producer",
            "space_id",
            "algorithm_id",
            "source_space_sha256",
            "options",
            "candidates",
            "audit",
        }
        optional_keys = {"capability_profile_sha256"}
        missing = sorted(required_keys - set(raw))
        unknown = sorted(
            set(raw) - required_keys - optional_keys - {"compiled_space_sha256"}
        )
        if missing or unknown or "compiled_space_sha256" not in raw:
            raise ValueError(
                "compiled search space keys mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        payload = dict(raw)
        digest = str(payload.pop("compiled_space_sha256", ""))
        if digest != canonical_sha256(payload):
            raise ValueError("compiled search SHA256 does not match its content")
        options = raw["options"]
        audit = raw["audit"]
        candidates = raw["candidates"]
        if not isinstance(options, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("compiled search options and audit must be objects")
        _expect_exact_keys(
            options,
            {"allow_algorithm_changes", "max_cartesian_product"},
            "compiled search options",
        )
        if not isinstance(candidates, list) or any(
            not isinstance(item, Mapping) for item in candidates
        ):
            raise ValueError("compiled search candidates must be objects")
        required_audit = {
            "cartesian_product_size",
            "candidate_count",
            "rejected_candidate_count",
            "rejections_by_constraint",
            "semantic_cohort_count",
            "by_semantic_cohort",
            "by_change_scope",
            "by_tuning_layer",
            "requires_engine_restart_count",
        }
        _expect_exact_keys(audit, required_audit, "compiled search audit")
        if not isinstance(audit["rejections_by_constraint"], Mapping):
            raise ValueError("constraint rejection audit must be an object")
        table = cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            space_id=str(raw["space_id"]),
            algorithm_id=str(raw["algorithm_id"]),
            source_space_sha256=str(raw["source_space_sha256"]),
            capability_profile_sha256=(
                str(raw["capability_profile_sha256"])
                if raw.get("capability_profile_sha256") is not None
                else None
            ),
            allow_algorithm_changes=options["allow_algorithm_changes"],
            max_cartesian_product=options["max_cartesian_product"],
            cartesian_product_size=audit["cartesian_product_size"],
            rejected_candidate_count=audit["rejected_candidate_count"],
            rejections_by_constraint=dict(audit["rejections_by_constraint"]),
            candidates=tuple(CompiledCandidate.from_dict(item) for item in candidates),
        )
        if raw["audit"] != table.audit():
            raise ValueError("compiled search audit does not match candidates")
        return table


def compile_search_space(
    space: DeploymentSearchSpace,
    *,
    capability_profile: RuntimeCapabilityProfile | None = None,
    allow_algorithm_changes: bool = False,
    max_cartesian_product: int = 100_000,
) -> CompiledSearchSpace:
    """Expand and constrain a search space without executing model work."""

    if not isinstance(allow_algorithm_changes, bool):
        raise ValueError("allow_algorithm_changes must be boolean")
    if (
        isinstance(max_cartesian_product, bool)
        or not isinstance(max_cartesian_product, int)
        or max_cartesian_product <= 0
    ):
        raise ValueError("max_cartesian_product must be a positive integer")
    semantic_knobs = [
        knob for knob in space.knobs if knob.semantic_effect == "changes_algorithm"
    ]
    if semantic_knobs and not allow_algorithm_changes:
        names = [knob.name for knob in semantic_knobs]
        raise ValueError(
            f"search space contains algorithm-changing knobs; explicit opt-in required: {names}"
        )
    capability_requirements = [
        requirement
        for knob in space.knobs
        for requirement in knob.capability_requirements
    ]
    if capability_requirements and capability_profile is None:
        raise ValueError("search space has capability requirements; profile required")
    knobs = tuple(sorted(space.knobs, key=lambda knob: knob.name))
    domains = tuple(knob.values() for knob in knobs)
    product_size = prod(len(values) for values in domains)
    if product_size > max_cartesian_product:
        raise ValueError(
            f"Cartesian product {product_size} exceeds limit {max_cartesian_product}"
        )

    rejection_counts: Counter[str] = Counter()
    candidates: list[CompiledCandidate] = []
    for selected in product(*domains):
        knob_values = dict(zip((knob.name for knob in knobs), selected, strict=True))
        violations = [
            constraint.label
            for constraint in space.constraints
            if not constraint.accepts(knob_values)
        ]
        if capability_profile is not None:
            for knob in knobs:
                selected_value = knob_values[knob.name]
                for requirement in knob.capability_requirements:
                    if not requirement.applies(selected_value):
                        continue
                    status = capability_profile.status_for(
                        requirement.capability_id
                    )
                    if status != "supported":
                        violations.append(
                            f"capability:{requirement.capability_id}:{status}"
                        )
        if violations:
            rejection_counts.update(violations)
            continue
        deployment_settings: dict[str, SettingValue] = dict(space.fixed_settings)
        semantic_settings: dict[str, SettingValue] = {}
        for knob, value in zip(knobs, selected, strict=True):
            target = (
                semantic_settings
                if knob.semantic_effect == "changes_algorithm"
                else deployment_settings
            )
            target[knob.setting_name] = value
        change_scopes = tuple(sorted({knob.change_scope for knob in knobs}))
        candidates.append(
            CompiledCandidate(
                candidate_id=_candidate_id(
                    space.space_id,
                    space.algorithm_id,
                    knob_values,
                    deployment_settings,
                    semantic_settings,
                ),
                semantic_cohort_id=_semantic_cohort_id(
                    space.algorithm_id, semantic_settings
                ),
                knob_values=dict(sorted(knob_values.items())),
                deployment_settings=dict(sorted(deployment_settings.items())),
                semantic_settings=dict(sorted(semantic_settings.items())),
                change_scopes=change_scopes,
                tuning_layers=tuple(sorted({knob.tuning_layer for knob in knobs})),
                requires_engine_restart="engine_restart" in change_scopes,
            )
        )
    candidates.sort(key=lambda candidate: candidate.candidate_id)
    if not candidates:
        raise ValueError("all search-space candidates were rejected by constraints")
    return CompiledSearchSpace(
        space_id=space.space_id,
        algorithm_id=space.algorithm_id,
        source_space_sha256=canonical_sha256(space.to_dict()),
        capability_profile_sha256=(
            capability_profile.sha256 if capability_profile is not None else None
        ),
        allow_algorithm_changes=allow_algorithm_changes,
        max_cartesian_product=max_cartesian_product,
        cartesian_product_size=product_size,
        rejected_candidate_count=product_size - len(candidates),
        rejections_by_constraint=dict(sorted(rejection_counts.items())),
        candidates=tuple(candidates),
    )


def configuration_from_candidate(
    compiled: CompiledSearchSpace, candidate_id: str
) -> ConfigurationSpec:
    """Export a deployment-only candidate for the calibration harness."""

    candidate = next(
        (item for item in compiled.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise KeyError(f"unknown compiled candidate: {candidate_id}")
    if candidate.semantic_settings:
        raise ValueError(
            "algorithm-changing candidates require a separate semantic contract"
        )
    return ConfigurationSpec(
        configuration_id=candidate.candidate_id,
        settings=setting_map_to_dict(candidate.deployment_settings),
        description=(
            f"Compiled from {compiled.space_id} ({compiled.source_space_sha256})"
        ),
    )
