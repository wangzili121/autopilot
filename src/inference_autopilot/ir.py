"""Strict, serializable performance graph contracts for inference algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping


_SYMBOL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_COMPOUND_ARITY = {
    "add": (2, None),
    "subtract": (2, 2),
    "multiply": (2, None),
    "minimum": (2, None),
    "maximum": (2, None),
    "ceiling_divide": (2, 2),
}


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


@dataclass(frozen=True, slots=True)
class IntExpr:
    """A small integer-expression AST used instead of executable expression strings."""

    op: str
    value: int | None = None
    name: str | None = None
    args: tuple["IntExpr", ...] = ()

    def __post_init__(self) -> None:
        if self.op == "literal":
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError("literal expressions require an integer value")
            if self.name is not None or self.args:
                raise ValueError("literal expressions cannot have a name or arguments")
            return
        if self.op == "symbol":
            if not isinstance(self.name, str) or not _SYMBOL_PATTERN.fullmatch(self.name):
                raise ValueError("symbol expressions require a snake_case name")
            if self.value is not None or self.args:
                raise ValueError("symbol expressions cannot have a value or arguments")
            return
        if self.op not in _COMPOUND_ARITY:
            raise ValueError(f"unknown integer expression operation: {self.op}")
        if self.value is not None or self.name is not None:
            raise ValueError("compound expressions cannot have a value or name")
        minimum, maximum = _COMPOUND_ARITY[self.op]
        if len(self.args) < minimum or (maximum is not None and len(self.args) > maximum):
            raise ValueError(f"{self.op} received an invalid number of arguments")

    @classmethod
    def literal(cls, value: int) -> "IntExpr":
        return cls("literal", value=value)

    @classmethod
    def symbol(cls, name: str) -> "IntExpr":
        return cls("symbol", name=name)

    @classmethod
    def compound(cls, op: str, *args: "IntExpr") -> "IntExpr":
        return cls(op, args=tuple(args))

    def symbols(self) -> frozenset[str]:
        if self.op == "symbol":
            assert self.name is not None
            return frozenset((self.name,))
        return frozenset().union(*(argument.symbols() for argument in self.args))

    def evaluate(self, bindings: Mapping[str, int]) -> int:
        if self.op == "literal":
            assert self.value is not None
            return self.value
        if self.op == "symbol":
            assert self.name is not None
            if self.name not in bindings:
                raise KeyError(f"missing integer expression binding: {self.name}")
            value = bindings[self.name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"integer expression binding {self.name} must be an integer")
            return value

        values = [argument.evaluate(bindings) for argument in self.args]
        if self.op == "add":
            return sum(values)
        if self.op == "subtract":
            return values[0] - values[1]
        if self.op == "multiply":
            result = 1
            for value in values:
                result *= value
            return result
        if self.op == "minimum":
            return min(values)
        if self.op == "maximum":
            return max(values)
        if self.op == "ceiling_divide":
            if values[1] <= 0:
                raise ValueError("ceiling_divide requires a positive divisor")
            return (values[0] + values[1] - 1) // values[1]
        raise AssertionError(f"unhandled integer expression operation: {self.op}")

    def to_dict(self) -> dict[str, Any]:
        if self.op == "literal":
            return {"op": self.op, "value": self.value}
        if self.op == "symbol":
            return {"op": self.op, "name": self.name}
        return {"op": self.op, "args": [argument.to_dict() for argument in self.args]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "IntExpr":
        op = str(raw.get("op", ""))
        if op == "literal":
            _expect_exact_keys(raw, {"op", "value"}, "literal integer expression")
            return cls.literal(raw["value"])
        if op == "symbol":
            _expect_exact_keys(raw, {"op", "name"}, "symbol integer expression")
            return cls.symbol(str(raw["name"]))
        _expect_exact_keys(raw, {"op", "args"}, "compound integer expression")
        arguments = raw["args"]
        if not isinstance(arguments, list) or any(
            not isinstance(argument, Mapping) for argument in arguments
        ):
            raise ValueError("compound integer expression args must be objects")
        return cls.compound(op, *(cls.from_dict(argument) for argument in arguments))


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    value_type: str
    default: int | bool
    category: str
    description: str
    minimum: int | None = None
    tunable: bool = False
    quality_sensitive: bool = False

    def __post_init__(self) -> None:
        if not _SYMBOL_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid parameter name: {self.name}")
        if self.value_type not in {"integer", "boolean"}:
            raise ValueError(f"unsupported parameter type: {self.value_type}")
        if self.category not in {"algorithm_budget", "algorithm_semantics", "workload_state"}:
            raise ValueError(f"unsupported parameter category: {self.category}")
        if self.value_type == "integer":
            if isinstance(self.default, bool) or not isinstance(self.default, int):
                raise ValueError(f"parameter {self.name} requires an integer default")
            if self.minimum is not None and self.default < self.minimum:
                raise ValueError(f"parameter {self.name} default is below its minimum")
        elif not isinstance(self.default, bool):
            raise ValueError(f"parameter {self.name} requires a boolean default")
        if self.minimum is not None and self.value_type != "integer":
            raise ValueError("only integer parameters can define a minimum")
        if self.category == "workload_state" and self.tunable:
            raise ValueError("workload state cannot be a tuning knob")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "default": self.default,
            "minimum": self.minimum,
            "category": self.category,
            "tunable": self.tunable,
            "quality_sensitive": self.quality_sensitive,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ParameterSpec":
        _expect_exact_keys(
            raw,
            {
                "name",
                "value_type",
                "default",
                "minimum",
                "category",
                "tunable",
                "quality_sensitive",
                "description",
            },
            "graph parameter",
        )
        minimum = raw["minimum"]
        if minimum is not None and (
            isinstance(minimum, bool) or not isinstance(minimum, int)
        ):
            raise ValueError("graph parameter minimum must be an integer or null")
        return cls(
            name=str(raw["name"]),
            value_type=str(raw["value_type"]),
            default=raw["default"],
            category=str(raw["category"]),
            description=str(raw["description"]),
            minimum=minimum,
            tunable=raw["tunable"],
            quality_sensitive=raw["quality_sensitive"],
        )


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    primitive: str
    resource_role: str
    multiplicity_upper_bound: IntExpr
    token_extent_upper_bound: IntExpr | None
    depends_on: tuple[str, ...] = ()
    readiness: str = "all_dependencies"
    batching_scope: str = "engine_continuous"
    cache_key_parts: tuple[str, ...] = ()
    tunable_runtime_fields: tuple[str, ...] = ()
    semantic_effect: str = "preserves_algorithm"
    description: str = ""

    def __post_init__(self) -> None:
        if not _SYMBOL_PATTERN.fullmatch(self.stage_id):
            raise ValueError(f"invalid stage id: {self.stage_id}")
        if self.primitive not in {"generate", "score", "reward", "reduce", "select"}:
            raise ValueError(f"unsupported stage primitive: {self.primitive}")
        if self.resource_role not in {"base_model", "proposal_model", "cpu"}:
            raise ValueError(f"unsupported resource role: {self.resource_role}")
        if self.readiness != "all_dependencies":
            raise ValueError(f"unsupported readiness policy: {self.readiness}")
        if self.batching_scope not in {"engine_continuous", "host_vectorized", "single_request"}:
            raise ValueError(f"unsupported batching scope: {self.batching_scope}")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"stage {self.stage_id} has duplicate dependencies")
        if self.semantic_effect not in {"preserves_algorithm", "changes_algorithm"}:
            raise ValueError(f"unsupported semantic effect: {self.semantic_effect}")

    def expressions(self) -> tuple[IntExpr, ...]:
        if self.token_extent_upper_bound is None:
            return (self.multiplicity_upper_bound,)
        return (self.multiplicity_upper_bound, self.token_extent_upper_bound)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "primitive": self.primitive,
            "resource_role": self.resource_role,
            "multiplicity_upper_bound": self.multiplicity_upper_bound.to_dict(),
            "token_extent_upper_bound": (
                None
                if self.token_extent_upper_bound is None
                else self.token_extent_upper_bound.to_dict()
            ),
            "depends_on": list(self.depends_on),
            "readiness": self.readiness,
            "batching_scope": self.batching_scope,
            "cache_key_parts": list(self.cache_key_parts),
            "tunable_runtime_fields": list(self.tunable_runtime_fields),
            "semantic_effect": self.semantic_effect,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageSpec":
        _expect_exact_keys(
            raw,
            {
                "stage_id",
                "primitive",
                "resource_role",
                "multiplicity_upper_bound",
                "token_extent_upper_bound",
                "depends_on",
                "readiness",
                "batching_scope",
                "cache_key_parts",
                "tunable_runtime_fields",
                "semantic_effect",
                "description",
            },
            "graph stage",
        )
        list_fields = ("depends_on", "cache_key_parts", "tunable_runtime_fields")
        if any(
            not isinstance(raw[name], list)
            or any(not isinstance(item, str) for item in raw[name])
            for name in list_fields
        ):
            raise ValueError("graph stage list fields must contain strings")
        multiplicity = raw["multiplicity_upper_bound"]
        token_extent = raw["token_extent_upper_bound"]
        if not isinstance(multiplicity, Mapping):
            raise ValueError("stage multiplicity_upper_bound must be an object")
        if token_extent is not None and not isinstance(token_extent, Mapping):
            raise ValueError("stage token_extent_upper_bound must be an object or null")
        return cls(
            stage_id=str(raw["stage_id"]),
            primitive=str(raw["primitive"]),
            resource_role=str(raw["resource_role"]),
            multiplicity_upper_bound=IntExpr.from_dict(multiplicity),
            token_extent_upper_bound=(
                None if token_extent is None else IntExpr.from_dict(token_extent)
            ),
            depends_on=tuple(raw["depends_on"]),
            readiness=str(raw["readiness"]),
            batching_scope=str(raw["batching_scope"]),
            cache_key_parts=tuple(raw["cache_key_parts"]),
            tunable_runtime_fields=tuple(raw["tunable_runtime_fields"]),
            semantic_effect=str(raw["semantic_effect"]),
            description=str(raw["description"]),
        )


@dataclass(frozen=True, slots=True)
class LoopSpec:
    stage_ids: tuple[str, ...]
    maximum_iterations: IntExpr
    state_variable: str
    state_update: str
    early_exit_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_ids:
            raise ValueError("a loop must contain at least one stage")
        if not _SYMBOL_PATTERN.fullmatch(self.state_variable):
            raise ValueError(f"invalid loop state variable: {self.state_variable}")
        if not self.state_update:
            raise ValueError("a loop must describe its state update")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_ids": list(self.stage_ids),
            "maximum_iterations": self.maximum_iterations.to_dict(),
            "state_variable": self.state_variable,
            "state_update": self.state_update,
            "early_exit_conditions": list(self.early_exit_conditions),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LoopSpec":
        _expect_exact_keys(
            raw,
            {
                "stage_ids",
                "maximum_iterations",
                "state_variable",
                "state_update",
                "early_exit_conditions",
            },
            "graph loop",
        )
        stage_ids = raw["stage_ids"]
        conditions = raw["early_exit_conditions"]
        maximum_iterations = raw["maximum_iterations"]
        if not isinstance(stage_ids, list) or any(
            not isinstance(item, str) for item in stage_ids
        ):
            raise ValueError("graph loop stage_ids must contain strings")
        if not isinstance(conditions, list) or any(
            not isinstance(item, str) for item in conditions
        ):
            raise ValueError("graph loop early_exit_conditions must contain strings")
        if not isinstance(maximum_iterations, Mapping):
            raise ValueError("graph loop maximum_iterations must be an object")
        return cls(
            stage_ids=tuple(stage_ids),
            maximum_iterations=IntExpr.from_dict(maximum_iterations),
            state_variable=str(raw["state_variable"]),
            state_update=str(raw["state_update"]),
            early_exit_conditions=tuple(conditions),
        )


@dataclass(frozen=True, slots=True)
class InferenceGraph:
    algorithm_id: str
    algorithm_semantics: str
    parameters: tuple[ParameterSpec, ...]
    stages: tuple[StageSpec, ...]
    loop: LoopSpec
    source_contract: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.algorithm_id:
            raise ValueError("algorithm_id cannot be empty")
        if self.algorithm_semantics not in {"exact", "biased_ablation"}:
            raise ValueError(f"unsupported algorithm semantics: {self.algorithm_semantics}")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("graph parameter names must be unique")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("graph stage ids must be unique")
        if set(self.loop.stage_ids) != set(stage_ids) or len(self.loop.stage_ids) != len(stage_ids):
            raise ValueError("loop stage_ids must contain every graph stage exactly once")

        known_stages = set(stage_ids)
        for stage in self.stages:
            unknown = sorted(set(stage.depends_on) - known_stages)
            if unknown:
                raise ValueError(f"stage {stage.stage_id} has unknown dependencies: {unknown}")
            if stage.stage_id in stage.depends_on:
                raise ValueError(f"stage {stage.stage_id} cannot depend on itself")
        self._validate_acyclic()

        known_symbols = set(parameter_names)
        expressions = [self.loop.maximum_iterations]
        expressions.extend(
            expression for stage in self.stages for expression in stage.expressions()
        )
        unknown_symbols = sorted(
            set().union(*(expression.symbols() for expression in expressions)) - known_symbols
        )
        if unknown_symbols:
            raise ValueError(f"graph expressions use undeclared symbols: {unknown_symbols}")

    def _validate_acyclic(self) -> None:
        dependencies = {stage.stage_id: set(stage.depends_on) for stage in self.stages}
        ready = [stage_id for stage_id, parents in dependencies.items() if not parents]
        visited = 0
        while ready:
            stage_id = ready.pop()
            visited += 1
            for child_id, parents in dependencies.items():
                if stage_id in parents:
                    parents.remove(stage_id)
                    if not parents:
                        ready.append(child_id)
        if visited != len(self.stages):
            raise ValueError("graph stage dependencies contain a cycle")

    @property
    def bindings(self) -> dict[str, int | bool]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "algorithm_semantics": self.algorithm_semantics,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "stages": [stage.to_dict() for stage in self.stages],
            "loop": self.loop.to_dict(),
            "source_contract": list(self.source_contract),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InferenceGraph":
        _expect_exact_keys(
            raw,
            {
                "schema_version",
                "algorithm_id",
                "algorithm_semantics",
                "parameters",
                "stages",
                "loop",
                "source_contract",
                "metadata",
            },
            "inference graph",
        )
        parameters = raw["parameters"]
        stages = raw["stages"]
        loop = raw["loop"]
        source_contract = raw["source_contract"]
        metadata = raw["metadata"]
        if not isinstance(parameters, list) or any(
            not isinstance(item, Mapping) for item in parameters
        ):
            raise ValueError("inference graph parameters must be objects")
        if not isinstance(stages, list) or any(
            not isinstance(item, Mapping) for item in stages
        ):
            raise ValueError("inference graph stages must be objects")
        if not isinstance(loop, Mapping):
            raise ValueError("inference graph loop must be an object")
        if not isinstance(source_contract, list) or any(
            not isinstance(item, str) for item in source_contract
        ):
            raise ValueError("inference graph source_contract must contain strings")
        if not isinstance(metadata, Mapping):
            raise ValueError("inference graph metadata must be an object")
        return cls(
            schema_version=str(raw["schema_version"]),
            algorithm_id=str(raw["algorithm_id"]),
            algorithm_semantics=str(raw["algorithm_semantics"]),
            parameters=tuple(ParameterSpec.from_dict(item) for item in parameters),
            stages=tuple(StageSpec.from_dict(item) for item in stages),
            loop=LoopSpec.from_dict(loop),
            source_contract=tuple(source_contract),
            metadata=dict(metadata),
        )
