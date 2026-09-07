"""Strict data contracts for reproducible paired calibration runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import re
from typing import Any


_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def require_digest(value: str, name: str) -> None:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def require_id(value: str, name: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must contain lowercase letters, numbers, '.', '_' or '-'")


def expect_keys(
    raw: Mapping[str, Any],
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - optional)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def require_object(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be an object")
    result = dict(raw)
    canonical_json(result)
    return result


def require_string_list(raw: Any, context: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(raw)


@dataclass(frozen=True, slots=True)
class ConfigurationSpec:
    configuration_id: str
    settings: Mapping[str, Any]
    description: str = ""

    def __post_init__(self) -> None:
        require_id(self.configuration_id, "configuration_id")
        if not self.settings:
            raise ValueError("configuration settings cannot be empty")
        canonical_json(dict(self.settings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "settings": dict(self.settings),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConfigurationSpec":
        expect_keys(
            raw,
            {"configuration_id", "settings", "description"},
            "configuration",
        )
        return cls(
            configuration_id=str(raw["configuration_id"]),
            settings=require_object(raw["settings"], "configuration settings"),
            description=str(raw["description"]),
        )


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    pattern: str
    blocks: int
    pair_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.pattern not in {"ABBA", "BAAB"}:
            raise ValueError("protocol pattern must be ABBA or BAAB")
        if isinstance(self.blocks, bool) or self.blocks <= 0:
            raise ValueError("protocol blocks must be a positive integer")
        if len(self.pair_seeds) != self.blocks * 2:
            raise ValueError("protocol requires exactly two pair seeds per block")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.pair_seeds):
            raise ValueError("pair seeds must be non-negative integers")
        if len(set(self.pair_seeds)) != len(self.pair_seeds):
            raise ValueError("pair seeds must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "blocks": self.blocks,
            "pair_seeds": list(self.pair_seeds),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProtocolSpec":
        expect_keys(raw, {"pattern", "blocks", "pair_seeds"}, "protocol")
        pair_seeds = raw["pair_seeds"]
        if not isinstance(pair_seeds, list):
            raise ValueError("protocol pair_seeds must be a list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in pair_seeds):
            raise ValueError("protocol pair_seeds must contain integers")
        blocks = raw["blocks"]
        if isinstance(blocks, bool) or not isinstance(blocks, int):
            raise ValueError("protocol blocks must be an integer")
        return cls(str(raw["pattern"]), blocks, tuple(pair_seeds))


@dataclass(frozen=True, slots=True)
class SemanticContract:
    algorithm_id: str
    semantic_class: str
    graph_sha256: str
    invariants: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_id(self.algorithm_id, "algorithm_id")
        if self.semantic_class not in {"exact", "approximate", "numerical_variant"}:
            raise ValueError(f"unsupported semantic class: {self.semantic_class}")
        require_digest(self.graph_sha256, "graph_sha256")
        if not self.invariants:
            raise ValueError("semantic invariants cannot be empty")
        canonical_json(dict(self.invariants))

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "semantic_class": self.semantic_class,
            "graph_sha256": self.graph_sha256,
            "invariants": dict(self.invariants),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SemanticContract":
        expect_keys(
            raw,
            {"algorithm_id", "semantic_class", "graph_sha256", "invariants"},
            "semantic contract",
        )
        return cls(
            algorithm_id=str(raw["algorithm_id"]),
            semantic_class=str(raw["semantic_class"]),
            graph_sha256=str(raw["graph_sha256"]),
            invariants=require_object(raw["invariants"], "semantic invariants"),
        )


@dataclass(frozen=True, slots=True)
class WorkloadContract:
    workload_id: str
    dataset_sha256: str
    arrival_trace_sha256: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_id(self.workload_id, "workload_id")
        require_digest(self.dataset_sha256, "dataset_sha256")
        require_digest(self.arrival_trace_sha256, "arrival_trace_sha256")
        if not self.parameters:
            raise ValueError("workload parameters cannot be empty")
        canonical_json(dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "dataset_sha256": self.dataset_sha256,
            "arrival_trace_sha256": self.arrival_trace_sha256,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkloadContract":
        expect_keys(
            raw,
            {"workload_id", "dataset_sha256", "arrival_trace_sha256", "parameters"},
            "workload contract",
        )
        return cls(
            workload_id=str(raw["workload_id"]),
            dataset_sha256=str(raw["dataset_sha256"]),
            arrival_trace_sha256=str(raw["arrival_trace_sha256"]),
            parameters=require_object(raw["parameters"], "workload parameters"),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    environment_id: str
    hardware: Mapping[str, Any]
    software: Mapping[str, Any]
    models: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_id(self.environment_id, "environment_id")
        for name in ("hardware", "software", "models"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"environment {name} cannot be empty")
            canonical_json(dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "hardware": dict(self.hardware),
            "software": dict(self.software),
            "models": dict(self.models),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EnvironmentContract":
        expect_keys(
            raw,
            {"environment_id", "hardware", "software", "models"},
            "environment contract",
        )
        return cls(
            environment_id=str(raw["environment_id"]),
            hardware=require_object(raw["hardware"], "environment hardware"),
            software=require_object(raw["software"], "environment software"),
            models=require_object(raw["models"], "environment models"),
        )


@dataclass(frozen=True, slots=True)
class MetricConstraint:
    metric: str
    operator: str
    value: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("constraint metric cannot be empty")
        if self.operator not in {"<=", ">="}:
            raise ValueError("constraint operator must be <= or >=")
        if isinstance(self.value, bool) or not isfinite(self.value):
            raise ValueError("constraint value must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "operator": self.operator, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricConstraint":
        expect_keys(raw, {"metric", "operator", "value"}, "metric constraint")
        value = raw["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("constraint value must be numeric")
        return cls(str(raw["metric"]), str(raw["operator"]), float(value))


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    primary_metric: str
    direction: str
    constraints: tuple[MetricConstraint, ...]

    def __post_init__(self) -> None:
        if not self.primary_metric:
            raise ValueError("primary metric cannot be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("objective direction must be maximize or minimize")
        metrics = [constraint.metric for constraint in self.constraints]
        if len(metrics) != len(set(metrics)):
            raise ValueError("objective constraint metrics must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObjectiveSpec":
        expect_keys(raw, {"primary_metric", "direction", "constraints"}, "objective")
        constraints = raw["constraints"]
        if not isinstance(constraints, list):
            raise ValueError("objective constraints must be a list")
        return cls(
            str(raw["primary_metric"]),
            str(raw["direction"]),
            tuple(
                MetricConstraint.from_dict(require_object(item, "constraint"))
                for item in constraints
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    campaign_id: str
    strong_baseline: bool
    protocol: ProtocolSpec
    semantic_contract: SemanticContract
    workload_contract: WorkloadContract
    environment_contract: EnvironmentContract
    objective: ObjectiveSpec
    required_metrics: tuple[str, ...]
    baseline: ConfigurationSpec
    candidates: tuple[ConfigurationSpec, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported calibration spec schema: {self.schema_version}")
        require_id(self.campaign_id, "campaign_id")
        if not isinstance(self.strong_baseline, bool):
            raise ValueError("strong_baseline must be a boolean")
        if not self.candidates:
            raise ValueError("calibration spec requires at least one candidate")
        configuration_ids = [self.baseline.configuration_id]
        configuration_ids.extend(candidate.configuration_id for candidate in self.candidates)
        if len(configuration_ids) != len(set(configuration_ids)):
            raise ValueError("calibration configuration ids must be unique")
        configurations = (self.baseline, *self.candidates)
        for configuration in configurations:
            conflicts = sorted(
                key
                for key, expected in self.semantic_contract.invariants.items()
                if key in configuration.settings
                and configuration.settings[key] != expected
            )
            if conflicts:
                raise ValueError(
                    f"configuration {configuration.configuration_id} overrides semantic "
                    f"invariants: {conflicts}"
                )
        if not self.required_metrics or len(self.required_metrics) != len(
            set(self.required_metrics)
        ):
            raise ValueError("required_metrics must be non-empty and unique")
        objective_metrics = {self.objective.primary_metric}
        objective_metrics.update(constraint.metric for constraint in self.objective.constraints)
        missing_metrics = sorted(objective_metrics - set(self.required_metrics))
        if missing_metrics:
            raise ValueError(f"required_metrics omit objective metrics: {missing_metrics}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "strong_baseline": self.strong_baseline,
            "protocol": self.protocol.to_dict(),
            "semantic_contract": self.semantic_contract.to_dict(),
            "workload_contract": self.workload_contract.to_dict(),
            "environment_contract": self.environment_contract.to_dict(),
            "objective": self.objective.to_dict(),
            "required_metrics": list(self.required_metrics),
            "baseline": self.baseline.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationSpec":
        required = {
            "schema_version",
            "campaign_id",
            "strong_baseline",
            "protocol",
            "semantic_contract",
            "workload_contract",
            "environment_contract",
            "objective",
            "required_metrics",
            "baseline",
            "candidates",
        }
        expect_keys(raw, required, "calibration spec")
        candidates = raw["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("calibration candidates must be a list")
        if not isinstance(raw["strong_baseline"], bool):
            raise ValueError("strong_baseline must be a boolean")
        return cls(
            schema_version=str(raw["schema_version"]),
            campaign_id=str(raw["campaign_id"]),
            strong_baseline=raw["strong_baseline"],
            protocol=ProtocolSpec.from_dict(require_object(raw["protocol"], "protocol")),
            semantic_contract=SemanticContract.from_dict(
                require_object(raw["semantic_contract"], "semantic contract")
            ),
            workload_contract=WorkloadContract.from_dict(
                require_object(raw["workload_contract"], "workload contract")
            ),
            environment_contract=EnvironmentContract.from_dict(
                require_object(raw["environment_contract"], "environment contract")
            ),
            objective=ObjectiveSpec.from_dict(require_object(raw["objective"], "objective")),
            required_metrics=require_string_list(raw["required_metrics"], "required_metrics"),
            baseline=ConfigurationSpec.from_dict(
                require_object(raw["baseline"], "baseline configuration")
            ),
            candidates=tuple(
                ConfigurationSpec.from_dict(require_object(item, "candidate configuration"))
                for item in candidates
            ),
        )


@dataclass(frozen=True, slots=True)
class PlannedRun:
    run_id: str
    comparison_group_id: str
    sequence_index: int
    group_sequence_index: int
    pair_index: int
    block_index: int
    variant_role: str
    configuration_id: str
    workload_seed: int
    expected_result_path: str

    def __post_init__(self) -> None:
        require_id(self.run_id, "run_id")
        require_id(self.comparison_group_id, "comparison_group_id")
        require_id(self.configuration_id, "configuration_id")
        for name in ("sequence_index", "group_sequence_index", "pair_index", "block_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.variant_role not in {"baseline", "candidate"}:
            raise ValueError("variant_role must be baseline or candidate")
        if isinstance(self.workload_seed, bool) or self.workload_seed < 0:
            raise ValueError("workload_seed must be a non-negative integer")
        if not self.expected_result_path:
            raise ValueError("expected_result_path cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "comparison_group_id": self.comparison_group_id,
            "sequence_index": self.sequence_index,
            "group_sequence_index": self.group_sequence_index,
            "pair_index": self.pair_index,
            "block_index": self.block_index,
            "variant_role": self.variant_role,
            "configuration_id": self.configuration_id,
            "workload_seed": self.workload_seed,
            "expected_result_path": self.expected_result_path,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlannedRun":
        required = {
            "run_id",
            "comparison_group_id",
            "sequence_index",
            "group_sequence_index",
            "pair_index",
            "block_index",
            "variant_role",
            "configuration_id",
            "workload_seed",
            "expected_result_path",
        }
        expect_keys(raw, required, "planned run")
        integer_fields = (
            "sequence_index",
            "group_sequence_index",
            "pair_index",
            "block_index",
            "workload_seed",
        )
        if any(
            isinstance(raw[name], bool) or not isinstance(raw[name], int)
            for name in integer_fields
        ):
            raise ValueError("planned run integer fields must contain integers")
        return cls(
            run_id=str(raw["run_id"]),
            comparison_group_id=str(raw["comparison_group_id"]),
            sequence_index=raw["sequence_index"],
            group_sequence_index=raw["group_sequence_index"],
            pair_index=raw["pair_index"],
            block_index=raw["block_index"],
            variant_role=str(raw["variant_role"]),
            configuration_id=str(raw["configuration_id"]),
            workload_seed=raw["workload_seed"],
            expected_result_path=str(raw["expected_result_path"]),
        )


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    spec_sha256: str
    spec: CalibrationSpec
    runs: tuple[PlannedRun, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported calibration plan schema: {self.schema_version}")
        require_digest(self.spec_sha256, "spec_sha256")
        if self.spec_sha256 != canonical_sha256(self.spec.to_dict()):
            raise ValueError("calibration plan spec_sha256 does not match embedded spec")
        if not self.runs:
            raise ValueError("calibration plan requires runs")
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("calibration plan run ids must be unique")
        sequence = [run.sequence_index for run in self.runs]
        if sequence != list(range(len(self.runs))):
            raise ValueError("calibration plan sequence indexes must be contiguous")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec_sha256": self.spec_sha256,
            "spec": self.spec.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationPlan":
        expect_keys(raw, {"schema_version", "spec_sha256", "spec", "runs"}, "calibration plan")
        runs = raw["runs"]
        if not isinstance(runs, list):
            raise ValueError("calibration plan runs must be a list")
        return cls(
            schema_version=str(raw["schema_version"]),
            spec_sha256=str(raw["spec_sha256"]),
            spec=CalibrationSpec.from_dict(require_object(raw["spec"], "embedded spec")),
            runs=tuple(PlannedRun.from_dict(require_object(item, "planned run")) for item in runs),
        )


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("artifact path cannot be empty")
        require_digest(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactReference":
        expect_keys(raw, {"path", "sha256"}, "artifact reference")
        return cls(str(raw["path"]), str(raw["sha256"]))


@dataclass(frozen=True, slots=True)
class RunObservation:
    run_id: str
    run_manifest_sha256: str
    started_at_unix: float
    finished_at_unix: float
    status: str
    metrics: Mapping[str, Any]
    artifact: ArtifactReference | None = None
    notes: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported run observation schema: {self.schema_version}")
        require_id(self.run_id, "run_id")
        require_digest(self.run_manifest_sha256, "run_manifest_sha256")
        if not isfinite(self.started_at_unix) or not isfinite(self.finished_at_unix):
            raise ValueError("observation timestamps must be finite")
        if self.finished_at_unix <= self.started_at_unix:
            raise ValueError("observation finish time must follow start time")
        if self.status not in {"success", "oom", "crash", "cancelled"}:
            raise ValueError(f"unsupported observation status: {self.status}")
        if self.status == "success" and not self.metrics:
            raise ValueError("successful observations require metrics")
        canonical_json(dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_manifest_sha256": self.run_manifest_sha256,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "status": self.status,
            "metrics": dict(self.metrics),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunObservation":
        required = {
            "schema_version",
            "run_id",
            "run_manifest_sha256",
            "started_at_unix",
            "finished_at_unix",
            "status",
            "metrics",
            "artifact",
            "notes",
        }
        expect_keys(raw, required, "run observation")
        for name in ("started_at_unix", "finished_at_unix"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
                raise ValueError(f"{name} must be numeric")
        artifact_raw = raw["artifact"]
        artifact = (
            None
            if artifact_raw is None
            else ArtifactReference.from_dict(require_object(artifact_raw, "artifact"))
        )
        return cls(
            schema_version=str(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            run_manifest_sha256=str(raw["run_manifest_sha256"]),
            started_at_unix=float(raw["started_at_unix"]),
            finished_at_unix=float(raw["finished_at_unix"]),
            status=str(raw["status"]),
            metrics=require_object(raw["metrics"], "observation metrics"),
            artifact=artifact,
            notes=require_string_list(raw["notes"], "observation notes"),
        )
