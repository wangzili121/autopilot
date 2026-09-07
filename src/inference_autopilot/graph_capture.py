"""Stage-aware planning of CUDA or ACL graph capture buckets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from inference_autopilot.calibration.models import canonical_sha256, require_digest, require_id


_GIB = 1024**3


def _exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _nonnegative_float(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{context} must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True, slots=True)
class ShapeObservation:
    stage_id: str
    shape_size: int
    count: int
    eager_latency_ms: float

    def __post_init__(self) -> None:
        require_id(self.stage_id, "shape observation stage_id")
        _positive_int(self.shape_size, "shape_size")
        _positive_int(self.count, "shape observation count")
        _nonnegative_float(self.eager_latency_ms, "eager_latency_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "shape_size": self.shape_size,
            "count": self.count,
            "eager_latency_ms": self.eager_latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ShapeObservation":
        _exact_keys(
            raw,
            {"stage_id", "shape_size", "count", "eager_latency_ms"},
            "shape observation",
        )
        return cls(
            stage_id=str(raw["stage_id"]),
            shape_size=_positive_int(raw["shape_size"], "shape_size"),
            count=_positive_int(raw["count"], "shape observation count"),
            eager_latency_ms=_nonnegative_float(
                raw["eager_latency_ms"], "eager_latency_ms"
            ),
        )


@dataclass(frozen=True, slots=True)
class CaptureCandidate:
    capture_size: int
    memory_bytes: int
    capture_time_ms: float
    replay_latency_ms_by_stage: Mapping[str, float]

    def __post_init__(self) -> None:
        _positive_int(self.capture_size, "capture_size")
        _nonnegative_int(self.memory_bytes, "capture candidate memory_bytes")
        _nonnegative_float(self.capture_time_ms, "capture_time_ms")
        if not self.replay_latency_ms_by_stage:
            raise ValueError("capture candidate requires stage replay latencies")
        for stage_id, latency in self.replay_latency_ms_by_stage.items():
            require_id(stage_id, "capture candidate stage_id")
            _nonnegative_float(latency, f"replay latency for {stage_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_size": self.capture_size,
            "memory_bytes": self.memory_bytes,
            "capture_time_ms": self.capture_time_ms,
            "replay_latency_ms_by_stage": dict(
                sorted(self.replay_latency_ms_by_stage.items())
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureCandidate":
        _exact_keys(
            raw,
            {
                "capture_size",
                "memory_bytes",
                "capture_time_ms",
                "replay_latency_ms_by_stage",
            },
            "capture candidate",
        )
        replay = raw["replay_latency_ms_by_stage"]
        if not isinstance(replay, Mapping):
            raise ValueError("replay_latency_ms_by_stage must be an object")
        return cls(
            capture_size=_positive_int(raw["capture_size"], "capture_size"),
            memory_bytes=_nonnegative_int(raw["memory_bytes"], "memory_bytes"),
            capture_time_ms=_nonnegative_float(raw["capture_time_ms"], "capture_time_ms"),
            replay_latency_ms_by_stage={
                str(stage): _nonnegative_float(value, f"replay latency for {stage}")
                for stage, value in replay.items()
            },
        )


@dataclass(frozen=True, slots=True)
class CaptureBudget:
    max_buckets: int
    memory_budget_bytes: int
    capture_time_budget_ms: float
    max_padding_ratio: float
    minimum_graph_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative_int(self.max_buckets, "max_buckets")
        _nonnegative_int(self.memory_budget_bytes, "memory_budget_bytes")
        _nonnegative_float(self.capture_time_budget_ms, "capture_time_budget_ms")
        _nonnegative_float(self.max_padding_ratio, "max_padding_ratio")
        if not 0.0 <= self.minimum_graph_hit_rate <= 1.0:
            raise ValueError("minimum_graph_hit_rate must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_buckets": self.max_buckets,
            "memory_budget_bytes": self.memory_budget_bytes,
            "capture_time_budget_ms": self.capture_time_budget_ms,
            "max_padding_ratio": self.max_padding_ratio,
            "minimum_graph_hit_rate": self.minimum_graph_hit_rate,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureBudget":
        _exact_keys(
            raw,
            {
                "max_buckets",
                "memory_budget_bytes",
                "capture_time_budget_ms",
                "max_padding_ratio",
                "minimum_graph_hit_rate",
            },
            "capture budget",
        )
        return cls(
            max_buckets=_nonnegative_int(raw["max_buckets"], "max_buckets"),
            memory_budget_bytes=_nonnegative_int(
                raw["memory_budget_bytes"], "memory_budget_bytes"
            ),
            capture_time_budget_ms=_nonnegative_float(
                raw["capture_time_budget_ms"], "capture_time_budget_ms"
            ),
            max_padding_ratio=_nonnegative_float(
                raw["max_padding_ratio"], "max_padding_ratio"
            ),
            minimum_graph_hit_rate=_nonnegative_float(
                raw["minimum_graph_hit_rate"], "minimum_graph_hit_rate"
            ),
        )


@dataclass(frozen=True, slots=True)
class CaptureObjective:
    amortization_windows: int
    padding_penalty_ms_per_unit: float = 0.0
    memory_penalty_ms_per_gib: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.amortization_windows, "amortization_windows")
        _nonnegative_float(
            self.padding_penalty_ms_per_unit, "padding_penalty_ms_per_unit"
        )
        _nonnegative_float(self.memory_penalty_ms_per_gib, "memory_penalty_ms_per_gib")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amortization_windows": self.amortization_windows,
            "padding_penalty_ms_per_unit": self.padding_penalty_ms_per_unit,
            "memory_penalty_ms_per_gib": self.memory_penalty_ms_per_gib,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureObjective":
        _exact_keys(
            raw,
            {
                "amortization_windows",
                "padding_penalty_ms_per_unit",
                "memory_penalty_ms_per_gib",
            },
            "capture objective",
        )
        return cls(
            amortization_windows=_positive_int(
                raw["amortization_windows"], "amortization_windows"
            ),
            padding_penalty_ms_per_unit=_nonnegative_float(
                raw["padding_penalty_ms_per_unit"], "padding_penalty_ms_per_unit"
            ),
            memory_penalty_ms_per_gib=_nonnegative_float(
                raw["memory_penalty_ms_per_gib"], "memory_penalty_ms_per_gib"
            ),
        )


@dataclass(frozen=True, slots=True)
class CapturePlanningSpec:
    profile_id: str
    algorithm_id: str
    engine_role: str
    backend: str
    graph_mode: str
    observations: tuple[ShapeObservation, ...]
    candidates: tuple[CaptureCandidate, ...]
    budget: CaptureBudget
    objective: CaptureObjective
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported capture planning spec: {self.schema_version}")
        require_id(self.profile_id, "capture profile_id")
        require_id(self.algorithm_id, "capture algorithm_id")
        require_id(self.engine_role, "capture engine_role")
        if self.backend not in {"acl", "cuda", "generic"}:
            raise ValueError(f"unsupported graph backend: {self.backend}")
        if not self.graph_mode:
            raise ValueError("graph_mode cannot be empty")
        if not self.observations or not self.candidates:
            raise ValueError("capture planning requires observations and candidates")
        observation_keys = [
            (observation.stage_id, observation.shape_size)
            for observation in self.observations
        ]
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("shape observations must be unique by stage and shape_size")
        capture_sizes = [candidate.capture_size for candidate in self.candidates]
        if len(capture_sizes) != len(set(capture_sizes)):
            raise ValueError("capture candidate sizes must be unique")
        stages = {observation.stage_id for observation in self.observations}
        for candidate in self.candidates:
            candidate_stages = set(candidate.replay_latency_ms_by_stage)
            if candidate_stages != stages:
                raise ValueError(
                    f"capture size {candidate.capture_size} stage profile mismatch; "
                    f"missing={sorted(stages - candidate_stages)}, "
                    f"unknown={sorted(candidate_stages - stages)}"
                )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "algorithm_id": self.algorithm_id,
            "engine_role": self.engine_role,
            "backend": self.backend,
            "graph_mode": self.graph_mode,
            "observations": [
                observation.to_dict()
                for observation in sorted(
                    self.observations,
                    key=lambda item: (item.shape_size, item.stage_id),
                )
            ],
            "candidates": [
                candidate.to_dict()
                for candidate in sorted(self.candidates, key=lambda item: item.capture_size)
            ],
            "budget": self.budget.to_dict(),
            "objective": self.objective.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapturePlanningSpec":
        _exact_keys(
            raw,
            {
                "schema_version",
                "profile_id",
                "algorithm_id",
                "engine_role",
                "backend",
                "graph_mode",
                "observations",
                "candidates",
                "budget",
                "objective",
            },
            "capture planning spec",
        )
        observations = raw["observations"]
        candidates = raw["candidates"]
        if not isinstance(observations, list) or any(
            not isinstance(item, Mapping) for item in observations
        ):
            raise ValueError("capture observations must be a list of objects")
        if not isinstance(candidates, list) or any(
            not isinstance(item, Mapping) for item in candidates
        ):
            raise ValueError("capture candidates must be a list of objects")
        if not isinstance(raw["budget"], Mapping) or not isinstance(
            raw["objective"], Mapping
        ):
            raise ValueError("capture budget and objective must be objects")
        return cls(
            schema_version=str(raw["schema_version"]),
            profile_id=str(raw["profile_id"]),
            algorithm_id=str(raw["algorithm_id"]),
            engine_role=str(raw["engine_role"]),
            backend=str(raw["backend"]),
            graph_mode=str(raw["graph_mode"]),
            observations=tuple(ShapeObservation.from_dict(item) for item in observations),
            candidates=tuple(CaptureCandidate.from_dict(item) for item in candidates),
            budget=CaptureBudget.from_dict(raw["budget"]),
            objective=CaptureObjective.from_dict(raw["objective"]),
        )


@dataclass(frozen=True, slots=True)
class CaptureAssignment:
    stage_id: str
    shape_size: int
    count: int
    execution: str
    capture_size: int | None
    latency_ms: float

    def __post_init__(self) -> None:
        require_id(self.stage_id, "capture assignment stage_id")
        _positive_int(self.shape_size, "assignment shape_size")
        _positive_int(self.count, "assignment count")
        if self.execution not in {"captured", "eager"}:
            raise ValueError(f"unsupported capture execution: {self.execution}")
        if (self.execution == "captured") != (self.capture_size is not None):
            raise ValueError("captured assignments require a capture_size")
        if self.capture_size is not None:
            _positive_int(self.capture_size, "assignment capture_size")
        _nonnegative_float(self.latency_ms, "assignment latency_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "shape_size": self.shape_size,
            "count": self.count,
            "execution": self.execution,
            "capture_size": self.capture_size,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureAssignment":
        _exact_keys(
            raw,
            {"stage_id", "shape_size", "count", "execution", "capture_size", "latency_ms"},
            "capture assignment",
        )
        capture_size = raw["capture_size"]
        if capture_size is not None:
            capture_size = _positive_int(capture_size, "assignment capture_size")
        return cls(
            stage_id=str(raw["stage_id"]),
            shape_size=_positive_int(raw["shape_size"], "assignment shape_size"),
            count=_positive_int(raw["count"], "assignment count"),
            execution=str(raw["execution"]),
            capture_size=capture_size,
            latency_ms=_nonnegative_float(raw["latency_ms"], "assignment latency_ms"),
        )


@dataclass(frozen=True, slots=True)
class CapturePlan:
    profile_id: str
    source_spec_sha256: str
    selected_capture_sizes: tuple[int, ...]
    assignments: tuple[CaptureAssignment, ...]
    metrics: Mapping[str, int | float]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        require_id(self.profile_id, "capture plan profile_id")
        require_digest(self.source_spec_sha256, "capture plan source_spec_sha256")
        if tuple(sorted(set(self.selected_capture_sizes))) != self.selected_capture_sizes:
            raise ValueError("selected capture sizes must be sorted and unique")
        for capture_size in self.selected_capture_sizes:
            _positive_int(capture_size, "selected capture size")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in self.metrics.values()
        ):
            raise ValueError("capture plan metrics must be finite numbers")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "profile_id": self.profile_id,
            "source_spec_sha256": self.source_spec_sha256,
            "selected_capture_sizes": list(self.selected_capture_sizes),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "metrics": dict(sorted(self.metrics.items())),
        }
        return {**payload, "capture_plan_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapturePlan":
        _exact_keys(
            raw,
            {
                "schema_version",
                "producer",
                "profile_id",
                "source_spec_sha256",
                "selected_capture_sizes",
                "assignments",
                "metrics",
                "capture_plan_sha256",
            },
            "capture plan",
        )
        payload = dict(raw)
        digest = str(payload.pop("capture_plan_sha256"))
        if digest != canonical_sha256(payload):
            raise ValueError("capture plan SHA256 does not match its content")
        assignments = raw["assignments"]
        selected = raw["selected_capture_sizes"]
        metrics = raw["metrics"]
        if not isinstance(assignments, list) or any(
            not isinstance(item, Mapping) for item in assignments
        ):
            raise ValueError("capture assignments must be a list of objects")
        if not isinstance(selected, list):
            raise ValueError("selected_capture_sizes must be a list")
        if not isinstance(metrics, Mapping):
            raise ValueError("capture plan metrics must be an object")
        return cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            profile_id=str(raw["profile_id"]),
            source_spec_sha256=str(raw["source_spec_sha256"]),
            selected_capture_sizes=tuple(
                _positive_int(value, "selected capture size") for value in selected
            ),
            assignments=tuple(CaptureAssignment.from_dict(item) for item in assignments),
            metrics=dict(metrics),
        )


@dataclass(frozen=True, slots=True)
class _Label:
    indices: tuple[int, ...]
    memory_bytes: int
    capture_time_ms: float
    runtime_latency_ms: float
    captured_count: int
    padding_units: int
    objective_ms: float


def _dominates(left: _Label, right: _Label) -> bool:
    return (
        left.memory_bytes <= right.memory_bytes
        and left.capture_time_ms <= right.capture_time_ms
        and left.objective_ms <= right.objective_ms + 1e-12
        and left.captured_count >= right.captured_count
        and (
            left.memory_bytes < right.memory_bytes
            or left.capture_time_ms < right.capture_time_ms
            or left.objective_ms < right.objective_ms - 1e-12
            or left.captured_count > right.captured_count
        )
    )


def _insert_pareto(frontier: list[_Label], candidate: _Label) -> None:
    if any(_dominates(existing, candidate) for existing in frontier):
        return
    frontier[:] = [existing for existing in frontier if not _dominates(candidate, existing)]
    frontier.append(candidate)
    frontier.sort(key=lambda label: (label.objective_ms, label.indices))


def _segment_cost(
    observations: tuple[ShapeObservation, ...],
    lower_exclusive: int,
    candidate: CaptureCandidate,
    spec: CapturePlanningSpec,
) -> tuple[float, int, int]:
    runtime = 0.0
    captured_count = 0
    padding_units = 0
    for observation in observations:
        if not lower_exclusive < observation.shape_size <= candidate.capture_size:
            continue
        padding_ratio = (candidate.capture_size - observation.shape_size) / observation.shape_size
        if padding_ratio <= spec.budget.max_padding_ratio + 1e-12:
            runtime += (
                observation.count
                * candidate.replay_latency_ms_by_stage[observation.stage_id]
            )
            captured_count += observation.count
            padding_units += observation.count * (
                candidate.capture_size - observation.shape_size
            )
        else:
            runtime += observation.count * observation.eager_latency_ms
    return runtime, captured_count, padding_units


def _fixed_objective_cost(candidate: CaptureCandidate, spec: CapturePlanningSpec) -> float:
    return (
        candidate.capture_time_ms / spec.objective.amortization_windows
        + candidate.memory_bytes / _GIB * spec.objective.memory_penalty_ms_per_gib
    )


def _assignments(
    spec: CapturePlanningSpec, selected: tuple[CaptureCandidate, ...]
) -> tuple[CaptureAssignment, ...]:
    assignments: list[CaptureAssignment] = []
    for observation in sorted(
        spec.observations, key=lambda item: (item.shape_size, item.stage_id)
    ):
        candidate = next(
            (
                item
                for item in selected
                if item.capture_size >= observation.shape_size
            ),
            None,
        )
        captured = candidate is not None and (
            (candidate.capture_size - observation.shape_size) / observation.shape_size
            <= spec.budget.max_padding_ratio + 1e-12
        )
        assignments.append(
            CaptureAssignment(
                stage_id=observation.stage_id,
                shape_size=observation.shape_size,
                count=observation.count,
                execution="captured" if captured else "eager",
                capture_size=candidate.capture_size if captured else None,
                latency_ms=(
                    candidate.replay_latency_ms_by_stage[observation.stage_id]
                    if captured
                    else observation.eager_latency_ms
                ),
            )
        )
    return tuple(assignments)


def plan_graph_capture(spec: CapturePlanningSpec) -> CapturePlan:
    """Find an exact Pareto-pruned bucket plan under hard resource budgets."""

    observations = tuple(
        sorted(spec.observations, key=lambda item: (item.shape_size, item.stage_id))
    )
    candidates = tuple(sorted(spec.candidates, key=lambda item: item.capture_size))
    total_count = sum(observation.count for observation in observations)
    baseline_latency = sum(
        observation.count * observation.eager_latency_ms
        for observation in observations
    )
    max_buckets = min(spec.budget.max_buckets, len(candidates))
    states: dict[tuple[int, int], list[_Label]] = {}

    for candidate_index, candidate in enumerate(candidates):
        if max_buckets == 0:
            break
        if (
            candidate.memory_bytes > spec.budget.memory_budget_bytes
            or candidate.capture_time_ms > spec.budget.capture_time_budget_ms
        ):
            continue
        runtime, captured, padding = _segment_cost(
            observations, 0, candidate, spec
        )
        objective = (
            runtime
            + padding * spec.objective.padding_penalty_ms_per_unit
            + _fixed_objective_cost(candidate, spec)
        )
        _insert_pareto(
            states.setdefault((candidate_index, 1), []),
            _Label(
                indices=(candidate_index,),
                memory_bytes=candidate.memory_bytes,
                capture_time_ms=candidate.capture_time_ms,
                runtime_latency_ms=runtime,
                captured_count=captured,
                padding_units=padding,
                objective_ms=objective,
            ),
        )

        for previous_index in range(candidate_index):
            previous_size = candidates[previous_index].capture_size
            segment_runtime, segment_captured, segment_padding = _segment_cost(
                observations, previous_size, candidate, spec
            )
            for bucket_count in range(1, max_buckets):
                for label in tuple(states.get((previous_index, bucket_count), ())):
                    memory = label.memory_bytes + candidate.memory_bytes
                    capture_time = label.capture_time_ms + candidate.capture_time_ms
                    if (
                        memory > spec.budget.memory_budget_bytes
                        or capture_time > spec.budget.capture_time_budget_ms
                    ):
                        continue
                    _insert_pareto(
                        states.setdefault((candidate_index, bucket_count + 1), []),
                        _Label(
                            indices=(*label.indices, candidate_index),
                            memory_bytes=memory,
                            capture_time_ms=capture_time,
                            runtime_latency_ms=(
                                label.runtime_latency_ms + segment_runtime
                            ),
                            captured_count=label.captured_count + segment_captured,
                            padding_units=label.padding_units + segment_padding,
                            objective_ms=(
                                label.objective_ms
                                + segment_runtime
                                + segment_padding
                                * spec.objective.padding_penalty_ms_per_unit
                                + _fixed_objective_cost(candidate, spec)
                            ),
                        ),
                    )

    complete: list[tuple[float, float, int, tuple[int, ...], _Label | None]] = []
    if spec.budget.minimum_graph_hit_rate == 0.0:
        complete.append((baseline_latency, baseline_latency, 0, (), None))
    for (last_index, _bucket_count), frontier in states.items():
        last_size = candidates[last_index].capture_size
        tail_latency = sum(
            observation.count * observation.eager_latency_ms
            for observation in observations
            if observation.shape_size > last_size
        )
        for label in frontier:
            hit_rate = label.captured_count / total_count
            if hit_rate + 1e-12 < spec.budget.minimum_graph_hit_rate:
                continue
            sizes = tuple(candidates[index].capture_size for index in label.indices)
            complete.append(
                (
                    label.objective_ms + tail_latency,
                    label.runtime_latency_ms + tail_latency,
                    len(label.indices),
                    sizes,
                    label,
                )
            )
    if not complete:
        raise ValueError("no graph capture plan satisfies the resource and hit-rate budgets")

    _objective, _runtime, _count, _sizes, best = min(
        complete, key=lambda item: (item[0], item[1], item[2], item[3])
    )
    selected = (
        () if best is None else tuple(candidates[index] for index in best.indices)
    )
    assignments = _assignments(spec, selected)
    runtime_latency = sum(
        assignment.count * assignment.latency_ms for assignment in assignments
    )
    captured_count = sum(
        assignment.count for assignment in assignments if assignment.execution == "captured"
    )
    padding_units = sum(
        assignment.count * (assignment.capture_size - assignment.shape_size)
        for assignment in assignments
        if assignment.capture_size is not None
    )
    total_shape_units = sum(
        assignment.count * assignment.shape_size for assignment in assignments
    )
    memory_bytes = sum(candidate.memory_bytes for candidate in selected)
    capture_time_ms = sum(candidate.capture_time_ms for candidate in selected)
    amortized_capture_ms = capture_time_ms / spec.objective.amortization_windows
    memory_penalty_ms = (
        memory_bytes / _GIB * spec.objective.memory_penalty_ms_per_gib
    )
    objective_ms = (
        runtime_latency
        + padding_units * spec.objective.padding_penalty_ms_per_unit
        + amortized_capture_ms
        + memory_penalty_ms
    )
    metrics: dict[str, int | float] = {
        "baseline_eager_latency_ms_per_profile_window": baseline_latency,
        "planned_runtime_latency_ms_per_profile_window": runtime_latency,
        "objective_ms_per_profile_window": objective_ms,
        "predicted_runtime_speedup": (
            baseline_latency / runtime_latency if runtime_latency > 0 else 1.0
        ),
        "graph_hit_rate": captured_count / total_count,
        "captured_execution_count": captured_count,
        "eager_execution_count": total_count - captured_count,
        "padding_units": padding_units,
        "padding_ratio": padding_units / total_shape_units,
        "selected_bucket_count": len(selected),
        "graph_memory_bytes": memory_bytes,
        "capture_time_ms": capture_time_ms,
        "amortized_capture_ms_per_profile_window": amortized_capture_ms,
        "memory_penalty_ms_per_profile_window": memory_penalty_ms,
    }
    return CapturePlan(
        profile_id=spec.profile_id,
        source_spec_sha256=spec.digest,
        selected_capture_sizes=tuple(candidate.capture_size for candidate in selected),
        assignments=assignments,
        metrics=metrics,
    )
