"""Extract graph-planning workload traces from Conditional IS run artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_sha256,
    require_digest,
    require_id,
    require_object,
)


_STAGE_BY_CALL = {
    ("base", "sample"): "candidate_generate",
    ("base", "score"): "target_score",
    ("proposal", "sample"): "proposal_rollout_generate",
    ("proposal", "score"): "proposal_score",
}


def _exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _finite_float(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _nonnegative_float(value: Any, context: str) -> float:
    parsed = _finite_float(value, context)
    if parsed < 0:
        raise ValueError(f"{context} must be non-negative")
    return parsed


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    ordered = sorted(values)
    return ordered[ceil(fraction * len(ordered)) - 1] if fraction > 0 else ordered[0]


@dataclass(frozen=True, slots=True)
class GraphTraceEvent:
    engine_role: str
    stage_id: str
    call_kind: str
    shape_size: int
    started_at_seconds: float
    duration_ms: float

    def __post_init__(self) -> None:
        if self.engine_role not in {"base", "proposal"}:
            raise ValueError(f"unsupported graph trace engine_role: {self.engine_role}")
        require_id(self.stage_id, "graph trace stage_id")
        if self.call_kind not in {"sample", "score"}:
            raise ValueError(f"unsupported graph trace call_kind: {self.call_kind}")
        expected_stage = _STAGE_BY_CALL.get((self.engine_role, self.call_kind))
        if self.stage_id != expected_stage:
            raise ValueError(
                "graph trace stage does not match engine role and call kind: "
                f"expected={expected_stage}, actual={self.stage_id}"
            )
        _positive_int(self.shape_size, "graph trace shape_size")
        _finite_float(self.started_at_seconds, "graph trace started_at_seconds")
        _nonnegative_float(self.duration_ms, "graph trace duration_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_role": self.engine_role,
            "stage_id": self.stage_id,
            "call_kind": self.call_kind,
            "shape_size": self.shape_size,
            "started_at_seconds": self.started_at_seconds,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphTraceEvent":
        _exact_keys(
            raw,
            {
                "engine_role",
                "stage_id",
                "call_kind",
                "shape_size",
                "started_at_seconds",
                "duration_ms",
            },
            "graph trace event",
        )
        return cls(
            engine_role=str(raw["engine_role"]),
            stage_id=str(raw["stage_id"]),
            call_kind=str(raw["call_kind"]),
            shape_size=_positive_int(raw["shape_size"], "graph trace shape_size"),
            started_at_seconds=_finite_float(
                raw["started_at_seconds"], "graph trace started_at_seconds"
            ),
            duration_ms=_nonnegative_float(
                raw["duration_ms"], "graph trace duration_ms"
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphWorkloadTrace:
    trace_id: str
    algorithm_id: str
    source_result_sha256: str
    events: tuple[GraphTraceEvent, ...]
    metadata: Mapping[str, Any]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported graph workload trace: {self.schema_version}")
        require_id(self.trace_id, "graph trace_id")
        require_id(self.algorithm_id, "graph trace algorithm_id")
        require_digest(self.source_result_sha256, "source result SHA256")
        if not self.events:
            raise ValueError("graph workload trace requires at least one event")

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "algorithm_id": self.algorithm_id,
            "source_result_sha256": self.source_result_sha256,
            "events": [event.to_dict() for event in self.events],
            "metadata": dict(sorted(self.metadata.items())),
        }

    def artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {**payload, "graph_workload_trace_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphWorkloadTrace":
        _exact_keys(
            raw,
            {
                "schema_version",
                "trace_id",
                "algorithm_id",
                "source_result_sha256",
                "events",
                "metadata",
                "graph_workload_trace_sha256",
            },
            "graph workload trace",
        )
        payload = dict(raw)
        digest = str(payload.pop("graph_workload_trace_sha256"))
        require_digest(digest, "graph workload trace SHA256")
        if canonical_sha256(payload) != digest:
            raise ValueError("graph workload trace SHA256 does not match its content")
        events = payload["events"]
        if not isinstance(events, list):
            raise ValueError("graph workload trace events must be an array")
        metadata = require_object(payload["metadata"], "graph workload trace metadata")
        return cls(
            schema_version=str(payload["schema_version"]),
            trace_id=str(payload["trace_id"]),
            algorithm_id=str(payload["algorithm_id"]),
            source_result_sha256=str(payload["source_result_sha256"]),
            events=tuple(
                GraphTraceEvent.from_dict(require_object(item, "graph trace event"))
                for item in events
            ),
            metadata=metadata,
        )

    def audit(self) -> dict[str, Any]:
        groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for event in self.events:
            groups[(event.engine_role, event.stage_id, event.shape_size)].append(
                event.duration_ms
            )
        histograms = []
        for (role, stage_id, shape_size), durations in sorted(groups.items()):
            histograms.append(
                {
                    "engine_role": role,
                    "stage_id": stage_id,
                    "shape_size": shape_size,
                    "count": len(durations),
                    "wall_service_ms_mean": sum(durations) / len(durations),
                    "wall_service_ms_p50": _percentile(durations, 0.50),
                    "wall_service_ms_p95": _percentile(durations, 0.95),
                }
            )
        return {
            "trace_id": self.trace_id,
            "event_count": len(self.events),
            "engine_roles": sorted({event.engine_role for event in self.events}),
            "stage_ids": sorted({event.stage_id for event in self.events}),
            "histograms": histograms,
            "request_group_boundaries_by_engine": {
                role: list(self.recommend_request_group_boundaries(role))
                for role in sorted({event.engine_role for event in self.events})
            },
            "request_group_boundaries_by_stage": {
                stage_id: list(
                    self.recommend_request_group_boundaries(
                        next(
                            event.engine_role
                            for event in self.events
                            if event.stage_id == stage_id
                        ),
                        stage_ids=(stage_id,),
                    )
                )
                for stage_id in sorted({event.stage_id for event in self.events})
            },
        }

    def recommend_request_group_boundaries(
        self,
        engine_role: str,
        *,
        stage_ids: Sequence[str] | None = None,
        maximum_candidates: int = 8,
    ) -> tuple[int, ...]:
        """Select API call-width peaks and quantiles without inventing sizes."""

        _positive_int(maximum_candidates, "maximum_candidates")
        selected_stages = None if stage_ids is None else set(stage_ids)
        if selected_stages is not None:
            for stage_id in selected_stages:
                require_id(stage_id, "request-group boundary stage_id")
        role_events = [
            event
            for event in self.events
            if event.engine_role == engine_role
            and (selected_stages is None or event.stage_id in selected_stages)
        ]
        if not role_events:
            raise ValueError(
                "trace has no events for engine role/stages: "
                f"role={engine_role}, stages={sorted(selected_stages or [])}"
            )
        counts: dict[int, int] = defaultdict(int)
        for event in role_events:
            counts[event.shape_size] += 1
        ordered = sorted(counts)
        if len(ordered) <= maximum_candidates:
            return tuple(ordered)

        selected = {ordered[0], ordered[-1]}
        peak_budget = min(2, max(0, maximum_candidates - len(selected)))
        peaks = sorted(counts, key=lambda size: (-counts[size], size))[:peak_budget]
        selected.update(peaks)

        slots = maximum_candidates - len(selected)
        if slots > 0:
            total = sum(counts.values())
            quantiles = [(index + 1) / (slots + 1) for index in range(slots)]
            for fraction in quantiles:
                threshold = fraction * total
                cumulative = 0
                for size in ordered:
                    cumulative += counts[size]
                    if cumulative >= threshold:
                        selected.add(size)
                        break

        if len(selected) < maximum_candidates:
            remaining = sorted(
                (size for size in ordered if size not in selected),
                key=lambda size: (-counts[size], size),
            )
            selected.update(remaining[: maximum_candidates - len(selected)])
        return tuple(sorted(selected))


def trace_from_chang_result(
    raw: Mapping[str, Any],
    *,
    trace_id: str,
) -> GraphWorkloadTrace:
    """Normalize a pressure-run result without treating call time as kernel latency."""

    require_id(trace_id, "graph trace_id")
    timeline = raw.get("model_call_timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("chang result requires a non-empty model_call_timeline")
    events: list[GraphTraceEvent] = []
    for index, item in enumerate(timeline):
        event = require_object(item, f"model_call_timeline[{index}]")
        role = str(event.get("role", ""))
        kind = str(event.get("kind", ""))
        stage_id = _STAGE_BY_CALL.get((role, kind))
        if stage_id is None:
            raise ValueError(f"unsupported chang model call: role={role}, kind={kind}")
        start = _finite_float(event.get("start_unix"), f"event {index} start_unix")
        end = _finite_float(event.get("end_unix"), f"event {index} end_unix")
        duration = _nonnegative_float(
            event.get("duration_seconds"), f"event {index} duration_seconds"
        )
        if end < start:
            raise ValueError(f"event {index} ends before it starts")
        if abs((end - start) - duration) > max(0.005, duration * 0.01):
            raise ValueError(f"event {index} duration is inconsistent with timestamps")
        events.append(
            GraphTraceEvent(
                engine_role=role,
                stage_id=stage_id,
                call_kind=kind,
                shape_size=_positive_int(
                    event.get("request_groups"), f"event {index} request_groups"
                ),
                started_at_seconds=start,
                duration_ms=duration * 1000.0,
            )
        )

    result_payload = dict(raw)
    runtime = require_object(raw.get("runtime", {}), "chang result runtime")
    algorithm = require_object(raw.get("algorithm", {}), "chang result algorithm")
    return GraphWorkloadTrace(
        trace_id=trace_id,
        algorithm_id="conditional_is_small_proposal",
        source_result_sha256=canonical_sha256(result_payload),
        events=tuple(sorted(events, key=lambda item: item.started_at_seconds)),
        metadata={
            "requests": raw.get("requests"),
            "workers": raw.get("workers"),
            "arrival_qps": raw.get("arrival_qps"),
            "base_max_num_seqs": runtime.get("base_max_num_seqs"),
            "proposal_max_num_seqs": runtime.get("proposal_max_num_seqs"),
            "rollout_count": algorithm.get("rollout_count"),
            "measurement_semantics": "backend_call_wall_service_time_including_queueing",
        },
    )
