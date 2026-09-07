"""Attest requested deployment settings against the runtime-resolved engine config."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median
from typing import Any

from inference_autopilot.calibration.models import (
    CalibrationPlan,
    canonical_sha256,
    require_digest,
    require_id,
    require_object,
)
from inference_autopilot.features import SelectorFeatureRow, SelectorFeatureTable
from inference_autopilot.stage_wavefront import plan_stage_wavefront


_ROLES = ("base", "proposal")
_ENGINE_START = re.compile(
    r"Initializing a V1 LLM engine .*?with config: model='([^']+)'([^\n]*)"
)
_DTYPE = re.compile(r"\bdtype=torch\.([A-Za-z0-9_]+)")
_MAX_SEQ_LEN = re.compile(r"\bmax_seq_len=([0-9]+)")
_PREFIX_CACHING = re.compile(r"\benable_prefix_caching=(True|False)")
_CHUNKED_PREFILL = re.compile(r"\benable_chunked_prefill=(True|False)")
_COMPILATION_MODE = re.compile(
    r"'mode': <CompilationMode\.([A-Z0-9_]+):"
)
_COMPILATION_BACKEND = re.compile(r"'backend': '([^']+)'")
_COMPILE_ENDPOINTS = re.compile(
    r"'compile_ranges_endpoints': \[([0-9, ]*)\]"
)
_GRAPH_MODE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.([A-Z0-9_]+):")
_GRAPH_CAPTURE_SIZES = re.compile(
    r"'cudagraph_capture_sizes': \[([0-9, ]*)\]"
)
_MAX_GRAPH_CAPTURE_SIZE = re.compile(r"'max_cudagraph_capture_size': ([0-9]+)")
_AVAILABLE_KV = re.compile(
    r"Available KV cache memory: ([0-9]+(?:\.[0-9]+)?) GiB"
)
_KV_TOKENS = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
_MAX_CONCURRENCY = re.compile(
    r"Maximum concurrency for [0-9,]+ tokens per request: "
    r"([0-9]+(?:\.[0-9]+)?)x"
)
_GRAPH_CAPTURE = re.compile(
    r"Graph capturing finished in ([0-9]+(?:\.[0-9]+)?) secs, "
    r"took ([0-9]+(?:\.[0-9]+)?) GiB"
)


def _expect_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return parsed


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string or null")
    return value


def _optional_integer_sequence(value: Any, context: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be an integer array or null")
    parsed = tuple(_positive_integer(item, context) for item in value)
    if not parsed:
        raise ValueError(f"{context} cannot be empty")
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(f"{context} must be sorted and unique")
    return parsed


def _required_match(pattern: re.Pattern[str], text: str, context: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"runner log must contain exactly one {context} record")
    value = matches[0]
    if isinstance(value, tuple):
        raise ValueError(f"internal parser error for {context}")
    return value


def _integer_list(raw: str, context: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw.split(","))
    if any(value <= 0 for value in values):
        raise ValueError(f"{context} values must be positive")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{context} values must be sorted and unique")
    return values


@dataclass(frozen=True, slots=True)
class EngineRuntimeClosure:
    role: str
    model_id: str
    requested_max_num_seqs: int
    effective_admission_max_num_seqs: int
    admission_capacity_source: str
    requested_max_num_batched_tokens: int
    requested_graph_mode: str | None
    requested_graph_capture_sizes: tuple[int, ...] | None
    resolved_dtype: str
    resolved_max_seq_len: int
    resolved_prefix_caching: bool
    resolved_chunked_prefill: bool
    resolved_compilation_mode: str
    resolved_compilation_backend: str
    resolved_compile_ranges_endpoints: tuple[int, ...]
    resolved_graph_mode: str
    resolved_graph_capture_sizes: tuple[int, ...]
    resolved_max_graph_capture_size: int
    graph_policy_source: str
    graph_policy_matches_requested: bool | None
    compile_range_matches_requested: bool
    graph_capacity_coverage_ratio: float
    uncovered_graph_capacity: int
    available_kv_cache_gib: float
    kv_cache_tokens: int
    max_model_concurrency: float
    graph_capture_seconds: float
    graph_capture_gib: float

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unsupported engine role: {self.role}")
        if not self.model_id:
            raise ValueError("runtime closure model_id cannot be empty")
        _positive_integer(self.requested_max_num_seqs, "requested_max_num_seqs")
        _positive_integer(
            self.effective_admission_max_num_seqs,
            "effective_admission_max_num_seqs",
        )
        if self.effective_admission_max_num_seqs > self.requested_max_num_seqs:
            raise ValueError("effective admission capacity exceeds engine capacity")
        if self.admission_capacity_source not in {
            "engine_scheduler",
            "stage_wavefront_plan",
        }:
            raise ValueError("unsupported admission_capacity_source")
        if (
            self.admission_capacity_source == "engine_scheduler"
            and self.effective_admission_max_num_seqs
            != self.requested_max_num_seqs
        ):
            raise ValueError("engine scheduler capacity must equal requested capacity")
        if self.admission_capacity_source == "stage_wavefront_plan" and (
            self.role != "proposal"
        ):
            raise ValueError("stage wavefront capacity applies only to proposal")
        _positive_integer(
            self.requested_max_num_batched_tokens,
            "requested_max_num_batched_tokens",
        )
        _optional_string(self.requested_graph_mode, "requested_graph_mode")
        _optional_integer_sequence(
            self.requested_graph_capture_sizes,
            "requested_graph_capture_sizes",
        )
        for name in (
            "resolved_dtype",
            "resolved_compilation_mode",
            "resolved_compilation_backend",
            "resolved_graph_mode",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        _positive_integer(self.resolved_max_seq_len, "resolved_max_seq_len")
        for name in ("resolved_prefix_caching", "resolved_chunked_prefill"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name, values in (
            ("resolved_compile_ranges_endpoints", self.resolved_compile_ranges_endpoints),
            ("resolved_graph_capture_sizes", self.resolved_graph_capture_sizes),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a non-empty sorted unique tuple")
            for value in values:
                _positive_integer(value, name)
        _positive_integer(
            self.resolved_max_graph_capture_size,
            "resolved_max_graph_capture_size",
        )
        if self.resolved_max_graph_capture_size != max(
            self.resolved_graph_capture_sizes
        ):
            raise ValueError("resolved graph maximum does not match capture sizes")
        if self.graph_policy_source not in {"explicit", "runtime_resolved"}:
            raise ValueError("unsupported graph_policy_source")
        if self.graph_policy_source == "explicit":
            if self.graph_policy_matches_requested is None:
                raise ValueError("explicit graph policy requires a match result")
        elif self.graph_policy_matches_requested is not None:
            raise ValueError("runtime-resolved graph policy cannot have a match result")
        for name in (
            "compile_range_matches_requested",
            "graph_policy_matches_requested",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or null")
        expected_gap = max(
            self.effective_admission_max_num_seqs
            - self.resolved_max_graph_capture_size,
            0,
        )
        if self.uncovered_graph_capacity != expected_gap:
            raise ValueError("uncovered graph capacity does not match resolved policy")
        expected_ratio = (
            min(
                self.resolved_max_graph_capture_size
                / self.effective_admission_max_num_seqs,
                1.0,
            )
        )
        if abs(self.graph_capacity_coverage_ratio - expected_ratio) > 1e-12:
            raise ValueError("graph capacity coverage ratio does not match policy")
        for name in (
            "graph_capacity_coverage_ratio",
            "available_kv_cache_gib",
            "max_model_concurrency",
            "graph_capture_seconds",
            "graph_capture_gib",
        ):
            _finite_nonnegative(getattr(self, name), name)
        _positive_integer(self.kv_cache_tokens, "kv_cache_tokens")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "requested_max_num_seqs": self.requested_max_num_seqs,
            "effective_admission_max_num_seqs": (
                self.effective_admission_max_num_seqs
            ),
            "admission_capacity_source": self.admission_capacity_source,
            "requested_max_num_batched_tokens": (
                self.requested_max_num_batched_tokens
            ),
            "requested_graph_mode": self.requested_graph_mode,
            "requested_graph_capture_sizes": (
                None
                if self.requested_graph_capture_sizes is None
                else list(self.requested_graph_capture_sizes)
            ),
            "resolved_dtype": self.resolved_dtype,
            "resolved_max_seq_len": self.resolved_max_seq_len,
            "resolved_prefix_caching": self.resolved_prefix_caching,
            "resolved_chunked_prefill": self.resolved_chunked_prefill,
            "resolved_compilation_mode": self.resolved_compilation_mode,
            "resolved_compilation_backend": self.resolved_compilation_backend,
            "resolved_compile_ranges_endpoints": list(
                self.resolved_compile_ranges_endpoints
            ),
            "resolved_graph_mode": self.resolved_graph_mode,
            "resolved_graph_capture_sizes": list(
                self.resolved_graph_capture_sizes
            ),
            "resolved_max_graph_capture_size": (
                self.resolved_max_graph_capture_size
            ),
            "graph_policy_source": self.graph_policy_source,
            "graph_policy_matches_requested": (
                self.graph_policy_matches_requested
            ),
            "compile_range_matches_requested": (
                self.compile_range_matches_requested
            ),
            "graph_capacity_coverage_ratio": self.graph_capacity_coverage_ratio,
            "uncovered_graph_capacity": self.uncovered_graph_capacity,
            "available_kv_cache_gib": self.available_kv_cache_gib,
            "kv_cache_tokens": self.kv_cache_tokens,
            "max_model_concurrency": self.max_model_concurrency,
            "graph_capture_seconds": self.graph_capture_seconds,
            "graph_capture_gib": self.graph_capture_gib,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EngineRuntimeClosure":
        keys = {
            "role",
            "model_id",
            "requested_max_num_seqs",
            "effective_admission_max_num_seqs",
            "admission_capacity_source",
            "requested_max_num_batched_tokens",
            "requested_graph_mode",
            "requested_graph_capture_sizes",
            "resolved_dtype",
            "resolved_max_seq_len",
            "resolved_prefix_caching",
            "resolved_chunked_prefill",
            "resolved_compilation_mode",
            "resolved_compilation_backend",
            "resolved_compile_ranges_endpoints",
            "resolved_graph_mode",
            "resolved_graph_capture_sizes",
            "resolved_max_graph_capture_size",
            "graph_policy_source",
            "graph_policy_matches_requested",
            "compile_range_matches_requested",
            "graph_capacity_coverage_ratio",
            "uncovered_graph_capacity",
            "available_kv_cache_gib",
            "kv_cache_tokens",
            "max_model_concurrency",
            "graph_capture_seconds",
            "graph_capture_gib",
        }
        _expect_keys(raw, keys, "engine runtime closure")
        return cls(
            role=str(raw["role"]),
            model_id=str(raw["model_id"]),
            requested_max_num_seqs=_positive_integer(
                raw["requested_max_num_seqs"], "requested_max_num_seqs"
            ),
            effective_admission_max_num_seqs=_positive_integer(
                raw["effective_admission_max_num_seqs"],
                "effective_admission_max_num_seqs",
            ),
            admission_capacity_source=str(raw["admission_capacity_source"]),
            requested_max_num_batched_tokens=_positive_integer(
                raw["requested_max_num_batched_tokens"],
                "requested_max_num_batched_tokens",
            ),
            requested_graph_mode=_optional_string(
                raw["requested_graph_mode"], "requested_graph_mode"
            ),
            requested_graph_capture_sizes=_optional_integer_sequence(
                raw["requested_graph_capture_sizes"],
                "requested_graph_capture_sizes",
            ),
            resolved_dtype=str(raw["resolved_dtype"]),
            resolved_max_seq_len=_positive_integer(
                raw["resolved_max_seq_len"], "resolved_max_seq_len"
            ),
            resolved_prefix_caching=raw["resolved_prefix_caching"],
            resolved_chunked_prefill=raw["resolved_chunked_prefill"],
            resolved_compilation_mode=str(raw["resolved_compilation_mode"]),
            resolved_compilation_backend=str(raw["resolved_compilation_backend"]),
            resolved_compile_ranges_endpoints=_optional_integer_sequence(
                raw["resolved_compile_ranges_endpoints"],
                "resolved_compile_ranges_endpoints",
            )
            or (),
            resolved_graph_mode=str(raw["resolved_graph_mode"]),
            resolved_graph_capture_sizes=_optional_integer_sequence(
                raw["resolved_graph_capture_sizes"],
                "resolved_graph_capture_sizes",
            )
            or (),
            resolved_max_graph_capture_size=_positive_integer(
                raw["resolved_max_graph_capture_size"],
                "resolved_max_graph_capture_size",
            ),
            graph_policy_source=str(raw["graph_policy_source"]),
            graph_policy_matches_requested=raw[
                "graph_policy_matches_requested"
            ],
            compile_range_matches_requested=raw[
                "compile_range_matches_requested"
            ],
            graph_capacity_coverage_ratio=_finite_nonnegative(
                raw["graph_capacity_coverage_ratio"],
                "graph_capacity_coverage_ratio",
            ),
            uncovered_graph_capacity=_nonnegative_integer(
                raw["uncovered_graph_capacity"], "uncovered_graph_capacity"
            ),
            available_kv_cache_gib=_finite_nonnegative(
                raw["available_kv_cache_gib"], "available_kv_cache_gib"
            ),
            kv_cache_tokens=_positive_integer(
                raw["kv_cache_tokens"], "kv_cache_tokens"
            ),
            max_model_concurrency=_finite_nonnegative(
                raw["max_model_concurrency"], "max_model_concurrency"
            ),
            graph_capture_seconds=_finite_nonnegative(
                raw["graph_capture_seconds"], "graph_capture_seconds"
            ),
            graph_capture_gib=_finite_nonnegative(
                raw["graph_capture_gib"], "graph_capture_gib"
            ),
        )


@dataclass(frozen=True, slots=True)
class RunRuntimeClosure:
    run_id: str
    sequence_index: int
    configuration_id: str
    workload_seed: int
    run_manifest_sha256: str
    effective_config_sha256: str
    runner_log_sha256: str
    engines: tuple[EngineRuntimeClosure, ...]

    def __post_init__(self) -> None:
        require_id(self.run_id, "runtime closure run_id")
        require_id(self.configuration_id, "runtime closure configuration_id")
        for name in ("sequence_index", "workload_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "run_manifest_sha256",
            "effective_config_sha256",
            "runner_log_sha256",
        ):
            require_digest(getattr(self, name), name)
        if tuple(engine.role for engine in self.engines) != _ROLES:
            raise ValueError("runtime closure must contain ordered base and proposal engines")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "configuration_id": self.configuration_id,
            "workload_seed": self.workload_seed,
            "run_manifest_sha256": self.run_manifest_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            "runner_log_sha256": self.runner_log_sha256,
            "engines": [engine.to_dict() for engine in self.engines],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunRuntimeClosure":
        keys = {
            "run_id",
            "sequence_index",
            "configuration_id",
            "workload_seed",
            "run_manifest_sha256",
            "effective_config_sha256",
            "runner_log_sha256",
            "engines",
        }
        _expect_keys(raw, keys, "run runtime closure")
        engines = raw["engines"]
        if not isinstance(engines, list):
            raise ValueError("runtime closure engines must be an array")
        return cls(
            run_id=str(raw["run_id"]),
            sequence_index=_nonnegative_integer(
                raw["sequence_index"], "sequence_index"
            ),
            configuration_id=str(raw["configuration_id"]),
            workload_seed=_nonnegative_integer(
                raw["workload_seed"], "workload_seed"
            ),
            run_manifest_sha256=str(raw["run_manifest_sha256"]),
            effective_config_sha256=str(raw["effective_config_sha256"]),
            runner_log_sha256=str(raw["runner_log_sha256"]),
            engines=tuple(
                EngineRuntimeClosure.from_dict(
                    require_object(item, "engine runtime closure")
                )
                for item in engines
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeClosureIssue:
    run_id: str
    message: str

    def __post_init__(self) -> None:
        require_id(self.run_id, "runtime closure issue run_id")
        if not self.message:
            raise ValueError("runtime closure issue message cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "message": self.message}


def _policy_summary(runs: Sequence[RunRuntimeClosure]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EngineRuntimeClosure]] = defaultdict(list)
    for run in runs:
        for engine in run.engines:
            grouped[(run.configuration_id, engine.role)].append(engine)
    policies = []
    for (configuration_id, role), engines in sorted(grouped.items()):
        signatures = {
            canonical_sha256(
                {
                    "compile_ranges_endpoints": list(
                        engine.resolved_compile_ranges_endpoints
                    ),
                    "graph_mode": engine.resolved_graph_mode,
                    "graph_capture_sizes": list(
                        engine.resolved_graph_capture_sizes
                    ),
                }
            )
            for engine in engines
        }
        representative = engines[0]
        policies.append(
            {
                "configuration_id": configuration_id,
                "role": role,
                "observation_count": len(engines),
                "unique_resolved_policy_count": len(signatures),
                "requested_max_num_seqs": representative.requested_max_num_seqs,
                "effective_admission_max_num_seqs": (
                    representative.effective_admission_max_num_seqs
                ),
                "admission_capacity_source": (
                    representative.admission_capacity_source
                ),
                "requested_max_num_batched_tokens": (
                    representative.requested_max_num_batched_tokens
                ),
                "resolved_compile_range_ceiling": max(
                    representative.resolved_compile_ranges_endpoints
                ),
                "resolved_graph_mode": representative.resolved_graph_mode,
                "resolved_graph_bucket_count": len(
                    representative.resolved_graph_capture_sizes
                ),
                "resolved_graph_capture_ceiling": (
                    representative.resolved_max_graph_capture_size
                ),
                "graph_capacity_coverage_ratio": (
                    representative.graph_capacity_coverage_ratio
                ),
                "uncovered_graph_capacity": (
                    representative.uncovered_graph_capacity
                ),
                "graph_policy_source": representative.graph_policy_source,
                "available_kv_cache_gib_median": median(
                    engine.available_kv_cache_gib for engine in engines
                ),
                "graph_capture_seconds_median": median(
                    engine.graph_capture_seconds for engine in engines
                ),
            }
        )
    return policies


def _derive_summary(runs: Sequence[RunRuntimeClosure]) -> dict[str, Any]:
    engines = [engine for run in runs for engine in run.engines]
    return {
        "engine_count": len(engines),
        "runtime_resolved_graph_policy_engine_count": sum(
            engine.graph_policy_source == "runtime_resolved" for engine in engines
        ),
        "explicit_graph_policy_engine_count": sum(
            engine.graph_policy_source == "explicit" for engine in engines
        ),
        "graph_policy_mismatch_count": sum(
            engine.graph_policy_matches_requested is False for engine in engines
        ),
        "compile_range_mismatch_count": sum(
            not engine.compile_range_matches_requested for engine in engines
        ),
        "graph_capacity_gap_engine_count": sum(
            engine.uncovered_graph_capacity > 0 for engine in engines
        ),
        "minimum_graph_capacity_coverage_ratio": (
            min(engine.graph_capacity_coverage_ratio for engine in engines)
            if engines
            else None
        ),
        "resolved_policies": _policy_summary(runs),
    }


@dataclass(frozen=True, slots=True)
class RuntimeClosureAssessment:
    plan_sha256: str
    campaign_id: str
    expected_run_count: int
    valid_run_count: int
    complete: bool
    issues: tuple[RuntimeClosureIssue, ...]
    runs: tuple[RunRuntimeClosure, ...]
    summary: Mapping[str, Any]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported runtime closure schema")
        if not self.producer:
            raise ValueError("runtime closure producer cannot be empty")
        require_digest(self.plan_sha256, "runtime closure plan_sha256")
        require_id(self.campaign_id, "runtime closure campaign_id")
        if not isinstance(self.complete, bool):
            raise ValueError("runtime closure complete must be boolean")
        _positive_integer(self.expected_run_count, "expected_run_count")
        _nonnegative_integer(self.valid_run_count, "valid_run_count")
        if self.valid_run_count != len(self.runs):
            raise ValueError("valid_run_count does not match runtime closure runs")
        if self.valid_run_count > self.expected_run_count:
            raise ValueError("valid runtime closure runs exceed expected runs")
        if self.complete != (
            self.valid_run_count == self.expected_run_count and not self.issues
        ):
            raise ValueError("runtime closure completeness does not match evidence")
        indexes = [run.sequence_index for run in self.runs]
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise ValueError("runtime closure sequence indexes must be sorted and unique")
        if dict(self.summary) != _derive_summary(self.runs):
            raise ValueError("runtime closure summary does not match run evidence")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "plan_sha256": self.plan_sha256,
            "campaign_id": self.campaign_id,
            "expected_run_count": self.expected_run_count,
            "valid_run_count": self.valid_run_count,
            "complete": self.complete,
            "issues": [issue.to_dict() for issue in self.issues],
            "runs": [run.to_dict() for run in self.runs],
            "summary": dict(self.summary),
        }
        return {**payload, "runtime_closure_assessment_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeClosureAssessment":
        keys = {
            "schema_version",
            "producer",
            "plan_sha256",
            "campaign_id",
            "expected_run_count",
            "valid_run_count",
            "complete",
            "issues",
            "runs",
            "summary",
            "runtime_closure_assessment_sha256",
        }
        _expect_keys(raw, keys, "runtime closure assessment")
        payload = dict(raw)
        digest = str(payload.pop("runtime_closure_assessment_sha256"))
        require_digest(digest, "runtime_closure_assessment_sha256")
        if digest != canonical_sha256(payload):
            raise ValueError("runtime closure assessment SHA256 does not match content")
        issues_raw = raw["issues"]
        runs_raw = raw["runs"]
        if not isinstance(issues_raw, list) or not isinstance(runs_raw, list):
            raise ValueError("runtime closure issues and runs must be arrays")
        issues = []
        for item in issues_raw:
            issue = require_object(item, "runtime closure issue")
            _expect_keys(issue, {"run_id", "message"}, "runtime closure issue")
            issues.append(
                RuntimeClosureIssue(str(issue["run_id"]), str(issue["message"]))
            )
        runs = tuple(
            RunRuntimeClosure.from_dict(require_object(item, "run runtime closure"))
            for item in runs_raw
        )
        summary = require_object(raw["summary"], "runtime closure summary")
        assessment = cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            plan_sha256=str(raw["plan_sha256"]),
            campaign_id=str(raw["campaign_id"]),
            expected_run_count=_positive_integer(
                raw["expected_run_count"], "expected_run_count"
            ),
            valid_run_count=_nonnegative_integer(
                raw["valid_run_count"], "valid_run_count"
            ),
            complete=raw["complete"],
            issues=tuple(issues),
            runs=runs,
            summary=dict(summary),
        )
        return assessment

    def audit(self) -> dict[str, Any]:
        parsed = RuntimeClosureAssessment.from_dict(self.to_dict())
        return {
            "campaign_id": parsed.campaign_id,
            "complete": parsed.complete,
            "expected_run_count": parsed.expected_run_count,
            "valid_run_count": parsed.valid_run_count,
            "issue_count": len(parsed.issues),
            **dict(parsed.summary),
        }


def _parse_boolean(pattern: re.Pattern[str], text: str, context: str) -> bool:
    return _required_match(pattern, text, context) == "True"


def _parse_engine(
    init_line: str,
    block: str,
    *,
    role: str,
    model_id: str,
    settings: Mapping[str, Any],
    effective_admission_max_num_seqs: int,
    admission_capacity_source: str,
) -> EngineRuntimeClosure:
    requested_seqs = _positive_integer(
        settings.get(f"{role}_max_num_seqs"), f"{role}_max_num_seqs"
    )
    requested_tokens = _positive_integer(
        settings.get(f"{role}_max_num_batched_tokens"),
        f"{role}_max_num_batched_tokens",
    )
    requested_mode = _optional_string(
        settings.get(f"{role}_graph_mode"), f"{role}_graph_mode"
    )
    requested_sizes = _optional_integer_sequence(
        settings.get(f"{role}_graph_capture_sizes"),
        f"{role}_graph_capture_sizes",
    )
    explicit_policy = requested_mode is not None or requested_sizes is not None
    if explicit_policy and (requested_mode is None or requested_sizes is None):
        raise ValueError(f"{role} graph policy must specify both mode and capture sizes")

    compile_endpoints = _integer_list(
        _required_match(
            _COMPILE_ENDPOINTS, init_line, f"{role} compile range endpoints"
        ),
        f"{role} compile range endpoints",
    )
    capture_sizes = _integer_list(
        _required_match(
            _GRAPH_CAPTURE_SIZES, init_line, f"{role} graph capture sizes"
        ),
        f"{role} graph capture sizes",
    )
    max_capture = int(
        _required_match(
            _MAX_GRAPH_CAPTURE_SIZE, init_line, f"{role} max graph capture size"
        )
    )
    graph_mode = _required_match(_GRAPH_MODE, init_line, f"{role} graph mode")
    capture = _GRAPH_CAPTURE.findall(block)
    if len(capture) != 1:
        raise ValueError(f"runner log must contain one {role} graph capture result")
    graph_matches = None
    if explicit_policy:
        graph_matches = requested_mode == graph_mode and requested_sizes == capture_sizes

    return EngineRuntimeClosure(
        role=role,
        model_id=model_id,
        requested_max_num_seqs=requested_seqs,
        effective_admission_max_num_seqs=effective_admission_max_num_seqs,
        admission_capacity_source=admission_capacity_source,
        requested_max_num_batched_tokens=requested_tokens,
        requested_graph_mode=requested_mode,
        requested_graph_capture_sizes=requested_sizes,
        resolved_dtype=_required_match(_DTYPE, init_line, f"{role} dtype"),
        resolved_max_seq_len=int(
            _required_match(_MAX_SEQ_LEN, init_line, f"{role} max sequence length")
        ),
        resolved_prefix_caching=_parse_boolean(
            _PREFIX_CACHING, init_line, f"{role} prefix caching"
        ),
        resolved_chunked_prefill=_parse_boolean(
            _CHUNKED_PREFILL, init_line, f"{role} chunked prefill"
        ),
        resolved_compilation_mode=_required_match(
            _COMPILATION_MODE, init_line, f"{role} compilation mode"
        ),
        resolved_compilation_backend=_required_match(
            _COMPILATION_BACKEND, init_line, f"{role} compilation backend"
        ),
        resolved_compile_ranges_endpoints=compile_endpoints,
        resolved_graph_mode=graph_mode,
        resolved_graph_capture_sizes=capture_sizes,
        resolved_max_graph_capture_size=max_capture,
        graph_policy_source="explicit" if explicit_policy else "runtime_resolved",
        graph_policy_matches_requested=graph_matches,
        compile_range_matches_requested=(
            max(compile_endpoints) == requested_tokens
        ),
        graph_capacity_coverage_ratio=(
            min(max_capture / effective_admission_max_num_seqs, 1.0)
        ),
        uncovered_graph_capacity=max(
            effective_admission_max_num_seqs - max_capture, 0
        ),
        available_kv_cache_gib=float(
            _required_match(_AVAILABLE_KV, block, f"{role} available KV cache")
        ),
        kv_cache_tokens=int(
            _required_match(_KV_TOKENS, block, f"{role} KV cache tokens").replace(
                ",", ""
            )
        ),
        max_model_concurrency=float(
            _required_match(_MAX_CONCURRENCY, block, f"{role} max concurrency")
        ),
        graph_capture_seconds=float(capture[0][0]),
        graph_capture_gib=float(capture[0][1]),
    )


def _load_run_closure(
    plan: CalibrationPlan, campaign_root: Path, run_id: str
) -> RunRuntimeClosure:
    expected = next(run for run in plan.runs if run.run_id == run_id)
    root = campaign_root / run_id
    manifest_path = root / "run-manifest.json"
    effective_path = root / "effective-config.json"
    log_path = root / "runner.log"
    for path in (manifest_path, effective_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing runtime closure artifact: {path}")
    manifest = require_object(
        json.loads(manifest_path.read_text(encoding="utf-8")), "run manifest"
    )
    effective = require_object(
        json.loads(effective_path.read_text(encoding="utf-8")), "effective config"
    )
    if require_object(manifest.get("run"), "manifest run") != expected.to_dict():
        raise ValueError(f"run manifest does not match plan for {run_id}")
    if manifest.get("plan_sha256") != canonical_sha256(plan.to_dict()):
        raise ValueError(f"run manifest plan binding mismatch for {run_id}")
    settings = require_object(
        effective.get("deployment_settings"), "effective deployment settings"
    )
    manifest_settings = require_object(
        require_object(manifest.get("configuration"), "manifest configuration").get(
            "settings"
        ),
        "manifest configuration settings",
    )
    if settings != manifest_settings:
        raise ValueError(f"effective deployment settings mismatch for {run_id}")

    wavefront_plan = None
    wavefront_mode = settings.get("proposal_stage_wavefront_mode", "off")
    if wavefront_mode in {"auto", "prefix_sharded"}:
        workload = plan.spec.workload_contract.parameters
        invariants = plan.spec.semantic_contract.invariants
        admission_unit_concurrency = min(
            int(workload["requests"]), int(workload["workers"])
        )
        sequences_per_admission_unit = int(invariants["rollout_count"])
        if wavefront_mode == "prefix_sharded":
            admission_unit_concurrency *= int(invariants["candidate_count"])
        else:
            sequences_per_admission_unit *= int(invariants["candidate_count"])
        wavefront_plan = plan_stage_wavefront(
            outer_concurrency=admission_unit_concurrency,
            sequences_per_group=sequences_per_admission_unit,
            graph_capture_ceiling=int(
                settings["proposal_graph_capture_ceiling"]
            ),
            scheduler_sequence_cap=int(settings["proposal_max_num_seqs"]),
            minimum_full_wave_utilization=float(
                settings["proposal_stage_wavefront_min_utilization"]
            ),
        )
    elif wavefront_mode != "off":
        raise ValueError(f"unsupported proposal stage-wavefront mode: {wavefront_mode}")

    log_text = log_path.read_text(encoding="utf-8")
    starts = list(_ENGINE_START.finditer(log_text))
    if len(starts) != len(_ROLES):
        raise ValueError("runner log must contain exactly two engine configurations")
    engines = []
    for index, role in enumerate(_ROLES):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log_text)
        uses_wavefront = role == "proposal" and wavefront_plan is not None
        effective_admission_capacity = (
            wavefront_plan.selected.sequences_per_wave
            if uses_wavefront
            else int(settings[f"{role}_max_num_seqs"])
        )
        engines.append(
            _parse_engine(
                starts[index].group(0),
                log_text[starts[index].start() : end],
                role=role,
                model_id=starts[index].group(1),
                settings=settings,
                effective_admission_max_num_seqs=effective_admission_capacity,
                admission_capacity_source=(
                    "stage_wavefront_plan" if uses_wavefront else "engine_scheduler"
                ),
            )
        )
    return RunRuntimeClosure(
        run_id=run_id,
        sequence_index=expected.sequence_index,
        configuration_id=expected.configuration_id,
        workload_seed=expected.workload_seed,
        run_manifest_sha256=_file_sha256(manifest_path),
        effective_config_sha256=_file_sha256(effective_path),
        runner_log_sha256=_file_sha256(log_path),
        engines=tuple(engines),
    )


def attest_runtime_closure(
    plan: CalibrationPlan, campaign_root: Path
) -> RuntimeClosureAssessment:
    """Bind every planned run to the configuration that vLLM actually resolved."""

    runs = []
    issues = []
    for expected in plan.runs:
        try:
            closure = _load_run_closure(plan, campaign_root, expected.run_id)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(RuntimeClosureIssue(expected.run_id, str(error)))
            continue
        runs.append(closure)
        mismatches = [
            engine.role
            for engine in closure.engines
            if engine.graph_policy_matches_requested is False
            or not engine.compile_range_matches_requested
        ]
        if mismatches:
            issues.append(
                RuntimeClosureIssue(
                    expected.run_id,
                    f"runtime did not honor requested settings for roles {mismatches}",
                )
            )
    ordered = tuple(sorted(runs, key=lambda run: run.sequence_index))
    return RuntimeClosureAssessment(
        plan_sha256=canonical_sha256(plan.to_dict()),
        campaign_id=plan.spec.campaign_id,
        expected_run_count=len(plan.runs),
        valid_run_count=len(ordered),
        complete=len(ordered) == len(plan.runs) and not issues,
        issues=tuple(issues),
        runs=ordered,
        summary=_derive_summary(ordered),
    )


def enrich_features_with_runtime_closure(
    table: SelectorFeatureTable, assessment: RuntimeClosureAssessment
) -> SelectorFeatureTable:
    """Join attested runtime defaults into model-ready scalar feature rows."""

    if not assessment.complete:
        raise ValueError("runtime closure assessment must be complete before enrichment")
    by_run = {run.run_id: run for run in assessment.runs}
    row_runs = {str(row.source["locator"]) for row in table.rows}
    if row_runs != set(by_run):
        raise ValueError("feature rows and runtime closure runs do not match exactly")
    assessment_digest = assessment.to_dict()["runtime_closure_assessment_sha256"]
    enriched = []
    for row in table.rows:
        closure = by_run[str(row.source["locator"])]
        static = dict(row.static_features)
        telemetry = dict(row.telemetry_features)
        static["runtime.closure_assessment_sha256"] = assessment_digest
        for engine in closure.engines:
            prefix = f"runtime.{engine.role}"
            static.update(
                {
                    f"{prefix}.dtype": engine.resolved_dtype,
                    f"{prefix}.max_seq_len": engine.resolved_max_seq_len,
                    f"{prefix}.prefix_caching": engine.resolved_prefix_caching,
                    f"{prefix}.chunked_prefill": engine.resolved_chunked_prefill,
                    f"{prefix}.compilation_mode": (
                        engine.resolved_compilation_mode
                    ),
                    f"{prefix}.compile_range_ceiling": max(
                        engine.resolved_compile_ranges_endpoints
                    ),
                    f"{prefix}.effective_admission_max_num_seqs": (
                        engine.effective_admission_max_num_seqs
                    ),
                    f"{prefix}.admission_capacity_source": (
                        engine.admission_capacity_source
                    ),
                    f"{prefix}.graph_mode": engine.resolved_graph_mode,
                    f"{prefix}.graph_bucket_count": len(
                        engine.resolved_graph_capture_sizes
                    ),
                    f"{prefix}.graph_capture_ceiling": (
                        engine.resolved_max_graph_capture_size
                    ),
                    f"{prefix}.graph_capacity_coverage_ratio": (
                        engine.graph_capacity_coverage_ratio
                    ),
                    f"{prefix}.uncovered_graph_capacity": (
                        engine.uncovered_graph_capacity
                    ),
                    f"{prefix}.graph_policy_source": engine.graph_policy_source,
                }
            )
            telemetry.update(
                {
                    f"runtime_closure.{engine.role}.available_kv_cache_gib": (
                        engine.available_kv_cache_gib
                    ),
                    f"runtime_closure.{engine.role}.kv_cache_tokens": (
                        engine.kv_cache_tokens
                    ),
                    f"runtime_closure.{engine.role}.max_model_concurrency": (
                        engine.max_model_concurrency
                    ),
                    f"runtime_closure.{engine.role}.graph_capture_seconds": (
                        engine.graph_capture_seconds
                    ),
                    f"runtime_closure.{engine.role}.graph_capture_gib": (
                        engine.graph_capture_gib
                    ),
                }
            )
        enriched.append(
            SelectorFeatureRow(
                row_id=row.row_id,
                source_kind=row.source_kind,
                source=row.source,
                campaign=row.campaign,
                variant=row.variant,
                cohort=row.cohort,
                evidence=row.evidence,
                eligibility=row.eligibility,
                static_features=static,
                telemetry_features=telemetry,
                targets=row.targets,
                missing=row.missing,
                tags=tuple(dict.fromkeys((*row.tags, "runtime_closure_attested"))),
            )
        )
    return SelectorFeatureTable(tuple(enriched))
