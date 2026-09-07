"""Content-addressed startup and cache-cost evidence for calibration campaigns."""

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


_ENGINE_START = re.compile(
    r"Initializing a V1 LLM engine .*?model='([^']+)'"
)
_WEIGHT_LOAD = re.compile(r"Loading weights took ([0-9]+(?:\.[0-9]+)?) seconds")
_CACHE_DIRECTORY = re.compile(r"/torch_compile_cache/([0-9a-f]+)/")
_DYNAMO = re.compile(r"Dynamo bytecode transform time: ([0-9]+(?:\.[0-9]+)?) s")
_GRAPH_COMPILE = re.compile(
    r"Compiling a graph for compile range \(([^)]*)\) takes "
    r"([0-9]+(?:\.[0-9]+)?) s"
)
_COMPILE_WARMUP = re.compile(
    r"torch\.compile and initial profiling/warmup run together took "
    r"([0-9]+(?:\.[0-9]+)?) s in total"
)
_GRAPH_CAPTURE = re.compile(
    r"Graph capturing finished in ([0-9]+(?:\.[0-9]+)?) secs, "
    r"took ([0-9]+(?:\.[0-9]+)?) GiB"
)
_ENGINE_INIT = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took "
    r"([0-9]+(?:\.[0-9]+)?) seconds"
)
_ROLES = ("base", "proposal")


def _expect_keys(
    raw: Mapping[str, Any], required: set[str], context: str
) -> None:
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _finite_nonnegative(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return parsed


def _optional_number(value: Any, context: str) -> float | None:
    return None if value is None else _finite_nonnegative(value, context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EngineStartupEvidence:
    role: str
    engine_fingerprint_sha256: str
    model_id: str
    cache_key: str | None
    weight_load_seconds: float | None
    dynamo_seconds: float | None
    graph_compile_seconds: float
    compile_warmup_seconds: float | None
    graph_capture_seconds: float | None
    graph_capture_gib: float | None
    engine_init_seconds: float
    compiler_invoked: bool
    graph_capture_invoked: bool

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unsupported engine role: {self.role}")
        require_digest(
            self.engine_fingerprint_sha256, "engine_fingerprint_sha256"
        )
        if not self.model_id:
            raise ValueError("startup model_id cannot be empty")
        if self.cache_key is not None and re.fullmatch(
            r"[0-9a-f]+", self.cache_key
        ) is None:
            raise ValueError("cache_key must be lowercase hexadecimal")
        for name in (
            "weight_load_seconds",
            "dynamo_seconds",
            "graph_compile_seconds",
            "compile_warmup_seconds",
            "graph_capture_seconds",
            "graph_capture_gib",
            "engine_init_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)
        if self.engine_init_seconds <= 0:
            raise ValueError("engine_init_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "engine_fingerprint_sha256": self.engine_fingerprint_sha256,
            "model_id": self.model_id,
            "cache_key": self.cache_key,
            "weight_load_seconds": self.weight_load_seconds,
            "dynamo_seconds": self.dynamo_seconds,
            "graph_compile_seconds": self.graph_compile_seconds,
            "compile_warmup_seconds": self.compile_warmup_seconds,
            "graph_capture_seconds": self.graph_capture_seconds,
            "graph_capture_gib": self.graph_capture_gib,
            "engine_init_seconds": self.engine_init_seconds,
            "compiler_invoked": self.compiler_invoked,
            "graph_capture_invoked": self.graph_capture_invoked,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EngineStartupEvidence":
        keys = {
            "role",
            "engine_fingerprint_sha256",
            "model_id",
            "cache_key",
            "weight_load_seconds",
            "dynamo_seconds",
            "graph_compile_seconds",
            "compile_warmup_seconds",
            "graph_capture_seconds",
            "graph_capture_gib",
            "engine_init_seconds",
            "compiler_invoked",
            "graph_capture_invoked",
        }
        _expect_keys(raw, keys, "engine startup evidence")
        for name in ("compiler_invoked", "graph_capture_invoked"):
            if not isinstance(raw[name], bool):
                raise ValueError(f"{name} must be boolean")
        cache_key = raw["cache_key"]
        if cache_key is not None and not isinstance(cache_key, str):
            raise ValueError("cache_key must be a string or null")
        return cls(
            role=str(raw["role"]),
            engine_fingerprint_sha256=str(raw["engine_fingerprint_sha256"]),
            model_id=str(raw["model_id"]),
            cache_key=cache_key,
            weight_load_seconds=_optional_number(
                raw["weight_load_seconds"], "weight_load_seconds"
            ),
            dynamo_seconds=_optional_number(raw["dynamo_seconds"], "dynamo_seconds"),
            graph_compile_seconds=_finite_nonnegative(
                raw["graph_compile_seconds"], "graph_compile_seconds"
            ),
            compile_warmup_seconds=_optional_number(
                raw["compile_warmup_seconds"], "compile_warmup_seconds"
            ),
            graph_capture_seconds=_optional_number(
                raw["graph_capture_seconds"], "graph_capture_seconds"
            ),
            graph_capture_gib=_optional_number(
                raw["graph_capture_gib"], "graph_capture_gib"
            ),
            engine_init_seconds=_finite_nonnegative(
                raw["engine_init_seconds"], "engine_init_seconds"
            ),
            compiler_invoked=raw["compiler_invoked"],
            graph_capture_invoked=raw["graph_capture_invoked"],
        )


@dataclass(frozen=True, slots=True)
class HarnessRunCost:
    run_id: str
    sequence_index: int
    configuration_id: str
    workload_seed: int
    run_manifest_sha256: str
    effective_config_sha256: str
    runner_log_sha256: str
    startup_seconds: float
    engines: tuple[EngineStartupEvidence, ...]

    def __post_init__(self) -> None:
        require_id(self.run_id, "harness run_id")
        require_id(self.configuration_id, "harness configuration_id")
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
            raise ValueError("harness run must contain ordered base and proposal engines")
        expected = sum(engine.engine_init_seconds for engine in self.engines)
        if abs(self.startup_seconds - expected) > 1e-9:
            raise ValueError("run startup_seconds does not match engine evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "configuration_id": self.configuration_id,
            "workload_seed": self.workload_seed,
            "run_manifest_sha256": self.run_manifest_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            "runner_log_sha256": self.runner_log_sha256,
            "startup_seconds": self.startup_seconds,
            "engines": [engine.to_dict() for engine in self.engines],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HarnessRunCost":
        keys = {
            "run_id",
            "sequence_index",
            "configuration_id",
            "workload_seed",
            "run_manifest_sha256",
            "effective_config_sha256",
            "runner_log_sha256",
            "startup_seconds",
            "engines",
        }
        _expect_keys(raw, keys, "harness run cost")
        engines = raw["engines"]
        if not isinstance(engines, list):
            raise ValueError("harness run engines must be an array")
        for name in ("sequence_index", "workload_seed"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"{name} must be an integer")
        return cls(
            run_id=str(raw["run_id"]),
            sequence_index=raw["sequence_index"],
            configuration_id=str(raw["configuration_id"]),
            workload_seed=raw["workload_seed"],
            run_manifest_sha256=str(raw["run_manifest_sha256"]),
            effective_config_sha256=str(raw["effective_config_sha256"]),
            runner_log_sha256=str(raw["runner_log_sha256"]),
            startup_seconds=_finite_nonnegative(
                raw["startup_seconds"], "startup_seconds"
            ),
            engines=tuple(
                EngineStartupEvidence.from_dict(
                    require_object(item, "engine startup evidence")
                )
                for item in engines
            ),
        )


@dataclass(frozen=True, slots=True)
class CacheReuseGroup:
    role: str
    engine_fingerprint_sha256: str
    run_ids: tuple[str, ...]
    cache_keys: tuple[str, ...]
    first_engine_init_seconds: float
    later_median_engine_init_seconds: float | None
    engine_init_reduction_fraction: float | None
    first_compile_warmup_seconds: float | None
    later_median_compile_warmup_seconds: float | None
    compile_warmup_reduction_fraction: float | None
    repeated_compiler_invocations: int
    repeated_graph_captures: int
    classification: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unsupported cache reuse role: {self.role}")
        require_digest(
            self.engine_fingerprint_sha256, "engine_fingerprint_sha256"
        )
        if not self.run_ids:
            raise ValueError("cache reuse group must contain runs")
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("cache reuse group run ids must be unique")
        if tuple(sorted(set(self.cache_keys))) != self.cache_keys:
            raise ValueError("cache reuse group cache keys must be sorted and unique")
        for key in self.cache_keys:
            if re.fullmatch(r"[0-9a-f]+", key) is None:
                raise ValueError("cache reuse keys must be lowercase hexadecimal")
        for name in (
            "first_engine_init_seconds",
            "later_median_engine_init_seconds",
            "first_compile_warmup_seconds",
            "later_median_compile_warmup_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)
        for name in ("repeated_compiler_invocations", "repeated_graph_captures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        allowed = {
            "single_observation",
            "cache_key_missing_or_changed",
            "cache_key_reused_compile_reduced",
            "cache_key_reused_compile_repeated",
            "cache_key_reused_without_compile",
        }
        if self.classification not in allowed:
            raise ValueError(
                f"unsupported cache reuse classification: {self.classification}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "engine_fingerprint_sha256": self.engine_fingerprint_sha256,
            "run_ids": list(self.run_ids),
            "cache_keys": list(self.cache_keys),
            "first_engine_init_seconds": self.first_engine_init_seconds,
            "later_median_engine_init_seconds": self.later_median_engine_init_seconds,
            "engine_init_reduction_fraction": self.engine_init_reduction_fraction,
            "first_compile_warmup_seconds": self.first_compile_warmup_seconds,
            "later_median_compile_warmup_seconds": (
                self.later_median_compile_warmup_seconds
            ),
            "compile_warmup_reduction_fraction": (
                self.compile_warmup_reduction_fraction
            ),
            "repeated_compiler_invocations": self.repeated_compiler_invocations,
            "repeated_graph_captures": self.repeated_graph_captures,
            "classification": self.classification,
        }


@dataclass(frozen=True, slots=True)
class HarnessCostIssue:
    run_id: str
    message: str

    def __post_init__(self) -> None:
        require_id(self.run_id, "harness cost issue run_id")
        if not self.message:
            raise ValueError("harness cost issue message cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "message": self.message}


@dataclass(frozen=True, slots=True)
class HarnessCostAssessment:
    plan_sha256: str
    campaign_id: str
    expected_run_count: int
    valid_run_count: int
    complete: bool
    issues: tuple[HarnessCostIssue, ...]
    runs: tuple[HarnessRunCost, ...]
    cache_reuse_groups: tuple[CacheReuseGroup, ...]
    summary: Mapping[str, Any]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported harness cost assessment schema: {self.schema_version}"
            )
        if not self.producer:
            raise ValueError("harness cost producer cannot be empty")
        require_digest(self.plan_sha256, "harness cost plan_sha256")
        require_id(self.campaign_id, "harness cost campaign_id")
        for name in ("expected_run_count", "valid_run_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.expected_run_count <= 0:
            raise ValueError("expected_run_count must be positive")
        if self.valid_run_count != len(self.runs):
            raise ValueError("valid_run_count does not match harness runs")
        if self.valid_run_count > self.expected_run_count:
            raise ValueError("valid_run_count exceeds expected_run_count")
        if self.complete != (
            self.valid_run_count == self.expected_run_count and not self.issues
        ):
            raise ValueError("harness cost completeness does not match evidence")
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("harness cost run ids must be unique")
        sequence = [run.sequence_index for run in self.runs]
        if sequence != sorted(sequence) or len(sequence) != len(set(sequence)):
            raise ValueError("harness cost runs must have unique sorted sequence indexes")
        if self.cache_reuse_groups != _derive_cache_groups(self.runs):
            raise ValueError("cache reuse groups do not match harness runs")
        if dict(self.summary) != _derive_summary(
            self.runs, self.cache_reuse_groups
        ):
            raise ValueError("harness cost summary does not match harness runs")

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
            "cache_reuse_groups": [
                group.to_dict() for group in self.cache_reuse_groups
            ],
            "summary": dict(self.summary),
        }
        return {**payload, "harness_cost_assessment_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HarnessCostAssessment":
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
            "cache_reuse_groups",
            "summary",
            "harness_cost_assessment_sha256",
        }
        _expect_keys(raw, keys, "harness cost assessment")
        payload = dict(raw)
        digest = str(payload.pop("harness_cost_assessment_sha256"))
        require_digest(digest, "harness_cost_assessment_sha256")
        if digest != canonical_sha256(payload):
            raise ValueError("harness cost assessment SHA256 does not match content")
        issues_raw = raw["issues"]
        runs_raw = raw["runs"]
        groups_raw = raw["cache_reuse_groups"]
        summary = require_object(raw["summary"], "harness cost summary")
        if not all(isinstance(items, list) for items in (issues_raw, runs_raw, groups_raw)):
            raise ValueError("harness cost issue, run, and group fields must be arrays")
        issues = []
        for item in issues_raw:
            issue = require_object(item, "harness cost issue")
            _expect_keys(issue, {"run_id", "message"}, "harness cost issue")
            issues.append(HarnessCostIssue(str(issue["run_id"]), str(issue["message"])))
        runs = tuple(
            HarnessRunCost.from_dict(require_object(item, "harness run cost"))
            for item in runs_raw
        )
        expected_groups = _derive_cache_groups(runs)
        actual_groups = tuple(
            _cache_group_from_dict(require_object(item, "cache reuse group"))
            for item in groups_raw
        )
        if actual_groups != expected_groups:
            raise ValueError("cache reuse groups do not match embedded run evidence")
        expected_summary = _derive_summary(runs, expected_groups)
        if dict(summary) != expected_summary:
            raise ValueError("harness cost summary does not match embedded run evidence")
        for name in ("expected_run_count", "valid_run_count"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"{name} must be an integer")
        expected_count = raw["expected_run_count"]
        valid_count = raw["valid_run_count"]
        complete = raw["complete"]
        if not isinstance(complete, bool):
            raise ValueError("harness cost complete must be boolean")
        if valid_count != len(runs):
            raise ValueError("valid_run_count does not match embedded runs")
        if complete != (valid_count == expected_count and not issues):
            raise ValueError("harness cost completeness does not match evidence")
        return cls(
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
            plan_sha256=str(raw["plan_sha256"]),
            campaign_id=str(raw["campaign_id"]),
            expected_run_count=expected_count,
            valid_run_count=valid_count,
            complete=complete,
            issues=tuple(issues),
            runs=runs,
            cache_reuse_groups=expected_groups,
            summary=expected_summary,
        )

    def audit(self) -> dict[str, Any]:
        parsed = HarnessCostAssessment.from_dict(self.to_dict())
        classifications: dict[str, int] = defaultdict(int)
        for group in parsed.cache_reuse_groups:
            classifications[group.classification] += 1
        return {
            "campaign_id": parsed.campaign_id,
            "complete": parsed.complete,
            "expected_run_count": parsed.expected_run_count,
            "valid_run_count": parsed.valid_run_count,
            "issue_count": len(parsed.issues),
            "cache_group_count": len(parsed.cache_reuse_groups),
            "cache_group_classifications": dict(sorted(classifications.items())),
            **dict(parsed.summary),
        }


def _single_match(pattern: re.Pattern[str], block: str) -> str | None:
    matches = pattern.findall(block)
    if len(matches) > 1:
        raise ValueError(f"startup log contains repeated {pattern.pattern!r} records")
    return None if not matches else str(matches[0])


def _engine_fingerprint(
    manifest: Mapping[str, Any], effective: Mapping[str, Any], role: str
) -> str:
    environment = require_object(
        manifest.get("environment_contract"), "run environment contract"
    )
    models = require_object(environment.get("models"), "environment models")
    software = require_object(environment.get("software"), "environment software")
    model = require_object(models.get(role), f"environment {role} model")
    deployment = require_object(
        effective.get("deployment_settings"), "effective deployment settings"
    )
    role_settings = {
        key: value
        for key, value in deployment.items()
        if key == "model_runner" or key.startswith(f"{role}_")
    }
    semantic = require_object(
        effective.get("semantic_invariants"), "effective semantic invariants"
    )
    fingerprint = {
        "schema_version": "1.0",
        "role": role,
        "model": model,
        "software": software,
        "source_config_sha256": effective.get("source_config_sha256"),
        "deployment_settings": role_settings,
        "semantic_runtime": {
            key: semantic.get(key)
            for key in ("automatic_prefix_caching", "chunked_prefill", "model_runner")
        },
    }
    return canonical_sha256(fingerprint)


def _parse_engine_block(
    block: str,
    *,
    role: str,
    model_id: str,
    fingerprint: str,
) -> EngineStartupEvidence:
    weight_load = _single_match(_WEIGHT_LOAD, block)
    cache_key = _single_match(_CACHE_DIRECTORY, block)
    dynamo = _single_match(_DYNAMO, block)
    compile_warmup = _single_match(_COMPILE_WARMUP, block)
    init = _single_match(_ENGINE_INIT, block)
    if init is None:
        raise ValueError(f"runner log omits {role} engine initialization cost")
    compile_records = _GRAPH_COMPILE.findall(block)
    capture_records = _GRAPH_CAPTURE.findall(block)
    if len(capture_records) > 1:
        raise ValueError(f"runner log contains repeated {role} graph capture costs")
    capture_seconds = None
    capture_gib = None
    if capture_records:
        capture_seconds = float(capture_records[0][0])
        capture_gib = float(capture_records[0][1])
    return EngineStartupEvidence(
        role=role,
        engine_fingerprint_sha256=fingerprint,
        model_id=model_id,
        cache_key=cache_key,
        weight_load_seconds=None if weight_load is None else float(weight_load),
        dynamo_seconds=None if dynamo is None else float(dynamo),
        graph_compile_seconds=sum(float(seconds) for _, seconds in compile_records),
        compile_warmup_seconds=(
            None if compile_warmup is None else float(compile_warmup)
        ),
        graph_capture_seconds=capture_seconds,
        graph_capture_gib=capture_gib,
        engine_init_seconds=float(init),
        compiler_invoked=bool(compile_records or compile_warmup),
        graph_capture_invoked=bool(capture_records),
    )


def _load_run_cost(
    plan: CalibrationPlan, campaign_root: Path, run_id: str
) -> HarnessRunCost:
    expected = next(run for run in plan.runs if run.run_id == run_id)
    root = campaign_root / run_id
    manifest_path = root / "run-manifest.json"
    effective_path = root / "effective-config.json"
    log_path = root / "runner.log"
    for path in (manifest_path, effective_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing harness artifact: {path}")
    manifest = require_object(
        json.loads(manifest_path.read_text(encoding="utf-8")), "run manifest"
    )
    effective = require_object(
        json.loads(effective_path.read_text(encoding="utf-8")), "effective config"
    )
    manifest_run = require_object(manifest.get("run"), "manifest run")
    if manifest_run != expected.to_dict():
        raise ValueError(f"run manifest does not match plan for {run_id}")
    plan_sha256 = canonical_sha256(plan.to_dict())
    if manifest.get("plan_sha256") != plan_sha256:
        raise ValueError(f"run manifest plan binding mismatch for {run_id}")
    deployment = require_object(
        effective.get("deployment_settings"), "effective deployment settings"
    )
    configuration = require_object(manifest.get("configuration"), "manifest configuration")
    manifest_settings = require_object(
        configuration.get("settings"), "manifest configuration settings"
    )
    if deployment != manifest_settings:
        raise ValueError(f"effective deployment settings mismatch for {run_id}")

    log_text = log_path.read_text(encoding="utf-8")
    starts = list(_ENGINE_START.finditer(log_text))
    if len(starts) != len(_ROLES):
        raise ValueError(
            f"runner log must contain exactly {len(_ROLES)} engine initializations"
        )
    engines = []
    for index, role in enumerate(_ROLES):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log_text)
        engines.append(
            _parse_engine_block(
                log_text[starts[index].start() : end],
                role=role,
                model_id=starts[index].group(1),
                fingerprint=_engine_fingerprint(manifest, effective, role),
            )
        )
    return HarnessRunCost(
        run_id=run_id,
        sequence_index=expected.sequence_index,
        configuration_id=expected.configuration_id,
        workload_seed=expected.workload_seed,
        run_manifest_sha256=_file_sha256(manifest_path),
        effective_config_sha256=_file_sha256(effective_path),
        runner_log_sha256=_file_sha256(log_path),
        startup_seconds=sum(engine.engine_init_seconds for engine in engines),
        engines=tuple(engines),
    )


def _reduction(first: float | None, later: float | None) -> float | None:
    if first is None or later is None or first == 0:
        return None
    return 1.0 - later / first


def _derive_cache_groups(
    runs: Sequence[HarnessRunCost],
) -> tuple[CacheReuseGroup, ...]:
    grouped: dict[tuple[str, str], list[tuple[HarnessRunCost, EngineStartupEvidence]]] = (
        defaultdict(list)
    )
    for run in sorted(runs, key=lambda item: item.sequence_index):
        for engine in run.engines:
            grouped[(engine.role, engine.engine_fingerprint_sha256)].append((run, engine))
    result = []
    for (role, fingerprint), observations in sorted(grouped.items()):
        first = observations[0][1]
        later = [engine for _, engine in observations[1:]]
        later_init = median([item.engine_init_seconds for item in later]) if later else None
        later_compile_values = [
            item.compile_warmup_seconds
            for item in later
            if item.compile_warmup_seconds is not None
        ]
        later_compile = median(later_compile_values) if later_compile_values else None
        cache_keys = tuple(
            sorted({engine.cache_key for _, engine in observations if engine.cache_key})
        )
        repeated_compiler = sum(item.compiler_invoked for item in later)
        repeated_capture = sum(item.graph_capture_invoked for item in later)
        if not later:
            classification = "single_observation"
        elif len(cache_keys) != 1:
            classification = "cache_key_missing_or_changed"
        elif repeated_compiler:
            reduction = _reduction(first.compile_warmup_seconds, later_compile)
            classification = (
                "cache_key_reused_compile_reduced"
                if reduction is not None and reduction >= 0.2
                else "cache_key_reused_compile_repeated"
            )
        else:
            classification = "cache_key_reused_without_compile"
        result.append(
            CacheReuseGroup(
                role=role,
                engine_fingerprint_sha256=fingerprint,
                run_ids=tuple(run.run_id for run, _ in observations),
                cache_keys=cache_keys,
                first_engine_init_seconds=first.engine_init_seconds,
                later_median_engine_init_seconds=later_init,
                engine_init_reduction_fraction=_reduction(
                    first.engine_init_seconds, later_init
                ),
                first_compile_warmup_seconds=first.compile_warmup_seconds,
                later_median_compile_warmup_seconds=later_compile,
                compile_warmup_reduction_fraction=_reduction(
                    first.compile_warmup_seconds, later_compile
                ),
                repeated_compiler_invocations=repeated_compiler,
                repeated_graph_captures=repeated_capture,
                classification=classification,
            )
        )
    return tuple(result)


def _cache_group_from_dict(raw: Mapping[str, Any]) -> CacheReuseGroup:
    keys = {
        "role",
        "engine_fingerprint_sha256",
        "run_ids",
        "cache_keys",
        "first_engine_init_seconds",
        "later_median_engine_init_seconds",
        "engine_init_reduction_fraction",
        "first_compile_warmup_seconds",
        "later_median_compile_warmup_seconds",
        "compile_warmup_reduction_fraction",
        "repeated_compiler_invocations",
        "repeated_graph_captures",
        "classification",
    }
    _expect_keys(raw, keys, "cache reuse group")
    run_ids = raw["run_ids"]
    cache_keys = raw["cache_keys"]
    if not isinstance(run_ids, list) or not isinstance(cache_keys, list):
        raise ValueError("cache reuse run ids and keys must be arrays")
    for name in ("repeated_compiler_invocations", "repeated_graph_captures"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int):
            raise ValueError(f"{name} must be an integer")
    return CacheReuseGroup(
        role=str(raw["role"]),
        engine_fingerprint_sha256=str(raw["engine_fingerprint_sha256"]),
        run_ids=tuple(str(item) for item in run_ids),
        cache_keys=tuple(str(item) for item in cache_keys),
        first_engine_init_seconds=_finite_nonnegative(
            raw["first_engine_init_seconds"], "first_engine_init_seconds"
        ),
        later_median_engine_init_seconds=_optional_number(
            raw["later_median_engine_init_seconds"],
            "later_median_engine_init_seconds",
        ),
        engine_init_reduction_fraction=_optional_number_allow_negative(
            raw["engine_init_reduction_fraction"], "engine_init_reduction_fraction"
        ),
        first_compile_warmup_seconds=_optional_number(
            raw["first_compile_warmup_seconds"], "first_compile_warmup_seconds"
        ),
        later_median_compile_warmup_seconds=_optional_number(
            raw["later_median_compile_warmup_seconds"],
            "later_median_compile_warmup_seconds",
        ),
        compile_warmup_reduction_fraction=_optional_number_allow_negative(
            raw["compile_warmup_reduction_fraction"],
            "compile_warmup_reduction_fraction",
        ),
        repeated_compiler_invocations=raw["repeated_compiler_invocations"],
        repeated_graph_captures=raw["repeated_graph_captures"],
        classification=str(raw["classification"]),
    )


def _optional_number_allow_negative(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _derive_summary(
    runs: Sequence[HarnessRunCost], groups: Sequence[CacheReuseGroup]
) -> dict[str, Any]:
    engines = [engine for run in runs for engine in run.engines]
    repeated_runs = sum(max(0, len(group.run_ids) - 1) for group in groups)
    repeated_compile = sum(group.repeated_compiler_invocations for group in groups)
    repeated_capture = sum(group.repeated_graph_captures for group in groups)
    repeated_init_seconds = 0.0
    by_group = {
        (group.role, group.engine_fingerprint_sha256): group for group in groups
    }
    first_seen: set[tuple[str, str]] = set()
    for run in sorted(runs, key=lambda item: item.sequence_index):
        for engine in run.engines:
            key = (engine.role, engine.engine_fingerprint_sha256)
            if key in first_seen:
                repeated_init_seconds += engine.engine_init_seconds
            else:
                first_seen.add(key)
            if key not in by_group:
                raise ValueError("summary engine group is missing")
    return {
        "total_engine_startup_seconds": sum(
            engine.engine_init_seconds for engine in engines
        ),
        "total_compile_warmup_seconds": sum(
            engine.compile_warmup_seconds or 0.0 for engine in engines
        ),
        "total_graph_capture_seconds": sum(
            engine.graph_capture_seconds or 0.0 for engine in engines
        ),
        "repeated_engine_observation_count": repeated_runs,
        "repeated_engine_init_seconds": repeated_init_seconds,
        "repeated_compiler_invocation_count": repeated_compile,
        "repeated_graph_capture_count": repeated_capture,
        "cache_key_reused_compile_repeated_group_count": sum(
            group.classification == "cache_key_reused_compile_repeated"
            for group in groups
        ),
    }


def assess_harness_cost(
    plan: CalibrationPlan, campaign_root: str | Path
) -> HarnessCostAssessment:
    """Extract startup costs without mixing them into steady-state objectives."""

    root = Path(campaign_root).expanduser().resolve()
    issues = []
    runs = []
    for expected in plan.runs:
        try:
            runs.append(_load_run_cost(plan, root, expected.run_id))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            issues.append(HarnessCostIssue(expected.run_id, str(error)))
    ordered = tuple(sorted(runs, key=lambda item: item.sequence_index))
    groups = _derive_cache_groups(ordered)
    summary = _derive_summary(ordered, groups)
    return HarnessCostAssessment(
        plan_sha256=canonical_sha256(plan.to_dict()),
        campaign_id=plan.spec.campaign_id,
        expected_run_count=len(plan.runs),
        valid_run_count=len(ordered),
        complete=len(ordered) == len(plan.runs) and not issues,
        issues=tuple(issues),
        runs=ordered,
        cache_reuse_groups=groups,
        summary=summary,
    )


__all__ = [
    "CacheReuseGroup",
    "EngineStartupEvidence",
    "HarnessCostAssessment",
    "HarnessCostIssue",
    "HarnessRunCost",
    "assess_harness_cost",
]
