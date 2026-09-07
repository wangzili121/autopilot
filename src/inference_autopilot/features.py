"""Auditable feature extraction for offline and future online selectors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

from inference_autopilot.calibration.models import (
    RunObservation,
    canonical_json,
    canonical_sha256,
)
from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidenceRecord,
    QualityAssessment,
)


FeatureValue = bool | int | float | str

_ROW_KEYS = {
    "row_id",
    "source_kind",
    "source",
    "campaign",
    "variant",
    "cohort",
    "evidence",
    "eligibility",
    "static_features",
    "telemetry_features",
    "targets",
    "missing",
    "tags",
}

_TARGET_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "performance.completed_qps": (("completed_qps",), ("qps",)),
    "performance.elapsed_seconds": (("elapsed_seconds",),),
    "latency.p50_seconds": (("latency_seconds", "p50"),),
    "latency.p95_seconds": (("latency_seconds", "p95"), ("p95_seconds",)),
    "latency.p99_seconds": (("latency_seconds", "p99"),),
    "quality.accuracy": (("accuracy",),),
    "resource.preemptions": (("preemptions",),),
    "resource.proposal_kv_peak_fraction": (("proposal_kv_peak_fraction",),),
}

_TELEMETRY_CONTAINERS = {
    "queue_wait_seconds": "telemetry.queue_wait_seconds",
    "service_seconds": "telemetry.service_seconds",
    "continuous_batching": "telemetry.batching",
    "vllm_runtime_metrics": "telemetry.runtime",
    "compute": "telemetry.compute",
}

_TELEMETRY_SCALARS = {
    "aicore_mean_percent",
    "hbm_bandwidth_mean_percent",
    "proposal_running_max",
    "proposal_waiting_max",
    "temporary_allocation_gib",
    "kv_usage_at_failure_fraction",
}

_STATIC_EXCLUDED_WORKLOAD = {
    "arrival_trace_sha256",
    "dataset_path",
    "dataset_sha256",
    "method",
    "sequence_index",
    "workload_id",
}

_STATIC_EXCLUDED_ALGORITHM = {
    "algorithm_id",
    "autopilot_semantic_class",
    "graph_sha256",
    "method",
    "request_diagnostics",
    "semantic_class",
}

_SELECTOR_REQUIREMENTS = {
    "identity.algorithm_id": (("cohort", "algorithm_id"),),
    "identity.semantic_class": (("cohort", "semantic_class"),),
    "workload.requests": (("static", "workload.requests"),),
    "workload.load_descriptor": (
        ("static", "workload.workers"),
        ("static", "workload.arrival_qps"),
    ),
    "workload.context_descriptor": (
        ("static", "workload.prompt_tokens.mean"),
        ("static", "workload.prompt_tokens_mean"),
        ("static", "workload.prompt_tokens_approx"),
        ("static", "workload.context_tokens_requested"),
        ("static", "workload.context_tokens"),
        ("static", "workload.prompt_length_regime"),
    ),
    "algorithm.candidate_count": (("static", "algorithm.candidate_count"),),
    "algorithm.rollout_count": (("static", "algorithm.rollout_count"),),
    "algorithm.block_size": (("static", "algorithm.block_size"),),
    "algorithm.total_length": (("static", "algorithm.total_length"),),
    "algorithm.apply_importance_correction": (
        ("static", "algorithm.apply_importance_correction"),
    ),
    "deployment.base_max_num_seqs": (
        ("static", "deployment.base_max_num_seqs"),
    ),
    "deployment.proposal_max_num_seqs": (
        ("static", "deployment.proposal_max_num_seqs"),
    ),
    "deployment.base_max_num_batched_tokens": (
        ("static", "deployment.base_max_num_batched_tokens"),
    ),
    "deployment.proposal_max_num_batched_tokens": (
        ("static", "deployment.proposal_max_num_batched_tokens"),
    ),
    "deployment.base_memory_fraction": (
        ("static", "deployment.base_memory_fraction"),
    ),
    "deployment.proposal_memory_fraction": (
        ("static", "deployment.proposal_memory_fraction"),
    ),
}

_PRIOR_REQUIREMENTS = {
    name: alternatives
    for name, alternatives in _SELECTOR_REQUIREMENTS.items()
    if name
    in {
        "identity.algorithm_id",
        "identity.semantic_class",
        "workload.requests",
        "workload.load_descriptor",
        "workload.context_descriptor",
        "deployment.base_max_num_seqs",
        "deployment.proposal_max_num_seqs",
    }
}

_FEASIBILITY_REQUIREMENTS = {
    name: alternatives
    for name, alternatives in _PRIOR_REQUIREMENTS.items()
    if name != "workload.load_descriptor"
}

_REQUIRED_RESPONSE_TARGETS = (
    "performance.completed_qps",
    "latency.p95_seconds",
)


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _is_feature_value(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and isfinite(value)


def _feature_map(raw: Mapping[str, Any], context: str) -> dict[str, FeatureValue]:
    result: dict[str, FeatureValue] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        if not _is_feature_value(value):
            raise ValueError(f"{context} value for {key} must be a finite scalar")
        result[key] = value
    canonical_json(result)
    return dict(sorted(result.items()))


def _flatten_scalars(
    raw: Mapping[str, Any], prefix: str, *, excluded: set[str] | None = None
) -> dict[str, FeatureValue]:
    excluded = excluded or set()
    flattened: dict[str, FeatureValue] = {}
    for key, value in sorted(raw.items()):
        name = str(key)
        if name in excluded or value is None:
            continue
        path = f"{prefix}.{name}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_scalars(value, path))
        elif _is_feature_value(value):
            flattened[path] = value
    return flattened


def _lookup(raw: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = raw
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first_scalar(raw: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _lookup(raw, path)
        if _is_feature_value(value):
            return value
    return None


def _merge_objects(*objects: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for raw in objects:
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                    merged[key] = _merge_objects(merged[key], value)
                else:
                    merged[key] = value
    return merged


def _graph_features(algorithm: Mapping[str, Any]) -> dict[str, FeatureValue]:
    names = (
        "candidate_count",
        "rollout_count",
        "block_size",
        "total_length",
    )
    values = [algorithm.get(name) for name in names]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        return {}
    candidate_count, rollout_count, block_size, total_length = values
    correction = algorithm.get("apply_importance_correction")
    if not isinstance(correction, bool):
        return {}
    steps = ceil(total_length / block_size)
    remaining_sum = sum(
        max(total_length - min((step + 1) * block_size, total_length), 0)
        for step in range(steps)
    )
    rollout_sequences = candidate_count * rollout_count * steps
    rollout_tokens = candidate_count * rollout_count * remaining_sum
    return {
        "graph.guidance_steps_upper": steps,
        "graph.candidate_sequences_per_request_upper": candidate_count * steps,
        "graph.proposal_sequences_per_request_upper": rollout_sequences,
        "graph.target_score_sequences_per_request_upper": (
            rollout_sequences if correction else 0
        ),
        "graph.parallel_width_upper": candidate_count * rollout_count,
        "graph.candidate_token_slots_per_request_upper": candidate_count * total_length,
        "graph.proposal_token_slots_per_request_upper": rollout_tokens,
        "graph.target_score_token_slots_per_request_upper": (
            rollout_tokens if correction else 0
        ),
    }


def _deployment_graph_features(
    configuration: Mapping[str, Any],
) -> dict[str, FeatureValue]:
    features: dict[str, FeatureValue] = {}
    for engine_role in ("base", "proposal"):
        captures = configuration.get(f"{engine_role}_graph_capture_sizes")
        capacity = configuration.get(f"{engine_role}_max_num_seqs")
        if (
            not isinstance(captures, (list, tuple))
            or not captures
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in captures
            )
        ):
            continue
        prefix = f"deployment.{engine_role}_graph"
        ceiling = max(captures)
        features[f"{prefix}_capture_ceiling"] = ceiling
        features[f"{prefix}_capture_bucket_count"] = len(set(captures))
        if isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0:
            features[f"{prefix}_capacity_coverage_ratio"] = ceiling / capacity
            features[f"{prefix}_uncaptured_capacity"] = max(capacity - ceiling, 0)
            features[f"{prefix}_covers_scheduler_capacity"] = ceiling >= capacity
    return features


def deployment_features(
    configuration: Mapping[str, Any],
) -> dict[str, FeatureValue]:
    """Project runner settings into stable scalar features used by selectors."""

    features = _flatten_scalars(configuration, "deployment")
    features.update(_deployment_graph_features(configuration))
    return dict(sorted(features.items()))


def _extract_static(
    workload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, FeatureValue]:
    features = _flatten_scalars(
        workload, "workload", excluded=_STATIC_EXCLUDED_WORKLOAD
    )
    features.update(deployment_features(configuration))
    features.update(
        _flatten_scalars(
            algorithm,
            "algorithm",
            excluded=_STATIC_EXCLUDED_ALGORITHM,
        )
    )
    features.update(_flatten_scalars(environment, "environment"))
    arrival_qps = features.get("workload.arrival_qps")
    if isinstance(arrival_qps, (int, float)) and not isinstance(arrival_qps, bool):
        features["workload.closed_loop"] = arrival_qps == 0
    features.update(_graph_features(algorithm))
    return dict(sorted(features.items()))


def _extract_telemetry(metrics: Mapping[str, Any]) -> dict[str, FeatureValue]:
    features: dict[str, FeatureValue] = {}
    for source_name, prefix in _TELEMETRY_CONTAINERS.items():
        raw = metrics.get(source_name)
        if isinstance(raw, Mapping):
            features.update(_flatten_scalars(raw, prefix))
    for name in sorted(_TELEMETRY_SCALARS):
        value = metrics.get(name)
        if _is_feature_value(value):
            features[f"telemetry.{name}"] = value
    return dict(sorted(features.items()))


def _extract_targets(
    metrics: Mapping[str, Any], status: str | None
) -> dict[str, FeatureValue]:
    targets: dict[str, FeatureValue] = {}
    for name, paths in _TARGET_PATHS.items():
        value = _first_scalar(metrics, paths)
        if value is not None:
            targets[name] = value
    percent = metrics.get("proposal_kv_peak_percent")
    if (
        "resource.proposal_kv_peak_fraction" not in targets
        and isinstance(percent, (int, float))
        and not isinstance(percent, bool)
        and isfinite(float(percent))
    ):
        targets["resource.proposal_kv_peak_fraction"] = float(percent) / 100.0
    failure = metrics.get("failure")
    if status is not None:
        targets["run.status"] = status
        targets["run.success"] = status == "success"
    elif _is_feature_value(failure):
        targets["run.status"] = str(failure).lower()
        targets["run.success"] = False
    elif "performance.completed_qps" in targets:
        targets["run.status"] = "success"
        targets["run.success"] = True
    return dict(sorted(targets.items()))


def _missing_requirements(
    requirements: Mapping[str, tuple[tuple[str, str], ...]],
    cohort: Mapping[str, Any],
    static: Mapping[str, FeatureValue],
) -> tuple[str, ...]:
    sources = {"cohort": cohort, "static": static}
    missing = []
    for name, alternatives in requirements.items():
        values = (sources[section].get(key) for section, key in alternatives)
        if not any(value is not None and value != "" and value != "unknown" for value in values):
            missing.append(name)
    return tuple(sorted(missing))


@dataclass(frozen=True, slots=True)
class FeatureEligibility:
    response_model_fit: bool
    prior_construction: bool
    diagnostic_analysis: bool
    feasibility_model: bool
    performance_claim: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or any(not reason for reason in self.reasons):
            raise ValueError("feature eligibility requires non-empty reasons")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_model_fit": self.response_model_fit,
            "prior_construction": self.prior_construction,
            "diagnostic_analysis": self.diagnostic_analysis,
            "feasibility_model": self.feasibility_model,
            "performance_claim": self.performance_claim,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeatureEligibility":
        keys = {
            "response_model_fit",
            "prior_construction",
            "diagnostic_analysis",
            "feasibility_model",
            "performance_claim",
            "reasons",
        }
        _expect_exact_keys(raw, keys, "feature eligibility")
        boolean_keys = keys - {"reasons"}
        if any(not isinstance(raw[key], bool) for key in boolean_keys):
            raise ValueError("feature eligibility flags must be booleans")
        reasons = raw["reasons"]
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise ValueError("feature eligibility reasons must be strings")
        return cls(
            response_model_fit=raw["response_model_fit"],
            prior_construction=raw["prior_construction"],
            diagnostic_analysis=raw["diagnostic_analysis"],
            feasibility_model=raw["feasibility_model"],
            performance_claim=raw["performance_claim"],
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class SelectorFeatureRow:
    row_id: str
    source_kind: str
    source: Mapping[str, Any]
    campaign: str
    variant: str
    cohort: Mapping[str, Any]
    evidence: Mapping[str, Any]
    eligibility: FeatureEligibility
    static_features: Mapping[str, FeatureValue]
    telemetry_features: Mapping[str, FeatureValue]
    targets: Mapping[str, FeatureValue]
    missing: Mapping[str, tuple[str, ...]]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.row_id or not self.source_kind or not self.campaign or not self.variant:
            raise ValueError("feature row identity fields cannot be empty")
        _feature_map(self.static_features, "static features")
        _feature_map(self.telemetry_features, "telemetry features")
        _feature_map(self.targets, "targets")
        for name in ("source", "cohort", "evidence", "missing"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"feature row {name} must be an object")
        source_keys = {"path", "sha256", "locator"}
        if set(self.source) not in (source_keys, source_keys | {"imported_format"}):
            raise ValueError("feature source has missing or unknown fields")
        if not isinstance(self.source.get("locator"), str) or not self.source["locator"]:
            raise ValueError("feature source locator must be a non-empty string")
        source_sha = self.source.get("sha256")
        if source_sha is not None and (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise ValueError("feature source sha256 must be null or a lowercase digest")
        cohort_keys = {
            "algorithm_id",
            "semantic_class",
            "graph_sha256",
            "workload_id",
            "environment_id",
        }
        _expect_exact_keys(self.cohort, cohort_keys, "feature cohort")
        for name in ("algorithm_id", "semantic_class"):
            if not isinstance(self.cohort[name], str) or not self.cohort[name]:
                raise ValueError(f"feature cohort {name} must be a non-empty string")
        evidence_keys = {"grade", "purpose", "claim_eligible", "reasons"}
        _expect_exact_keys(self.evidence, evidence_keys, "feature evidence")
        if self.evidence["grade"] == "ungraded":
            if (
                self.evidence["purpose"] != "pending_assessment"
                or self.evidence["claim_eligible"] is not False
                or not isinstance(self.evidence["reasons"], list)
            ):
                raise ValueError("ungraded feature evidence must be pending assessment")
        else:
            QualityAssessment.from_dict(self.evidence)
        expected_missing = {
            "selector_requirements",
            "prior_requirements",
            "feasibility_requirements",
            "targets",
        }
        _expect_exact_keys(self.missing, expected_missing, "feature missingness")
        for name, values in self.missing.items():
            if not isinstance(values, tuple) or any(not value for value in values):
                raise ValueError(f"feature missingness {name} must be non-empty strings")
        expected_eligibility = _eligibility(
            str(self.evidence["grade"]),
            self.missing["selector_requirements"],
            self.missing["prior_requirements"],
            self.missing["feasibility_requirements"],
            self.missing["targets"],
            tuple(str(reason) for reason in self.evidence["reasons"]),
        )
        if self.eligibility != expected_eligibility:
            raise ValueError("feature eligibility does not match evidence and missingness")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("feature row tags must be unique")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "source_kind": self.source_kind,
            "source": dict(self.source),
            "campaign": self.campaign,
            "variant": self.variant,
            "cohort": dict(self.cohort),
            "evidence": dict(self.evidence),
            "eligibility": self.eligibility.to_dict(),
            "static_features": dict(sorted(self.static_features.items())),
            "telemetry_features": dict(sorted(self.telemetry_features.items())),
            "targets": dict(sorted(self.targets.items())),
            "missing": {key: list(values) for key, values in sorted(self.missing.items())},
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectorFeatureRow":
        _expect_exact_keys(raw, _ROW_KEYS, "selector feature row")
        object_keys = {
            "source",
            "cohort",
            "evidence",
            "eligibility",
            "static_features",
            "telemetry_features",
            "targets",
            "missing",
        }
        if any(not isinstance(raw[key], Mapping) for key in object_keys):
            raise ValueError("selector feature row object fields must be objects")
        missing_raw = raw["missing"]
        if any(not isinstance(value, list) for value in missing_raw.values()):
            raise ValueError("feature missingness entries must be lists")
        tags = raw["tags"]
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("feature row tags must be strings")
        return cls(
            row_id=str(raw["row_id"]),
            source_kind=str(raw["source_kind"]),
            source=dict(raw["source"]),
            campaign=str(raw["campaign"]),
            variant=str(raw["variant"]),
            cohort=dict(raw["cohort"]),
            evidence=dict(raw["evidence"]),
            eligibility=FeatureEligibility.from_dict(raw["eligibility"]),
            static_features=_feature_map(raw["static_features"], "static features"),
            telemetry_features=_feature_map(
                raw["telemetry_features"], "telemetry features"
            ),
            targets=_feature_map(raw["targets"], "targets"),
            missing={key: tuple(str(item) for item in value) for key, value in missing_raw.items()},
            tags=tuple(tags),
        )


@dataclass(frozen=True, slots=True)
class SelectorFeatureTable:
    rows: tuple[SelectorFeatureRow, ...]
    schema_version: str = "1.0"
    producer: str = "inference-autopilot/0.1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported selector feature schema: {self.schema_version}")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("selector feature row ids must be unique")

    def feature_catalog(self) -> dict[str, list[str]]:
        return {
            "static": sorted({key for row in self.rows for key in row.static_features}),
            "telemetry": sorted(
                {key for row in self.rows for key in row.telemetry_features}
            ),
            "targets": sorted({key for row in self.rows for key in row.targets}),
        }

    def audit(self) -> dict[str, Any]:
        grades = Counter(str(row.evidence.get("grade", "unknown")) for row in self.rows)
        semantics = Counter(
            str(row.cohort.get("semantic_class", "unknown")) for row in self.rows
        )
        sources = Counter(row.source_kind for row in self.rows)
        eligibility_names = (
            "response_model_fit",
            "prior_construction",
            "diagnostic_analysis",
            "feasibility_model",
            "performance_claim",
        )
        eligibility = {
            name: sum(bool(getattr(row.eligibility, name)) for row in self.rows)
            for name in eligibility_names
        }
        missing_selector = Counter(
            name
            for row in self.rows
            for name in row.missing["selector_requirements"]
        )
        coverage: dict[str, dict[str, int]] = {}
        for kind, attribute in (
            ("static", "static_features"),
            ("telemetry", "telemetry_features"),
            ("targets", "targets"),
        ):
            counts = Counter(key for row in self.rows for key in getattr(row, attribute))
            coverage[kind] = dict(sorted(counts.items()))
        return {
            "row_count": len(self.rows),
            "by_source_kind": dict(sorted(sources.items())),
            "by_evidence_grade": dict(sorted(grades.items())),
            "by_semantic_class": dict(sorted(semantics.items())),
            "eligibility": eligibility,
            "missing_selector_requirements": dict(sorted(missing_selector.items())),
            "feature_coverage": coverage,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "feature_catalog": self.feature_catalog(),
            "rows": [row.to_dict() for row in self.rows],
            "audit": self.audit(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectorFeatureTable":
        keys = {"schema_version", "producer", "feature_catalog", "rows", "audit"}
        _expect_exact_keys(raw, keys, "selector feature table")
        rows = raw["rows"]
        if not isinstance(rows, list):
            raise ValueError("selector feature rows must be a list")
        table = cls(
            rows=tuple(
                SelectorFeatureRow.from_dict(item)
                for item in rows
                if isinstance(item, Mapping)
            ),
            schema_version=str(raw["schema_version"]),
            producer=str(raw["producer"]),
        )
        if len(table.rows) != len(rows):
            raise ValueError("each selector feature row must be an object")
        if raw["feature_catalog"] != table.feature_catalog():
            raise ValueError("selector feature catalog does not match rows")
        if raw["audit"] != table.audit():
            raise ValueError("selector feature audit does not match rows")
        return table


def _cohort(
    algorithm: Mapping[str, Any], workload: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "algorithm_id": algorithm.get("algorithm_id")
        or algorithm.get("method")
        or workload.get("method")
        or "unknown",
        "semantic_class": algorithm.get("semantic_class")
        or algorithm.get("autopilot_semantic_class")
        or "unknown",
        "graph_sha256": algorithm.get("graph_sha256"),
        "workload_id": workload.get("workload_id"),
        "environment_id": environment.get("environment_id"),
    }


def _eligibility(
    grade: str,
    selector_missing: tuple[str, ...],
    prior_missing: tuple[str, ...],
    feasibility_missing: tuple[str, ...],
    missing_targets: tuple[str, ...],
    reasons: tuple[str, ...],
) -> FeatureEligibility:
    has_response_targets = not missing_targets
    is_a = grade == EvidenceGrade.A_FORMAL_PAIRED.value
    is_b = grade == EvidenceGrade.B_CONTROLLED_SINGLE.value
    is_c = grade == EvidenceGrade.C_DIAGNOSTIC.value
    is_x = grade == EvidenceGrade.X_EXCLUDED.value
    eligibility_reasons = list(reasons)
    if selector_missing:
        eligibility_reasons.append(
            f"selector context incomplete: {list(selector_missing)}"
        )
    if missing_targets:
        eligibility_reasons.append(f"response targets missing: {list(missing_targets)}")
    if is_x and feasibility_missing:
        eligibility_reasons.append(
            f"feasibility context incomplete: {list(feasibility_missing)}"
        )
    return FeatureEligibility(
        response_model_fit=is_a and not selector_missing and has_response_targets,
        prior_construction=is_b and not prior_missing and has_response_targets,
        diagnostic_analysis=is_c or grade == "ungraded",
        feasibility_model=is_x and not feasibility_missing,
        performance_claim=is_a,
        reasons=tuple(dict.fromkeys(eligibility_reasons)),
    )


def _build_row(
    *,
    row_id: str,
    source_kind: str,
    source: Mapping[str, Any],
    campaign: str,
    variant: str,
    workload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    environment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    status: str | None,
    evidence: Mapping[str, Any],
    tags: tuple[str, ...],
) -> SelectorFeatureRow:
    actual_workload = metrics.get("workload")
    actual_algorithm = metrics.get("algorithm")
    merged_workload = _merge_objects(workload, actual_workload)
    merged_algorithm = _merge_objects(algorithm, actual_algorithm)
    cohort = _cohort(merged_algorithm, merged_workload, environment)
    static = _extract_static(
        merged_workload, configuration, merged_algorithm, environment
    )
    telemetry = _extract_telemetry(metrics)
    targets = _extract_targets(metrics, status)
    selector_missing = _missing_requirements(
        _SELECTOR_REQUIREMENTS, cohort, static
    )
    prior_missing = _missing_requirements(_PRIOR_REQUIREMENTS, cohort, static)
    feasibility_missing = _missing_requirements(
        _FEASIBILITY_REQUIREMENTS, cohort, static
    )
    missing_targets = tuple(
        name for name in _REQUIRED_RESPONSE_TARGETS if name not in targets
    )
    grade = str(evidence["grade"])
    reasons_raw = evidence.get("reasons", ())
    reasons = tuple(str(reason) for reason in reasons_raw)
    return SelectorFeatureRow(
        row_id=row_id,
        source_kind=source_kind,
        source=dict(source),
        campaign=campaign,
        variant=variant,
        cohort=cohort,
        evidence=dict(evidence),
        eligibility=_eligibility(
            grade,
            selector_missing,
            prior_missing,
            feasibility_missing,
            missing_targets,
            reasons,
        ),
        static_features=static,
        telemetry_features=telemetry,
        targets=targets,
        missing={
            "selector_requirements": selector_missing,
            "prior_requirements": prior_missing,
            "feasibility_requirements": feasibility_missing,
            "targets": missing_targets,
        },
        tags=tags,
    )


def features_from_record(record: EvidenceRecord) -> SelectorFeatureRow:
    """Extract a selector row without changing an evidence record's grade."""

    status_raw = record.metrics.get("run_status")
    status = str(status_raw) if isinstance(status_raw, str) else None
    return _build_row(
        row_id=record.record_id,
        source_kind="evidence_record",
        source=record.source.to_dict(),
        campaign=record.campaign,
        variant=record.variant,
        workload=record.workload,
        configuration=record.configuration,
        algorithm=record.algorithm,
        environment=record.environment,
        metrics=record.metrics,
        status=status,
        evidence=record.quality.to_dict(),
        tags=record.tags,
    )


def features_from_ledger(ledger: EvidenceLedger) -> SelectorFeatureTable:
    """Build a sparse, audited table from normalized evidence."""

    return SelectorFeatureTable(tuple(features_from_record(record) for record in ledger.records))


def features_from_run(
    manifest_raw: Mapping[str, Any], observation: RunObservation
) -> SelectorFeatureTable:
    """Extract one pending row from a manifest-bound observation.

    The row stays ungraded until the calibration assessor verifies its comparison
    group. This prevents a successful process exit from becoming fit evidence.
    """

    manifest = dict(manifest_raw)
    manifest_digest = str(manifest.pop("run_manifest_sha256", ""))
    if manifest_digest != canonical_sha256(manifest):
        raise ValueError("run manifest SHA256 does not match its content")
    run = manifest.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("run manifest lacks a run object")
    if observation.run_id != run.get("run_id"):
        raise ValueError("observation run id does not match the run manifest")
    if observation.run_manifest_sha256 != manifest_digest:
        raise ValueError("observation is not bound to the run manifest")
    semantic = manifest.get("semantic_contract")
    workload_contract = manifest.get("workload_contract")
    environment = manifest.get("environment_contract")
    configuration = manifest.get("configuration")
    if not all(
        isinstance(value, Mapping)
        for value in (semantic, workload_contract, environment, configuration)
    ):
        raise ValueError("run manifest contracts must be objects")
    workload = workload_contract.get("parameters")
    settings = configuration.get("settings")
    invariants = semantic.get("invariants")
    if not all(isinstance(value, Mapping) for value in (workload, settings, invariants)):
        raise ValueError("run manifest parameters, settings and invariants must be objects")
    algorithm = {
        "algorithm_id": semantic.get("algorithm_id"),
        "semantic_class": semantic.get("semantic_class"),
        "graph_sha256": semantic.get("graph_sha256"),
        **dict(invariants),
    }
    source = {
        "path": observation.artifact.path if observation.artifact else None,
        "sha256": observation.artifact.sha256 if observation.artifact else None,
        "locator": observation.run_id,
    }
    row = _build_row(
        row_id=observation.run_id,
        source_kind="run_observation",
        source=source,
        campaign=str(run.get("comparison_group_id", observation.run_id)),
        variant=str(run.get("configuration_id", "unknown")),
        workload={
            "workload_id": workload_contract.get("workload_id"),
            **dict(workload),
        },
        configuration=dict(settings),
        algorithm=algorithm,
        environment=dict(environment),
        metrics=observation.metrics,
        status=observation.status,
        evidence={
            "grade": "ungraded",
            "purpose": "pending_assessment",
            "claim_eligible": False,
            "reasons": [
                "observation must pass its complete paired calibration group before grading"
            ],
        },
        tags=("pending_assessment", str(run.get("variant_role", "unknown"))),
    )
    return SelectorFeatureTable((row,))
