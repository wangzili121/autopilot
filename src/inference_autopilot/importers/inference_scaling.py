"""Conservative importer for legacy inference-scaling benchmark JSON."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from inference_autopilot.evidence import (
    EvidenceGrade,
    EvidenceLedger,
    EvidencePurpose,
    EvidenceRecord,
    ImportRejection,
    QualityAssessment,
    SourceArtifact,
)


JsonObject = dict[str, Any]
Parser = Callable[[JsonObject, SourceArtifact, str], list[EvidenceRecord]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select(raw: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: raw[key] for key in keys if key in raw}


def _require_object(raw: Any, context: str) -> JsonObject:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    return raw


def _require_list(raw: Any, context: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a JSON list")
    return raw


def _controlled(reason: str, *additional_reasons: str) -> QualityAssessment:
    return QualityAssessment(
        grade=EvidenceGrade.B_CONTROLLED_SINGLE,
        purpose=EvidencePurpose.CALIBRATION_ONLY,
        claim_eligible=False,
        reasons=(
            reason,
            *additional_reasons,
            "no paired repeats or run-order control encoded in source artifact",
        ),
    )


def _diagnostic(reason: str) -> QualityAssessment:
    return QualityAssessment(
        grade=EvidenceGrade.C_DIAGNOSTIC,
        purpose=EvidencePurpose.DIAGNOSTIC_ONLY,
        claim_eligible=False,
        reasons=(reason,),
    )


def _constraint(reason: str) -> QualityAssessment:
    return QualityAssessment(
        grade=EvidenceGrade.X_EXCLUDED,
        purpose=EvidencePurpose.CONSTRAINT_ONLY,
        claim_eligible=False,
        reasons=(reason, "excluded from throughput-model fitting"),
    )


def _source_with_locator(source: SourceArtifact, locator: str) -> SourceArtifact:
    return SourceArtifact(source.path, source.sha256, source.imported_format, locator)


def _record_id(source: SourceArtifact, locator: str) -> str:
    normalized = locator.replace("/", "_").replace("[", "_").replace("]", "")
    return f"{source.sha256[:16]}:{normalized}"


def _parse_capacity_sweep(
    data: JsonObject, source: SourceArtifact, campaign: str
) -> list[EvidenceRecord]:
    workload_raw = _require_object(data.get("workload"), "capacity sweep workload")
    runs = _require_list(data.get("runs"), "capacity sweep runs")
    if not runs:
        raise ValueError("capacity sweep runs cannot be empty")
    configuration_common = _select(
        workload_raw,
        (
            "base_memory_fraction",
            "proposal_memory_fraction",
            "graph_mode",
            "sampler",
        ),
    )
    workload = _select(
        workload_raw,
        ("method", "dataset", "requests", "workers", "arrival_qps"),
    )
    environment = _select(workload_raw, ("dtype",))
    records: list[EvidenceRecord] = []
    for index, raw_run in enumerate(runs):
        run = _require_object(raw_run, f"capacity sweep run {index}")
        locator = f"runs[{index}]"
        configuration = dict(configuration_common)
        configuration.update(
            _select(
                run,
                (
                    "base_max_num_seqs",
                    "proposal_max_num_seqs",
                    "base_max_num_batched_tokens",
                    "proposal_max_num_batched_tokens",
                ),
            )
        )
        metrics = _select(
            run,
            (
                "elapsed_seconds",
                "completed_qps",
                "p95_seconds",
                "aicore_mean_percent",
                "hbm_bandwidth_mean_percent",
                "proposal_kv_peak_percent",
                "preemptions",
            ),
        )
        if "completed_qps" not in metrics:
            raise ValueError(f"capacity sweep run {index} lacks completed_qps")
        variant = (
            f"base{configuration.get('base_max_num_seqs', 'unknown')}_"
            f"proposal{configuration.get('proposal_max_num_seqs', 'unknown')}"
        )
        records.append(
            EvidenceRecord(
                record_id=_record_id(source, locator),
                campaign=campaign,
                variant=variant,
                source=_source_with_locator(source, locator),
                workload=workload,
                configuration=configuration,
                algorithm={
                    "method": workload_raw.get("method"),
                    "autopilot_semantic_class": "legacy_unverified_exact_candidate",
                },
                environment=environment,
                metrics=metrics,
                quality=_controlled("single-observation controlled capacity sweep"),
                tags=("end_to_end", "capacity_sweep", "legacy_unverified_semantics"),
            )
        )
    return records


def _parse_named_config_sweep(
    data: JsonObject, source: SourceArtifact, campaign: str
) -> list[EvidenceRecord]:
    named_configs = {
        "small": _require_object(data.get("small"), "small configuration"),
        "large": _require_object(data.get("large"), "large configuration"),
    }
    results = _require_list(data.get("results"), "named configuration results")
    records: list[EvidenceRecord] = []
    for index, raw_result in enumerate(results):
        result = _require_object(raw_result, f"named configuration result {index}")
        config_name = result.get("config")
        if config_name not in named_configs:
            raise ValueError(f"result {index} references unknown configuration: {config_name}")
        named = named_configs[str(config_name)]
        configuration = {
            "base_max_num_seqs": named.get("base_max_num_seqs"),
            "proposal_max_num_seqs": named.get("proposal_max_num_seqs"),
            "base_max_num_batched_tokens": named.get("base_token_budget"),
            "proposal_max_num_batched_tokens": named.get("proposal_token_budget"),
        }
        locator = f"results[{index}]"
        records.append(
            EvidenceRecord(
                record_id=_record_id(source, locator),
                campaign=campaign,
                variant=f"requests{result.get('requests', 'unknown')}_{config_name}",
                source=_source_with_locator(source, locator),
                workload={
                    "method": data.get("path"),
                    "requests": result.get("requests"),
                },
                configuration=configuration,
                algorithm={
                    "method": data.get("path"),
                    "autopilot_semantic_class": "legacy_unverified_exact_candidate",
                },
                environment={},
                metrics=_select(
                    result,
                    (
                        "qps",
                        "p95_seconds",
                        "proposal_running_max",
                        "proposal_waiting_max",
                    ),
                ),
                quality=_controlled("single-observation low-pressure capacity comparison"),
                tags=(
                    "end_to_end",
                    "load_sweep",
                    "capacity_sweep",
                    "legacy_unverified_semantics",
                ),
            )
        )
    return records


def _kv_variant(name: str) -> tuple[str, bool | None]:
    if name.startswith("shared_"):
        prefix_mode = "shared"
    elif name.startswith("unique_"):
        prefix_mode = "unique"
    else:
        prefix_mode = "unknown"
    if name.endswith("_no_apc"):
        return prefix_mode, False
    if name.endswith("_apc"):
        return prefix_mode, True
    return prefix_mode, None


def _parse_kv_pressure(
    data: JsonObject, source: SourceArtifact, campaign: str
) -> list[EvidenceRecord]:
    experiments = _require_list(data.get("experiments"), "KV pressure experiments")
    models = _require_object(data.get("models"), "KV pressure models")
    common_configuration = _select(data, ("base_max_num_seqs", "proposal_max_num_seqs"))
    records: list[EvidenceRecord] = []
    for index, raw_experiment in enumerate(experiments):
        experiment = _require_object(raw_experiment, f"KV pressure experiment {index}")
        name = str(experiment.get("name", f"experiment_{index}"))
        prefix_mode, prefix_caching = _kv_variant(name)
        locator = f"experiments[{index}]"
        configuration = dict(common_configuration)
        configuration["prefix_caching"] = prefix_caching
        records.append(
            EvidenceRecord(
                record_id=_record_id(source, locator),
                campaign=campaign,
                variant=name,
                source=_source_with_locator(source, locator),
                workload={
                    "method": data.get("path"),
                    "requests": data.get("requests"),
                    "prefix_mode": prefix_mode,
                    "prompt_tokens_mean": experiment.get("prompt_tokens_mean"),
                },
                configuration=configuration,
                algorithm={
                    "method": data.get("path"),
                    "autopilot_semantic_class": "legacy_unverified_exact_candidate",
                },
                environment={"models": models},
                metrics={
                    "completed_qps": experiment.get("completed_qps"),
                    "p95_seconds": experiment.get("p95_seconds"),
                    "proposal_kv_peak_fraction": experiment.get("proposal_kv_peak"),
                    "preemptions": experiment.get("preemptions"),
                },
                quality=_controlled("single-observation KV/APC pressure experiment"),
                tags=("end_to_end", "kv_pressure", "legacy_unverified_semantics"),
            )
        )

    if "unique_8k_failure" in data:
        failure = _require_object(data["unique_8k_failure"], "unique 8K failure")
        locator = "unique_8k_failure"
        records.append(
            EvidenceRecord(
                record_id=_record_id(source, locator),
                campaign=campaign,
                variant="unique_8k_failure",
                source=_source_with_locator(source, locator),
                workload={
                    "method": data.get("path"),
                    "requests": data.get("requests"),
                    "prefix_mode": "unique",
                    "prompt_tokens_approx": failure.get("prompt_tokens_approx"),
                },
                configuration=common_configuration,
                algorithm={
                    "method": data.get("path"),
                    "autopilot_semantic_class": "legacy_unverified_exact_candidate",
                },
                environment={"models": models},
                metrics={
                    "failure": failure.get("failure"),
                    "temporary_allocation_gib": failure.get("temporary_allocation_gib"),
                    "kv_usage_at_failure_fraction": failure.get("kv_usage_at_failure"),
                },
                quality=_constraint("run failed before producing a valid performance observation"),
                tags=("failure", "memory_constraint", "long_context"),
            )
        )
    return records


def _parse_full_run(
    data: JsonObject, source: SourceArtifact, campaign: str
) -> list[EvidenceRecord]:
    runtime = _require_object(data.get("runtime"), "full-run runtime")
    workload_raw = _require_object(data.get("workload"), "full-run workload")
    algorithm = _require_object(data.get("algorithm", {}), "full-run algorithm")
    workload = {
        "method": data.get("method"),
        "requests": data.get("requests"),
        "workers": data.get("workers"),
        "arrival_qps": data.get("arrival_qps"),
        **_select(
            workload_raw,
            (
                "dataset",
                "context_tokens_requested",
                "prefix_mode",
                "prefix_caching",
                "prompt_tokens",
            ),
        ),
    }
    metrics = {
        **_select(data, ("elapsed_seconds", "completed_qps", "accuracy")),
        **_select(
            data,
            (
                "latency_seconds",
                "queue_wait_seconds",
                "service_seconds",
                "continuous_batching",
                "vllm_runtime_metrics",
                "compute",
            ),
        ),
    }
    if "completed_qps" not in metrics:
        raise ValueError("full-run artifact lacks completed_qps")
    semantic_class = "exact_algorithm"
    semantic_reason: str | None = None
    if algorithm.get("adaptive_rollout_pruning") or algorithm.get("selective_rescoring"):
        semantic_class = "approximate_algorithm"
        semantic_reason = (
            "algorithm changes rollout or target-scoring work and requires its own cohort"
        )
    elif any(marker in source.path.lower() for marker in ("selected_token", "tiled")):
        semantic_class = "numerical_runtime_variant"
        semantic_reason = "runtime scoring path is not established as trajectory-bit-exact"
    normalized_algorithm = dict(algorithm)
    normalized_algorithm["autopilot_semantic_class"] = semantic_class

    diagnostic_markers = ("smoke", "micro", "simulation", "simulated")
    lower_path = source.path.lower()
    semantic_reasons = () if semantic_reason is None else (semantic_reason,)
    quality = (
        _diagnostic("source path marks this run as smoke, microbenchmark or simulation")
        if any(marker in lower_path for marker in diagnostic_markers)
        else _controlled("single complete end-to-end run", *semantic_reasons)
    )
    semantic_tag = {
        "exact_algorithm": "exact_algorithm",
        "approximate_algorithm": "approximate_algorithm",
        "numerical_runtime_variant": "runtime_numerical_variant",
    }[semantic_class]
    locator = "run"
    return [
        EvidenceRecord(
            record_id=_record_id(source, locator),
            campaign=campaign,
            variant=(
                f"requests{data.get('requests', 'unknown')}_"
                f"base{runtime.get('base_max_num_seqs', 'unknown')}_"
                f"proposal{runtime.get('proposal_max_num_seqs', 'unknown')}"
            ),
            source=_source_with_locator(source, locator),
            workload=workload,
            configuration=runtime,
            algorithm=normalized_algorithm,
            environment={"dtype": data.get("dtype")},
            metrics=metrics,
            quality=quality,
            tags=("end_to_end", "full_run", semantic_tag),
        )
    ]


def _parse_v5_method_comparison(
    data: JsonObject, source: SourceArtifact, campaign: str
) -> list[EvidenceRecord]:
    methods = _require_object(data.get("methods"), "v5 method results")
    algorithm_config = _require_object(data.get("algorithm_config"), "v5 algorithm config")
    runtime_config = _require_object(data.get("runtime_config"), "v5 runtime config")
    environment = _require_object(data.get("environment", {}), "v5 environment")
    evaluation = _require_object(data.get("evaluation", {}), "v5 evaluation")
    models = _require_object(data.get("models", {}), "v5 models")
    if not methods:
        raise ValueError("v5 method results cannot be empty")

    records: list[EvidenceRecord] = []
    for method_name, raw_metrics in methods.items():
        method = str(method_name)
        metrics_source = _require_object(raw_metrics, f"v5 method {method}")
        metrics = _select(
            metrics_source,
            (
                "synchronous_seconds",
                "asynchronous_continuous_batching_seconds",
                "wall_time_gain_fraction",
                "wall_time_speedup_synchronous_over_asynchronous",
                "synchronous_over_asynchronous_flops",
                "synchronous_accuracy",
                "asynchronous_accuracy",
                "synchronous_mean_output_tokens",
                "asynchronous_mean_output_tokens",
                "output_exact_match_count",
                "output_exact_match_fraction",
                "outputs_bitwise_equal",
                "answer_match_count",
                "answer_match_fraction",
                "mean_common_prefix_fraction",
                "median_common_prefix_fraction",
                "synchronous_compute",
                "asynchronous_compute",
                "continuous_batching",
            ),
        )
        if not metrics:
            raise ValueError(f"v5 method {method} contains no supported metrics")
        method_algorithm = algorithm_config.get(method, {})
        if not isinstance(method_algorithm, dict):
            method_algorithm = {}
        normalized_algorithm = {
            "method": method,
            **method_algorithm,
            **_select(algorithm_config, ("max_new_tokens", "sampling")),
            "autopilot_semantic_class": "legacy_comparison",
        }
        problem_indices = evaluation.get("problem_indices")
        problem_count = len(problem_indices) if isinstance(problem_indices, list) else None
        locator = f"methods/{method}"
        records.append(
            EvidenceRecord(
                record_id=_record_id(source, locator),
                campaign=campaign,
                variant=method,
                source=_source_with_locator(source, locator),
                workload={
                    "benchmark": data.get("benchmark"),
                    "method": method,
                    "workers": data.get("workers"),
                    "problem_count": problem_count,
                    "dataset_path": evaluation.get("dataset_path"),
                    "dataset_sha256": evaluation.get("dataset_sha256"),
                },
                configuration=runtime_config,
                algorithm=normalized_algorithm,
                environment={
                    **environment,
                    "models": models,
                    "runtime_backend": data.get("runtime_backend"),
                    "runtime_backend_classes": data.get("runtime_backend_classes"),
                },
                metrics=metrics,
                quality=_diagnostic(
                    "legacy synchronous-versus-asynchronous comparison predates the "
                    "current strong baseline"
                ),
                tags=("legacy", "diagnostic", "weak_baseline_comparison"),
            )
        )
    return records


def _detect_parser(data: JsonObject) -> tuple[str, Parser] | None:
    if isinstance(data.get("workload"), dict) and isinstance(data.get("runs"), list):
        return "capacity_sweep_v1", _parse_capacity_sweep
    if all(key in data for key in ("small", "large", "results")):
        return "named_capacity_load_sweep_v1", _parse_named_config_sweep
    if isinstance(data.get("models"), dict) and isinstance(data.get("experiments"), list):
        return "kv_pressure_v1", _parse_kv_pressure
    if all(key in data for key in ("method", "workload", "runtime", "completed_qps")):
        return "full_end_to_end_run_v1", _parse_full_run
    if all(
        key in data
        for key in ("algorithm_config", "benchmark", "methods", "runtime_config")
    ):
        return "method_comparison_v5", _parse_v5_method_comparison
    return None


def import_results(source_path: str | Path) -> EvidenceLedger:
    """Import a JSON file or recursively scan a directory into an evidence ledger."""

    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_file():
        paths = [source_path]
        root = source_path.parent
    else:
        paths = sorted(path for path in source_path.rglob("*.json") if path.is_file())
        root = source_path

    records: list[EvidenceRecord] = []
    rejections: list[ImportRejection] = []
    seen_digests: dict[str, str] = {}
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        digest: str | None = None
        try:
            digest = _sha256(path)
            if digest in seen_digests:
                rejections.append(
                    ImportRejection(
                        relative_path,
                        digest,
                        f"duplicate content of {seen_digests[digest]}",
                    )
                )
                continue
            seen_digests[digest] = relative_path
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = _require_object(raw, "top-level JSON")
            detected = _detect_parser(data)
            if detected is None:
                keys = ", ".join(sorted(data)[:12])
                rejections.append(
                    ImportRejection(
                        relative_path,
                        digest,
                        f"unsupported legacy schema; keys=[{keys}]",
                    )
                )
                continue
            imported_format, parser = detected
            source = SourceArtifact(relative_path, digest, imported_format, "file")
            records.extend(parser(data, source, path.stem))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            rejections.append(ImportRejection(relative_path, digest, f"import failed: {error}"))
    return EvidenceLedger(records=tuple(records), rejections=tuple(rejections))
