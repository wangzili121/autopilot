"""Prepare and ingest runs for chang's conditional-IS small-proposal path."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import time
from typing import Any

from inference_autopilot.calibration.models import (
    ArtifactReference,
    CalibrationPlan,
    RunObservation,
    canonical_sha256,
    require_object,
)
from inference_autopilot.calibration.planning import build_run_manifest
from inference_autopilot.stage_wavefront import plan_stage_wavefront


ADAPTER_ID = "chang-pressure-v1"
_ALGORITHM_ID = "conditional_is_small_proposal"
_REQUIRED_SOURCE_FILES = (
    "experiments/arllm/gsm8k_async_benchmark.py",
    "experiments/arllm/gsm8k_reproduction.py",
    "experiments/arllm/runtime.py",
    "src/inference_scaling/arllm/algorithms/conditional_is.py",
    "src/inference_scaling/arllm/backends/batching.py",
    "src/inference_scaling/arllm/backends/loader.py",
    "src/inference_scaling/arllm/backends/vllm_backend.py",
    "src/inference_scaling/arllm/config.py",
)
_REQUIRED_SETTING_PATHS = {
    "base_max_num_seqs": ("vllm", "base", "max_num_seqs"),
    "proposal_max_num_seqs": ("vllm", "proposal", "max_num_seqs"),
    "base_max_num_batched_tokens": (
        "vllm",
        "base",
        "max_num_batched_tokens",
    ),
    "proposal_max_num_batched_tokens": (
        "vllm",
        "proposal",
        "max_num_batched_tokens",
    ),
    "base_memory_fraction": ("vllm", "base", "gpu_memory_utilization"),
    "proposal_memory_fraction": (
        "vllm",
        "proposal",
        "gpu_memory_utilization",
    ),
    "base_batch_wait_seconds": ("autopilot", "base_batch_wait_seconds"),
    "proposal_batch_wait_seconds": (
        "autopilot",
        "proposal_batch_wait_seconds",
    ),
    "base_score_priority": ("vllm", "base", "score_priority"),
}
_GRAPH_SETTING_NAMES = frozenset(
    {
        "base_graph_mode",
        "base_graph_capture_sizes",
        "proposal_graph_mode",
        "proposal_graph_capture_sizes",
    }
)
_ENVIRONMENT_SETTING_NAMES = frozenset({"model_runner"})
_STAGE_WAVEFRONT_REQUIRED_SETTING_NAMES = frozenset(
    {
        "proposal_stage_wavefront_mode",
        "proposal_graph_capture_ceiling",
        "proposal_stage_wavefront_max_wait_seconds",
        "proposal_stage_wavefront_min_utilization",
    }
)
_STAGE_WAVEFRONT_OPTIONAL_SETTING_NAMES = frozenset(
    {
        "proposal_stage_wavefront_max_shard_sequences",
        "proposal_stage_wavefront_max_shard_prefill_tokens",
        "proposal_stage_wavefront_max_shards_per_parent",
        "proposal_stage_wavefront_max_inflight_waves",
    }
)
_STAGE_WAVEFRONT_SETTING_NAMES = (
    _STAGE_WAVEFRONT_REQUIRED_SETTING_NAMES
    | _STAGE_WAVEFRONT_OPTIONAL_SETTING_NAMES
)
_SETTING_PATHS = {
    **_REQUIRED_SETTING_PATHS,
    **{
        name: ()
        for name in (
            _GRAPH_SETTING_NAMES
            | _ENVIRONMENT_SETTING_NAMES
            | _STAGE_WAVEFRONT_SETTING_NAMES
        )
    },
}
_PLACEHOLDER_MARKERS = ("replace-me", "replace-with-")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as stream:
        return require_object(tomllib.load(stream), "chang source config")


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    digest = str(payload.pop("run_manifest_sha256", ""))
    if digest != canonical_sha256(payload):
        raise ValueError("run manifest SHA256 does not match its content")
    return {**payload, "run_manifest_sha256": digest}


def _resolve_under(root: Path, path: Path, description: str) -> Path:
    root = root.expanduser().resolve()
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{description} must be inside source_repo")
    if not resolved.is_file():
        raise ValueError(f"{description} does not exist: {resolved}")
    return resolved


def _resolve_external(root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def _lookup(raw: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = raw
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if passed else "fail",
            "required_for_formal": required,
            "detail": detail,
        }
    )


def _has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
    if isinstance(value, Mapping):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_placeholder(item) for item in value)
    return False


def _looks_like_placeholder_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and len(set(value)) == 1


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def _source_snapshot(source_repo: Path) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    sources: dict[str, str] = {}
    for relative in _REQUIRED_SOURCE_FILES:
        path = source_repo / relative
        if path.is_file():
            hashes[relative] = _file_sha256(path)
            sources[relative] = path.read_text(encoding="utf-8")
    return hashes, sources


def source_snapshot_sha256(source_repo: Path) -> str:
    hashes, _sources = _source_snapshot(source_repo)
    return canonical_sha256(hashes)


def _autopilot_snapshot() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(package_root)): _file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }


def autopilot_snapshot_sha256() -> str:
    """Fingerprint the complete plugin-side Python implementation."""

    return canonical_sha256(_autopilot_snapshot())


def _git_revision(source_repo: Path) -> tuple[str | None, bool | None]:
    if not (source_repo / ".git").exists():
        return None, None
    revision = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(source_repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _semantic_values(config: Mapping[str, Any]) -> dict[str, Any]:
    conditional = require_object(config.get("conditional_is"), "conditional_is config")
    generation = require_object(config.get("generation"), "generation config")
    vllm = require_object(config.get("vllm"), "vllm config")
    engine_kwargs = require_object(
        vllm.get("engine_kwargs", {}), "vllm.engine_kwargs config"
    )
    return {
        "candidate_count": conditional.get("candidate_count"),
        "rollout_count": conditional.get("rollout_count"),
        "block_size": conditional.get("block_size"),
        "total_length": generation.get("max_new_tokens"),
        "apply_importance_correction": conditional.get(
            "apply_importance_correction", True
        ),
        "automatic_prefix_caching": vllm.get("enable_prefix_caching", True),
        "chunked_prefill": engine_kwargs.get("enable_chunked_prefill", False),
    }


def arrival_trace_sha256(parameters: Mapping[str, Any]) -> str:
    """Fingerprint the deterministic arrival offsets used by the pressure runner."""

    requests = int(parameters["requests"])
    arrival_qps = float(parameters.get("arrival_qps", 0.0))
    offsets = [0.0] * requests
    if arrival_qps > 0:
        offsets = [index / arrival_qps for index in range(requests)]
    return canonical_sha256(
        {
            "schema_version": "1.0",
            "kind": "deterministic_fixed_rate",
            "offset_seconds": offsets,
        }
    )


@dataclass(frozen=True, slots=True)
class ChangRunBundle:
    manifest: Mapping[str, Any]
    launch: Mapping[str, Any]
    effective_config: Mapping[str, Any]

    @property
    def formal_eligible(self) -> bool:
        return bool(self.launch["compatibility"]["formal_eligible"])

    def write(self, output_dir: Path) -> None:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("run-manifest.json", self.manifest),
            ("launch.json", self.launch),
            ("effective-config.json", self.effective_config),
        ):
            path = output_dir / name
            serialized = json.dumps(
                payload, indent=2, sort_keys=True, ensure_ascii=True
            ) + "\n"
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)


def build_chang_run_bundle(
    plan: CalibrationPlan,
    run_id: str,
    *,
    source_repo: Path,
    source_config: Path,
    data: Path,
    output_dir: Path,
    python_executable: str = "python3",
) -> ChangRunBundle:
    """Create a dry-run bundle and report whether it can support formal evidence."""

    source_repo = source_repo.expanduser().resolve()
    if not source_repo.is_dir():
        raise ValueError(f"source_repo does not exist: {source_repo}")
    source_config = _resolve_under(source_repo, source_config, "source_config")
    data = _resolve_external(source_repo, data)
    output_dir = output_dir.expanduser().resolve()
    manifest = _manifest_payload(build_run_manifest(plan, run_id))
    config = _load_toml(source_config)
    source_hashes, source_text = _source_snapshot(source_repo)
    autopilot_hashes = _autopilot_snapshot()
    autopilot_snapshot = canonical_sha256(autopilot_hashes)
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "algorithm_id",
        manifest["semantic_contract"]["algorithm_id"] == _ALGORITHM_ID,
        f"adapter requires {_ALGORITHM_ID}",
    )
    missing_sources = sorted(set(_REQUIRED_SOURCE_FILES) - set(source_hashes))
    _check(
        checks,
        "source_layout",
        not missing_sources,
        "all required chang entrypoints found"
        if not missing_sources
        else f"missing source files: {missing_sources}",
    )
    async_source = source_text.get(
        "experiments/arllm/gsm8k_async_benchmark.py", ""
    )
    _check(
        checks,
        "small_proposal_execution_path",
        all(
            marker in async_source
            for marker in (
                "conditional_is_small_proposal",
                "proposal_backend",
                "rollout_backend",
            )
        ),
        "async entrypoint must load and pass a distinct proposal backend",
    )
    graph_digest = manifest["semantic_contract"].get("graph_sha256")
    _check(
        checks,
        "graph_attestation",
        not _looks_like_placeholder_digest(graph_digest),
        "graph digest is not an obvious placeholder",
    )

    settings = require_object(
        manifest["configuration"]["settings"], "deployment settings"
    )
    unknown_settings = sorted(set(settings) - set(_SETTING_PATHS))
    missing_settings = sorted(set(_REQUIRED_SETTING_PATHS) - set(settings))
    _check(
        checks,
        "deployment_settings",
        not unknown_settings and not missing_settings,
        f"unknown={unknown_settings}, missing={missing_settings}",
    )
    supplied_graph_settings = set(settings).intersection(_GRAPH_SETTING_NAMES)
    graph_settings_complete = supplied_graph_settings in {
        frozenset(),
        _GRAPH_SETTING_NAMES,
    }
    _check(
        checks,
        "graph_policy_settings",
        graph_settings_complete,
        "graph policy is omitted or supplies mode and capture sizes for both engines"
        if graph_settings_complete
        else f"partial graph policy settings: {sorted(supplied_graph_settings)}",
    )
    if supplied_graph_settings == _GRAPH_SETTING_NAMES:
        invalid_graph_roles: list[str] = []
        for role in ("base", "proposal"):
            mode = settings[f"{role}_graph_mode"]
            sizes = settings[f"{role}_graph_capture_sizes"]
            valid_mode = (
                isinstance(mode, str)
                and bool(mode)
                and mode.replace("_", "").isalnum()
                and mode.upper() == mode
            )
            valid_sizes = (
                isinstance(sizes, list)
                and not any(isinstance(size, bool) for size in sizes)
                and all(_is_positive_integer(size) for size in sizes)
                and sizes == sorted(set(sizes))
                and ((mode == "NONE" and not sizes) or (mode != "NONE" and bool(sizes)))
            )
            if not valid_mode or not valid_sizes:
                invalid_graph_roles.append(role)
        _check(
            checks,
            "graph_policy_values",
            not invalid_graph_roles,
            "graph modes and capture sizes are valid"
            if not invalid_graph_roles
            else f"invalid graph policy roles: {invalid_graph_roles}",
        )
    supplied_wavefront_settings = set(settings).intersection(
        _STAGE_WAVEFRONT_SETTING_NAMES
    )
    wavefront_settings_complete = not supplied_wavefront_settings or (
        _STAGE_WAVEFRONT_REQUIRED_SETTING_NAMES
        <= supplied_wavefront_settings
        <= _STAGE_WAVEFRONT_SETTING_NAMES
    )
    _check(
        checks,
        "stage_wavefront_settings",
        wavefront_settings_complete,
        "stage-wavefront policy is omitted or fully specified"
        if wavefront_settings_complete
        else f"partial stage-wavefront settings: {sorted(supplied_wavefront_settings)}",
    )
    if wavefront_settings_complete and supplied_wavefront_settings:
        wavefront_mode = settings["proposal_stage_wavefront_mode"]
        capture_ceiling = settings["proposal_graph_capture_ceiling"]
        wavefront_wait = settings["proposal_stage_wavefront_max_wait_seconds"]
        minimum_utilization = settings[
            "proposal_stage_wavefront_min_utilization"
        ]
        maximum_parent_shards = settings.get(
            "proposal_stage_wavefront_max_shards_per_parent"
        )
        maximum_shard_sequences = settings.get(
            "proposal_stage_wavefront_max_shard_sequences"
        )
        maximum_shard_prefill_tokens = settings.get(
            "proposal_stage_wavefront_max_shard_prefill_tokens"
        )
        maximum_inflight_waves = settings.get(
            "proposal_stage_wavefront_max_inflight_waves"
        )
        shard_setting_valid = (
            wavefront_mode == "prefix_sharded"
            and _is_positive_integer(maximum_parent_shards)
            and _is_positive_integer(maximum_shard_sequences)
            and maximum_shard_sequences
            <= min(capture_ceiling, settings.get("proposal_max_num_seqs", 0))
            and _is_positive_integer(maximum_shard_prefill_tokens)
            and maximum_shard_prefill_tokens
            <= settings.get("proposal_max_num_batched_tokens", 0)
            and _is_positive_integer(maximum_inflight_waves)
            and maximum_inflight_waves
            <= min(capture_ceiling, settings.get("proposal_max_num_seqs", 0))
        ) or (
            wavefront_mode in {"off", "auto"}
            and maximum_parent_shards is None
            and maximum_shard_sequences is None
            and maximum_shard_prefill_tokens is None
            and maximum_inflight_waves is None
        )
        valid_wavefront_values = (
            isinstance(wavefront_mode, str)
            and wavefront_mode in {"off", "auto", "prefix_sharded"}
            and _is_positive_integer(capture_ceiling)
            and _is_nonnegative_number(wavefront_wait)
            and isinstance(minimum_utilization, (int, float))
            and not isinstance(minimum_utilization, bool)
            and 0.0 < float(minimum_utilization) <= 1.0
            and shard_setting_valid
        )
        _check(
            checks,
            "stage_wavefront_values",
            valid_wavefront_values,
            "stage-wavefront mode, ceiling, wait, utilization and sharding are valid"
            if valid_wavefront_values
            else "invalid stage-wavefront mode, ceiling, wait, utilization or sharding",
        )
        proposal_capture_sizes = settings.get("proposal_graph_capture_sizes")
        graph_ceiling_attested = (
            supplied_graph_settings == _GRAPH_SETTING_NAMES
            and isinstance(proposal_capture_sizes, list)
            and bool(proposal_capture_sizes)
            and all(_is_positive_integer(size) for size in proposal_capture_sizes)
            and _is_positive_integer(capture_ceiling)
            and capture_ceiling == max(proposal_capture_sizes)
        )
        _check(
            checks,
            "stage_wavefront_graph_ceiling",
            wavefront_mode == "off" or graph_ceiling_attested,
            "stage wavefront ceiling matches the explicit proposal graph policy"
            if wavefront_mode == "off" or graph_ceiling_attested
            else "stage wavefront requires an explicit proposal graph policy whose maximum capture size equals the declared ceiling",
        )
        invariants_for_wavefront = require_object(
            manifest["semantic_contract"]["invariants"], "semantic invariants"
        )
        candidate_count = invariants_for_wavefront.get("candidate_count")
        rollout_count = invariants_for_wavefront.get("rollout_count")
        admission_unit_sequences = (
            rollout_count
            if wavefront_mode == "prefix_sharded"
            else (
                candidate_count * rollout_count
                if _is_positive_integer(candidate_count)
                and _is_positive_integer(rollout_count)
                else None
            )
        )
        group_fits = (
            wavefront_mode == "off"
            or (
                _is_positive_integer(admission_unit_sequences)
                and _is_positive_integer(capture_ceiling)
                and _is_positive_integer(settings.get("proposal_max_num_seqs"))
                and (
                    wavefront_mode != "prefix_sharded"
                    or admission_unit_sequences <= maximum_shard_sequences
                )
                and admission_unit_sequences
                <= min(capture_ceiling, settings["proposal_max_num_seqs"])
            )
        )
        _check(
            checks,
            "stage_wavefront_group_capacity",
            group_fits,
            "one atomic proposal admission unit fits the graph-eligible scheduler capacity"
            if group_fits
            else "one atomic proposal admission unit exceeds the graph-eligible scheduler capacity",
        )
        workload_parameters_for_wavefront = require_object(
            manifest["workload_contract"]["parameters"],
            "workload parameters",
        )
        context_tokens = workload_parameters_for_wavefront.get("context_tokens", 0)
        proposal_token_cap = settings.get("proposal_max_num_batched_tokens")
        group_prefill_lower_bound_fits = (
            wavefront_mode in {"off", "prefix_sharded"}
            or (
                _is_nonnegative_number(context_tokens)
                and _is_positive_integer(candidate_count)
                and _is_positive_integer(rollout_count)
                and _is_positive_integer(proposal_token_cap)
                and context_tokens * candidate_count * rollout_count
                <= proposal_token_cap
            )
        )
        _check(
            checks,
            "stage_wavefront_group_prefill_lower_bound",
            group_prefill_lower_bound_fits,
            "one nominal proposal group fits the token cap under the declared context lower bound"
            if group_prefill_lower_bound_fits
            else "one nominal proposal group already exceeds the token cap under the declared context lower bound; use a validated sharding policy",
        )
        prefix_run_prefill_lower_bound_fits = (
            wavefront_mode != "prefix_sharded"
            or (
                _is_nonnegative_number(context_tokens)
                and _is_positive_integer(rollout_count)
                and _is_positive_integer(maximum_shard_prefill_tokens)
                and context_tokens * rollout_count
                <= maximum_shard_prefill_tokens
            )
        )
        _check(
            checks,
            "stage_wavefront_prefix_run_prefill_lower_bound",
            prefix_run_prefill_lower_bound_fits,
            "one repeated-prefix rollout run fits the token cap under the declared context lower bound"
            if prefix_run_prefill_lower_bound_fits
            else "one repeated-prefix rollout run exceeds the token cap under the declared context lower bound",
        )
    capacity_keys = (
        "base_max_num_seqs",
        "proposal_max_num_seqs",
        "base_max_num_batched_tokens",
        "proposal_max_num_batched_tokens",
    )
    invalid_capacities = [
        name for name in capacity_keys if not _is_positive_integer(settings.get(name))
    ]
    _check(
        checks,
        "capacity_values",
        not invalid_capacities,
        f"invalid positive integer settings: {invalid_capacities}",
    )
    wait_keys = ("base_batch_wait_seconds", "proposal_batch_wait_seconds")
    invalid_waits = [
        name for name in wait_keys if not _is_nonnegative_number(settings.get(name))
    ]
    _check(
        checks,
        "batch_wait_values",
        not invalid_waits,
        f"invalid non-negative wait settings: {invalid_waits}",
    )
    model_runner = settings.get("model_runner")
    _check(
        checks,
        "model_runner_value",
        model_runner is None or model_runner in {"MRV1", "MRV2"},
        "model runner is omitted or selects MRV1/MRV2"
        if model_runner is None or model_runner in {"MRV1", "MRV2"}
        else f"invalid model runner: {model_runner!r}",
    )
    memory_sum = float(settings.get("base_memory_fraction", 1.0)) + float(
        settings.get("proposal_memory_fraction", 1.0)
    )
    _check(
        checks,
        "memory_fraction",
        0.0 < memory_sum < 1.0,
        f"base plus proposal memory fraction is {memory_sum}",
    )

    loader_source = source_text.get(
        "src/inference_scaling/arllm/backends/loader.py", ""
    )
    backend_source = source_text.get(
        "src/inference_scaling/arllm/backends/vllm_backend.py", ""
    )
    score_priority_supported = "score_priority" in loader_source and (
        "score_priority" in backend_source
    )
    score_priority_requested = settings.get("base_score_priority") is not None
    _check(
        checks,
        "base_score_priority_capability",
        not score_priority_requested or score_priority_supported,
        "requested priority is supported by the source runtime"
        if score_priority_supported
        else "source runtime does not expose score_priority",
    )
    _check(
        checks,
        "runtime_telemetry_capability",
        "runtime_metrics" in backend_source,
        "vLLM backend must expose runtime_metrics for KV and preemption signals",
    )

    actual_semantics = _semantic_values(config)
    invariants = require_object(
        manifest["semantic_contract"]["invariants"], "semantic invariants"
    )
    comparable_invariants = {
        key: value for key, value in invariants.items() if key in actual_semantics
    }
    semantic_mismatches = {
        key: {"expected": expected, "source_config": actual_semantics.get(key)}
        for key, expected in comparable_invariants.items()
        if actual_semantics.get(key) != expected
    }
    _check(
        checks,
        "semantic_invariants",
        not semantic_mismatches,
        "source config matches verifiable semantic invariants"
        if not semantic_mismatches
        else f"mismatches={semantic_mismatches}",
    )
    models = require_object(config.get("models"), "models config")
    missing_models = []
    for role in ("base", "proposal"):
        configured = Path(str(models.get(role, ""))).expanduser()
        if not configured.is_absolute():
            configured = source_repo / configured
        if not configured.exists():
            missing_models.append(f"{role}:{configured.resolve()}")
    _check(
        checks,
        "model_paths",
        not missing_models,
        "base and proposal model paths exist"
        if not missing_models
        else f"missing model paths: {missing_models}",
    )
    environment = require_object(
        manifest["environment_contract"], "environment contract"
    )
    environment_models = require_object(
        environment.get("models"), "environment models"
    )
    model_hash_mismatches: dict[str, Any] = {}
    for role in ("base", "proposal"):
        role_contract = environment_models.get(role)
        expected_hash = (
            role_contract.get("weights_sha256")
            if isinstance(role_contract, Mapping)
            else None
        )
        configured_hash = models.get(f"{role}_weight_sha256")
        if expected_hash != configured_hash or _has_placeholder(expected_hash):
            model_hash_mismatches[role] = {
                "environment": expected_hash,
                "source_config": configured_hash,
            }
    _check(
        checks,
        "model_weight_attestation",
        not model_hash_mismatches,
        "model weight hashes match the source config"
        if not model_hash_mismatches
        else f"mismatches={model_hash_mismatches}",
    )

    workload = require_object(
        manifest["workload_contract"], "workload contract"
    )
    workload_parameters = require_object(
        workload["parameters"], "workload parameters"
    )
    supported_workload_keys = {
        "dataset",
        "requests",
        "workers",
        "arrival_qps",
        "prefix_mode",
        "prompt_length_regime",
        "context_tokens",
    }
    unknown_workload = sorted(set(workload_parameters) - supported_workload_keys)
    _check(
        checks,
        "workload_parameters",
        not unknown_workload,
        f"unsupported workload parameters: {unknown_workload}"
        if unknown_workload
        else "workload parameters are supported",
    )
    workload_values_valid = (
        _is_positive_integer(workload_parameters.get("requests"))
        and _is_positive_integer(workload_parameters.get("workers"))
        and _is_nonnegative_number(workload_parameters.get("arrival_qps", 0.0))
        and _is_nonnegative_number(workload_parameters.get("context_tokens", 0))
        and workload_parameters.get("prefix_mode", "unique") in {"unique", "shared"}
    )
    _check(
        checks,
        "workload_values",
        workload_values_valid,
        "requests/workers, arrival rate, context length and prefix mode are valid",
    )
    actual_dataset_sha = _file_sha256(data) if data.is_file() else None
    _check(
        checks,
        "dataset_file",
        data.is_file(),
        f"dataset file {'exists' if data.is_file() else 'is missing'}: {data}",
    )
    _check(
        checks,
        "dataset_sha256",
        actual_dataset_sha == workload["dataset_sha256"],
        f"actual={actual_dataset_sha}, contract={workload['dataset_sha256']}",
    )
    actual_arrival_sha = arrival_trace_sha256(workload_parameters)
    _check(
        checks,
        "arrival_trace_sha256",
        actual_arrival_sha == workload["arrival_trace_sha256"],
        f"actual={actual_arrival_sha}, contract={workload['arrival_trace_sha256']}",
    )
    source_snapshot = canonical_sha256(source_hashes)
    software = require_object(environment.get("software"), "environment software")
    expected_snapshot = software.get("inference_scaling_source_sha256")
    expected_revision = software.get("inference_scaling_commit")
    try:
        actual_revision, dirty = _git_revision(source_repo)
    except (OSError, subprocess.SubprocessError) as error:
        actual_revision, dirty = None, None
        revision_detail = f"git inspection failed: {error}"
    else:
        revision_detail = (
            f"revision={actual_revision}, dirty={dirty}, snapshot={source_snapshot}"
        )
    source_attested = expected_snapshot == source_snapshot or (
        actual_revision is not None
        and not dirty
        and expected_revision == actual_revision
    )
    _check(
        checks,
        "source_attestation",
        source_attested,
        revision_detail,
    )
    _check(
        checks,
        "environment_attestation",
        not _has_placeholder(environment),
        "environment contract contains no placeholders",
    )

    required_failures = [
        check["name"]
        for check in checks
        if check["required_for_formal"] and check["status"] != "pass"
    ]
    native_result = output_dir / "native-result.json"
    observation = output_dir / "observation.json"
    bundle_manifest = output_dir / "run-manifest.json"
    launch_manifest = output_dir / "launch.json"
    launch = {
        "schema_version": "1.0",
        "adapter_id": ADAPTER_ID,
        "created_at_unix": time.time(),
        "run_id": run_id,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "working_directory": str(source_repo),
        "command": [
            python_executable,
            "-m",
            "inference_autopilot.runners.chang_pressure",
            "execute",
            "--launch-manifest",
            str(launch_manifest),
            "--run-manifest",
            str(bundle_manifest),
            "--source-repo",
            str(source_repo),
            "--source-config",
            str(source_config),
            "--data",
            str(data),
            "--native-result",
            str(native_result),
            "--observation",
            str(observation),
            "--expected-source-config-sha256",
            _file_sha256(source_config),
            "--expected-source-snapshot-sha256",
            source_snapshot,
        ],
        "paths": {
            "source_repo": str(source_repo),
            "source_config": str(source_config),
            "data": str(data),
            "native_result": str(native_result),
            "observation": str(observation),
        },
        "source": {
            "source_config_sha256": _file_sha256(source_config),
            "dataset_sha256": actual_dataset_sha,
            "snapshot_sha256": source_snapshot,
            "git_revision": actual_revision,
            "git_dirty": dirty,
            "implementation_sha256": source_hashes,
        },
        "autopilot": {
            "snapshot_sha256": autopilot_snapshot,
            "implementation_sha256": autopilot_hashes,
        },
        "compatibility": {
            "formal_eligible": not required_failures,
            "failed_checks": required_failures,
            "checks": checks,
        },
    }
    effective_config = {
        "schema_version": "1.0",
        "adapter_id": ADAPTER_ID,
        "algorithm_id": _ALGORITHM_ID,
        "workload_seed": manifest["workload_contract"]["workload_seed"],
        "source_config": str(source_config),
        "source_config_sha256": _file_sha256(source_config),
        "autopilot_snapshot_sha256": autopilot_snapshot,
        "deployment_settings": settings,
        "workload_parameters": workload_parameters,
        "semantic_invariants": invariants,
    }
    return ChangRunBundle(manifest, launch, effective_config)


def _native_metric(raw: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value = _lookup(raw, path)
    if value is None:
        raise ValueError(f"native result is missing {'.'.join(path)}")
    return value


def _maximum_runtime_metric(
    raw: Mapping[str, Any],
    metric: str,
    roles: tuple[str, ...],
    *,
    allow_missing: bool = False,
) -> float | None:
    values: list[float] = []
    missing: list[str] = []
    for role in roles:
        value = _lookup(
            raw,
            ("vllm_runtime_metrics", role, metric, "maximum"),
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        else:
            missing.append(role)
    if missing:
        if allow_missing:
            return None
        raise ValueError(f"native result lacks {metric} telemetry for roles {missing}")
    return max(values)


def _validate_native_binding(
    manifest: Mapping[str, Any], native: Mapping[str, Any]
) -> None:
    expected_digest = manifest["run_manifest_sha256"]
    if native.get("run_manifest_sha256") != expected_digest:
        raise ValueError("native result is not bound to this run manifest")
    if native.get("method") != _ALGORITHM_ID:
        raise ValueError("native result is not conditional_is_small_proposal")
    if native.get("formal_preflight") is not True:
        raise ValueError("native result did not pass the formal launch preflight")
    expected_settings = manifest["configuration"]["settings"]
    actual_settings = native.get("runtime")
    if actual_settings != expected_settings:
        raise ValueError("native runtime settings do not match the run manifest")
    stage_wavefront = require_object(
        native.get("proposal_stage_wavefront", {"enabled": False}),
        "native proposal stage wavefront",
    )
    stage_enabled = stage_wavefront.get("enabled")
    if not isinstance(stage_enabled, bool):
        raise ValueError("native proposal stage wavefront enabled flag is invalid")
    stage_mode = expected_settings.get("proposal_stage_wavefront_mode", "off")
    if stage_enabled != (stage_mode != "off"):
        raise ValueError("native proposal stage wavefront mode does not match manifest")
    actual_stage_mode = stage_wavefront.get(
        "mode", "auto" if stage_enabled else "off"
    )
    if actual_stage_mode != stage_mode:
        raise ValueError("native proposal stage wavefront mode does not match manifest")
    if stage_enabled:
        workload = require_object(
            manifest["workload_contract"]["parameters"], "workload parameters"
        )
        invariants = require_object(
            manifest["semantic_contract"]["invariants"], "semantic invariants"
        )
        sequences_per_admission_unit = int(invariants["rollout_count"])
        if stage_mode != "prefix_sharded":
            sequences_per_admission_unit *= int(invariants["candidate_count"])
        admission_unit_concurrency = min(
            int(workload["requests"]), int(workload["workers"])
        )
        if stage_mode == "prefix_sharded":
            admission_unit_concurrency *= int(invariants["candidate_count"])
        expected_plan = plan_stage_wavefront(
            outer_concurrency=admission_unit_concurrency,
            sequences_per_group=sequences_per_admission_unit,
            graph_capture_ceiling=int(
                expected_settings["proposal_graph_capture_ceiling"]
            ),
            scheduler_sequence_cap=int(
                expected_settings["proposal_max_num_seqs"]
            ),
            minimum_full_wave_utilization=float(
                expected_settings["proposal_stage_wavefront_min_utilization"]
            ),
        )
        if stage_wavefront.get("plan") != expected_plan.to_dict():
            raise ValueError("native proposal stage wavefront plan does not match manifest")
        runtime = require_object(
            stage_wavefront.get("runtime"), "native proposal stage wavefront runtime"
        )
        expected_sequences = expected_plan.selected.sequences_per_wave
        expected_runtime = {
            "max_wave_sequences": expected_sequences,
            "target_wave_sequences": expected_sequences,
            "max_wave_prefill_tokens": expected_settings[
                "proposal_max_num_batched_tokens"
            ],
            "max_wait_seconds": expected_settings[
                "proposal_stage_wavefront_max_wait_seconds"
            ],
        }
        if stage_mode == "prefix_sharded":
            expected_runtime["max_shards_per_parent_per_wave"] = (
                expected_settings[
                    "proposal_stage_wavefront_max_shards_per_parent"
                ]
            )
            expected_runtime["max_shard_sequences"] = expected_settings[
                "proposal_stage_wavefront_max_shard_sequences"
            ]
            expected_runtime["max_shard_prefill_tokens"] = expected_settings[
                "proposal_stage_wavefront_max_shard_prefill_tokens"
            ]
            expected_runtime["max_inflight_waves"] = expected_settings[
                "proposal_stage_wavefront_max_inflight_waves"
            ]
            expected_runtime["max_inflight_sequences"] = expected_sequences
        mismatches = {
            key: {"expected": expected, "actual": runtime.get(key)}
            for key, expected in expected_runtime.items()
            if runtime.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "native proposal stage wavefront runtime does not match manifest: "
                f"{mismatches}"
            )
        if stage_mode == "prefix_sharded":
            _validate_prefix_sharded_runtime(
                runtime,
                max_wave_sequences=expected_sequences,
                max_wave_prefill_tokens=int(
                    expected_settings["proposal_max_num_batched_tokens"]
                ),
                max_shards_per_parent_per_wave=int(
                    expected_settings[
                        "proposal_stage_wavefront_max_shards_per_parent"
                    ]
                ),
                max_shard_sequences=int(
                    expected_settings[
                        "proposal_stage_wavefront_max_shard_sequences"
                    ]
                ),
                max_shard_prefill_tokens=int(
                    expected_settings[
                        "proposal_stage_wavefront_max_shard_prefill_tokens"
                    ]
                ),
                max_inflight_waves=int(
                    expected_settings[
                        "proposal_stage_wavefront_max_inflight_waves"
                    ]
                ),
            )
    expected_workload = manifest["workload_contract"]["parameters"]
    actual_workload = native.get("workload_parameters")
    if actual_workload != expected_workload:
        raise ValueError("native workload parameters do not match the run manifest")
    expected_semantics = manifest["semantic_contract"]["invariants"]
    actual_semantics = native.get("semantic_invariants")
    if actual_semantics != expected_semantics:
        raise ValueError("native semantic invariants do not match the run manifest")
    expected_snapshot = manifest["environment_contract"]["software"].get(
        "inference_scaling_source_sha256"
    )
    if (
        expected_snapshot is not None
        and native.get("source_snapshot_sha256") != expected_snapshot
    ):
        raise ValueError("native source snapshot does not match the environment contract")
    launch_digest = native.get("launch_manifest_sha256")
    if (
        not isinstance(launch_digest, str)
        or len(launch_digest) != 64
        or any(character not in "0123456789abcdef" for character in launch_digest)
    ):
        raise ValueError("native result lacks a valid launch manifest digest")


def _validated_wavefront_release_counts(
    runtime: Mapping[str, Any],
    wave_count: int,
    *,
    required: bool,
) -> dict[str, int]:
    if required and "release_reason_counts" not in runtime:
        raise ValueError(
            "proposal stage wavefront runtime lacks required release reasons"
        )
    raw_counts = require_object(
        runtime.get("release_reason_counts", {}),
        "proposal stage wavefront release reason counts",
    )
    allowed = {
        "target_sequences",
        "sequence_capacity",
        "prefill_token_capacity",
        "sequence_and_prefill_capacity",
        "collection_timeout",
        "shutdown",
    }
    if any(
        name not in allowed
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in raw_counts.items()
    ):
        raise ValueError("proposal stage wavefront release reasons are invalid")
    counts = {str(name): int(count) for name, count in raw_counts.items()}
    if required and sum(counts.values()) != wave_count:
        raise ValueError(
            "proposal stage wavefront release reason count does not match wave count"
        )
    waves = runtime.get("waves", [])
    if required:
        if not isinstance(waves, list) or len(waves) != wave_count:
            raise ValueError(
                "proposal stage wavefront wave records do not match wave count"
            )
        observed = Counter(
            str(require_object(wave, "proposal stage wavefront wave").get(
                "release_reason", ""
            ))
            for wave in waves
        )
        if dict(observed) != counts:
            raise ValueError(
                "proposal stage wavefront release reasons do not match wave records"
            )
    return counts


def _validated_nonnegative_int(
    runtime: Mapping[str, Any], name: str
) -> int:
    value = runtime.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"prefix-sharded wavefront runtime has invalid {name} telemetry"
        )
    return value


def _validated_nonnegative_float(
    runtime: Mapping[str, Any], name: str
) -> float:
    value = runtime.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"prefix-sharded wavefront runtime has invalid {name} telemetry"
        )
    return float(value)


def _validate_prefix_sharded_runtime(
    runtime: Mapping[str, Any],
    *,
    max_wave_sequences: int,
    max_wave_prefill_tokens: int,
    max_shards_per_parent_per_wave: int,
    max_shard_sequences: int,
    max_shard_prefill_tokens: int,
    max_inflight_waves: int,
) -> None:
    """Fail closed on the semantic and fairness evidence for sharded runs."""

    if runtime.get("mode") != "prefix_sharded":
        raise ValueError("prefix-sharded wavefront runtime mode is missing")
    counts = {
        name: _validated_nonnegative_int(runtime, name)
        for name in (
            "admitted_groups",
            "admitted_parent_groups",
            "admitted_shards",
            "admitted_sequences",
            "admitted_prefill_tokens",
            "completed_parent_groups",
            "repeated_prefix_run_count",
            "preserved_prefix_run_count",
            "forced_split_run_count",
            "oversized_atomic_run_count",
            "cancelled_shard_count",
            "wave_count",
            "partial_wave_count",
            "oversized_wave_count",
            "fairness_limited_wave_count",
            "fairness_violation_count",
            "cross_parent_wave_count",
            "parent_groups_across_waves",
            "maximum_shards_for_one_parent",
            "inflight_waves",
            "inflight_sequences",
            "maximum_inflight_waves",
            "maximum_inflight_sequences",
            "sequence_credit_stall_count",
            "streaming_sequence_credit_release_count",
            "batch_sequence_credit_release_count",
            "dispatch_failure_count",
        )
    }
    for name in (
        "mean_admission_wait_seconds",
        "maximum_admission_wait_seconds",
        "mean_wave_sequence_utilization",
        "mean_wave_prefill_utilization",
        "mean_parent_groups_per_wave",
        "mean_parent_completion_span_seconds",
        "maximum_parent_completion_span_seconds",
        "sequence_credit_stall_seconds",
    ):
        _validated_nonnegative_float(runtime, name)
    if counts["admitted_parent_groups"] <= 0:
        raise ValueError("prefix-sharded wavefront admitted no parent groups")
    if counts["admitted_groups"] != counts["admitted_parent_groups"]:
        raise ValueError("prefix-sharded admitted group aliases disagree")
    if counts["completed_parent_groups"] != counts["admitted_parent_groups"]:
        raise ValueError("prefix-sharded parent completion barrier is incomplete")
    if counts["admitted_shards"] < counts["admitted_parent_groups"]:
        raise ValueError("prefix-sharded runtime reports fewer shards than parents")
    if (
        counts["preserved_prefix_run_count"]
        + counts["forced_split_run_count"]
        != counts["repeated_prefix_run_count"]
    ):
        raise ValueError("prefix-sharded prefix-run accounting is inconsistent")
    zero_required = (
        "oversized_atomic_run_count",
        "cancelled_shard_count",
        "oversized_wave_count",
        "fairness_violation_count",
        "inflight_waves",
        "inflight_sequences",
        "dispatch_failure_count",
    )
    violations = {
        name: counts[name] for name in zero_required if counts[name] != 0
    }
    if violations:
        raise ValueError(
            "prefix-sharded runtime violated formal safety invariants: "
            f"{violations}"
        )
    if (
        counts["maximum_shards_for_one_parent"]
        > max_shards_per_parent_per_wave
    ):
        raise ValueError("prefix-sharded runtime exceeded its parent fairness cap")
    if counts["wave_count"] <= 0:
        raise ValueError("prefix-sharded wavefront emitted no waves")
    if counts["maximum_inflight_waves"] <= 0:
        raise ValueError("prefix-sharded wavefront did not dispatch any wave")
    if counts["maximum_inflight_waves"] > max_inflight_waves:
        raise ValueError("prefix-sharded runtime exceeded its in-flight wave cap")
    if counts["maximum_inflight_sequences"] > max_wave_sequences:
        raise ValueError("prefix-sharded runtime exceeded its sequence-credit cap")
    if (
        counts["streaming_sequence_credit_release_count"]
        + counts["batch_sequence_credit_release_count"]
        != counts["admitted_sequences"]
    ):
        raise ValueError("prefix-sharded sequence-credit releases are incomplete")
    _validated_wavefront_release_counts(
        runtime, counts["wave_count"], required=True
    )
    waves = runtime["waves"]
    observed_shards = 0
    observed_sequences = 0
    observed_prefill_tokens = 0
    observed_parent_groups = 0
    observed_cross_parent_waves = 0
    observed_maximum_inflight_waves = 0
    observed_maximum_inflight_sequences = 0
    for raw_wave in waves:
        wave = require_object(raw_wave, "prefix-sharded wavefront wave")
        shards = wave.get("shards")
        sequences = wave.get("sequences")
        prefill_tokens = wave.get("prefill_tokens")
        maximum_parent_shards = wave.get("maximum_shards_for_one_parent")
        parent_groups = wave.get("parent_groups")
        maximum_shard_sequences_observed = wave.get(
            "maximum_shard_sequences"
        )
        maximum_shard_prefill_observed = wave.get(
            "maximum_shard_prefill_tokens"
        )
        inflight_waves = wave.get("inflight_waves_after_admission")
        inflight_sequences = wave.get("inflight_sequences_after_admission")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                shards,
                sequences,
                prefill_tokens,
                maximum_parent_shards,
                parent_groups,
                maximum_shard_sequences_observed,
                maximum_shard_prefill_observed,
                inflight_waves,
                inflight_sequences,
            )
        ):
            raise ValueError("prefix-sharded wave record has invalid counters")
        if not shards or not sequences:
            raise ValueError("prefix-sharded wave record is empty")
        if sequences > max_wave_sequences or prefill_tokens > max_wave_prefill_tokens:
            raise ValueError("prefix-sharded wave record exceeds an admission cap")
        if maximum_parent_shards > max_shards_per_parent_per_wave:
            raise ValueError("prefix-sharded wave record violates parent fairness")
        if maximum_shard_sequences_observed > max_shard_sequences:
            raise ValueError("prefix-sharded wave record exceeds its shard sequence cap")
        if maximum_shard_prefill_observed > max_shard_prefill_tokens:
            raise ValueError("prefix-sharded wave record exceeds its shard token cap")
        if wave.get("fairness_violation") is not False:
            raise ValueError("prefix-sharded wave lacks a clean fairness attestation")
        if inflight_waves <= 0 or inflight_waves > max_inflight_waves:
            raise ValueError("prefix-sharded wave exceeded its in-flight wave cap")
        if inflight_sequences <= 0 or inflight_sequences > max_wave_sequences:
            raise ValueError("prefix-sharded wave exceeded its sequence-credit cap")
        observed_shards += shards
        observed_sequences += sequences
        observed_prefill_tokens += prefill_tokens
        observed_parent_groups += parent_groups
        observed_cross_parent_waves += int(parent_groups > 1)
        observed_maximum_inflight_waves = max(
            observed_maximum_inflight_waves, inflight_waves
        )
        observed_maximum_inflight_sequences = max(
            observed_maximum_inflight_sequences, inflight_sequences
        )
    if observed_shards != counts["admitted_shards"]:
        raise ValueError("prefix-sharded wave shard accounting is inconsistent")
    if observed_sequences != counts["admitted_sequences"]:
        raise ValueError("prefix-sharded wave sequence accounting is inconsistent")
    if observed_prefill_tokens != counts["admitted_prefill_tokens"]:
        raise ValueError("prefix-sharded wave token accounting is inconsistent")
    if observed_parent_groups != counts["parent_groups_across_waves"]:
        raise ValueError("prefix-sharded wave parent accounting is inconsistent")
    if observed_cross_parent_waves != counts["cross_parent_wave_count"]:
        raise ValueError("prefix-sharded cross-parent accounting is inconsistent")
    if observed_maximum_inflight_waves != counts["maximum_inflight_waves"]:
        raise ValueError("prefix-sharded in-flight wave accounting is inconsistent")
    if observed_maximum_inflight_sequences != counts["maximum_inflight_sequences"]:
        raise ValueError("prefix-sharded sequence-credit accounting is inconsistent")


def observation_from_chang_result(
    manifest_raw: Mapping[str, Any], native_result: Path
) -> RunObservation:
    """Convert a manifest-bound pressure result into a standard observation."""

    manifest = _manifest_payload(manifest_raw)
    native_result = native_result.expanduser().resolve()
    native = require_object(
        json.loads(native_result.read_text(encoding="utf-8")), "chang native result"
    )
    _validate_native_binding(manifest, native)
    started = float(_native_metric(native, ("started_at_unix",)))
    finished = float(_native_metric(native, ("finished_at_unix",)))
    preemptions = _maximum_runtime_metric(
        native,
        "vllm:num_preemptions",
        ("base", "proposal"),
        allow_missing=True,
    )
    stage_wavefront = require_object(
        native.get("proposal_stage_wavefront", {"enabled": False}),
        "proposal stage wavefront",
    )
    stage_wavefront_enabled = stage_wavefront.get("enabled") is True
    stage_wavefront_runtime = require_object(
        stage_wavefront.get("runtime", {}) if stage_wavefront_enabled else {},
        "proposal stage wavefront runtime",
    )
    wave_count = int(stage_wavefront_runtime.get("wave_count", 0))
    partial_wave_count = int(
        stage_wavefront_runtime.get("partial_wave_count", 0)
    )
    required_release_metrics = {
        "proposal_stage_wavefront_target_release_fraction",
        "proposal_stage_wavefront_sequence_capacity_release_fraction",
        "proposal_stage_wavefront_prefill_capacity_release_fraction",
        "proposal_stage_wavefront_timeout_release_fraction",
    }.intersection(manifest["required_metrics"])
    release_reason_counts = _validated_wavefront_release_counts(
        stage_wavefront_runtime,
        wave_count,
        required=bool(stage_wavefront_enabled and required_release_metrics),
    )
    target_release_count = int(release_reason_counts.get("target_sequences", 0))
    sequence_release_count = int(
        release_reason_counts.get("sequence_capacity", 0)
    ) + int(release_reason_counts.get("sequence_and_prefill_capacity", 0))
    prefill_release_count = int(
        release_reason_counts.get("prefill_token_capacity", 0)
    ) + int(release_reason_counts.get("sequence_and_prefill_capacity", 0))
    timeout_release_count = int(
        release_reason_counts.get("collection_timeout", 0)
    )
    metrics = {
        "completed_qps": _native_metric(native, ("completed_qps",)),
        "latency_seconds": _native_metric(native, ("latency_seconds",)),
        "accuracy": _native_metric(native, ("accuracy",)),
        "proposal_kv_peak_fraction": _maximum_runtime_metric(
            native,
            "vllm:kv_cache_usage_perc",
            ("proposal",),
        ),
        "elapsed_seconds": _native_metric(native, ("elapsed_seconds",)),
        "queue_wait_seconds": _native_metric(native, ("queue_wait_seconds",)),
        "service_seconds": _native_metric(native, ("service_seconds",)),
        "continuous_batching": _native_metric(native, ("continuous_batching",)),
        "vllm_runtime_metrics": _native_metric(native, ("vllm_runtime_metrics",)),
        "compute": _native_metric(native, ("compute",)),
        "algorithm": _native_metric(native, ("algorithm",)),
        "workload": _native_metric(native, ("workload",)),
        "proposal_stage_wavefront_enabled": float(stage_wavefront_enabled),
        "proposal_stage_wavefront_wave_count": float(wave_count),
        "proposal_stage_wavefront_partial_wave_fraction": (
            partial_wave_count / wave_count if wave_count else 0.0
        ),
        "proposal_stage_wavefront_oversized_wave_count": float(
            stage_wavefront_runtime.get("oversized_wave_count", 0)
        ),
        "proposal_stage_wavefront_target_release_fraction": (
            target_release_count / wave_count if wave_count else 0.0
        ),
        "proposal_stage_wavefront_sequence_capacity_release_fraction": (
            sequence_release_count / wave_count if wave_count else 0.0
        ),
        "proposal_stage_wavefront_prefill_capacity_release_fraction": (
            prefill_release_count / wave_count if wave_count else 0.0
        ),
        "proposal_stage_wavefront_timeout_release_fraction": (
            timeout_release_count / wave_count if wave_count else 0.0
        ),
        "proposal_stage_wavefront_mean_admission_wait_seconds": float(
            stage_wavefront_runtime.get("mean_admission_wait_seconds", 0.0)
        ),
        "proposal_stage_wavefront_max_admission_wait_seconds": float(
            stage_wavefront_runtime.get("maximum_admission_wait_seconds", 0.0)
        ),
        "proposal_stage_wavefront_admitted_parent_groups": float(
            stage_wavefront_runtime.get("admitted_parent_groups", 0)
        ),
        "proposal_stage_wavefront_admitted_shards": float(
            stage_wavefront_runtime.get("admitted_shards", 0)
        ),
        "proposal_stage_wavefront_completed_parent_groups": float(
            stage_wavefront_runtime.get("completed_parent_groups", 0)
        ),
        "proposal_stage_wavefront_repeated_prefix_run_count": float(
            stage_wavefront_runtime.get("repeated_prefix_run_count", 0)
        ),
        "proposal_stage_wavefront_preserved_prefix_run_count": float(
            stage_wavefront_runtime.get("preserved_prefix_run_count", 0)
        ),
        "proposal_stage_wavefront_forced_split_run_count": float(
            stage_wavefront_runtime.get("forced_split_run_count", 0)
        ),
        "proposal_stage_wavefront_oversized_atomic_run_count": float(
            stage_wavefront_runtime.get("oversized_atomic_run_count", 0)
        ),
        "proposal_stage_wavefront_cancelled_shard_count": float(
            stage_wavefront_runtime.get("cancelled_shard_count", 0)
        ),
        "proposal_stage_wavefront_fairness_violation_count": float(
            stage_wavefront_runtime.get("fairness_violation_count", 0)
        ),
        "proposal_stage_wavefront_cross_parent_wave_fraction": (
            float(stage_wavefront_runtime.get("cross_parent_wave_count", 0))
            / wave_count
            if wave_count
            else 0.0
        ),
        "proposal_stage_wavefront_mean_parent_groups_per_wave": float(
            stage_wavefront_runtime.get("mean_parent_groups_per_wave", 0.0)
        ),
        "proposal_stage_wavefront_mean_wave_sequence_utilization": float(
            stage_wavefront_runtime.get(
                "mean_wave_sequence_utilization", 0.0
            )
        ),
        "proposal_stage_wavefront_mean_wave_prefill_utilization": float(
            stage_wavefront_runtime.get(
                "mean_wave_prefill_utilization", 0.0
            )
        ),
        "proposal_stage_wavefront_max_shards_for_one_parent": float(
            stage_wavefront_runtime.get("maximum_shards_for_one_parent", 0)
        ),
        "proposal_stage_wavefront_mean_parent_completion_span_seconds": float(
            stage_wavefront_runtime.get(
                "mean_parent_completion_span_seconds", 0.0
            )
        ),
        "proposal_stage_wavefront_max_parent_completion_span_seconds": float(
            stage_wavefront_runtime.get(
                "maximum_parent_completion_span_seconds", 0.0
            )
        ),
        "proposal_stage_wavefront_max_inflight_waves": float(
            stage_wavefront_runtime.get("maximum_inflight_waves", 0)
        ),
        "proposal_stage_wavefront_max_inflight_sequences": float(
            stage_wavefront_runtime.get("maximum_inflight_sequences", 0)
        ),
        "proposal_stage_wavefront_sequence_credit_stall_count": float(
            stage_wavefront_runtime.get("sequence_credit_stall_count", 0)
        ),
        "proposal_stage_wavefront_sequence_credit_stall_seconds": float(
            stage_wavefront_runtime.get("sequence_credit_stall_seconds", 0.0)
        ),
        "proposal_stage_wavefront_dispatch_failure_count": float(
            stage_wavefront_runtime.get("dispatch_failure_count", 0)
        ),
        "proposal_stage_wavefront_streaming_sequence_credit_release_count": float(
            stage_wavefront_runtime.get(
                "streaming_sequence_credit_release_count", 0
            )
        ),
        "proposal_stage_wavefront_batch_sequence_credit_release_count": float(
            stage_wavefront_runtime.get("batch_sequence_credit_release_count", 0)
        ),
        "proposal_stage_wavefront_streaming_credit_fraction": (
            float(
                stage_wavefront_runtime.get(
                    "streaming_sequence_credit_release_count", 0
                )
            )
            / float(stage_wavefront_runtime.get("admitted_sequences", 0))
            if stage_wavefront_runtime.get("admitted_sequences", 0)
            else 0.0
        ),
    }
    if preemptions is not None:
        metrics["preemptions"] = preemptions
    observation = RunObservation(
        run_id=str(manifest["run"]["run_id"]),
        run_manifest_sha256=str(manifest["run_manifest_sha256"]),
        started_at_unix=started,
        finished_at_unix=finished,
        status="success",
        metrics=metrics,
        artifact=ArtifactReference(str(native_result), _file_sha256(native_result)),
        notes=(
            f"converted by {ADAPTER_ID}",
            f"launch_manifest_sha256={native['launch_manifest_sha256']}",
        ),
    )
    missing = []
    for name in manifest["required_metrics"]:
        value = _lookup(metrics, tuple(str(name).split(".")))
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
        ):
            missing.append(name)
    missing.sort()
    if missing:
        raise ValueError(f"converted observation lacks required metrics: {missing}")
    return observation


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return _manifest_payload(require_object(raw, "run manifest"))


def write_json(payload: Mapping[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
