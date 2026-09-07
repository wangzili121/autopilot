"""Open-loop execution worker for chang's small-proposal Conditional IS path."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, replace
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

from inference_autopilot.calibration.models import RunObservation, require_object
from inference_autopilot.runners.prefix_sharded_wavefront import (
    PrefixShardedStageWavefrontAdmissionBackend,
)
from inference_autopilot.runners.stage_wavefront import (
    StageWavefrontAdmissionBackend,
)
from inference_autopilot.stage_wavefront import (
    StageWavefrontPlan,
    plan_stage_wavefront,
)
from inference_autopilot.runners.chang import (
    _ALGORITHM_ID,
    _file_sha256,
    _load_toml,
    _semantic_values,
    autopilot_snapshot_sha256,
    load_manifest,
    observation_from_chang_result,
    source_snapshot_sha256,
    write_json,
)


class _CallTraceBackend:
    def __init__(self, backend: Any, role: str, measurement_semantics: str) -> None:
        self.backend = backend
        self.role = role
        self.measurement_semantics = measurement_semantics
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    @staticmethod
    def _distribution(values: Sequence[int]) -> dict[str, float | int]:
        if not values:
            return {"minimum": 0, "mean": 0.0, "maximum": 0, "total": 0}
        return {
            "minimum": min(values),
            "mean": sum(values) / len(values),
            "maximum": max(values),
            "total": sum(values),
        }

    @classmethod
    def _shape_summary(
        cls, kind: str, requests: Sequence[Any]
    ) -> dict[str, Any]:
        prefix_tokens: list[int] = []
        token_extents: list[int] = []
        shape_tokens: list[int] = []
        if kind == "sample":
            for request in requests:
                prefix = len(getattr(request, "prefix", ()))
                extent = int(getattr(request, "max_new_tokens", 0))
                prefix_tokens.append(prefix)
                token_extents.append(extent)
                shape_tokens.append(prefix + extent)
        else:
            for request in requests:
                prefix = len(getattr(request, "prefix", ()))
                for continuation in getattr(request, "continuations", ()):
                    extent = len(continuation)
                    prefix_tokens.append(prefix)
                    token_extents.append(extent)
                    shape_tokens.append(prefix + extent)
        return {
            "sequence_count": len(shape_tokens),
            "prefix_tokens": cls._distribution(prefix_tokens),
            "token_extent": cls._distribution(token_extents),
            "shape_tokens": cls._distribution(shape_tokens),
        }

    def _record(
        self,
        kind: str,
        requests: Sequence[Any],
        started: float,
        finished: float,
    ) -> None:
        event = {
            "role": self.role,
            "kind": kind,
            "measurement_semantics": self.measurement_semantics,
            "start_unix": started,
            "end_unix": finished,
            "duration_seconds": finished - started,
            "request_groups": len(requests),
            **self._shape_summary(kind, requests),
        }
        with self._lock:
            self.events.append(event)

    def _call(self, kind: str, requests: Sequence[Any]) -> Any:
        started = time.time()
        try:
            return getattr(self.backend, f"{kind}_batch")(requests)
        finally:
            finished = time.time()
            self._record(kind, requests, started, finished)

    def sample_batch(self, requests: Sequence[Any]) -> Any:
        return self._call("sample", requests)

    def sample_batch_with_callback(
        self, requests: Sequence[Any], on_complete: Any
    ) -> Any:
        callback = getattr(self.backend, "sample_batch_with_callback", None)
        if not callable(callback):
            samples = self.sample_batch(requests)
            for index, sample in enumerate(samples):
                on_complete(index, sample)
            return samples
        started = time.time()
        try:
            return callback(requests, on_complete)
        finally:
            self._record("sample", requests, started, time.time())

    def score_batch(self, requests: Sequence[Any]) -> Any:
        return self._call("score", requests)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _summarize_runtime_metrics(
    samples: Sequence[dict[str, Any]], role: str
) -> dict[str, dict[str, float]]:
    names = sorted(
        {
            name
            for sample in samples
            for name in require_object(sample.get(role, {}), f"{role} runtime metrics")
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = [
            float(sample[role][name])
            for sample in samples
            if name in sample.get(role, {})
            and isinstance(sample[role][name], (int, float))
            and not isinstance(sample[role][name], bool)
        ]
        if values:
            summary[name] = {
                "mean": sum(values) / len(values),
                "maximum": max(values),
                "final": values[-1],
            }
    return summary


def _context_text(request_index: int, target_tokens: int, shared: bool) -> str:
    owner = 0 if shared else request_index
    sentences = [
        (
            f"Reference record {owner}, entry {entry}: this background sentence is "
            "context only and does not change the arithmetic problem."
        )
        for entry in range(max(32, target_tokens // 12 + 32))
    ]
    return "\n".join(sentences)


def _problem_with_context(
    backend: Any,
    problem: Any,
    request_index: int,
    target_tokens: int,
    shared: bool,
) -> Any:
    if target_tokens <= 0:
        return problem
    text = _context_text(request_index, target_tokens, shared)
    token_ids = backend.encode(text, add_special_tokens=False)
    truncated = backend.decode(token_ids[:target_tokens])
    return replace(
        problem,
        question=(
            "Use the following reference as inert background; solve only the final "
            f"math problem.\n\n{truncated}\n\nMath problem:\n{problem.question}"
        ),
    )


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    table = config
    for part in path[:-1]:
        child = table.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"configuration path {'.'.join(path)} crosses a non-table")
        table = child
    table[path[-1]] = value


def _apply_model_runner_environment(settings: Mapping[str, Any]) -> None:
    model_runner = settings.get("model_runner")
    if model_runner is None:
        return
    values = {"MRV1": "0", "MRV2": "1"}
    if model_runner not in values:
        raise ValueError("model_runner must be MRV1 or MRV2")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = values[model_runner]


def _build_stage_wavefront_plan(
    settings: Mapping[str, Any],
    workload: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> StageWavefrontPlan | None:
    mode = settings.get("proposal_stage_wavefront_mode", "off")
    if mode == "off":
        return None
    if mode not in {"auto", "prefix_sharded"}:
        raise ValueError(
            "proposal_stage_wavefront_mode must be off, auto or prefix_sharded"
        )
    required = (
        "proposal_graph_capture_ceiling",
        "proposal_stage_wavefront_max_wait_seconds",
        "proposal_stage_wavefront_min_utilization",
    )
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError(f"auto proposal stage wavefront lacks settings: {missing}")
    wait_seconds = settings["proposal_stage_wavefront_max_wait_seconds"]
    if (
        not isinstance(wait_seconds, (int, float))
        or isinstance(wait_seconds, bool)
        or wait_seconds < 0
    ):
        raise ValueError(
            "proposal_stage_wavefront_max_wait_seconds must be non-negative"
        )
    parent_sequences = int(invariants["candidate_count"]) * int(
        invariants["rollout_count"]
    )
    sequences_per_admission_unit = (
        parent_sequences
        if mode == "auto"
        else int(invariants["rollout_count"])
    )
    context_token_lower_bound = int(workload.get("context_tokens", 0))
    proposal_token_cap = int(settings["proposal_max_num_batched_tokens"])
    lower_bound_sequences = (
        parent_sequences
        if mode == "auto"
        else int(invariants["rollout_count"])
    )
    if context_token_lower_bound * lower_bound_sequences > proposal_token_cap:
        raise ValueError(
            "one proposal group exceeds the stage-wavefront token cap under "
            "the declared context lower bound; validated group sharding is required"
            if mode == "auto"
            else "one repeated-prefix rollout run exceeds the stage-wavefront "
            "token cap under the declared context lower bound"
        )
    if mode == "prefix_sharded":
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
        if (
            isinstance(maximum_parent_shards, bool)
            or not isinstance(maximum_parent_shards, int)
            or maximum_parent_shards <= 0
            or isinstance(maximum_shard_sequences, bool)
            or not isinstance(maximum_shard_sequences, int)
            or maximum_shard_sequences < int(invariants["rollout_count"])
            or maximum_shard_sequences
            > min(
                int(settings["proposal_graph_capture_ceiling"]),
                int(settings["proposal_max_num_seqs"]),
            )
            or isinstance(maximum_shard_prefill_tokens, bool)
            or not isinstance(maximum_shard_prefill_tokens, int)
            or maximum_shard_prefill_tokens <= 0
            or maximum_shard_prefill_tokens > proposal_token_cap
            or isinstance(maximum_inflight_waves, bool)
            or not isinstance(maximum_inflight_waves, int)
            or maximum_inflight_waves <= 0
            or maximum_inflight_waves
            > min(
                int(settings["proposal_graph_capture_ceiling"]),
                int(settings["proposal_max_num_seqs"]),
            )
        ):
            raise ValueError(
                "prefix_sharded proposal stage wavefront requires valid shard "
                "sequence, prefill-token, parent-fairness and in-flight wave caps"
            )
        if (
            context_token_lower_bound * int(invariants["rollout_count"])
            > maximum_shard_prefill_tokens
        ):
            raise ValueError(
                "one repeated-prefix rollout run exceeds the configured shard "
                "token cap under the declared context lower bound"
            )
    admission_unit_concurrency = min(
        int(workload["requests"]), int(workload["workers"])
    )
    if mode == "prefix_sharded":
        admission_unit_concurrency *= int(invariants["candidate_count"])
    return plan_stage_wavefront(
        outer_concurrency=admission_unit_concurrency,
        sequences_per_group=sequences_per_admission_unit,
        graph_capture_ceiling=int(settings["proposal_graph_capture_ceiling"]),
        scheduler_sequence_cap=int(settings["proposal_max_num_seqs"]),
        minimum_full_wave_utilization=float(
            settings["proposal_stage_wavefront_min_utilization"]
        ),
    )


def _validate_and_apply(
    manifest: dict[str, Any], config: dict[str, Any], source_repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    semantic_contract = require_object(
        manifest["semantic_contract"], "semantic contract"
    )
    if semantic_contract["algorithm_id"] != _ALGORITHM_ID:
        raise ValueError(f"worker only supports {_ALGORITHM_ID}")
    invariants = require_object(
        semantic_contract["invariants"], "semantic invariants"
    )
    actual_semantics = _semantic_values(config)
    mismatches = {
        key: {"expected": expected, "source_config": actual_semantics.get(key)}
        for key, expected in invariants.items()
        if key in actual_semantics and actual_semantics[key] != expected
    }
    if mismatches:
        raise ValueError(f"source config violates semantic invariants: {mismatches}")

    settings = require_object(
        manifest["configuration"]["settings"], "deployment settings"
    )
    runtime_paths = {
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
        "base_memory_fraction": (
            "vllm",
            "base",
            "gpu_memory_utilization",
        ),
        "proposal_memory_fraction": (
            "vllm",
            "proposal",
            "gpu_memory_utilization",
        ),
    }
    for name, path in runtime_paths.items():
        _set_nested(config, path, settings[name])
    graph_setting_names = {
        f"{role}_{suffix}"
        for role in ("base", "proposal")
        for suffix in ("graph_mode", "graph_capture_sizes")
    }
    supplied_graph_settings = set(settings).intersection(graph_setting_names)
    if supplied_graph_settings and supplied_graph_settings != graph_setting_names:
        raise ValueError("graph policy requires mode and capture sizes for both engines")
    if supplied_graph_settings:
        for role in ("base", "proposal"):
            mode = settings[f"{role}_graph_mode"]
            sizes = settings[f"{role}_graph_capture_sizes"]
            if (
                not isinstance(mode, str)
                or not mode
                or mode.upper() != mode
                or not mode.replace("_", "").isalnum()
            ):
                raise ValueError(f"invalid {role} graph mode")
            if (
                not isinstance(sizes, list)
                or any(isinstance(size, bool) for size in sizes)
                or not all(isinstance(size, int) and size > 0 for size in sizes)
                or sizes != sorted(set(sizes))
                or (mode == "NONE" and sizes)
                or (mode != "NONE" and not sizes)
            ):
                raise ValueError(f"invalid {role} graph capture sizes")
            _set_nested(
                config,
                ("vllm", role, "engine_kwargs", "compilation_config"),
                {
                    "cudagraph_mode": mode,
                    **(
                        {"cudagraph_capture_sizes": list(sizes)}
                        if sizes
                        else {}
                    ),
                },
            )
    config.setdefault("runtime", {})["backend"] = "vllm"
    config.setdefault("run", {})["seed"] = manifest["workload_contract"][
        "workload_seed"
    ]
    models = require_object(config.get("models"), "models config")
    for role in ("base", "proposal"):
        model_path = Path(str(models[role])).expanduser()
        if not model_path.is_absolute():
            model_path = source_repo / model_path
        config["models"][role] = str(model_path.resolve())

    score_priority = settings.get("base_score_priority")
    if score_priority is not None:
        loader = source_repo / "src/inference_scaling/arllm/backends/loader.py"
        backend = source_repo / "src/inference_scaling/arllm/backends/vllm_backend.py"
        if "score_priority" not in loader.read_text(
            encoding="utf-8"
        ) or "score_priority" not in backend.read_text(encoding="utf-8"):
            raise ValueError("source runtime does not support base_score_priority")
        _set_nested(config, ("vllm", "base", "score_priority"), score_priority)
        _set_nested(
            config,
            ("vllm", "base", "engine_kwargs", "scheduling_policy"),
            "priority",
        )

    memory_sum = float(settings["base_memory_fraction"]) + float(
        settings["proposal_memory_fraction"]
    )
    if not 0.0 < memory_sum < 1.0:
        raise ValueError("base and proposal memory fractions must sum to less than one")
    workload = require_object(
        manifest["workload_contract"]["parameters"], "workload parameters"
    )
    return config, settings, workload


def _run_one_with_optional_diagnostics(
    callback: Any,
    method: str,
    base: Any,
    proposal: Any,
    raw_base: Any,
    prompt: Any,
    problem: Any,
    config: dict[str, Any],
    root_seed: int,
    diagnostics: dict[str, Any],
) -> Any:
    arguments = (
        method,
        base,
        proposal,
        raw_base,
        prompt,
        problem,
        config,
        root_seed,
    )
    if "request_diagnostics" in inspect.signature(callback).parameters:
        return callback(*arguments, diagnostics)
    return callback(*arguments)


def execute(
    manifest_path: Path,
    *,
    launch_manifest: Path,
    source_repo: Path,
    source_config: Path,
    data: Path,
    native_result: Path,
    expected_source_config_sha256: str,
    expected_source_snapshot_sha256: str,
    allow_nonformal: bool = False,
) -> dict[str, Any]:
    """Execute one manifest-bound run. Heavy runtime imports happen only here."""

    manifest = load_manifest(manifest_path)
    launch_manifest = launch_manifest.expanduser().resolve()
    launch = require_object(
        json.loads(launch_manifest.read_text(encoding="utf-8")), "launch manifest"
    )
    if launch.get("adapter_id") != "chang-pressure-v1":
        raise ValueError("launch manifest uses an unsupported adapter")
    if launch.get("run_manifest_sha256") != manifest["run_manifest_sha256"]:
        raise ValueError("launch manifest is not bound to the run manifest")
    compatibility = require_object(
        launch.get("compatibility"), "launch compatibility"
    )
    if not compatibility.get("formal_eligible") and not allow_nonformal:
        raise ValueError(
            "launch preflight is not formal-eligible: "
            f"{compatibility.get('failed_checks', [])}"
        )
    launch_autopilot = require_object(
        launch.get("autopilot"), "launch autopilot snapshot"
    )
    actual_autopilot_snapshot = autopilot_snapshot_sha256()
    if launch_autopilot.get("snapshot_sha256") != actual_autopilot_snapshot:
        raise ValueError(
            "Autopilot implementation changed after run-bundle preparation"
        )
    launch_source = require_object(launch.get("source"), "launch source")
    if launch_source.get("source_config_sha256") != expected_source_config_sha256:
        raise ValueError("source config digest differs from launch preflight")
    if launch_source.get("snapshot_sha256") != expected_source_snapshot_sha256:
        raise ValueError("source snapshot digest differs from launch preflight")
    source_repo = source_repo.expanduser().resolve()
    source_config = source_config.expanduser().resolve()
    data = data.expanduser().resolve()
    if _file_sha256(source_config) != expected_source_config_sha256:
        raise ValueError("source config changed after run-bundle preparation")
    actual_source_snapshot = source_snapshot_sha256(source_repo)
    if actual_source_snapshot != expected_source_snapshot_sha256:
        raise ValueError("inference-scaling source changed after run-bundle preparation")
    if _file_sha256(data) != manifest["workload_contract"]["dataset_sha256"]:
        raise ValueError("dataset SHA256 does not match the run manifest")

    config, settings, workload = _validate_and_apply(
        manifest, _load_toml(source_config), source_repo
    )
    _apply_model_runner_environment(settings)
    for path in (source_repo, source_repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from experiments.arllm.gsm8k_async_benchmark import (  # type: ignore[import-not-found]
        _compute_delta,
        _run_one,
        _sampling,
    )
    from experiments.arllm.gsm8k_reproduction import (  # type: ignore[import-not-found]
        _load_backend,
        _prompt_tokens,
    )
    from experiments.arllm.runtime import (  # type: ignore[import-not-found]
        validate_model_artifacts,
    )
    from inference_scaling.arllm.backends import (  # type: ignore[import-not-found]
        ContinuousBatchingBackend,
        ScoreCachingBackend,
        close_backend,
    )
    from inference_scaling.arllm.types import (  # type: ignore[import-not-found]
        GenerationRequest,
    )
    from inference_scaling.shared.evaluation import (  # type: ignore[import-not-found]
        extract_numeric_answer,
        load_gsm8k,
        select_problems,
    )

    validate_model_artifacts(config, {"base", "proposal"})
    request_count = int(workload["requests"])
    worker_count = min(int(workload["workers"]), request_count)
    arrival_qps = float(workload.get("arrival_qps", 0.0))
    context_tokens = int(workload.get("context_tokens", 0))
    prefix_mode = str(workload.get("prefix_mode", "unique"))
    if request_count <= 0 or worker_count <= 0:
        raise ValueError("requests and workers must be positive")
    if arrival_qps < 0 or context_tokens < 0:
        raise ValueError("arrival_qps and context_tokens must be non-negative")
    if prefix_mode not in {"unique", "shared"}:
        raise ValueError("prefix_mode must be unique or shared")
    invariant_snapshot = require_object(
        manifest["semantic_contract"]["invariants"], "semantic invariants"
    )
    stage_wavefront_plan = _build_stage_wavefront_plan(
        settings, workload, invariant_snapshot
    )
    problems = select_problems(
        load_gsm8k(data),
        request_count,
        seed=int(config["run"]["subset_seed"]),
    )
    raw_base = _load_backend(str(config["models"]["base"]), config)
    raw_proposal = None
    try:
        raw_proposal = _load_backend(str(config["models"]["proposal"]), config)
        if (
            settings.get("proposal_stage_wavefront_mode") == "prefix_sharded"
            and int(settings["proposal_stage_wavefront_max_inflight_waves"]) > 1
            and not bool(
                getattr(raw_proposal, "supports_native_continuous_batching", False)
            )
        ):
            raise RuntimeError(
                "pipelined prefix-sharded admission requires a native asynchronous "
                "continuous-batching proposal backend"
            )
        if raw_base.tokenizer.get_vocab() != raw_proposal.tokenizer.get_vocab():
            raise ValueError("base and proposal tokenizers must match")
        problems = tuple(
            _problem_with_context(
                raw_base,
                problem,
                index,
                context_tokens,
                prefix_mode == "shared",
            )
            for index, problem in enumerate(problems)
        )
        prompts = [_prompt_tokens(raw_base, problem) for problem in problems]
        prompt_lengths = [len(prompt) for prompt in prompts]
        sampling = _sampling(raw_base, config)
        root_seed = int(config["run"]["seed"])
        raw_base.sample_batch(
            [GenerationRequest(prompts[0], 2, sampling, root_seed, "warmup-base")]
        )
        raw_proposal.sample_batch(
            [
                GenerationRequest(
                    prompts[0], 2, sampling, root_seed + 1, "warmup-proposal"
                )
            ]
        )
        base_before = raw_base.snapshot()
        proposal_before = raw_proposal.snapshot()
        request_events: list[dict[str, Any]] = []
        outputs: list[Any | None] = [None] * request_count
        diagnostics: list[dict[str, Any] | None] = [None] * request_count
        runtime_samples: list[dict[str, Any]] = []
        event_lock = threading.Lock()

        with ExitStack() as stack:
            max_batch_size = int(config["runtime"]["max_batch_size"])
            max_batch_tokens = int(config["runtime"]["max_batch_tokens"])
            engine_base = _CallTraceBackend(
                raw_base, "base", "engine_batch_wall_service_time"
            )
            engine_proposal = _CallTraceBackend(
                raw_proposal, "proposal", "engine_batch_wall_service_time"
            )
            base_batching = stack.enter_context(
                ContinuousBatchingBackend(
                    engine_base,
                    max_batch_size=max_batch_size,
                    max_batch_tokens=max_batch_tokens,
                    batch_wait_seconds=float(settings["base_batch_wait_seconds"]),
                )
            )
            proposal_batching = stack.enter_context(
                ContinuousBatchingBackend(
                    engine_proposal,
                    max_batch_size=max_batch_size,
                    max_batch_tokens=max_batch_tokens,
                    batch_wait_seconds=float(
                        settings["proposal_batch_wait_seconds"]
                    ),
                )
            )
            stage_wavefront_backend = None
            proposal_admission = proposal_batching
            wavefront_mode = str(
                settings.get("proposal_stage_wavefront_mode", "off")
            )
            if stage_wavefront_plan is not None:
                selected_wave = stage_wavefront_plan.selected
                admission_type = (
                    PrefixShardedStageWavefrontAdmissionBackend
                    if wavefront_mode == "prefix_sharded"
                    else StageWavefrontAdmissionBackend
                )
                admission_kwargs = {
                    "max_wave_sequences": selected_wave.sequences_per_wave,
                    "target_wave_sequences": selected_wave.sequences_per_wave,
                    "max_wait_seconds": float(
                        settings[
                            "proposal_stage_wavefront_max_wait_seconds"
                        ]
                    ),
                    "max_wave_prefill_tokens": int(
                        settings["proposal_max_num_batched_tokens"]
                    ),
                }
                if wavefront_mode == "prefix_sharded":
                    admission_kwargs["max_shards_per_parent_per_wave"] = int(
                        settings[
                            "proposal_stage_wavefront_max_shards_per_parent"
                        ]
                    )
                    admission_kwargs["max_shard_sequences"] = int(
                        settings[
                            "proposal_stage_wavefront_max_shard_sequences"
                        ]
                    )
                    admission_kwargs["max_shard_prefill_tokens"] = int(
                        settings[
                            "proposal_stage_wavefront_max_shard_prefill_tokens"
                        ]
                    )
                    admission_kwargs["max_inflight_waves"] = int(
                        settings[
                            "proposal_stage_wavefront_max_inflight_waves"
                        ]
                    )
                admission_backend = (
                    engine_proposal
                    if wavefront_mode == "prefix_sharded"
                    else proposal_batching
                )
                stage_wavefront_backend = stack.enter_context(
                    admission_type(
                        admission_backend,
                        **admission_kwargs,
                    )
                )
                proposal_admission = stage_wavefront_backend
            traced_base = _CallTraceBackend(
                base_batching,
                "base",
                "algorithm_call_wall_service_time_including_batch_queueing",
            )
            traced_proposal = _CallTraceBackend(
                proposal_admission,
                "proposal",
                "algorithm_call_wall_service_time_including_batch_queueing",
            )
            base = ScoreCachingBackend(traced_base)
            proposal = ScoreCachingBackend(traced_proposal)
            origin = time.time()
            metrics_stop = threading.Event()

            def collect_runtime_metrics() -> None:
                while not metrics_stop.is_set():
                    sample: dict[str, Any] = {"time_unix": time.time()}
                    for role, backend in (
                        ("base", raw_base),
                        ("proposal", raw_proposal),
                    ):
                        callback = getattr(backend, "runtime_metrics", None)
                        sample[role] = callback() if callable(callback) else {}
                    with event_lock:
                        runtime_samples.append(sample)
                    metrics_stop.wait(1.0)

            metrics_thread = threading.Thread(
                target=collect_runtime_metrics,
                name="autopilot-vllm-runtime-metrics",
                daemon=True,
            )
            metrics_thread.start()

            def run_request(index: int) -> Any:
                started = time.time()
                request_diagnostics: dict[str, Any] = {}
                output = _run_one_with_optional_diagnostics(
                    _run_one,
                    _ALGORITHM_ID,
                    base,
                    proposal,
                    raw_base,
                    prompts[index],
                    problems[index],
                    config,
                    root_seed,
                    request_diagnostics,
                )
                ended = time.time()
                outputs[index] = output
                diagnostics[index] = request_diagnostics
                with event_lock:
                    request_events.append(
                        {
                            "request_index": index,
                            "scheduled_unix": (
                                origin + index / arrival_qps
                                if arrival_qps > 0
                                else origin
                            ),
                            "start_unix": started,
                            "end_unix": ended,
                        }
                    )
                return output

            futures = []
            try:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    for index in range(request_count):
                        if arrival_qps > 0:
                            delay = origin + index / arrival_qps - time.time()
                            if delay > 0:
                                time.sleep(delay)
                        futures.append(executor.submit(run_request, index))
                    for future in as_completed(futures):
                        future.result()
            finally:
                metrics_stop.set()
                metrics_thread.join(timeout=5.0)
            completed = time.time()
            stage_runtime_snapshot = (
                stage_wavefront_backend.snapshot()
                if stage_wavefront_backend is not None
                else None
            )
            proposal_batching_snapshot = asdict(proposal_batching.snapshot())
            if (
                wavefront_mode == "prefix_sharded"
                and stage_runtime_snapshot is not None
            ):
                proposal_batching_snapshot.update(
                    {
                        "sample_batches": stage_runtime_snapshot.wave_count,
                        "sample_requests": stage_runtime_snapshot.admitted_sequences,
                        "maximum_sample_batch": max(
                            (
                                int(wave["sequences"])
                                for wave in stage_runtime_snapshot.waves
                            ),
                            default=0,
                        ),
                    }
                )
            batching = {
                "base": asdict(base_batching.snapshot()),
                "proposal": proposal_batching_snapshot,
            }
            stage_wavefront = {
                "enabled": stage_wavefront_plan is not None,
                "mode": settings.get("proposal_stage_wavefront_mode", "off"),
                **(
                    {
                        "plan": stage_wavefront_plan.to_dict(),
                        "runtime": stage_runtime_snapshot.to_dict(),
                    }
                    if stage_wavefront_plan is not None
                    and stage_wavefront_backend is not None
                    else {}
                ),
            }

        base_after = raw_base.snapshot()
        proposal_after = raw_proposal.snapshot()
        if any(output is None for output in outputs):
            raise RuntimeError("a pressure-test request completed without output")
        token_outputs = [tuple(output) for output in outputs if output is not None]
        numeric_answers = [
            extract_numeric_answer(raw_base.decode(output)) for output in token_outputs
        ]
        latencies = [
            event["end_unix"] - event["scheduled_unix"] for event in request_events
        ]
        queue_waits = [
            event["start_unix"] - event["scheduled_unix"] for event in request_events
        ]
        services = [
            event["end_unix"] - event["start_unix"] for event in request_events
        ]
        runtime_metrics = {
            "base": _summarize_runtime_metrics(runtime_samples, "base"),
            "proposal": _summarize_runtime_metrics(runtime_samples, "proposal"),
            "sample_count": len(runtime_samples),
        }
        payload = {
            "schema_version": "1.0",
            "adapter_id": "chang-pressure-v1",
            "run_id": manifest["run"]["run_id"],
            "run_manifest_sha256": manifest["run_manifest_sha256"],
            "source_snapshot_sha256": actual_source_snapshot,
            "autopilot_snapshot_sha256": actual_autopilot_snapshot,
            "launch_manifest_sha256": _file_sha256(launch_manifest),
            "formal_preflight": compatibility.get("formal_eligible") is True,
            "method": _ALGORITHM_ID,
            "started_at_unix": origin,
            "finished_at_unix": completed,
            "elapsed_seconds": completed - origin,
            "completed_qps": request_count / (completed - origin),
            "latency_seconds": {
                "mean": sum(latencies) / len(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "queue_wait_seconds": {
                "mean": sum(queue_waits) / len(queue_waits),
                "p95": _percentile(queue_waits, 0.95),
            },
            "service_seconds": {
                "mean": sum(services) / len(services),
                "p95": _percentile(services, 0.95),
            },
            "accuracy": sum(
                answer == problem.gold_answer
                for answer, problem in zip(numeric_answers, problems, strict=True)
            )
            / len(problems),
            "runtime": settings,
            "workload_parameters": workload,
            "semantic_invariants": invariant_snapshot,
            "workload": {
                "dataset_sha256": _file_sha256(data),
                "context_tokens_requested": context_tokens,
                "prefix_mode": prefix_mode,
                "prompt_tokens": {
                    "minimum": min(prompt_lengths),
                    "mean": sum(prompt_lengths) / len(prompt_lengths),
                    "maximum": max(prompt_lengths),
                },
            },
            "continuous_batching": batching,
            "proposal_stage_wavefront": stage_wavefront,
            "vllm_runtime_metrics": runtime_metrics,
            "compute": _compute_delta(
                base_before,
                base_after,
                proposal_before,
                proposal_after,
            ),
            "algorithm": {
                "candidate_count": invariant_snapshot["candidate_count"],
                "rollout_count": invariant_snapshot["rollout_count"],
                "block_size": invariant_snapshot["block_size"],
                "total_length": invariant_snapshot["total_length"],
                "apply_importance_correction": invariant_snapshot[
                    "apply_importance_correction"
                ],
                "request_diagnostics": diagnostics,
            },
            "requests_timeline": sorted(
                request_events, key=lambda event: event["request_index"]
            ),
            "model_call_timeline": traced_base.events + traced_proposal.events,
            "engine_batch_timeline": engine_base.events + engine_proposal.events,
            "outputs": [
                {
                    "request_index": index,
                    "problem_index": problem.index,
                    "token_ids": list(output),
                    "numeric_answer": str(answer) if answer is not None else None,
                    "gold_answer": str(problem.gold_answer),
                }
                for index, (problem, output, answer) in enumerate(
                    zip(problems, token_outputs, numeric_answers, strict=True)
                )
            ],
        }
        write_json(payload, native_result)
        return payload
    finally:
        close_backend(raw_proposal)
        close_backend(raw_base)


def _failure_observation(
    manifest_path: Path, started: float, error: BaseException
) -> RunObservation:
    manifest = load_manifest(manifest_path)
    finished = max(time.time(), started + 1e-9)
    message = f"{type(error).__name__}: {error}"
    status = (
        "oom"
        if "out of memory" in message.lower() or "oom" in message.lower()
        else "crash"
    )
    return RunObservation(
        run_id=str(manifest["run"]["run_id"]),
        run_manifest_sha256=str(manifest["run_manifest_sha256"]),
        started_at_unix=started,
        finished_at_unix=finished,
        status=status,
        metrics={"failure": message},
        notes=("worker failed before producing a complete native result",),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chang-pressure-runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--launch-manifest", type=Path, required=True)
    execute_parser.add_argument("--run-manifest", type=Path, required=True)
    execute_parser.add_argument("--source-repo", type=Path, required=True)
    execute_parser.add_argument("--source-config", type=Path, required=True)
    execute_parser.add_argument("--data", type=Path, required=True)
    execute_parser.add_argument("--native-result", type=Path, required=True)
    execute_parser.add_argument("--observation", type=Path, required=True)
    execute_parser.add_argument(
        "--expected-source-config-sha256", required=True
    )
    execute_parser.add_argument(
        "--expected-source-snapshot-sha256", required=True
    )
    execute_parser.add_argument("--allow-nonformal", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    try:
        execute(
            args.run_manifest,
            launch_manifest=args.launch_manifest,
            source_repo=args.source_repo,
            source_config=args.source_config,
            data=args.data,
            native_result=args.native_result,
            expected_source_config_sha256=args.expected_source_config_sha256,
            expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
            allow_nonformal=args.allow_nonformal,
        )
        observation = observation_from_chang_result(
            load_manifest(args.run_manifest), args.native_result
        )
        write_json(observation.to_dict(), args.observation)
        return 0
    except BaseException as error:
        try:
            observation = _failure_observation(args.run_manifest, started, error)
            write_json(observation.to_dict(), args.observation)
        except BaseException:
            pass
        print(f"chang-pressure-runner: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
