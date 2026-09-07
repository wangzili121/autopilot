"""Run chang's pressure entrypoint and flush vLLM graph metrics before shutdown."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from inference_autopilot.calibration.models import canonical_sha256, require_object
from inference_autopilot.graph_experiment import (
    GraphExperimentPlan,
    GraphPolicy,
    apply_graph_policy_to_config,
)
from inference_autopilot.runners.stage_wavefront import (
    StageWavefrontAdmissionBackend,
)
from inference_autopilot.stage_wavefront import (
    StageWavefrontPlan,
    plan_stage_wavefront,
)


def _flush_graph_metrics(backend: Any, role: str) -> None:
    engine = getattr(backend, "_engine", None)
    callback = getattr(engine, "do_log_stats", None)
    if callback is None:
        print(
            f"[inference-autopilot] graph-stats-unavailable role={role}",
            flush=True,
        )
        return
    print(
        f"[inference-autopilot] graph-stats-begin role={role}",
        flush=True,
    )
    result = callback()
    if inspect.isawaitable(result):
        runner = getattr(backend, "_runner", None)
        run = getattr(runner, "run", None)
        if run is None:
            raise RuntimeError("async vLLM graph metrics require the backend loop runner")
        run(result)
    sys.stdout.flush()
    sys.stderr.flush()
    settle_seconds = float(
        os.environ.get("INFERENCE_AUTOPILOT_GRAPH_STATS_SETTLE_SECONDS", "2")
    )
    if settle_seconds < 0:
        raise ValueError("graph stats settle seconds must be non-negative")
    time.sleep(settle_seconds)
    sys.stdout.flush()
    sys.stderr.flush()
    print(
        f"[inference-autopilot] graph-stats-end role={role}",
        flush=True,
    )


def _load_graph_policy() -> tuple[GraphExperimentPlan, Any, GraphPolicy] | None:
    plan_path = os.environ.get("INFERENCE_AUTOPILOT_GRAPH_PLAN")
    run_id = os.environ.get("INFERENCE_AUTOPILOT_GRAPH_RUN_ID")
    if bool(plan_path) != bool(run_id):
        raise ValueError("graph plan and graph run id must be provided together")
    if not plan_path or not run_id:
        return None
    raw = require_object(
        json.loads(Path(plan_path).read_text(encoding="utf-8")),
        "graph experiment plan",
    )
    plan = GraphExperimentPlan.from_dict(raw)
    run, policy = plan.policy_for_run(run_id)
    print(
        "[inference-autopilot] "
        f"graph-plan-sha256={raw['graph_experiment_plan_sha256']} "
        f"run-id={run.run_id} policy-id={policy.policy_id} "
        f"workload-seed={run.workload_seed}",
        flush=True,
    )
    return plan, run, policy


def _output_path() -> Path | None:
    try:
        index = sys.argv.index("--output")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return None


def _prompt_prefix(tokenizer: Any, token_count: int) -> list[int]:
    if token_count < 0:
        raise ValueError("prompt prefix token count must be non-negative")
    if token_count == 0:
        return []
    phrase = (
        "Background context for infrastructure profiling only. "
        "It is unrelated to the question. "
    )
    tokens = list(tokenizer.encode(phrase, add_special_tokens=False))
    if not tokens:
        raise ValueError("tokenizer produced an empty profiling prefix")
    repeats = (token_count + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:token_count]


def _prepend_prompt_prefix(tokens: Any, prefix: list[int]) -> Any:
    if not prefix:
        return tokens
    if isinstance(tokens, tuple):
        return tuple(prefix) + tokens
    if isinstance(tokens, list):
        return prefix + tokens
    raise TypeError(
        "profiling prompt tokens must be a list or tuple, "
        f"got {type(tokens).__name__}"
    )


def _apply_profile_runtime_overrides(
    config: dict[str, Any], environment: dict[str, str]
) -> dict[str, int]:
    overrides = {}
    specifications = (
        (
            "proposal_max_num_batched_tokens",
            "INFERENCE_AUTOPILOT_PROPOSAL_MAX_NUM_BATCHED_TOKENS",
            "proposal",
            "max_num_batched_tokens",
        ),
    )
    for label, environment_name, role, field in specifications:
        raw = environment.get(environment_name)
        if raw is None:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{environment_name} must be a positive integer")
        config.setdefault("vllm", {}).setdefault(role, {})[field] = value
        overrides[label] = value
    return overrides


def _argument_int(arguments: list[str], name: str, default: int) -> int:
    try:
        value = int(arguments[arguments.index(name) + 1])
    except ValueError:
        return default
    except IndexError as error:
        raise ValueError(f"{name} requires an integer value") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _build_stage_wavefront_plan(
    config: dict[str, Any],
    environment: dict[str, str],
    arguments: list[str],
) -> tuple[StageWavefrontPlan | None, float]:
    mode = environment.get(
        "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT", "off"
    ).lower()
    wait_seconds = float(
        environment.get(
            "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS",
            "0.05",
        )
    )
    if wait_seconds < 0:
        raise ValueError("proposal stage wavefront max wait must be non-negative")
    if mode == "off":
        return None, wait_seconds
    if mode != "auto":
        raise ValueError("proposal stage wavefront mode must be off or auto")
    raw_ceiling = environment.get(
        "INFERENCE_AUTOPILOT_PROPOSAL_GRAPH_CAPTURE_CEILING"
    )
    if raw_ceiling is None:
        raise ValueError(
            "auto proposal stage wavefront requires a graph capture ceiling"
        )
    graph_ceiling = int(raw_ceiling)
    minimum_utilization = float(
        environment.get(
            "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT_MIN_UTILIZATION",
            "0.65",
        )
    )
    requests = _argument_int(arguments, "--requests", 96)
    workers = _argument_int(arguments, "--workers", 64)
    section = config["conditional_is"]
    sequences_per_group = int(section["candidate_count"]) * int(
        section["rollout_count"]
    )
    scheduler_cap = int(config["vllm"]["proposal"]["max_num_seqs"])
    return (
        plan_stage_wavefront(
            outer_concurrency=min(requests, workers),
            sequences_per_group=sequences_per_group,
            graph_capture_ceiling=graph_ceiling,
            scheduler_sequence_cap=scheduler_cap,
            minimum_full_wave_utilization=minimum_utilization,
        ),
        wait_seconds,
    )


def main() -> None:
    pressure = importlib.import_module(
        "experiments.arllm.run_small_proposal_pressure"
    )
    original_close = pressure.close_backend
    original_load = pressure._load_backend
    original_prompt_tokens = pressure._prompt_tokens
    original_toml_load = pressure.tomllib.load
    original_batching_context = pressure._batching_context
    roles_by_identity: dict[int, str] = {}
    engine_load_seconds: dict[str, float] = {}
    prompt_token_lengths: list[int] = []
    prompt_prefix_tokens = int(
        os.environ.get("INFERENCE_AUTOPILOT_PROMPT_PREFIX_TOKENS", "0")
    )
    if prompt_prefix_tokens < 0:
        raise ValueError("prompt prefix token count must be non-negative")
    prefixes_by_tokenizer: dict[int, list[int]] = {}
    effective_run_seeds: dict[str, int] = {}
    effective_runtime_overrides: dict[str, int] = {}
    stage_wavefront_plan: StageWavefrontPlan | None = None
    stage_wavefront_backend: StageWavefrontAdmissionBackend | None = None
    stage_wavefront_wait_seconds = 0.05
    graph_context = _load_graph_policy()

    def load_with_policy(stream: Any) -> dict[str, Any]:
        nonlocal stage_wavefront_plan, stage_wavefront_wait_seconds
        config = original_toml_load(stream)
        effective_runtime_overrides.update(
            _apply_profile_runtime_overrides(config, dict(os.environ))
        )
        run_config = config.setdefault("run", {})
        for config_name, environment_name in (
            ("seed", "INFERENCE_AUTOPILOT_RUN_SEED"),
            ("subset_seed", "INFERENCE_AUTOPILOT_SUBSET_SEED"),
        ):
            override = os.environ.get(environment_name)
            if override is not None:
                value = int(override)
                if value < 0:
                    raise ValueError(f"{environment_name} must be non-negative")
                run_config[config_name] = value
            effective_run_seeds[config_name] = int(run_config[config_name])
        if graph_context is not None:
            _plan, run, policy = graph_context
            apply_graph_policy_to_config(
                config,
                policy,
                workload_seed=run.workload_seed,
            )
        stage_wavefront_plan, stage_wavefront_wait_seconds = (
            _build_stage_wavefront_plan(config, dict(os.environ), list(sys.argv))
        )
        return config

    def load_with_role(*args: Any, **kwargs: Any) -> Any:
        role_index = len(roles_by_identity)
        if role_index >= 2:
            raise RuntimeError("graph profiler expected exactly two model backends")
        role = ("base", "proposal")[role_index]
        start = time.perf_counter()
        backend = original_load(*args, **kwargs)
        duration = time.perf_counter() - start
        roles_by_identity[id(backend)] = role
        engine_load_seconds[role] = duration
        print(
            "[inference-autopilot] "
            f"engine-load-complete role={role} duration-seconds={duration:.9f}",
            flush=True,
        )
        return backend

    def close_with_graph_metrics(backend: Any) -> None:
        role = roles_by_identity.get(id(backend), "unknown")
        try:
            _flush_graph_metrics(backend, role)
        finally:
            original_close(backend)

    def batching_with_stage_wavefront(
        stack: Any, backend: Any, config: dict[str, Any]
    ) -> Any:
        nonlocal stage_wavefront_backend
        batching = original_batching_context(stack, backend, config)
        role = roles_by_identity.get(id(backend), "unknown")
        if role != "proposal" or stage_wavefront_plan is None:
            return batching
        if stage_wavefront_backend is not None:
            raise RuntimeError("proposal stage wavefront was initialized twice")
        selected = stage_wavefront_plan.selected
        stage_wavefront_backend = stack.enter_context(
            StageWavefrontAdmissionBackend(
                batching,
                max_wave_sequences=selected.sequences_per_wave,
                target_wave_sequences=selected.sequences_per_wave,
                max_wait_seconds=stage_wavefront_wait_seconds,
                max_wave_prefill_tokens=int(
                    config["vllm"]["proposal"]["max_num_batched_tokens"]
                ),
            )
        )
        print(
            "[inference-autopilot] "
            "proposal-stage-wavefront-enabled "
            f"groups-per-wave={selected.groups_per_wave} "
            f"sequences-per-wave={selected.sequences_per_wave} "
            f"max-wait-seconds={stage_wavefront_wait_seconds}",
            flush=True,
        )
        return stage_wavefront_backend

    def prompt_tokens_with_profile(backend: Any, problem: Any) -> Any:
        tokens = original_prompt_tokens(backend, problem)
        if prompt_prefix_tokens:
            identity = id(backend.tokenizer)
            prefix = prefixes_by_tokenizer.get(identity)
            if prefix is None:
                prefix = _prompt_prefix(backend.tokenizer, prompt_prefix_tokens)
                prefixes_by_tokenizer[identity] = prefix
            tokens = _prepend_prompt_prefix(tokens, prefix)
        prompt_token_lengths.append(len(tokens))
        return tokens

    pressure._load_backend = load_with_role
    pressure.close_backend = close_with_graph_metrics
    pressure._prompt_tokens = prompt_tokens_with_profile
    pressure.tomllib.load = load_with_policy
    pressure._batching_context = batching_with_stage_wavefront
    pressure.main()
    output = _output_path()
    if output is None:
        raise ValueError("graph profiler requires the pressure runner --output")
    payload = require_object(
        json.loads(output.read_text(encoding="utf-8")),
        "chang pressure result",
    )
    payload["inference_autopilot_profile"] = {
        "engine_load_seconds": dict(sorted(engine_load_seconds.items())),
        "prompt_tokens": {
            "count": len(prompt_token_lengths),
            "minimum": min(prompt_token_lengths),
            "maximum": max(prompt_token_lengths),
            "mean": sum(prompt_token_lengths) / len(prompt_token_lengths),
            "synthetic_prefix_tokens": prompt_prefix_tokens,
        },
        "run_seeds": dict(sorted(effective_run_seeds.items())),
        "runtime_overrides": dict(sorted(effective_runtime_overrides.items())),
        "proposal_stage_wavefront": {
            "enabled": stage_wavefront_plan is not None,
            **(
                {
                    "plan": stage_wavefront_plan.to_dict(),
                    "runtime": stage_wavefront_backend.snapshot().to_dict(),
                }
                if stage_wavefront_plan is not None
                and stage_wavefront_backend is not None
                else {}
            ),
        },
    }
    if graph_context is not None:
        plan, run, policy = graph_context
        payload["graph_experiment"] = {
            "campaign_id": plan.campaign_id,
            "source_profile_sha256": plan.source_profile_sha256,
            "run": run.to_dict(),
            "policy": policy.to_dict(),
            "policy_sha256": canonical_sha256(policy.to_dict()),
        }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
