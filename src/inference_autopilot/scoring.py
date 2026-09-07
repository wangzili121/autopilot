"""Memory-aware planning contracts for exact trajectory scoring reductions."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from math import exp, isfinite, log, prod
from typing import Any, Iterable

from inference_autopilot.calibration.models import canonical_sha256, require_id
from inference_autopilot.search_space import RuntimeCapabilityProfile


_STATISTICS = {"selected_logprob", "topk_mean_logprob", "entropy"}


def streaming_score_statistics_reference(
    logits: Iterable[float],
    *,
    selected_token_id: int | None = None,
    top_k: int = 0,
) -> dict[str, float]:
    """One-pass exact reductions used as the contract for a future device kernel."""

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if selected_token_id is not None and (
        isinstance(selected_token_id, bool)
        or not isinstance(selected_token_id, int)
        or selected_token_id < 0
    ):
        raise ValueError("selected_token_id must be a non-negative integer")
    values = iter(logits)
    running_max = float("-inf")
    scaled_sum = 0.0
    scaled_logit_sum = 0.0
    selected_logit: float | None = None
    top_logits: list[float] = []
    count = 0
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if not isfinite(value):
            raise ValueError("score logits must be finite")
        if index == selected_token_id:
            selected_logit = value
        if top_k:
            if len(top_logits) < top_k:
                heappush(top_logits, value)
            elif value > top_logits[0]:
                heapreplace(top_logits, value)
        if value > running_max:
            scale = 0.0 if count == 0 else exp(running_max - value)
            scaled_sum = scaled_sum * scale + 1.0
            scaled_logit_sum = scaled_logit_sum * scale + value
            running_max = value
        else:
            weight = exp(value - running_max)
            scaled_sum += weight
            scaled_logit_sum += weight * value
        count += 1
    if count == 0:
        raise ValueError("score logits cannot be empty")
    if selected_token_id is not None and selected_logit is None:
        raise ValueError("selected_token_id is outside the vocabulary")
    if top_k > count:
        raise ValueError("top_k must be between zero and the vocabulary size")

    log_partition = running_max + log(scaled_sum)
    result = {
        "log_partition": log_partition,
        "entropy": log_partition - scaled_logit_sum / scaled_sum,
    }
    if selected_logit is not None:
        result["selected_logprob"] = selected_logit - log_partition
    if top_k:
        result["topk_mean_logprob"] = (
            sum(top_logits) / top_k - log_partition
        )
    return result


@dataclass(frozen=True, slots=True)
class ScoreRequirement:
    statistics: tuple[str, ...]
    top_k: int = 0

    def __post_init__(self) -> None:
        if not self.statistics or tuple(sorted(set(self.statistics))) != self.statistics:
            raise ValueError("score statistics must be sorted and unique")
        unknown = sorted(set(self.statistics) - _STATISTICS)
        if unknown:
            raise ValueError(f"unsupported score statistics: {unknown}")
        needs_top_k = "topk_mean_logprob" in self.statistics
        if needs_top_k != (self.top_k > 0):
            raise ValueError("top_k must be positive exactly when top-k scoring is required")

    def to_dict(self) -> dict[str, Any]:
        return {"statistics": list(self.statistics), "top_k": self.top_k}


def requirement_for_reward(
    reward: str,
    *,
    importance_correction: bool,
    consilience_top_k: int = 5,
) -> ScoreRequirement:
    """Translate algorithm semantics into exact statistics the scorer must return."""

    statistics: set[str] = set()
    if importance_correction:
        statistics.add("selected_logprob")
    if reward == "consilience":
        statistics.add("topk_mean_logprob")
    elif reward in {"self_certainty", "entropy"}:
        statistics.add("entropy")
    elif reward not in {"self_consistency", "frozen_consensus", "exact"}:
        raise ValueError(f"unsupported reward for scoring planning: {reward}")
    if not statistics:
        raise ValueError("this algorithm/reward combination requires no model scorer")
    return ScoreRequirement(
        tuple(sorted(statistics)),
        consilience_top_k if "topk_mean_logprob" in statistics else 0,
    )


@dataclass(frozen=True, slots=True)
class ScoreWorkload:
    positions: int
    vocab_size: int
    logits_element_bytes: int = 2
    accumulator_bytes: int = 4

    def __post_init__(self) -> None:
        for name in (
            "positions",
            "vocab_size",
            "logits_element_bytes",
            "accumulator_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"score workload {name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "positions": self.positions,
            "vocab_size": self.vocab_size,
            "logits_element_bytes": self.logits_element_bytes,
            "accumulator_bytes": self.accumulator_bytes,
        }


@dataclass(frozen=True, slots=True)
class ScoreImplementation:
    implementation_id: str
    capability_id: str
    statistics: tuple[str, ...]
    materializes_full_logprobs: bool
    fuses_lm_head: bool
    token_chunk_size: int
    vocab_tile_size: int

    def __post_init__(self) -> None:
        require_id(self.implementation_id, "score implementation_id")
        require_id(self.capability_id, "score capability_id")
        if tuple(sorted(set(self.statistics))) != self.statistics:
            raise ValueError("implementation statistics must be sorted and unique")
        if set(self.statistics) - _STATISTICS:
            raise ValueError("implementation contains unsupported statistics")
        if self.token_chunk_size <= 0 or self.vocab_tile_size <= 0:
            raise ValueError("score tile sizes must be positive")

    def supports(self, requirement: ScoreRequirement) -> bool:
        return set(requirement.statistics).issubset(self.statistics)


@dataclass(frozen=True, slots=True)
class ScoreMemoryEstimate:
    input_logits_bytes: int
    reduction_workspace_bytes: int
    output_bytes: int
    estimated_peak_bytes: int

    @property
    def incremental_reduction_bytes(self) -> int:
        return self.reduction_workspace_bytes + self.output_bytes

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_logits_bytes": self.input_logits_bytes,
            "reduction_workspace_bytes": self.reduction_workspace_bytes,
            "output_bytes": self.output_bytes,
            "incremental_reduction_bytes": self.incremental_reduction_bytes,
            "incremental_reduction_gib": self.incremental_reduction_bytes
            / (1024**3),
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "estimated_peak_gib": self.estimated_peak_bytes / (1024**3),
        }


def estimate_score_memory(
    requirement: ScoreRequirement,
    workload: ScoreWorkload,
    implementation: ScoreImplementation,
) -> ScoreMemoryEstimate:
    if not implementation.supports(requirement):
        raise ValueError("score implementation does not satisfy required statistics")
    positions = workload.positions
    vocab_size = workload.vocab_size
    token_chunk = min(positions, implementation.token_chunk_size)
    vocab_tile = min(vocab_size, implementation.vocab_tile_size)

    if implementation.fuses_lm_head:
        input_logits = prod((token_chunk, vocab_tile, workload.logits_element_bytes))
    else:
        input_logits = prod((positions, vocab_size, workload.logits_element_bytes))
    if implementation.materializes_full_logprobs:
        workspace = prod((positions, vocab_size, workload.accumulator_bytes))
    else:
        tile = prod((token_chunk, vocab_tile, workload.accumulator_bytes))
        running_scalars = 2
        if "selected_logprob" in requirement.statistics:
            running_scalars += 1
        if "entropy" in requirement.statistics:
            running_scalars += 1
        if "topk_mean_logprob" in requirement.statistics:
            running_scalars += requirement.top_k
        workspace = tile + token_chunk * running_scalars * workload.accumulator_bytes

    output_scalars = positions * len(requirement.statistics)
    output = output_scalars * workload.accumulator_bytes
    return ScoreMemoryEstimate(
        input_logits_bytes=input_logits,
        reduction_workspace_bytes=workspace,
        output_bytes=output,
        estimated_peak_bytes=input_logits + workspace + output,
    )


def _implementations(
    workload: ScoreWorkload,
    token_chunk_sizes: Iterable[int],
    vocab_tile_sizes: Iterable[int],
) -> tuple[ScoreImplementation, ...]:
    all_statistics = tuple(sorted(_STATISTICS))
    implementations = [
        ScoreImplementation(
            implementation_id="native-full-logsoftmax",
            capability_id="score.native_full_logsoftmax",
            statistics=all_statistics,
            materializes_full_logprobs=True,
            fuses_lm_head=False,
            token_chunk_size=workload.positions,
            vocab_tile_size=workload.vocab_size,
        )
    ]
    for token_chunk, vocab_tile in prod_grid(token_chunk_sizes, vocab_tile_sizes):
        implementations.append(
            ScoreImplementation(
                implementation_id=f"streaming-reduction-t{token_chunk}-v{vocab_tile}",
                capability_id="score.streaming_tiled_reduction",
                statistics=all_statistics,
                materializes_full_logprobs=False,
                fuses_lm_head=False,
                token_chunk_size=token_chunk,
                vocab_tile_size=vocab_tile,
            )
        )
    return tuple(implementations)


def prod_grid(first: Iterable[int], second: Iterable[int]) -> tuple[tuple[int, int], ...]:
    first_values = tuple(sorted(set(first)))
    second_values = tuple(sorted(set(second)))
    if not first_values or not second_values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (*first_values, *second_values)
    ):
        raise ValueError("score tile domains must contain positive integers")
    return tuple((left, right) for left in first_values for right in second_values)


def build_score_reduction_plan(
    requirement: ScoreRequirement,
    workload: ScoreWorkload,
    *,
    memory_budget_bytes: int,
    capability_profile: RuntimeCapabilityProfile | None = None,
    token_chunk_sizes: Iterable[int] = (64, 128, 256, 512),
    vocab_tile_sizes: Iterable[int] = (1024, 2048, 4096, 8192),
) -> dict[str, Any]:
    if memory_budget_bytes <= 0:
        raise ValueError("score memory budget must be positive")
    candidates = []
    for implementation in _implementations(
        workload, token_chunk_sizes, vocab_tile_sizes
    ):
        estimate = estimate_score_memory(requirement, workload, implementation)
        capability_status = (
            capability_profile.status_for(implementation.capability_id)
            if capability_profile is not None
            else "unknown"
        )
        candidates.append(
            {
                "implementation_id": implementation.implementation_id,
                "capability_id": implementation.capability_id,
                "capability_status": capability_status,
                "materializes_full_logprobs": implementation.materializes_full_logprobs,
                "fuses_lm_head": implementation.fuses_lm_head,
                "token_chunk_size": implementation.token_chunk_size,
                "vocab_tile_size": implementation.vocab_tile_size,
                "memory": estimate.to_dict(),
                "within_memory_budget": (
                    estimate.incremental_reduction_bytes <= memory_budget_bytes
                ),
                "runnable": capability_status == "supported"
                and estimate.incremental_reduction_bytes <= memory_budget_bytes,
            }
        )
    candidates.sort(
        key=lambda item: (
            not item["runnable"],
            item["memory"]["estimated_peak_bytes"],
            item["implementation_id"],
        )
    )
    payload = {
        "schema_version": "1.0",
        "requirement": requirement.to_dict(),
        "workload": workload.to_dict(),
        "memory_budget_bytes": memory_budget_bytes,
        "capability_profile_sha256": (
            capability_profile.sha256 if capability_profile is not None else None
        ),
        "candidates": candidates,
    }
    return {**payload, "score_reduction_plan_sha256": canonical_sha256(payload)}
