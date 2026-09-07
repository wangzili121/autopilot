"""Exact memory-bounded reduction for vLLM selected prompt log-probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any


_ENABLED_ENV = "INFERENCE_AUTOPILOT_TILED_PROMPT_LOGPROBS"
_TOKEN_CHUNK_ENV = "INFERENCE_AUTOPILOT_SCORE_TOKEN_CHUNK_SIZE"
_VOCAB_TILE_ENV = "INFERENCE_AUTOPILOT_SCORE_VOCAB_TILE_SIZE"


def _positive_environment_integer(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TiledPromptLogprobsConfig:
    enabled: bool = False
    token_chunk_size: int = 512
    vocab_tile_size: int = 32768

    def __post_init__(self) -> None:
        if self.token_chunk_size <= 0 or self.vocab_tile_size <= 0:
            raise ValueError("prompt-logprob tile sizes must be positive")

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "TiledPromptLogprobsConfig":
        values = os.environ if environment is None else environment
        raw_enabled = values.get(_ENABLED_ENV, "0")
        if raw_enabled not in {"0", "1"}:
            raise ValueError(f"{_ENABLED_ENV} must be 0 or 1")
        return cls(
            enabled=raw_enabled == "1",
            token_chunk_size=_positive_environment_integer(
                values, _TOKEN_CHUNK_ENV, 512
            ),
            vocab_tile_size=_positive_environment_integer(
                values, _VOCAB_TILE_ENV, 32768
            ),
        )


def tiled_selected_prompt_logprobs(
    logits: Any,
    token_ids: Any,
    config: TiledPromptLogprobsConfig,
) -> tuple[Any, Any, Any]:
    """Return vLLM-compatible selected IDs, log-probabilities, and exact ranks."""

    import torch

    if logits.ndim != 2:
        raise ValueError("prompt logits must be a two-dimensional tensor")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("prompt token IDs must match the logits row count")
    if token_ids.dtype != torch.int64:
        raise ValueError("prompt token IDs must use int64")
    positions, vocab_size = logits.shape
    selected_logprobs = torch.empty(
        (positions, 1), dtype=torch.float32, device=logits.device
    )
    selected_ranks = torch.empty(
        (positions,), dtype=torch.int64, device=logits.device
    )
    for token_start in range(0, positions, config.token_chunk_size):
        token_end = min(token_start + config.token_chunk_size, positions)
        rows = logits[token_start:token_end]
        row_token_ids = token_ids[token_start:token_end]
        selected_logits = rows.gather(1, row_token_ids[:, None]).squeeze(1)
        log_partition = torch.full(
            (token_end - token_start,),
            float("-inf"),
            dtype=torch.float32,
            device=logits.device,
        )
        ranks = torch.zeros(
            (token_end - token_start,), dtype=torch.int64, device=logits.device
        )
        for vocab_start in range(0, vocab_size, config.vocab_tile_size):
            vocab_end = min(vocab_start + config.vocab_tile_size, vocab_size)
            source_tile = rows[:, vocab_start:vocab_end]
            tile_partition = torch.logsumexp(source_tile.float(), dim=-1)
            log_partition = torch.logaddexp(log_partition, tile_partition)
            ranks.add_((source_tile >= selected_logits[:, None]).sum(dim=-1))
        selected_logprobs[token_start:token_end, 0] = (
            selected_logits.float() - log_partition
        )
        selected_ranks[token_start:token_end] = ranks

    return token_ids[:, None].to(torch.int32), selected_logprobs, selected_ranks


def maybe_tiled_selected_prompt_logprobs(
    logits: Any,
    token_ids: Any,
    num_prompt_logprobs: int,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Any, Any, Any] | None:
    """Use the tiled path only for selected-only prompt-logprob requests."""

    config = TiledPromptLogprobsConfig.from_environment(environment)
    if not config.enabled or num_prompt_logprobs != 0:
        return None
    return tiled_selected_prompt_logprobs(logits, token_ids, config)
