#!/usr/bin/env python3
"""Benchmark exact full and vocabulary-tiled scoring reductions on one NPU."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_int_csv(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item) for item in raw.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=_positive_int, required=True)
    parser.add_argument("--vocab-size", type=_positive_int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-chunk-sizes", type=_positive_int_csv, default=(64, 128, 256))
    parser.add_argument("--vocab-tile-sizes", type=_positive_int_csv, default=(2048, 4096, 8192))
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _native_reduction(torch: Any, logits: Any, selected_ids: Any, top_k: int) -> dict[str, Any]:
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    outputs = {
        "selected_logprob": logprobs.gather(1, selected_ids[:, None]).squeeze(1)
    }
    if top_k:
        outputs["topk_mean_logprob"] = torch.topk(
            logprobs, k=top_k, dim=-1
        ).values.mean(dim=-1)
    return outputs


def _tiled_reduction(
    torch: Any,
    logits: Any,
    selected_ids: Any,
    top_k: int,
    token_chunk_size: int,
    vocab_tile_size: int,
) -> dict[str, Any]:
    selected_parts = []
    topk_parts = []
    positions, vocab_size = logits.shape
    for token_start in range(0, positions, token_chunk_size):
        token_end = min(token_start + token_chunk_size, positions)
        rows = logits[token_start:token_end]
        row_ids = selected_ids[token_start:token_end]
        selected_logits = rows.gather(1, row_ids[:, None]).squeeze(1).float()
        log_partition = torch.full(
            (token_end - token_start,),
            float("-inf"),
            dtype=torch.float32,
            device=logits.device,
        )
        top_values = None
        for vocab_start in range(0, vocab_size, vocab_tile_size):
            vocab_end = min(vocab_start + vocab_tile_size, vocab_size)
            tile = rows[:, vocab_start:vocab_end].float()
            tile_partition = torch.logsumexp(tile, dim=-1)
            log_partition = torch.logaddexp(log_partition, tile_partition)
            if top_k:
                if top_values is None:
                    top_values = torch.topk(tile, k=top_k, dim=-1).values
                else:
                    top_values = torch.topk(
                        torch.cat((top_values, tile), dim=-1),
                        k=top_k,
                        dim=-1,
                    ).values
        selected_parts.append(selected_logits - log_partition)
        if top_k:
            assert top_values is not None
            topk_parts.append(top_values.mean(dim=-1) - log_partition)
    outputs = {"selected_logprob": torch.cat(selected_parts)}
    if top_k:
        outputs["topk_mean_logprob"] = torch.cat(topk_parts)
    return outputs


def _measure(
    torch: Any,
    operation: Callable[[], dict[str, Any]],
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    baseline_allocated = torch.npu.memory_allocated()
    baseline_reserved = torch.npu.memory_reserved()
    torch.npu.reset_peak_memory_stats()
    elapsed = []
    output = None
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter()
            current = operation()
            torch.npu.synchronize()
            elapsed.append(time.perf_counter() - start)
            output = current
    assert output is not None
    peak_allocated = torch.npu.max_memory_allocated()
    peak_reserved = torch.npu.max_memory_reserved()
    measurement = {
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": median(elapsed),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_incremental_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_incremental_reserved_bytes": max(0, peak_reserved - baseline_reserved),
    }
    return output, measurement


def _errors(torch: Any, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    errors = {}
    for name in sorted(expected):
        delta = (expected[name] - actual[name]).abs()
        errors[name] = {
            "max_absolute_error": delta.max().item(),
            "mean_absolute_error": delta.mean().item(),
        }
    return errors


def main() -> int:
    args = _parser().parse_args()
    if args.top_k < 0 or args.top_k > args.vocab_size:
        raise SystemExit("--top-k must be between zero and vocab size")

    import torch
    import torch_npu

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    device = torch.device("npu:0")
    logits = torch.empty(
        (args.positions, args.vocab_size), dtype=torch.float16, device=device
    ).normal_()
    selected_ids = torch.randint(
        0, args.vocab_size, (args.positions,), dtype=torch.int64, device=device
    )
    torch.npu.synchronize()

    warm_positions = min(args.positions, 16)
    warm_vocab = min(args.vocab_size, 4096)
    warm_logits = logits[:warm_positions, :warm_vocab]
    warm_ids = selected_ids[:warm_positions] % warm_vocab
    with torch.inference_mode():
        _native_reduction(torch, warm_logits, warm_ids, args.top_k)
        for token_chunk in args.token_chunk_sizes:
            for vocab_tile in args.vocab_tile_sizes:
                _tiled_reduction(
                    torch,
                    warm_logits,
                    warm_ids,
                    args.top_k,
                    token_chunk,
                    vocab_tile,
                )
    torch.npu.synchronize()

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "device": str(device),
        "device_name": torch.npu.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "workload": {
            "positions": args.positions,
            "vocab_size": args.vocab_size,
            "top_k": args.top_k,
            "dtype": str(logits.dtype),
            "input_logits_bytes": logits.numel() * logits.element_size(),
            "seed": args.seed,
        },
        "native": None,
        "tiled_candidates": [],
    }

    native_output = None
    try:
        native_output, native_measurement = _measure(
            torch,
            lambda: _native_reduction(torch, logits, selected_ids, args.top_k),
            args.repeats,
        )
        result["native"] = {"status": "success", **native_measurement}
    except RuntimeError as exc:
        result["native"] = {"status": "failed", "error": str(exc)}
        torch.npu.empty_cache()

    tiled_failed = False
    for token_chunk in args.token_chunk_sizes:
        for vocab_tile in args.vocab_tile_sizes:
            candidate: dict[str, Any] = {
                "token_chunk_size": token_chunk,
                "vocab_tile_size": vocab_tile,
            }
            try:
                tiled_output, measurement = _measure(
                    torch,
                    lambda tc=token_chunk, vt=vocab_tile: _tiled_reduction(
                        torch, logits, selected_ids, args.top_k, tc, vt
                    ),
                    args.repeats,
                )
                candidate.update({"status": "success", **measurement})
                if native_output is not None:
                    candidate["error_against_native"] = _errors(
                        torch, native_output, tiled_output
                    )
                del tiled_output
            except RuntimeError as exc:
                candidate.update({"status": "failed", "error": str(exc)})
                tiled_failed = True
                torch.npu.empty_cache()
            result["tiled_candidates"].append(candidate)

    successful = [
        candidate
        for candidate in result["tiled_candidates"]
        if candidate["status"] == "success"
    ]
    if successful:
        fastest = min(successful, key=lambda item: item["median_elapsed_seconds"])
        lowest_memory = min(
            successful, key=lambda item: item["peak_incremental_allocated_bytes"]
        )
        result["best_observed"] = {
            "fastest": {
                "token_chunk_size": fastest["token_chunk_size"],
                "vocab_tile_size": fastest["vocab_tile_size"],
            },
            "lowest_peak_incremental_allocated": {
                "token_chunk_size": lowest_memory["token_chunk_size"],
                "vocab_tile_size": lowest_memory["vocab_tile_size"],
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["best_observed"], sort_keys=True))
    return 1 if tiled_failed or not successful else 0


if __name__ == "__main__":
    raise SystemExit(main())
