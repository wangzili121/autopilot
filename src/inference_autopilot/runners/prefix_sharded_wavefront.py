"""Prefix-preserving sharded admission for synchronous algorithm callers."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from typing import Any

from inference_autopilot.runners.stage_wavefront import conditional_is_stage_key


_MISSING = object()


def _prefill_tokens(request: Any) -> int:
    prefix = getattr(request, "prefix", ())
    return len(prefix)


def _same_prefix_run(left: Any, right: Any) -> bool:
    return (
        getattr(left, "prefix", None) == getattr(right, "prefix", None)
        and getattr(left, "sampling", None) == getattr(right, "sampling", None)
        and getattr(left, "max_new_tokens", None)
        == getattr(right, "max_new_tokens", None)
    )


@dataclass(frozen=True, slots=True)
class PrefixShard:
    """One order-preserving request shard and its parent-index mapping."""

    requests: tuple[Any, ...]
    parent_indexes: tuple[int, ...]
    prefill_token_count: int
    prefix_run_ids: tuple[int, ...]

    @property
    def sequence_count(self) -> int:
        return len(self.requests)

    def descriptor(self) -> dict[str, Any]:
        return {
            "parent_indexes": list(self.parent_indexes),
            "sequences": self.sequence_count,
            "prefill_tokens": self.prefill_token_count,
            "prefix_run_ids": list(self.prefix_run_ids),
        }


@dataclass(frozen=True, slots=True)
class PrefixShardPlan:
    """A semantic-preserving split of one synchronous parent call."""

    shards: tuple[PrefixShard, ...]
    repeated_prefix_run_count: int
    preserved_prefix_run_count: int
    forced_split_run_count: int
    oversized_atomic_run_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_count": len(self.shards),
            "repeated_prefix_run_count": self.repeated_prefix_run_count,
            "preserved_prefix_run_count": self.preserved_prefix_run_count,
            "forced_split_run_count": self.forced_split_run_count,
            "oversized_atomic_run_count": self.oversized_atomic_run_count,
            "shards": [shard.descriptor() for shard in self.shards],
        }


def plan_prefix_shards(
    requests: Sequence[Any],
    *,
    max_shard_sequences: int,
    max_shard_prefill_tokens: int,
) -> PrefixShardPlan:
    """Split only between repeated-prefix runs unless a run cannot fit."""

    if max_shard_sequences <= 0 or max_shard_prefill_tokens <= 0:
        raise ValueError("prefix shard limits must be positive")
    materialized = tuple(requests)
    if not materialized:
        return PrefixShardPlan((), 0, 0, 0, 0)

    runs: list[tuple[int, list[tuple[int, Any]]]] = []
    for parent_index, request in enumerate(materialized):
        if not runs or not _same_prefix_run(runs[-1][1][-1][1], request):
            runs.append((len(runs), []))
        runs[-1][1].append((parent_index, request))

    units: list[tuple[int, tuple[tuple[int, Any], ...]]] = []
    preserved = 0
    forced = 0
    oversized_atomic = 0

    def fits(entries: Sequence[tuple[int, Any]]) -> bool:
        return len(entries) <= max_shard_sequences and sum(
            _prefill_tokens(request) for _index, request in entries
        ) <= max_shard_prefill_tokens

    for run_id, run in runs:
        if fits(run):
            units.append((run_id, tuple(run)))
            preserved += 1
            continue
        forced += 1
        current: list[tuple[int, Any]] = []
        for entry in run:
            if current and not fits((*current, entry)):
                units.append((run_id, tuple(current)))
                current = []
            current.append(entry)
            if not fits(current):
                oversized_atomic += 1
                units.append((run_id, tuple(current)))
                current = []
        if current:
            units.append((run_id, tuple(current)))

    shards: list[PrefixShard] = []
    current_units: list[tuple[int, tuple[tuple[int, Any], ...]]] = []

    def flatten_units(
        source: Sequence[tuple[int, tuple[tuple[int, Any], ...]]],
    ) -> list[tuple[int, Any]]:
        return [entry for _run_id, unit in source for entry in unit]

    def flush() -> None:
        if not current_units:
            return
        entries = flatten_units(current_units)
        shards.append(
            PrefixShard(
                requests=tuple(request for _index, request in entries),
                parent_indexes=tuple(index for index, _request in entries),
                prefill_token_count=sum(
                    _prefill_tokens(request) for _index, request in entries
                ),
                prefix_run_ids=tuple(run_id for run_id, _unit in current_units),
            )
        )
        current_units.clear()

    for unit in units:
        candidate = flatten_units((*current_units, unit))
        if current_units and not fits(candidate):
            flush()
        current_units.append(unit)
        if not fits(flatten_units(current_units)):
            flush()
    flush()

    flattened_indexes = tuple(
        index for shard in shards for index in shard.parent_indexes
    )
    if flattened_indexes != tuple(range(len(materialized))):
        raise RuntimeError("prefix sharding changed parent request order")
    return PrefixShardPlan(
        shards=tuple(shards),
        repeated_prefix_run_count=len(runs),
        preserved_prefix_run_count=preserved,
        forced_split_run_count=forced,
        oversized_atomic_run_count=oversized_atomic,
    )


@dataclass(slots=True)
class _Parent:
    ordinal: int
    requests: tuple[Any, ...]
    callback: Any | None
    future: Future
    enqueued_at: float
    outputs: list[Any]
    remaining_shards: int
    first_admitted_at: float | None = None
    failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class _ShardTicket:
    parent: _Parent
    shard_index: int
    requests: tuple[Any, ...]
    parent_indexes: tuple[int, ...]
    stage_key: tuple[Any, ...]
    enqueued_at: float
    prefill_token_count: int

    @property
    def sequence_count(self) -> int:
        return len(self.requests)


@dataclass(frozen=True, slots=True)
class PrefixShardedWavefrontSnapshot:
    max_wave_sequences: int
    target_wave_sequences: int
    max_wait_seconds: float
    max_wave_prefill_tokens: int
    max_shard_sequences: int
    max_shard_prefill_tokens: int
    max_shards_per_parent_per_wave: int
    max_inflight_waves: int
    max_inflight_sequences: int
    admitted_groups: int
    admitted_parent_groups: int
    admitted_shards: int
    admitted_sequences: int
    admitted_prefill_tokens: int
    completed_parent_groups: int
    repeated_prefix_run_count: int
    preserved_prefix_run_count: int
    forced_split_run_count: int
    oversized_atomic_run_count: int
    cancelled_shard_count: int
    wave_count: int
    partial_wave_count: int
    oversized_wave_count: int
    fairness_limited_wave_count: int
    fairness_violation_count: int
    cross_parent_wave_count: int
    parent_groups_across_waves: int
    maximum_shards_for_one_parent: int
    maximum_queued_groups: int
    maximum_queued_shards: int
    maximum_queued_sequences: int
    inflight_waves: int
    inflight_sequences: int
    maximum_inflight_waves: int
    maximum_inflight_sequences: int
    sequence_credit_stall_count: int
    sequence_credit_stall_seconds: float
    streaming_sequence_credit_release_count: int
    batch_sequence_credit_release_count: int
    dispatch_failure_count: int
    mean_admission_wait_seconds: float
    maximum_admission_wait_seconds: float
    mean_wave_sequence_utilization: float
    mean_wave_prefill_utilization: float
    mean_parent_groups_per_wave: float
    mean_parent_completion_span_seconds: float
    maximum_parent_completion_span_seconds: float
    release_reason_counts: tuple[tuple[str, int], ...]
    waves: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "prefix_sharded",
            "max_wave_sequences": self.max_wave_sequences,
            "target_wave_sequences": self.target_wave_sequences,
            "max_wait_seconds": self.max_wait_seconds,
            "max_wave_prefill_tokens": self.max_wave_prefill_tokens,
            "max_shard_sequences": self.max_shard_sequences,
            "max_shard_prefill_tokens": self.max_shard_prefill_tokens,
            "max_shards_per_parent_per_wave": (
                self.max_shards_per_parent_per_wave
            ),
            "max_inflight_waves": self.max_inflight_waves,
            "max_inflight_sequences": self.max_inflight_sequences,
            "admitted_groups": self.admitted_groups,
            "admitted_parent_groups": self.admitted_parent_groups,
            "admitted_shards": self.admitted_shards,
            "admitted_sequences": self.admitted_sequences,
            "admitted_prefill_tokens": self.admitted_prefill_tokens,
            "completed_parent_groups": self.completed_parent_groups,
            "repeated_prefix_run_count": self.repeated_prefix_run_count,
            "preserved_prefix_run_count": self.preserved_prefix_run_count,
            "forced_split_run_count": self.forced_split_run_count,
            "oversized_atomic_run_count": self.oversized_atomic_run_count,
            "cancelled_shard_count": self.cancelled_shard_count,
            "wave_count": self.wave_count,
            "partial_wave_count": self.partial_wave_count,
            "oversized_wave_count": self.oversized_wave_count,
            "fairness_limited_wave_count": self.fairness_limited_wave_count,
            "fairness_violation_count": self.fairness_violation_count,
            "cross_parent_wave_count": self.cross_parent_wave_count,
            "parent_groups_across_waves": self.parent_groups_across_waves,
            "maximum_shards_for_one_parent": (
                self.maximum_shards_for_one_parent
            ),
            "maximum_queued_groups": self.maximum_queued_groups,
            "maximum_queued_shards": self.maximum_queued_shards,
            "maximum_queued_sequences": self.maximum_queued_sequences,
            "inflight_waves": self.inflight_waves,
            "inflight_sequences": self.inflight_sequences,
            "maximum_inflight_waves": self.maximum_inflight_waves,
            "maximum_inflight_sequences": self.maximum_inflight_sequences,
            "sequence_credit_stall_count": self.sequence_credit_stall_count,
            "sequence_credit_stall_seconds": self.sequence_credit_stall_seconds,
            "streaming_sequence_credit_release_count": (
                self.streaming_sequence_credit_release_count
            ),
            "batch_sequence_credit_release_count": (
                self.batch_sequence_credit_release_count
            ),
            "dispatch_failure_count": self.dispatch_failure_count,
            "mean_admission_wait_seconds": self.mean_admission_wait_seconds,
            "maximum_admission_wait_seconds": (
                self.maximum_admission_wait_seconds
            ),
            "mean_wave_sequence_utilization": (
                self.mean_wave_sequence_utilization
            ),
            "mean_wave_prefill_utilization": (
                self.mean_wave_prefill_utilization
            ),
            "mean_parent_groups_per_wave": self.mean_parent_groups_per_wave,
            "mean_parent_completion_span_seconds": (
                self.mean_parent_completion_span_seconds
            ),
            "maximum_parent_completion_span_seconds": (
                self.maximum_parent_completion_span_seconds
            ),
            "release_reason_counts": dict(self.release_reason_counts),
            "waves": list(self.waves),
        }


class PrefixShardedStageWavefrontAdmissionBackend:
    """Pack prefix-preserving shards while retaining each parent call barrier."""

    def __init__(
        self,
        backend: Any,
        *,
        max_wave_sequences: int,
        target_wave_sequences: int | None = None,
        max_wait_seconds: float = 0.05,
        max_wave_prefill_tokens: int,
        max_shard_sequences: int | None = None,
        max_shard_prefill_tokens: int | None = None,
        max_shards_per_parent_per_wave: int = 1,
        max_inflight_waves: int = 1,
    ) -> None:
        if max_wave_sequences <= 0:
            raise ValueError("max_wave_sequences must be positive")
        target = (
            max_wave_sequences
            if target_wave_sequences is None
            else target_wave_sequences
        )
        if target <= 0 or target > max_wave_sequences:
            raise ValueError(
                "target_wave_sequences must lie in [1, max_wave_sequences]"
            )
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be non-negative")
        if max_wave_prefill_tokens <= 0:
            raise ValueError("max_wave_prefill_tokens must be positive")
        shard_sequences = (
            max_wave_sequences
            if max_shard_sequences is None
            else max_shard_sequences
        )
        shard_prefill_tokens = (
            max_wave_prefill_tokens
            if max_shard_prefill_tokens is None
            else max_shard_prefill_tokens
        )
        if shard_sequences <= 0 or shard_sequences > max_wave_sequences:
            raise ValueError(
                "max_shard_sequences must lie in [1, max_wave_sequences]"
            )
        if (
            shard_prefill_tokens <= 0
            or shard_prefill_tokens > max_wave_prefill_tokens
        ):
            raise ValueError(
                "max_shard_prefill_tokens must lie in "
                "[1, max_wave_prefill_tokens]"
            )
        if max_shards_per_parent_per_wave <= 0:
            raise ValueError("max_shards_per_parent_per_wave must be positive")
        if max_inflight_waves <= 0:
            raise ValueError("max_inflight_waves must be positive")
        self._backend = backend
        self._max_wave_sequences = int(max_wave_sequences)
        self._target_wave_sequences = int(target)
        self._max_wait_seconds = float(max_wait_seconds)
        self._max_wave_prefill_tokens = int(max_wave_prefill_tokens)
        self._max_shard_sequences = int(shard_sequences)
        self._max_shard_prefill_tokens = int(shard_prefill_tokens)
        self._max_shards_per_parent_per_wave = int(
            max_shards_per_parent_per_wave
        )
        self._max_inflight_waves = int(max_inflight_waves)
        # Sequence credits tie pipelining to the graph-eligible scheduler domain.
        self._max_inflight_sequences = int(max_wave_sequences)
        self._condition = threading.Condition()
        self._queue: deque[_ShardTicket] = deque()
        self._next_parent_ordinal = 0
        self._stopping = False
        self._admitted_parent_ordinals: set[int] = set()
        self._admitted_shards = 0
        self._admitted_sequences = 0
        self._admitted_prefill_tokens = 0
        self._completed_parent_groups = 0
        self._repeated_prefix_run_count = 0
        self._preserved_prefix_run_count = 0
        self._forced_split_run_count = 0
        self._oversized_atomic_run_count = 0
        self._cancelled_shard_count = 0
        self._maximum_queued_groups = 0
        self._maximum_queued_shards = 0
        self._maximum_queued_sequences = 0
        self._inflight_waves = 0
        self._inflight_sequences = 0
        self._maximum_inflight_waves = 0
        self._maximum_inflight_sequences = 0
        self._sequence_credit_stall_count = 0
        self._sequence_credit_stall_seconds = 0.0
        self._streaming_sequence_credit_release_count = 0
        self._batch_sequence_credit_release_count = 0
        self._dispatch_failure_count = 0
        self._admission_wait_total = 0.0
        self._maximum_admission_wait = 0.0
        self._parent_completion_span_total = 0.0
        self._maximum_parent_completion_span = 0.0
        self._partial_wave_count = 0
        self._oversized_wave_count = 0
        self._fairness_limited_wave_count = 0
        self._fairness_violation_count = 0
        self._cross_parent_wave_count = 0
        self._parent_groups_across_waves = 0
        self._maximum_shards_for_one_parent = 0
        self._waves: list[dict[str, Any]] = []
        self._dispatch_executor = ThreadPoolExecutor(
            max_workers=self._max_inflight_waves,
            thread_name_prefix="inference-autopilot-prefix-wave",
        )
        self._worker = threading.Thread(
            target=self._run,
            name="inference-autopilot-prefix-sharded-wavefront",
            daemon=True,
        )
        self._worker.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _pack_locked(
        self,
    ) -> tuple[list[_ShardTicket], int, int, bool, bool, bool]:
        packed: list[_ShardTicket] = []
        sequences = 0
        prefill_tokens = 0
        per_parent: Counter[int] = Counter()
        sequence_blocked = False
        prefill_blocked = False
        fairness_blocked = False
        for ticket in self._queue:
            parent_ordinal = ticket.parent.ordinal
            if (
                per_parent[parent_ordinal]
                >= self._max_shards_per_parent_per_wave
            ):
                fairness_blocked = True
                continue
            exceeds_sequences = (
                sequences + ticket.sequence_count > self._max_wave_sequences
            )
            exceeds_prefill = (
                prefill_tokens + ticket.prefill_token_count
                > self._max_wave_prefill_tokens
            )
            if packed and (exceeds_sequences or exceeds_prefill):
                sequence_blocked = sequence_blocked or exceeds_sequences
                prefill_blocked = prefill_blocked or exceeds_prefill
                continue
            packed.append(ticket)
            per_parent[parent_ordinal] += 1
            sequences += ticket.sequence_count
            prefill_tokens += ticket.prefill_token_count
            if sequences >= self._target_wave_sequences:
                break
            if (
                sequences >= self._max_wave_sequences
                or prefill_tokens >= self._max_wave_prefill_tokens
            ):
                break
        return (
            packed,
            sequences,
            prefill_tokens,
            sequence_blocked,
            prefill_blocked,
            fairness_blocked,
        )

    def _take_wave_locked(
        self,
    ) -> tuple[list[_ShardTicket], int, int, str, bool] | None:
        while not self._queue:
            if self._stopping:
                return None
            self._condition.wait()
        while True:
            (
                packed,
                sequences,
                prefill_tokens,
                sequence_blocked,
                prefill_blocked,
                fairness_blocked,
            ) = self._pack_locked()
            remaining = (
                min(ticket.parent.enqueued_at for ticket in self._queue)
                + self._max_wait_seconds
                - time.monotonic()
            )
            if sequences >= self._target_wave_sequences:
                release_reason = "target_sequences"
            elif sequence_blocked and prefill_blocked:
                release_reason = "sequence_and_prefill_capacity"
            elif sequence_blocked:
                release_reason = "sequence_capacity"
            elif prefill_blocked:
                release_reason = "prefill_token_capacity"
            elif remaining <= 0:
                release_reason = "collection_timeout"
            elif self._stopping:
                release_reason = "shutdown"
            else:
                release_reason = ""
            if release_reason:
                selected = {id(ticket) for ticket in packed}
                self._queue = deque(
                    ticket
                    for ticket in self._queue
                    if id(ticket) not in selected
                )
                return (
                    packed,
                    sequences,
                    prefill_tokens,
                    release_reason,
                    fairness_blocked,
                )
            self._condition.wait(timeout=remaining)

    @staticmethod
    def _stage_counts(tickets: Sequence[_ShardTicket]) -> list[dict[str, Any]]:
        counts = Counter(ticket.stage_key for ticket in tickets)
        return [
            {"stage_key": [str(value) for value in key], "shards": count}
            for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
        ]

    def _record_wave(
        self,
        tickets: Sequence[_ShardTicket],
        sequences: int,
        prefill_tokens: int,
        admitted_at: float,
        release_reason: str,
        fairness_limited: bool,
        inflight_waves: int,
        inflight_sequences: int,
    ) -> None:
        waits = [admitted_at - ticket.parent.enqueued_at for ticket in tickets]
        parent_counts = Counter(ticket.parent.ordinal for ticket in tickets)
        maximum_parent_shards = max(parent_counts.values())
        for ticket in tickets:
            with ticket.parent.lock:
                if ticket.parent.first_admitted_at is None:
                    ticket.parent.first_admitted_at = admitted_at
        with self._condition:
            self._admitted_parent_ordinals.update(parent_counts)
            self._admitted_shards += len(tickets)
            self._admitted_sequences += sequences
            self._admitted_prefill_tokens += prefill_tokens
            self._admission_wait_total += sum(waits)
            self._maximum_admission_wait = max(
                self._maximum_admission_wait, *waits
            )
            partial = sequences < self._target_wave_sequences
            sequence_limit_exceeded = sequences > self._max_wave_sequences
            prefill_limit_exceeded = (
                prefill_tokens > self._max_wave_prefill_tokens
            )
            fairness_violation = (
                maximum_parent_shards > self._max_shards_per_parent_per_wave
            )
            self._partial_wave_count += int(partial)
            self._oversized_wave_count += int(
                sequence_limit_exceeded or prefill_limit_exceeded
            )
            self._fairness_limited_wave_count += int(fairness_limited)
            self._fairness_violation_count += int(fairness_violation)
            self._cross_parent_wave_count += int(len(parent_counts) > 1)
            self._parent_groups_across_waves += len(parent_counts)
            self._maximum_shards_for_one_parent = max(
                self._maximum_shards_for_one_parent, maximum_parent_shards
            )
            self._waves.append(
                {
                    "wave_id": len(self._waves) + 1,
                    "stage_composition": self._stage_counts(tickets),
                    "parent_groups": len(parent_counts),
                    "shards": len(tickets),
                    "groups": len(tickets),
                    "sequences": sequences,
                    "prefill_tokens": prefill_tokens,
                    "partial": partial,
                    "sequence_limit_exceeded": sequence_limit_exceeded,
                    "prefill_limit_exceeded": prefill_limit_exceeded,
                    "release_reason": release_reason,
                    "fairness_limited": fairness_limited,
                    "fairness_violation": fairness_violation,
                    "maximum_shards_for_one_parent": maximum_parent_shards,
                    "maximum_shard_sequences": max(
                        ticket.sequence_count for ticket in tickets
                    ),
                    "maximum_shard_prefill_tokens": max(
                        ticket.prefill_token_count for ticket in tickets
                    ),
                    "oldest_parent_wait_seconds": max(waits),
                    "maximum_wait_seconds": max(waits),
                    "inflight_waves_after_admission": inflight_waves,
                    "inflight_sequences_after_admission": inflight_sequences,
                }
            )

    @staticmethod
    def _record_sample(parent: _Parent, index: int, sample: Any) -> None:
        callback = None
        with parent.lock:
            if parent.failed:
                return
            if parent.outputs[index] is not _MISSING:
                return
            parent.outputs[index] = sample
            callback = parent.callback
        if callback is not None:
            callback(index, sample)

    def _complete_ticket(self, ticket: _ShardTicket, completed_at: float) -> None:
        result: list[Any] | None = None
        span: float | None = None
        with ticket.parent.lock:
            if ticket.parent.failed:
                return
            ticket.parent.remaining_shards -= 1
            if ticket.parent.remaining_shards < 0:
                raise RuntimeError("parent shard accounting became negative")
            if ticket.parent.remaining_shards == 0:
                if any(value is _MISSING for value in ticket.parent.outputs):
                    raise RuntimeError("sharded backend omitted a parent result")
                if ticket.parent.first_admitted_at is None:
                    raise RuntimeError("completed parent was never admitted")
                result = list(ticket.parent.outputs)
                span = completed_at - ticket.parent.first_admitted_at
        if result is None or span is None:
            return
        with self._condition:
            self._completed_parent_groups += 1
            self._parent_completion_span_total += span
            self._maximum_parent_completion_span = max(
                self._maximum_parent_completion_span, span
            )
        ticket.parent.future.set_result(result)

    def _fail_parents(
        self, tickets: Sequence[_ShardTicket], error: BaseException
    ) -> None:
        parents = {ticket.parent.ordinal: ticket.parent for ticket in tickets}
        with self._condition:
            failed_ordinals = set(parents)
            retained: deque[_ShardTicket] = deque()
            for queued in self._queue:
                if queued.parent.ordinal in failed_ordinals:
                    self._cancelled_shard_count += 1
                else:
                    retained.append(queued)
            self._queue = retained
            self._condition.notify_all()
        for parent in parents.values():
            with parent.lock:
                parent.failed = True
            if not parent.future.done():
                parent.future.set_exception(error)

    def _release_sequence_credits(
        self, sequences: int, *, streaming: bool
    ) -> None:
        if sequences <= 0:
            return
        with self._condition:
            self._inflight_sequences -= sequences
            if streaming:
                self._streaming_sequence_credit_release_count += sequences
            else:
                self._batch_sequence_credit_release_count += sequences
            if self._inflight_sequences < 0:
                raise RuntimeError("wavefront dispatch credit accounting became negative")
            self._condition.notify_all()

    def _release_wave_credit(self) -> None:
        with self._condition:
            self._inflight_waves -= 1
            if self._inflight_waves < 0:
                raise RuntimeError("wavefront dispatch credit accounting became negative")
            self._condition.notify_all()

    def _dispatch(
        self, tickets: Sequence[_ShardTicket], sequences: int
    ) -> None:
        requests = [request for ticket in tickets for request in ticket.requests]
        ownership = [
            (ticket.parent, parent_index)
            for ticket in tickets
            for parent_index in ticket.parent_indexes
        ]
        released_indexes: set[int] = set()
        release_lock = threading.Lock()

        def release_completed(index: int, *, streaming: bool) -> None:
            if index < 0 or index >= sequences:
                raise IndexError("sharded backend callback index is out of range")
            with release_lock:
                if index in released_indexes:
                    return
                released_indexes.add(index)
            self._release_sequence_credits(1, streaming=streaming)

        try:
            callback = getattr(self._backend, "sample_batch_with_callback", None)
            if callable(callback):

                def on_complete(index: int, sample: Any) -> None:
                    parent, parent_index = ownership[index]
                    self._record_sample(parent, parent_index, sample)
                    release_completed(index, streaming=True)

                samples = callback(requests, on_complete)
            else:
                samples = self._backend.sample_batch(requests)
            if len(samples) != sequences:
                raise RuntimeError(
                    "sharded wavefront backend returned an invalid sample count"
                )
            for index, ((parent, parent_index), sample) in enumerate(
                zip(ownership, samples, strict=True)
            ):
                self._record_sample(parent, parent_index, sample)
                release_completed(index, streaming=False)
        except BaseException as error:
            with release_lock:
                remaining_sequences = sequences - len(released_indexes)
            self._release_sequence_credits(
                remaining_sequences, streaming=False
            )
            self._release_wave_credit()
            with self._condition:
                self._dispatch_failure_count += 1
            self._fail_parents(tickets, error)
            return

        completed_at = time.monotonic()
        self._release_wave_credit()
        for ticket in tickets:
            self._complete_ticket(ticket, completed_at)

    def _run(self) -> None:
        while True:
            with self._condition:
                wave = self._take_wave_locked()
            if wave is None:
                return
            (
                tickets,
                sequences,
                prefill_tokens,
                release_reason,
                fairness_limited,
            ) = wave
            stall_started: float | None = None
            with self._condition:
                while (
                    self._inflight_waves >= self._max_inflight_waves
                    or (
                        self._inflight_waves > 0
                        and self._inflight_sequences + sequences
                        > self._max_inflight_sequences
                    )
                ):
                    if stall_started is None:
                        stall_started = time.monotonic()
                        self._sequence_credit_stall_count += 1
                    self._condition.wait()
                if stall_started is not None:
                    self._sequence_credit_stall_seconds += (
                        time.monotonic() - stall_started
                    )
                self._inflight_waves += 1
                self._inflight_sequences += sequences
                self._maximum_inflight_waves = max(
                    self._maximum_inflight_waves, self._inflight_waves
                )
                self._maximum_inflight_sequences = max(
                    self._maximum_inflight_sequences, self._inflight_sequences
                )
                inflight_waves = self._inflight_waves
                inflight_sequences = self._inflight_sequences
            admitted_at = time.monotonic()
            self._record_wave(
                tickets,
                sequences,
                prefill_tokens,
                admitted_at,
                release_reason,
                fairness_limited,
                inflight_waves,
                inflight_sequences,
            )
            try:
                self._dispatch_executor.submit(
                    self._dispatch, tickets, sequences
                )
            except BaseException as error:
                with self._condition:
                    self._dispatch_failure_count += 1
                self._release_sequence_credits(sequences, streaming=False)
                self._release_wave_credit()
                self._fail_parents(tickets, error)

    def _submit(self, requests: Sequence[Any], callback: Any | None) -> Future:
        materialized = tuple(requests)
        if not materialized:
            future = Future()
            future.set_result([])
            return future
        shard_plan = plan_prefix_shards(
            materialized,
            max_shard_sequences=self._max_shard_sequences,
            max_shard_prefill_tokens=self._max_shard_prefill_tokens,
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("prefix-sharded wavefront backend is closed")
            self._next_parent_ordinal += 1
            enqueued_at = time.monotonic()
            parent = _Parent(
                ordinal=self._next_parent_ordinal,
                requests=materialized,
                callback=callback,
                future=Future(),
                enqueued_at=enqueued_at,
                outputs=[_MISSING] * len(materialized),
                remaining_shards=len(shard_plan.shards),
            )
            for shard_index, shard in enumerate(shard_plan.shards):
                self._queue.append(
                    _ShardTicket(
                        parent=parent,
                        shard_index=shard_index,
                        requests=shard.requests,
                        parent_indexes=shard.parent_indexes,
                        stage_key=conditional_is_stage_key(shard.requests),
                        enqueued_at=enqueued_at,
                        prefill_token_count=shard.prefill_token_count,
                    )
                )
            self._repeated_prefix_run_count += (
                shard_plan.repeated_prefix_run_count
            )
            self._preserved_prefix_run_count += (
                shard_plan.preserved_prefix_run_count
            )
            self._forced_split_run_count += shard_plan.forced_split_run_count
            self._oversized_atomic_run_count += (
                shard_plan.oversized_atomic_run_count
            )
            queued_parents = len(
                {ticket.parent.ordinal for ticket in self._queue}
            )
            self._maximum_queued_groups = max(
                self._maximum_queued_groups, queued_parents
            )
            self._maximum_queued_shards = max(
                self._maximum_queued_shards, len(self._queue)
            )
            self._maximum_queued_sequences = max(
                self._maximum_queued_sequences,
                sum(ticket.sequence_count for ticket in self._queue),
            )
            self._condition.notify_all()
            return parent.future

    def sample_batch(self, requests: Sequence[Any]) -> Any:
        return self._submit(requests, None).result()

    def sample_batch_with_callback(
        self, requests: Sequence[Any], on_complete: Any
    ) -> Any:
        return self._submit(requests, on_complete).result()

    def score_batch(self, requests: Sequence[Any]) -> Any:
        return self._backend.score_batch(requests)

    def snapshot(self) -> PrefixShardedWavefrontSnapshot:
        with self._condition:
            mean_wait = (
                self._admission_wait_total / self._admitted_shards
                if self._admitted_shards
                else 0.0
            )
            mean_span = (
                self._parent_completion_span_total
                / self._completed_parent_groups
                if self._completed_parent_groups
                else 0.0
            )
            wave_count = len(self._waves)
            mean_sequence_utilization = (
                self._admitted_sequences
                / (wave_count * self._max_wave_sequences)
                if wave_count
                else 0.0
            )
            mean_prefill_utilization = (
                self._admitted_prefill_tokens
                / (wave_count * self._max_wave_prefill_tokens)
                if wave_count
                else 0.0
            )
            mean_parent_groups = (
                self._parent_groups_across_waves / wave_count
                if wave_count
                else 0.0
            )
            return PrefixShardedWavefrontSnapshot(
                max_wave_sequences=self._max_wave_sequences,
                target_wave_sequences=self._target_wave_sequences,
                max_wait_seconds=self._max_wait_seconds,
                max_wave_prefill_tokens=self._max_wave_prefill_tokens,
                max_shard_sequences=self._max_shard_sequences,
                max_shard_prefill_tokens=self._max_shard_prefill_tokens,
                max_shards_per_parent_per_wave=(
                    self._max_shards_per_parent_per_wave
                ),
                max_inflight_waves=self._max_inflight_waves,
                max_inflight_sequences=self._max_inflight_sequences,
                admitted_groups=len(self._admitted_parent_ordinals),
                admitted_parent_groups=len(self._admitted_parent_ordinals),
                admitted_shards=self._admitted_shards,
                admitted_sequences=self._admitted_sequences,
                admitted_prefill_tokens=self._admitted_prefill_tokens,
                completed_parent_groups=self._completed_parent_groups,
                repeated_prefix_run_count=self._repeated_prefix_run_count,
                preserved_prefix_run_count=self._preserved_prefix_run_count,
                forced_split_run_count=self._forced_split_run_count,
                oversized_atomic_run_count=self._oversized_atomic_run_count,
                cancelled_shard_count=self._cancelled_shard_count,
                wave_count=len(self._waves),
                partial_wave_count=self._partial_wave_count,
                oversized_wave_count=self._oversized_wave_count,
                fairness_limited_wave_count=self._fairness_limited_wave_count,
                fairness_violation_count=self._fairness_violation_count,
                cross_parent_wave_count=self._cross_parent_wave_count,
                parent_groups_across_waves=self._parent_groups_across_waves,
                maximum_shards_for_one_parent=(
                    self._maximum_shards_for_one_parent
                ),
                maximum_queued_groups=self._maximum_queued_groups,
                maximum_queued_shards=self._maximum_queued_shards,
                maximum_queued_sequences=self._maximum_queued_sequences,
                inflight_waves=self._inflight_waves,
                inflight_sequences=self._inflight_sequences,
                maximum_inflight_waves=self._maximum_inflight_waves,
                maximum_inflight_sequences=self._maximum_inflight_sequences,
                sequence_credit_stall_count=self._sequence_credit_stall_count,
                sequence_credit_stall_seconds=self._sequence_credit_stall_seconds,
                streaming_sequence_credit_release_count=(
                    self._streaming_sequence_credit_release_count
                ),
                batch_sequence_credit_release_count=(
                    self._batch_sequence_credit_release_count
                ),
                dispatch_failure_count=self._dispatch_failure_count,
                mean_admission_wait_seconds=mean_wait,
                maximum_admission_wait_seconds=self._maximum_admission_wait,
                mean_wave_sequence_utilization=mean_sequence_utilization,
                mean_wave_prefill_utilization=mean_prefill_utilization,
                mean_parent_groups_per_wave=mean_parent_groups,
                mean_parent_completion_span_seconds=mean_span,
                maximum_parent_completion_span_seconds=(
                    self._maximum_parent_completion_span
                ),
                release_reason_counts=tuple(
                    sorted(
                        Counter(
                            str(wave["release_reason"])
                            for wave in self._waves
                        ).items()
                    )
                ),
                waves=tuple(dict(wave) for wave in self._waves),
            )

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._worker.join()
        self._dispatch_executor.shutdown(wait=True)

    def __enter__(self) -> "PrefixShardedStageWavefrontAdmissionBackend":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "PrefixShard",
    "PrefixShardPlan",
    "PrefixShardedStageWavefrontAdmissionBackend",
    "PrefixShardedWavefrontSnapshot",
    "plan_prefix_shards",
]
