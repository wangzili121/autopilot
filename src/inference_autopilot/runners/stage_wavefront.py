"""Runtime admission control for phase-aligned algorithm waves."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Sequence


_STEP_PATTERN = re.compile(r"(?:^|:)step:(\d+)(?::|$)")


@dataclass(slots=True)
class _Ticket:
    requests: tuple[Any, ...]
    callback: Any | None
    future: Future
    stage_key: tuple[Any, ...]
    ordinal: int
    enqueued_at: float
    prefill_token_count: int

    @property
    def sequence_count(self) -> int:
        return len(self.requests)


@dataclass(frozen=True, slots=True)
class StageWavefrontSnapshot:
    max_wave_sequences: int
    target_wave_sequences: int
    max_wait_seconds: float
    max_wave_prefill_tokens: int | None
    admitted_groups: int
    admitted_sequences: int
    wave_count: int
    partial_wave_count: int
    oversized_wave_count: int
    maximum_queued_groups: int
    maximum_queued_sequences: int
    mean_admission_wait_seconds: float
    maximum_admission_wait_seconds: float
    release_reason_counts: tuple[tuple[str, int], ...]
    waves: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_wave_sequences": self.max_wave_sequences,
            "target_wave_sequences": self.target_wave_sequences,
            "max_wait_seconds": self.max_wait_seconds,
            "max_wave_prefill_tokens": self.max_wave_prefill_tokens,
            "admitted_groups": self.admitted_groups,
            "admitted_sequences": self.admitted_sequences,
            "wave_count": self.wave_count,
            "partial_wave_count": self.partial_wave_count,
            "oversized_wave_count": self.oversized_wave_count,
            "maximum_queued_groups": self.maximum_queued_groups,
            "maximum_queued_sequences": self.maximum_queued_sequences,
            "mean_admission_wait_seconds": self.mean_admission_wait_seconds,
            "maximum_admission_wait_seconds": self.maximum_admission_wait_seconds,
            "release_reason_counts": dict(self.release_reason_counts),
            "waves": list(self.waves),
        }


def conditional_is_stage_key(requests: Sequence[Any]) -> tuple[Any, ...]:
    """Extract stage metadata without making it an admission barrier."""

    stages: set[int] = set()
    lengths: set[int] = set()
    for request in requests:
        match = _STEP_PATTERN.search(str(getattr(request, "request_id", "")))
        if match is not None:
            stages.add(int(match.group(1)))
        lengths.add(int(getattr(request, "max_new_tokens", -1)))
    stage: int | str = next(iter(stages)) if len(stages) == 1 else "unknown"
    return ("conditional_is_rollout", stage, tuple(sorted(lengths)))


class StageWavefrontAdmissionBackend:
    """Merge synchronous algorithm groups into non-overlapping engine waves.

    The dispatcher batches every admitted wave into one backend call. Groups may
    come from different Conditional IS iteration numbers because all are the same
    proposal-generation operation. Requests within each caller group are kept in
    order, and outputs and completion callbacks are mapped back to that group.
    """

    def __init__(
        self,
        backend: Any,
        *,
        max_wave_sequences: int,
        target_wave_sequences: int | None = None,
        max_wait_seconds: float = 0.05,
        max_wave_prefill_tokens: int | None = None,
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
        if max_wave_prefill_tokens is not None and max_wave_prefill_tokens <= 0:
            raise ValueError("max_wave_prefill_tokens must be positive when set")
        self._backend = backend
        self._max_wave_sequences = int(max_wave_sequences)
        self._target_wave_sequences = int(target)
        self._max_wait_seconds = float(max_wait_seconds)
        self._max_wave_prefill_tokens = max_wave_prefill_tokens
        self._condition = threading.Condition()
        self._queue: deque[_Ticket] = deque()
        self._next_ordinal = 0
        self._stopping = False
        self._admitted_groups = 0
        self._admitted_sequences = 0
        self._admission_wait_total = 0.0
        self._maximum_admission_wait = 0.0
        self._maximum_queued_groups = 0
        self._maximum_queued_sequences = 0
        self._partial_wave_count = 0
        self._oversized_wave_count = 0
        self._waves: list[dict[str, Any]] = []
        self._worker = threading.Thread(
            target=self._run,
            name="inference-autopilot-stage-wavefront",
            daemon=True,
        )
        self._worker.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _pack_locked(
        self,
    ) -> tuple[list[_Ticket], int, int, bool, bool]:
        packed: list[_Ticket] = []
        sequences = 0
        prefill_tokens = 0
        sequence_blocked = False
        prefill_blocked = False
        for ticket in self._queue:
            exceeds_sequences = (
                sequences + ticket.sequence_count > self._max_wave_sequences
            )
            exceeds_prefill = (
                self._max_wave_prefill_tokens is not None
                and prefill_tokens + ticket.prefill_token_count
                > self._max_wave_prefill_tokens
            )
            if packed and (exceeds_sequences or exceeds_prefill):
                sequence_blocked = exceeds_sequences
                prefill_blocked = exceeds_prefill
                break
            packed.append(ticket)
            sequences += ticket.sequence_count
            prefill_tokens += ticket.prefill_token_count
            if sequences >= self._max_wave_sequences or (
                self._max_wave_prefill_tokens is not None
                and prefill_tokens >= self._max_wave_prefill_tokens
            ):
                break
        return (
            packed,
            sequences,
            prefill_tokens,
            sequence_blocked,
            prefill_blocked,
        )

    def _take_wave_locked(
        self,
    ) -> tuple[list[_Ticket], int, int, str] | None:
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
            ) = self._pack_locked()
            remaining = (
                self._queue[0].enqueued_at
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
                for expected in packed:
                    ticket = self._queue.popleft()
                    assert ticket is expected
                return packed, sequences, prefill_tokens, release_reason
            self._condition.wait(timeout=remaining)

    @staticmethod
    def _stage_counts(tickets: Sequence[_Ticket]) -> list[dict[str, Any]]:
        counts = Counter(ticket.stage_key for ticket in tickets)
        return [
            {
                "stage_key": [str(value) for value in key],
                "groups": count,
            }
            for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
        ]

    def _record_wave(
        self,
        tickets: Sequence[_Ticket],
        sequences: int,
        prefill_tokens: int,
        admitted_at: float,
        release_reason: str,
    ) -> None:
        waits = [admitted_at - ticket.enqueued_at for ticket in tickets]
        with self._condition:
            self._admitted_groups += len(tickets)
            self._admitted_sequences += sequences
            self._admission_wait_total += sum(waits)
            self._maximum_admission_wait = max(
                self._maximum_admission_wait, *waits
            )
            partial = sequences < self._target_wave_sequences
            sequence_limit_exceeded = sequences > self._max_wave_sequences
            prefill_limit_exceeded = (
                self._max_wave_prefill_tokens is not None
                and prefill_tokens > self._max_wave_prefill_tokens
            )
            self._partial_wave_count += int(partial)
            self._oversized_wave_count += int(
                sequence_limit_exceeded or prefill_limit_exceeded
            )
            self._waves.append(
                {
                    "wave_id": len(self._waves) + 1,
                    "stage_composition": self._stage_counts(tickets),
                    "groups": len(tickets),
                    "sequences": sequences,
                    "prefill_tokens": prefill_tokens,
                    "partial": partial,
                    "sequence_limit_exceeded": sequence_limit_exceeded,
                    "prefill_limit_exceeded": prefill_limit_exceeded,
                    "release_reason": release_reason,
                    "maximum_wait_seconds": max(waits),
                }
            )

    @staticmethod
    def _complete_callbacks(
        tickets: Sequence[_Ticket], samples: Sequence[Any]
    ) -> None:
        offset = 0
        for ticket in tickets:
            if ticket.callback is not None:
                for local_index, sample in enumerate(
                    samples[offset : offset + ticket.sequence_count]
                ):
                    ticket.callback(local_index, sample)
            offset += ticket.sequence_count

    def _dispatch(self, tickets: Sequence[_Ticket], sequences: int) -> None:
        requests = [request for ticket in tickets for request in ticket.requests]
        try:
            callback = getattr(self._backend, "sample_batch_with_callback", None)
            if callback is not None and any(
                ticket.callback is not None for ticket in tickets
            ):
                ownership: list[tuple[_Ticket, int]] = []
                for ticket in tickets:
                    ownership.extend(
                        (ticket, local_index)
                        for local_index in range(ticket.sequence_count)
                    )

                def on_complete(index: int, sample: Any) -> None:
                    ticket, local_index = ownership[index]
                    if ticket.callback is not None:
                        ticket.callback(local_index, sample)

                samples = callback(requests, on_complete)
            else:
                samples = self._backend.sample_batch(requests)
                self._complete_callbacks(tickets, samples)
            if len(samples) != sequences:
                raise RuntimeError(
                    "wavefront backend returned an invalid number of samples"
                )
            offset = 0
            for ticket in tickets:
                end = offset + ticket.sequence_count
                ticket.future.set_result(list(samples[offset:end]))
                offset = end
        except BaseException as error:
            for ticket in tickets:
                if not ticket.future.done():
                    ticket.future.set_exception(error)

    def _run(self) -> None:
        while True:
            with self._condition:
                wave = self._take_wave_locked()
            if wave is None:
                return
            tickets, sequences, prefill_tokens, release_reason = wave
            admitted_at = time.monotonic()
            self._record_wave(
                tickets,
                sequences,
                prefill_tokens,
                admitted_at,
                release_reason,
            )
            self._dispatch(tickets, sequences)

    def _submit(self, requests: Sequence[Any], callback: Any | None) -> Future:
        if not requests:
            future = Future()
            future.set_result([])
            return future
        requests = tuple(requests)
        with self._condition:
            if self._stopping:
                raise RuntimeError("stage wavefront backend is closed")
            self._next_ordinal += 1
            ticket = _Ticket(
                requests=requests,
                callback=callback,
                future=Future(),
                stage_key=conditional_is_stage_key(requests),
                ordinal=self._next_ordinal,
                enqueued_at=time.monotonic(),
                prefill_token_count=sum(
                    len(getattr(request, "prefix", ())) for request in requests
                ),
            )
            self._queue.append(ticket)
            queued_sequences = sum(item.sequence_count for item in self._queue)
            self._maximum_queued_groups = max(
                self._maximum_queued_groups, len(self._queue)
            )
            self._maximum_queued_sequences = max(
                self._maximum_queued_sequences, queued_sequences
            )
            self._condition.notify_all()
            return ticket.future

    def sample_batch(self, requests: Sequence[Any]) -> Any:
        return self._submit(requests, None).result()

    def sample_batch_with_callback(
        self,
        requests: Sequence[Any],
        on_complete: Any,
    ) -> Any:
        return self._submit(requests, on_complete).result()

    def score_batch(self, requests: Sequence[Any]) -> Any:
        return self._backend.score_batch(requests)

    def snapshot(self) -> StageWavefrontSnapshot:
        with self._condition:
            mean_wait = (
                self._admission_wait_total / self._admitted_groups
                if self._admitted_groups
                else 0.0
            )
            return StageWavefrontSnapshot(
                max_wave_sequences=self._max_wave_sequences,
                target_wave_sequences=self._target_wave_sequences,
                max_wait_seconds=self._max_wait_seconds,
                max_wave_prefill_tokens=self._max_wave_prefill_tokens,
                admitted_groups=self._admitted_groups,
                admitted_sequences=self._admitted_sequences,
                wave_count=len(self._waves),
                partial_wave_count=self._partial_wave_count,
                oversized_wave_count=self._oversized_wave_count,
                maximum_queued_groups=self._maximum_queued_groups,
                maximum_queued_sequences=self._maximum_queued_sequences,
                mean_admission_wait_seconds=mean_wait,
                maximum_admission_wait_seconds=self._maximum_admission_wait,
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
            if self._stopping:
                return
            self._stopping = True
            self._condition.notify_all()
        self._worker.join()

    def __enter__(self) -> "StageWavefrontAdmissionBackend":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "StageWavefrontAdmissionBackend",
    "StageWavefrontSnapshot",
    "conditional_is_stage_key",
]
