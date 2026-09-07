from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from unittest import TestCase

from inference_autopilot.runners.prefix_sharded_wavefront import (
    PrefixShardedStageWavefrontAdmissionBackend,
    plan_prefix_shards,
)


@dataclass(frozen=True)
class _Request:
    request_id: str
    prefix: tuple[int, ...]
    max_new_tokens: int
    sampling: str
    seed: int


def _parent(owner: int, *, prefix_tokens: int = 2000) -> list[_Request]:
    return [
        _Request(
            request_id=(
                f"owner:{owner}:step:0:candidate:{candidate}:rollout:{rollout}"
            ),
            prefix=(owner, candidate) + (1,) * (prefix_tokens - 2),
            max_new_tokens=48,
            sampling="full-support",
            seed=owner * 100 + candidate * 10 + rollout,
        )
        for candidate in range(4)
        for rollout in range(3)
    ]


class _CallbackBackend:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.lock = threading.Lock()
        self.call_sizes: list[int] = []
        self.calls: list[list[_Request]] = []
        self.fail_on_call = fail_on_call

    def _samples(self, requests):
        with self.lock:
            self.call_sizes.append(len(requests))
            self.calls.append(list(requests))
            call_number = len(self.call_sizes)
        if self.fail_on_call == call_number:
            raise RuntimeError("injected shard failure")
        return [f"sample:{request.request_id}" for request in requests]

    def sample_batch_with_callback(self, requests, callback):
        samples = self._samples(requests)
        for index in reversed(range(len(samples))):
            callback(index, samples[index])
        time.sleep(0.002)
        return samples

    def sample_batch(self, requests):
        return self._samples(requests)

    def score_batch(self, requests):
        return list(requests)


class _BlockingBackend(_CallbackBackend):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.two_calls_started = threading.Event()
        self.active_calls = 0
        self.maximum_active_calls = 0

    def sample_batch(self, requests):
        with self.lock:
            self.call_sizes.append(len(requests))
            self.calls.append(list(requests))
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls, self.active_calls
            )
            if self.active_calls >= 2:
                self.two_calls_started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test backend was not released")
        with self.lock:
            self.active_calls -= 1
        return [f"sample:{request.request_id}" for request in requests]

    def sample_batch_with_callback(self, requests, callback):
        samples = self.sample_batch(requests)
        for index, sample in enumerate(samples):
            callback(index, sample)
        return samples


class _IncrementalCallbackBackend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.call_count = 0
        self.first_sequence_released = threading.Event()
        self.second_call_started = threading.Event()
        self.finish_first_call = threading.Event()

    def sample_batch_with_callback(self, requests, callback):
        with self.lock:
            self.call_count += 1
            call_number = self.call_count
        samples = [f"sample:{request.request_id}" for request in requests]
        if call_number == 1:
            callback(0, samples[0])
            self.first_sequence_released.set()
            if not self.finish_first_call.wait(timeout=2.0):
                raise TimeoutError("first streaming call was not released")
            for index in range(1, len(samples)):
                callback(index, samples[index])
        else:
            self.second_call_started.set()
            for index, sample in enumerate(samples):
                callback(index, sample)
        return samples

    def sample_batch(self, requests):
        return self.sample_batch_with_callback(requests, lambda _index, _sample: None)

    def score_batch(self, requests):
        return list(requests)


class PrefixShardPlanningTests(TestCase):
    def test_preserves_repeated_prefix_runs_and_request_identity(self) -> None:
        requests = _parent(0)

        plan = plan_prefix_shards(
            requests,
            max_shard_sequences=6,
            max_shard_prefill_tokens=12000,
        )

        self.assertEqual([shard.sequence_count for shard in plan.shards], [6, 6])
        self.assertEqual(plan.repeated_prefix_run_count, 4)
        self.assertEqual(plan.preserved_prefix_run_count, 4)
        self.assertEqual(plan.forced_split_run_count, 0)
        self.assertEqual(plan.oversized_atomic_run_count, 0)
        flattened = [request for shard in plan.shards for request in shard.requests]
        self.assertEqual(flattened, requests)
        self.assertTrue(
            all(actual is expected for actual, expected in zip(flattened, requests))
        )
        self.assertEqual(
            [request.seed for request in flattened],
            [request.seed for request in requests],
        )

    def test_forces_a_run_split_only_when_the_run_cannot_fit(self) -> None:
        requests = _parent(0, prefix_tokens=5000)[:3]

        plan = plan_prefix_shards(
            requests,
            max_shard_sequences=4,
            max_shard_prefill_tokens=9000,
        )

        self.assertEqual([shard.sequence_count for shard in plan.shards], [1, 1, 1])
        self.assertEqual(plan.repeated_prefix_run_count, 1)
        self.assertEqual(plan.preserved_prefix_run_count, 0)
        self.assertEqual(plan.forced_split_run_count, 1)
        self.assertEqual(plan.oversized_atomic_run_count, 0)

    def test_attests_an_individually_oversized_request(self) -> None:
        plan = plan_prefix_shards(
            _parent(0, prefix_tokens=10000)[:1],
            max_shard_sequences=4,
            max_shard_prefill_tokens=9000,
        )

        self.assertEqual(len(plan.shards), 1)
        self.assertEqual(plan.forced_split_run_count, 1)
        self.assertEqual(plan.oversized_atomic_run_count, 1)


class PrefixShardedWavefrontRuntimeTests(TestCase):
    def test_reorders_callbacks_to_parent_indexes_and_preserves_barrier(self) -> None:
        backend = _CallbackBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=6,
            target_wave_sequences=6,
            max_wave_prefill_tokens=12000,
            max_shards_per_parent_per_wave=1,
            max_wait_seconds=0.02,
        )
        requests = _parent(2)
        callbacks: list[tuple[int, str]] = []

        outputs = admission.sample_batch_with_callback(
            requests, lambda index, sample: callbacks.append((index, sample))
        )

        self.assertEqual(
            outputs,
            [f"sample:{request.request_id}" for request in requests],
        )
        self.assertCountEqual(
            callbacks,
            [
                (index, f"sample:{request.request_id}")
                for index, request in enumerate(requests)
            ],
        )
        self.assertEqual(len(callbacks), len(requests))
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.completed_parent_groups, 1)
        self.assertEqual(snapshot.admitted_shards, 2)
        self.assertEqual(snapshot.maximum_shards_for_one_parent, 1)
        self.assertEqual(snapshot.fairness_violation_count, 0)
        admission.close()

    def test_fairness_cap_mixes_independent_parents(self) -> None:
        backend = _CallbackBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=6,
            target_wave_sequences=6,
            max_wave_prefill_tokens=18000,
            max_shard_sequences=3,
            max_shard_prefill_tokens=7000,
            max_shards_per_parent_per_wave=1,
            max_wait_seconds=0.05,
        )
        barrier = threading.Barrier(4)

        def run(owner: int):
            barrier.wait()
            return admission.sample_batch(_parent(owner))

        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(run, range(4)))

        self.assertEqual(
            outputs,
            [
                [f"sample:{request.request_id}" for request in _parent(owner)]
                for owner in range(4)
            ],
        )
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.admitted_parent_groups, 4)
        self.assertEqual(snapshot.admitted_shards, 16)
        self.assertEqual(snapshot.completed_parent_groups, 4)
        self.assertEqual(snapshot.fairness_violation_count, 0)
        self.assertLessEqual(snapshot.maximum_shards_for_one_parent, 1)
        self.assertTrue(all(wave["parent_groups"] == 2 for wave in snapshot.waves))
        self.assertTrue(
            all(wave["maximum_shard_sequences"] == 3 for wave in snapshot.waves)
        )
        admission.close()

    def test_short_parent_remains_one_indivisible_shard(self) -> None:
        backend = _CallbackBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=24,
            max_wave_prefill_tokens=131072,
            max_wait_seconds=0,
        )
        requests = _parent(0, prefix_tokens=100)

        outputs = admission.sample_batch(requests)

        self.assertEqual(len(outputs), len(requests))
        self.assertEqual(backend.call_sizes, [12])
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.admitted_shards, 1)
        self.assertEqual(snapshot.forced_split_run_count, 0)
        self.assertEqual(snapshot.oversized_atomic_run_count, 0)
        admission.close()

    def test_error_fails_parent_and_cancels_unsubmitted_shards(self) -> None:
        backend = _CallbackBackend(fail_on_call=1)
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=6,
            max_wave_prefill_tokens=12000,
            max_shards_per_parent_per_wave=1,
            max_wait_seconds=0,
        )

        with self.assertRaisesRegex(RuntimeError, "injected shard failure"):
            admission.sample_batch(_parent(0))

        snapshot = admission.snapshot()
        self.assertEqual(snapshot.cancelled_shard_count, 1)
        self.assertEqual(snapshot.completed_parent_groups, 0)
        admission.close()

    def test_pipelines_partial_waves_with_bounded_sequence_credits(self) -> None:
        backend = _BlockingBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=6,
            target_wave_sequences=6,
            max_wave_prefill_tokens=6000,
            max_shard_sequences=3,
            max_shard_prefill_tokens=6000,
            max_shards_per_parent_per_wave=1,
            max_inflight_waves=2,
            max_wait_seconds=0,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(admission.sample_batch, _parent(0)[:3])
            deadline = time.monotonic() + 1.0
            while not backend.call_sizes and time.monotonic() < deadline:
                time.sleep(0.001)
            second = executor.submit(admission.sample_batch, _parent(1)[:3])
            self.assertTrue(backend.two_calls_started.wait(timeout=1.0))
            backend.release.set()
            self.assertEqual(len(first.result()), 3)
            self.assertEqual(len(second.result()), 3)

        snapshot = admission.snapshot()
        self.assertEqual(backend.maximum_active_calls, 2)
        self.assertEqual(snapshot.max_inflight_waves, 2)
        self.assertEqual(snapshot.max_inflight_sequences, 6)
        self.assertEqual(snapshot.maximum_inflight_waves, 2)
        self.assertEqual(snapshot.maximum_inflight_sequences, 6)
        self.assertEqual(snapshot.inflight_waves, 0)
        self.assertEqual(snapshot.inflight_sequences, 0)
        self.assertEqual(snapshot.dispatch_failure_count, 0)
        self.assertTrue(
            all(
                wave["inflight_sequences_after_admission"] <= 6
                for wave in snapshot.waves
            )
        )
        admission.close()

    def test_max_inflight_wave_cap_blocks_a_second_dispatch(self) -> None:
        backend = _BlockingBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=6,
            target_wave_sequences=6,
            max_wave_prefill_tokens=6000,
            max_shard_sequences=3,
            max_shard_prefill_tokens=6000,
            max_shards_per_parent_per_wave=1,
            max_inflight_waves=1,
            max_wait_seconds=0,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(admission.sample_batch, _parent(0)[:3])
            deadline = time.monotonic() + 1.0
            while not backend.call_sizes and time.monotonic() < deadline:
                time.sleep(0.001)
            second = executor.submit(admission.sample_batch, _parent(1)[:3])
            time.sleep(0.02)
            self.assertFalse(backend.two_calls_started.is_set())
            backend.release.set()
            self.assertEqual(len(first.result()), 3)
            self.assertEqual(len(second.result()), 3)

        snapshot = admission.snapshot()
        self.assertEqual(backend.maximum_active_calls, 1)
        self.assertEqual(snapshot.maximum_inflight_waves, 1)
        self.assertGreaterEqual(snapshot.sequence_credit_stall_count, 1)
        self.assertGreater(snapshot.sequence_credit_stall_seconds, 0.0)
        admission.close()

    def test_streaming_completion_refills_sequence_credit_before_wave_end(self) -> None:
        backend = _IncrementalCallbackBackend()
        admission = PrefixShardedStageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=3,
            target_wave_sequences=3,
            max_wave_prefill_tokens=6000,
            max_shard_sequences=3,
            max_shard_prefill_tokens=6000,
            max_shards_per_parent_per_wave=1,
            max_inflight_waves=3,
            max_wait_seconds=0,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(admission.sample_batch, _parent(0)[:3])
            self.assertTrue(backend.first_sequence_released.wait(timeout=1.0))
            second = executor.submit(admission.sample_batch, _parent(1)[:1])
            self.assertTrue(backend.second_call_started.wait(timeout=1.0))
            self.assertFalse(first.done())
            backend.finish_first_call.set()
            self.assertEqual(len(first.result()), 3)
            self.assertEqual(len(second.result()), 1)

        snapshot = admission.snapshot()
        self.assertEqual(snapshot.maximum_inflight_sequences, 3)
        self.assertEqual(snapshot.maximum_inflight_waves, 2)
        self.assertEqual(snapshot.streaming_sequence_credit_release_count, 4)
        self.assertEqual(snapshot.batch_sequence_credit_release_count, 0)
        self.assertEqual(snapshot.inflight_sequences, 0)
        admission.close()


if __name__ == "__main__":
    import unittest

    unittest.main()
