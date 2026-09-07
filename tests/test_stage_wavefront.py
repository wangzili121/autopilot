from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from unittest import TestCase

from inference_autopilot.runners.stage_wavefront import (
    StageWavefrontAdmissionBackend,
    conditional_is_stage_key,
)
from inference_autopilot.stage_wavefront import plan_stage_wavefront


@dataclass(frozen=True)
class _Request:
    request_id: str
    max_new_tokens: int
    prefix: tuple[int, ...] = ()


class _Backend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.call_sizes: list[int] = []

    def sample_batch(self, requests):
        with self.lock:
            self.active_calls += 1
            self.call_sizes.append(len(requests))
            self.maximum_active_calls = max(
                self.maximum_active_calls, self.active_calls
            )
        time.sleep(0.02)
        with self.lock:
            self.active_calls -= 1
        return list(requests)

    def score_batch(self, requests):
        return list(requests)


def _requests(
    owner: int, *, step: int = 0, prefix_tokens: int = 0
) -> list[_Request]:
    return [
        _Request(
            f"owner:{owner}:step:{step}:rollout:{index}",
            48,
            (1,) * prefix_tokens,
        )
        for index in range(2)
    ]


class StageWavefrontPlanningTests(TestCase):
    def test_current_p96_shape_selects_exact_six_wave_partition(self) -> None:
        plan = plan_stage_wavefront(
            outer_concurrency=96,
            sequences_per_group=24,
            graph_capture_ceiling=512,
            scheduler_sequence_cap=768,
        )

        self.assertEqual(plan.effective_sequence_cap, 512)
        self.assertEqual(plan.selection_class, "exact_partition_above_minimum_utilization")
        self.assertEqual(plan.selected.groups_per_wave, 16)
        self.assertEqual(plan.selected.sequences_per_wave, 384)
        self.assertEqual(plan.selected.wave_count, 6)
        self.assertTrue(plan.selected.exact_partition)

    def test_high_utilization_policy_selects_five_wave_challenger(self) -> None:
        plan = plan_stage_wavefront(
            outer_concurrency=96,
            sequences_per_group=24,
            graph_capture_ceiling=512,
            scheduler_sequence_cap=768,
            minimum_full_wave_utilization=0.8,
        )

        self.assertEqual(plan.selection_class, "balanced_tail_and_utilization")
        self.assertEqual(plan.selected.groups_per_wave, 20)
        self.assertEqual(plan.selected.sequences_per_wave, 480)
        self.assertEqual(plan.selected.wave_count, 5)
        self.assertEqual(plan.selected.tail_groups, 16)
        self.assertFalse(plan.selected.exact_partition)

    def test_scheduler_cap_can_be_tighter_than_graph_ceiling(self) -> None:
        plan = plan_stage_wavefront(
            outer_concurrency=12,
            sequences_per_group=8,
            graph_capture_ceiling=512,
            scheduler_sequence_cap=64,
        )

        self.assertEqual(plan.effective_sequence_cap, 64)
        self.assertLessEqual(plan.selected.sequences_per_wave, 64)

    def test_rejects_an_indivisible_group_above_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "one algorithm group exceeds"):
            plan_stage_wavefront(
                outer_concurrency=4,
                sequences_per_group=65,
                graph_capture_ceiling=64,
                scheduler_sequence_cap=128,
            )


class StageWavefrontRuntimeTests(TestCase):
    def test_stage_key_separates_algorithm_steps(self) -> None:
        self.assertEqual(
            conditional_is_stage_key(_requests(0, step=2)),
            ("conditional_is_rollout", 2, (48,)),
        )

    def test_non_overlapping_full_waves_preserve_request_groups(self) -> None:
        backend = _Backend()
        admission = StageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=4,
            max_wait_seconds=0.2,
        )
        barrier = threading.Barrier(4)

        def run(owner: int):
            requests = _requests(owner)
            barrier.wait()
            return admission.sample_batch(requests)

        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(run, range(4)))

        self.assertEqual(outputs, [_requests(owner) for owner in range(4)])
        self.assertEqual(backend.maximum_active_calls, 1)
        self.assertEqual(backend.call_sizes, [4, 4])
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.wave_count, 2)
        self.assertEqual(snapshot.partial_wave_count, 0)
        self.assertEqual(
            dict(snapshot.release_reason_counts), {"target_sequences": 2}
        )
        self.assertEqual(
            [(wave["groups"], wave["sequences"]) for wave in snapshot.waves],
            [(2, 4), (2, 4)],
        )
        admission.close()

    def test_timeout_releases_a_partial_tail_wave(self) -> None:
        backend = _Backend()
        admission = StageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=4,
            max_wait_seconds=0.01,
        )

        self.assertEqual(admission.sample_batch(_requests(0)), _requests(0))
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.wave_count, 1)
        self.assertEqual(snapshot.partial_wave_count, 1)
        self.assertEqual(
            dict(snapshot.release_reason_counts), {"collection_timeout": 1}
        )
        self.assertEqual(snapshot.waves[0]["sequences"], 2)
        admission.close()

    def test_dynamic_prefill_budget_splits_a_sequence_feasible_wave(self) -> None:
        backend = _Backend()
        admission = StageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=4,
            max_wave_prefill_tokens=6,
            max_wait_seconds=0.2,
        )
        barrier = threading.Barrier(2)

        def run(owner: int):
            requests = _requests(owner, prefix_tokens=2)
            barrier.wait()
            return admission.sample_batch(requests)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(run, range(2)))

        snapshot = admission.snapshot()
        self.assertEqual(backend.call_sizes, [2, 2])
        self.assertEqual(snapshot.wave_count, 2)
        self.assertEqual(snapshot.partial_wave_count, 2)
        self.assertEqual(
            dict(snapshot.release_reason_counts),
            {"collection_timeout": 1, "prefill_token_capacity": 1},
        )
        self.assertEqual(
            [wave["prefill_tokens"] for wave in snapshot.waves], [4, 4]
        )
        admission.close()

    def test_oversized_indivisible_group_is_released_and_attested(self) -> None:
        backend = _Backend()
        admission = StageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=4,
            max_wave_prefill_tokens=3,
            max_wait_seconds=0.2,
        )

        requests = _requests(0, prefix_tokens=2)
        self.assertEqual(admission.sample_batch(requests), requests)
        snapshot = admission.snapshot()
        self.assertEqual(snapshot.oversized_wave_count, 1)
        self.assertTrue(snapshot.waves[0]["prefill_limit_exceeded"])
        self.assertFalse(snapshot.waves[0]["sequence_limit_exceeded"])
        admission.close()

    def test_merged_dispatch_maps_callbacks_back_to_each_group(self) -> None:
        backend = _Backend()
        admission = StageWavefrontAdmissionBackend(
            backend,
            max_wave_sequences=4,
            max_wait_seconds=0.2,
        )
        barrier = threading.Barrier(2)
        callbacks: list[tuple[int, int, _Request]] = []
        callback_lock = threading.Lock()

        def run(owner: int):
            requests = _requests(owner)
            barrier.wait()

            def completed(index: int, sample: _Request) -> None:
                with callback_lock:
                    callbacks.append((owner, index, sample))

            return admission.sample_batch_with_callback(requests, completed)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(run, range(2)))

        self.assertEqual(outputs, [_requests(owner) for owner in range(2)])
        self.assertCountEqual(
            callbacks,
            [
                (owner, index, request)
                for owner in range(2)
                for index, request in enumerate(_requests(owner))
            ],
        )
        admission.close()
