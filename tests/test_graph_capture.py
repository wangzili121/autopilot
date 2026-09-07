from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import unittest

from inference_autopilot.graph_capture import (
    CapturePlan,
    CapturePlanningSpec,
    plan_graph_capture,
)


def _profile() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "profile_id": "test-base-acl",
        "algorithm_id": "conditional_is_small_proposal",
        "engine_role": "base_model",
        "backend": "acl",
        "graph_mode": "piecewise",
        "observations": [
            {
                "stage_id": "candidate_generate",
                "shape_size": 1,
                "count": 100,
                "eager_latency_ms": 2.0,
            },
            {
                "stage_id": "candidate_generate",
                "shape_size": 4,
                "count": 40,
                "eager_latency_ms": 3.0,
            },
            {
                "stage_id": "target_score",
                "shape_size": 8,
                "count": 30,
                "eager_latency_ms": 6.0,
            },
            {
                "stage_id": "target_score",
                "shape_size": 16,
                "count": 10,
                "eager_latency_ms": 10.0,
            },
        ],
        "candidates": [
            {
                "capture_size": 1,
                "memory_bytes": 10,
                "capture_time_ms": 2.0,
                "replay_latency_ms_by_stage": {
                    "candidate_generate": 0.8,
                    "target_score": 1.5,
                },
            },
            {
                "capture_size": 4,
                "memory_bytes": 12,
                "capture_time_ms": 2.5,
                "replay_latency_ms_by_stage": {
                    "candidate_generate": 1.0,
                    "target_score": 1.8,
                },
            },
            {
                "capture_size": 8,
                "memory_bytes": 14,
                "capture_time_ms": 3.0,
                "replay_latency_ms_by_stage": {
                    "candidate_generate": 1.4,
                    "target_score": 2.2,
                },
            },
            {
                "capture_size": 16,
                "memory_bytes": 18,
                "capture_time_ms": 4.0,
                "replay_latency_ms_by_stage": {
                    "candidate_generate": 2.3,
                    "target_score": 3.4,
                },
            },
        ],
        "budget": {
            "max_buckets": 2,
            "memory_budget_bytes": 30,
            "capture_time_budget_ms": 7.0,
            "max_padding_ratio": 3.0,
            "minimum_graph_hit_rate": 0.0,
        },
        "objective": {
            "amortization_windows": 10,
            "padding_penalty_ms_per_unit": 0.01,
            "memory_penalty_ms_per_gib": 0.0,
        },
    }


def _exhaustive_objective(
    spec: CapturePlanningSpec, selected_sizes: tuple[int, ...]
) -> float | None:
    candidates = {candidate.capture_size: candidate for candidate in spec.candidates}
    selected = tuple(candidates[size] for size in selected_sizes)
    memory = sum(candidate.memory_bytes for candidate in selected)
    capture_time = sum(candidate.capture_time_ms for candidate in selected)
    if memory > spec.budget.memory_budget_bytes:
        return None
    if capture_time > spec.budget.capture_time_budget_ms:
        return None
    runtime = 0.0
    captured_count = 0
    padding = 0
    total_count = sum(observation.count for observation in spec.observations)
    for observation in spec.observations:
        candidate = next(
            (
                item
                for item in selected
                if item.capture_size >= observation.shape_size
            ),
            None,
        )
        captured = candidate is not None and (
            (candidate.capture_size - observation.shape_size) / observation.shape_size
            <= spec.budget.max_padding_ratio
        )
        if captured:
            runtime += (
                observation.count
                * candidate.replay_latency_ms_by_stage[observation.stage_id]
            )
            captured_count += observation.count
            padding += observation.count * (
                candidate.capture_size - observation.shape_size
            )
        else:
            runtime += observation.count * observation.eager_latency_ms
    if captured_count / total_count < spec.budget.minimum_graph_hit_rate:
        return None
    return (
        runtime
        + capture_time / spec.objective.amortization_windows
        + padding * spec.objective.padding_penalty_ms_per_unit
    )


class GraphCapturePlannerTest(unittest.TestCase):
    def test_dynamic_program_matches_exhaustive_search(self) -> None:
        spec = CapturePlanningSpec.from_dict(_profile())
        plan = plan_graph_capture(spec)
        sizes = tuple(sorted(candidate.capture_size for candidate in spec.candidates))
        exhaustive = []
        for count in range(spec.budget.max_buckets + 1):
            for selected in combinations(sizes, count):
                objective = _exhaustive_objective(spec, selected)
                if objective is not None:
                    exhaustive.append((objective, selected))

        expected_objective, expected_sizes = min(exhaustive)
        self.assertEqual(plan.selected_capture_sizes, expected_sizes)
        self.assertAlmostEqual(
            plan.metrics["objective_ms_per_profile_window"], expected_objective
        )
        self.assertGreater(plan.metrics["predicted_runtime_speedup"], 1.0)

    def test_all_eager_wins_when_capture_is_slower(self) -> None:
        raw = _profile()
        for candidate in raw["candidates"]:
            candidate["replay_latency_ms_by_stage"] = {
                "candidate_generate": 20.0,
                "target_score": 20.0,
            }
        plan = plan_graph_capture(CapturePlanningSpec.from_dict(raw))

        self.assertEqual(plan.selected_capture_sizes, ())
        self.assertEqual(plan.metrics["graph_hit_rate"], 0.0)
        self.assertEqual(plan.metrics["predicted_runtime_speedup"], 1.0)

    def test_hit_rate_and_resource_constraints_can_make_plan_infeasible(self) -> None:
        raw = _profile()
        raw["budget"]["minimum_graph_hit_rate"] = 1.0
        raw["budget"]["memory_budget_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "no graph capture plan"):
            plan_graph_capture(CapturePlanningSpec.from_dict(raw))

    def test_stage_profiles_are_complete_and_plan_digest_detects_tampering(self) -> None:
        raw = _profile()
        del raw["candidates"][0]["replay_latency_ms_by_stage"]["target_score"]
        with self.assertRaisesRegex(ValueError, "stage profile mismatch"):
            CapturePlanningSpec.from_dict(raw)

        plan_payload = plan_graph_capture(
            CapturePlanningSpec.from_dict(_profile())
        ).to_dict()
        self.assertEqual(CapturePlan.from_dict(plan_payload).to_dict(), plan_payload)
        tampered = deepcopy(plan_payload)
        tampered["metrics"]["graph_hit_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "SHA256"):
            CapturePlan.from_dict(tampered)

    def test_repository_acl_example_is_plannable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads(
            (
                root
                / "examples"
                / "conditional-is-base.acl-graph-profile.example.json"
            ).read_text(encoding="utf-8")
        )
        plan = plan_graph_capture(CapturePlanningSpec.from_dict(raw))

        self.assertEqual(plan.selected_capture_sizes, (1, 8, 16, 32))
        self.assertGreaterEqual(plan.metrics["graph_hit_rate"], 0.85)
        self.assertLessEqual(plan.metrics["graph_memory_bytes"], 268435456)


if __name__ == "__main__":
    unittest.main()
