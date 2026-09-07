from __future__ import annotations

from math import exp, log
import unittest

from inference_autopilot.scoring import (
    ScoreImplementation,
    ScoreWorkload,
    build_score_reduction_plan,
    estimate_score_memory,
    requirement_for_reward,
    streaming_score_statistics_reference,
)
from inference_autopilot.search_space import RuntimeCapabilityProfile


class ScoringPlanTest(unittest.TestCase):
    def test_streaming_reference_matches_naive_full_logsoftmax(self) -> None:
        logits = [1.25, -0.5, 3.0, 0.75, 2.5, -2.0]
        log_partition = log(sum(exp(value) for value in logits))
        logprobs = [value - log_partition for value in logits]
        probabilities = [exp(value) for value in logprobs]
        expected_entropy = -sum(
            probability * logprob
            for probability, logprob in zip(probabilities, logprobs, strict=True)
        )
        result = streaming_score_statistics_reference(
            logits, selected_token_id=3, top_k=3
        )

        self.assertAlmostEqual(result["selected_logprob"], logprobs[3])
        self.assertAlmostEqual(
            result["topk_mean_logprob"],
            sum(sorted(logprobs, reverse=True)[:3]) / 3,
        )
        self.assertAlmostEqual(result["entropy"], expected_entropy)

    def test_conditional_is_and_consilience_require_different_statistics(self) -> None:
        conditional_is = requirement_for_reward(
            "self_consistency", importance_correction=True
        )
        consilience = requirement_for_reward(
            "consilience", importance_correction=True, consilience_top_k=7
        )

        self.assertEqual(conditional_is.statistics, ("selected_logprob",))
        self.assertEqual(
            consilience.statistics,
            ("selected_logprob", "topk_mean_logprob"),
        )
        self.assertEqual(consilience.top_k, 7)

    def test_native_workspace_matches_observed_long_context_allocation(self) -> None:
        requirement = requirement_for_reward(
            "self_consistency", importance_correction=True
        )
        workload = ScoreWorkload(positions=3584, vocab_size=151936)
        native = ScoreImplementation(
            "native-full-logsoftmax",
            "score.native_full_logsoftmax",
            ("entropy", "selected_logprob", "topk_mean_logprob"),
            True,
            False,
            3584,
            151936,
        )
        estimate = estimate_score_memory(requirement, workload, native)

        self.assertEqual(
            estimate.reduction_workspace_bytes,
            3584 * 151936 * 4,
        )
        self.assertAlmostEqual(
            estimate.reduction_workspace_bytes / (1024**3), 2.028564453125
        )

    def test_streaming_topk_reduction_avoids_full_fp32_workspace(self) -> None:
        requirement = requirement_for_reward(
            "consilience", importance_correction=True, consilience_top_k=5
        )
        workload = ScoreWorkload(positions=3584, vocab_size=151936)
        streaming = ScoreImplementation(
            "streaming-reduction-t256-v4096",
            "score.streaming_tiled_reduction",
            ("entropy", "selected_logprob", "topk_mean_logprob"),
            False,
            False,
            256,
            4096,
        )
        estimate = estimate_score_memory(requirement, workload, streaming)

        self.assertLess(estimate.reduction_workspace_bytes, 5 * 1024 * 1024)

    def test_budget_applies_to_incremental_reduction_memory(self) -> None:
        requirement = requirement_for_reward(
            "consilience", importance_correction=True, consilience_top_k=5
        )
        workload = ScoreWorkload(positions=3584, vocab_size=151936)
        profile = RuntimeCapabilityProfile.from_dict(
            {
                "schema_version": "1.0",
                "profile_id": "score-capabilities",
                "environment_id": "test-environment",
                "capabilities": {
                    "score.native_full_logsoftmax": {
                        "status": "supported",
                        "reason": "runtime implementation",
                        "evidence_sha256": [],
                    }
                },
            }
        )
        plan = build_score_reduction_plan(
            requirement,
            workload,
            memory_budget_bytes=int(1.91 * 1024**3),
            capability_profile=profile,
            token_chunk_sizes=(256,),
            vocab_tile_sizes=(4096,),
        )

        by_id = {
            candidate["implementation_id"]: candidate
            for candidate in plan["candidates"]
        }
        native = by_id["native-full-logsoftmax"]
        tiled = by_id["streaming-reduction-t256-v4096"]
        self.assertFalse(native["within_memory_budget"])
        self.assertTrue(tiled["within_memory_budget"])
        self.assertGreater(
            tiled["memory"]["estimated_peak_bytes"],
            tiled["memory"]["incremental_reduction_bytes"],
        )

    def test_plan_never_marks_unknown_implementation_runnable(self) -> None:
        profile = RuntimeCapabilityProfile.from_dict(
            {
                "schema_version": "1.0",
                "profile_id": "score-capabilities",
                "environment_id": "test-environment",
                "capabilities": {
                    "score.native_full_logsoftmax": {
                        "status": "supported",
                        "reason": "runtime implementation",
                        "evidence_sha256": [],
                    }
                },
            }
        )
        plan = build_score_reduction_plan(
            requirement_for_reward("consilience", importance_correction=True),
            ScoreWorkload(positions=128, vocab_size=151936),
            memory_budget_bytes=4 * 1024**3,
            capability_profile=profile,
            token_chunk_sizes=(64,),
            vocab_tile_sizes=(4096,),
        )

        by_id = {candidate["implementation_id"]: candidate for candidate in plan["candidates"]}
        self.assertTrue(by_id["native-full-logsoftmax"]["runnable"])
        self.assertFalse(by_id["streaming-reduction-t64-v4096"]["runnable"])
        self.assertEqual(
            by_id["streaming-reduction-t64-v4096"]["capability_status"],
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
