from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from unittest import TestCase
from unittest.mock import patch

from scripts.run_chang_graph_profile import (
    _apply_profile_runtime_overrides,
    _argument_int,
    _build_stage_wavefront_plan,
    _flush_graph_metrics,
    _prepend_prompt_prefix,
    _prompt_prefix,
)


class _Engine:
    def do_log_stats(self) -> None:
        print("vllm graph table")


class _Backend:
    _engine = _Engine()


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.call = (text, add_special_tokens)
        return [11, 22, 33]


class GraphProfileWrapperTests(TestCase):
    def test_auto_stage_wavefront_uses_algorithm_and_runtime_shape(self) -> None:
        config = {
            "conditional_is": {"candidate_count": 8, "rollout_count": 3},
            "vllm": {"proposal": {"max_num_seqs": 768}},
        }

        plan, wait_seconds = _build_stage_wavefront_plan(
            config,
            {
                "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT": "auto",
                "INFERENCE_AUTOPILOT_PROPOSAL_GRAPH_CAPTURE_CEILING": "512",
                "INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT_MAX_WAIT_SECONDS": "0.02",
            },
            ["runner", "--requests", "96", "--workers", "96"],
        )

        assert plan is not None
        self.assertEqual(plan.selected.groups_per_wave, 16)
        self.assertEqual(plan.selected.sequences_per_wave, 384)
        self.assertEqual(wait_seconds, 0.02)

    def test_stage_wavefront_is_fail_closed_without_capture_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a graph capture ceiling"):
            _build_stage_wavefront_plan(
                {
                    "conditional_is": {"candidate_count": 8, "rollout_count": 3},
                    "vllm": {"proposal": {"max_num_seqs": 768}},
                },
                {"INFERENCE_AUTOPILOT_PROPOSAL_STAGE_WAVEFRONT": "auto"},
                ["runner"],
            )

    def test_argument_int_uses_default_only_when_flag_is_absent(self) -> None:
        self.assertEqual(_argument_int(["runner"], "--workers", 64), 64)
        self.assertEqual(
            _argument_int(["runner", "--workers", "96"], "--workers", 64),
            96,
        )
        with self.assertRaisesRegex(ValueError, "requires an integer"):
            _argument_int(["runner", "--workers"], "--workers", 64)

    def test_profile_runtime_override_changes_only_proposal_token_budget(self) -> None:
        config = {
            "vllm": {
                "base": {"max_num_batched_tokens": 32768},
                "proposal": {"max_num_batched_tokens": 131072},
            }
        }

        overrides = _apply_profile_runtime_overrides(
            config,
            {"INFERENCE_AUTOPILOT_PROPOSAL_MAX_NUM_BATCHED_TOKENS": "768"},
        )

        self.assertEqual(config["vllm"]["base"]["max_num_batched_tokens"], 32768)
        self.assertEqual(config["vllm"]["proposal"]["max_num_batched_tokens"], 768)
        self.assertEqual(overrides, {"proposal_max_num_batched_tokens": 768})

    def test_profile_runtime_override_rejects_nonpositive_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            _apply_profile_runtime_overrides(
                {},
                {"INFERENCE_AUTOPILOT_PROPOSAL_MAX_NUM_BATCHED_TOKENS": "0"},
            )

    def test_graph_stats_are_drained_between_stdout_markers(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"INFERENCE_AUTOPILOT_GRAPH_STATS_SETTLE_SECONDS": "0"},
            ),
            redirect_stdout(output),
        ):
            _flush_graph_metrics(_Backend(), "proposal")

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "[inference-autopilot] graph-stats-begin role=proposal",
                "vllm graph table",
                "[inference-autopilot] graph-stats-end role=proposal",
            ],
        )

    def test_negative_settle_window_is_rejected(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"INFERENCE_AUTOPILOT_GRAPH_STATS_SETTLE_SECONDS": "-1"},
            ),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                _flush_graph_metrics(_Backend(), "base")

    def test_prompt_prefix_has_exact_requested_token_count(self) -> None:
        tokenizer = _Tokenizer()

        self.assertEqual(_prompt_prefix(tokenizer, 0), [])
        self.assertEqual(
            _prompt_prefix(tokenizer, 8),
            [11, 22, 33, 11, 22, 33, 11, 22],
        )
        self.assertFalse(tokenizer.call[1])
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            _prompt_prefix(tokenizer, -1)

    def test_prepend_prompt_prefix_preserves_prompt_container_type(self) -> None:
        tuple_tokens = (3, 4)
        list_tokens = [3, 4]

        self.assertEqual(_prepend_prompt_prefix(tuple_tokens, [1, 2]), (1, 2, 3, 4))
        self.assertIsInstance(_prepend_prompt_prefix(tuple_tokens, [1, 2]), tuple)
        self.assertEqual(_prepend_prompt_prefix(list_tokens, [1, 2]), [1, 2, 3, 4])
        self.assertIsInstance(_prepend_prompt_prefix(list_tokens, [1, 2]), list)
        self.assertIs(_prepend_prompt_prefix(tuple_tokens, []), tuple_tokens)
        with self.assertRaisesRegex(TypeError, "must be a list or tuple"):
            _prepend_prompt_prefix("3 4", [1, 2])
