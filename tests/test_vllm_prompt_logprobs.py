from __future__ import annotations

import unittest

from inference_autopilot.runtime.vllm_prompt_logprobs import (
    TiledPromptLogprobsConfig,
    maybe_tiled_selected_prompt_logprobs,
)


class TiledPromptLogprobsConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_and_use_measured_pareto_tile(self) -> None:
        config = TiledPromptLogprobsConfig.from_environment({})

        self.assertFalse(config.enabled)
        self.assertEqual(config.token_chunk_size, 512)
        self.assertEqual(config.vocab_tile_size, 32768)

    def test_environment_enables_and_overrides_tiles(self) -> None:
        config = TiledPromptLogprobsConfig.from_environment(
            {
                "INFERENCE_AUTOPILOT_TILED_PROMPT_LOGPROBS": "1",
                "INFERENCE_AUTOPILOT_SCORE_TOKEN_CHUNK_SIZE": "256",
                "INFERENCE_AUTOPILOT_SCORE_VOCAB_TILE_SIZE": "16384",
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.token_chunk_size, 256)
        self.assertEqual(config.vocab_tile_size, 16384)

    def test_invalid_environment_fails_before_torch_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            TiledPromptLogprobsConfig.from_environment(
                {"INFERENCE_AUTOPILOT_TILED_PROMPT_LOGPROBS": "yes"}
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            TiledPromptLogprobsConfig.from_environment(
                {"INFERENCE_AUTOPILOT_SCORE_TOKEN_CHUNK_SIZE": "0"}
            )

    def test_disabled_or_topk_requests_do_not_import_torch(self) -> None:
        self.assertIsNone(
            maybe_tiled_selected_prompt_logprobs(
                object(),
                object(),
                0,
                environment={"INFERENCE_AUTOPILOT_TILED_PROMPT_LOGPROBS": "0"},
            )
        )
        self.assertIsNone(
            maybe_tiled_selected_prompt_logprobs(
                object(),
                object(),
                2,
                environment={"INFERENCE_AUTOPILOT_TILED_PROMPT_LOGPROBS": "1"},
            )
        )


if __name__ == "__main__":
    unittest.main()
