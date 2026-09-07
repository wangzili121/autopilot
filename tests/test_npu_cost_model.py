from __future__ import annotations

from dataclasses import replace
import unittest

from inference_autopilot.npu_cost_model import (
    NPUCalibrationCorpus,
    NPUStageCostModel,
    StageBucketCalibration,
    StageCalibrationProfile,
    StageCostQuery,
    StageLatencyMeasurement,
    fit_npu_stage_cost_model,
    predict_serial_graph_cost,
    predict_stage_cost,
)


def _measurement(shape: int, extent: int, elapsed: float) -> StageLatencyMeasurement:
    return StageLatencyMeasurement(
        shape_tokens=shape,
        token_extent=extent,
        elapsed_seconds=elapsed,
        repetitions=4,
        stddev_seconds=0.01,
    )


def _corpus() -> NPUCalibrationCorpus:
    return NPUCalibrationCorpus(
        calibration_id="ascend-stage-smoke",
        environment_sha256="a" * 64,
        model_set_sha256="b" * 64,
        configuration_sha256="c" * 64,
        relative_error_floor=0.03,
        interval_multiplier=2.0,
        stages=(
            StageCalibrationProfile(
                stage_id="candidate_generate",
                primitive="generate",
                resource_role="base_model",
                shape_axis="active_context_tokens",
                buckets=(
                    StageBucketCalibration(
                        bucket_id="short-b8",
                        batch_size=8,
                        minimum_shape_tokens=1,
                        maximum_shape_tokens=512,
                        measurements=(
                            _measurement(128, 16, 1.8),
                            _measurement(128, 32, 2.6),
                            _measurement(256, 48, 3.4),
                        ),
                    ),
                ),
            ),
            StageCalibrationProfile(
                stage_id="target_score",
                primitive="score",
                resource_role="base_model",
                shape_axis="scored_sequence_tokens",
                buckets=(
                    StageBucketCalibration(
                        bucket_id="medium-b4",
                        batch_size=4,
                        minimum_shape_tokens=513,
                        maximum_shape_tokens=4096,
                        measurements=(
                            _measurement(1024, 128, 1.14),
                            _measurement(2048, 256, 1.78),
                        ),
                    ),
                ),
            ),
        ),
    )


class NPUCostModelTest(unittest.TestCase):
    def test_fit_recovers_affine_bucket_model_and_round_trips(self) -> None:
        corpus = _corpus()
        model = fit_npu_stage_cost_model(corpus)
        fit = next(item for item in model.fits if item.stage_id == "candidate_generate")

        self.assertAlmostEqual(fit.intercept_seconds, 1.0)
        self.assertAlmostEqual(fit.seconds_per_token, 0.05)
        self.assertEqual(fit.weighted_sample_count, 12)
        self.assertEqual(NPUStageCostModel.from_dict(model.to_dict()), model)
        self.assertEqual(model.calibration_sha256, corpus.sha256)

    def test_supported_prediction_has_nonzero_uncertainty_floor(self) -> None:
        model = fit_npu_stage_cost_model(_corpus())
        prediction = predict_stage_cost(
            model,
            StageCostQuery(
                stage_id="candidate_generate",
                batch_size=8,
                shape_tokens=200,
                token_extent=24,
                invocations=2,
            ),
        )

        self.assertEqual(prediction.status, "supported")
        self.assertEqual(prediction.reason, "inside_calibrated_bucket")
        self.assertAlmostEqual(prediction.predicted_seconds, 4.4)
        self.assertLess(prediction.lower_seconds, prediction.predicted_seconds)
        self.assertGreater(prediction.upper_seconds, prediction.predicted_seconds)

    def test_model_abstains_for_unseen_batch_shape_and_extent(self) -> None:
        model = fit_npu_stage_cost_model(_corpus())
        unseen_batch = predict_stage_cost(
            model, StageCostQuery("candidate_generate", 16, 200, 24)
        )
        unseen_shape = predict_stage_cost(
            model, StageCostQuery("candidate_generate", 8, 1024, 24)
        )
        unseen_extent = predict_stage_cost(
            model, StageCostQuery("candidate_generate", 8, 200, 96)
        )

        self.assertEqual(unseen_batch.reason, "unsupported_batch_size")
        self.assertIsNone(unseen_batch.predicted_seconds)
        self.assertEqual(unseen_shape.reason, "shape_outside_calibration")
        self.assertEqual(unseen_extent.reason, "token_extent_outside_calibration")
        self.assertIsNotNone(unseen_extent.predicted_seconds)
        self.assertEqual(unseen_extent.status, "abstain")

    def test_serial_composition_refuses_incomplete_graph(self) -> None:
        model = fit_npu_stage_cost_model(_corpus())
        supported = predict_serial_graph_cost(
            model,
            (
                StageCostQuery("candidate_generate", 8, 128, 16, 2),
                StageCostQuery("target_score", 4, 1024, 128, 1),
            ),
        )
        incomplete = predict_serial_graph_cost(
            model,
            (
                StageCostQuery("candidate_generate", 8, 128, 16, 2),
                StageCostQuery("target_score", 8, 1024, 128, 1),
            ),
        )

        self.assertEqual(supported.status, "supported")
        self.assertAlmostEqual(supported.predicted_seconds, 4.74)
        self.assertEqual(supported.composition, "serial_upper_bound")
        self.assertEqual(len(supported.to_dict()["prediction_sha256"]), 64)
        self.assertEqual(incomplete.status, "abstain")
        self.assertIsNone(incomplete.predicted_seconds)

    def test_overlapping_buckets_and_negative_slopes_fail_closed(self) -> None:
        stage = _corpus().stages[0]
        overlapping = replace(
            stage.buckets[0],
            bucket_id="overlap",
            minimum_shape_tokens=100,
            maximum_shape_tokens=1024,
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            replace(stage, buckets=(*stage.buckets, overlapping))

        decreasing = replace(
            stage.buckets[0],
            measurements=(
                _measurement(128, 16, 2.0),
                _measurement(128, 32, 1.0),
            ),
        )
        bad_corpus = replace(
            _corpus(), stages=(replace(stage, buckets=(decreasing,)),)
        )
        with self.assertRaisesRegex(ValueError, "negative latency slope"):
            fit_npu_stage_cost_model(bad_corpus)


if __name__ == "__main__":
    unittest.main()
