"""Sparse, uncertainty-aware stage cost modeling for NPU inference graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

from inference_autopilot.calibration.models import (
    canonical_sha256,
    expect_keys,
    require_digest,
    require_id,
    require_object,
)


_PRIMITIVES = {"generate", "score", "reward", "reduce", "select"}
_RESOURCE_ROLES = {"base_model", "proposal_model", "cpu"}


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _finite_float(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{context} must be {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class StageLatencyMeasurement:
    """One repeated measurement inside a compiler/runtime shape bucket."""

    shape_tokens: int
    token_extent: int
    elapsed_seconds: float
    repetitions: int
    stddev_seconds: float

    def __post_init__(self) -> None:
        _positive_int(self.shape_tokens, "measurement shape_tokens")
        _non_negative_int(self.token_extent, "measurement token_extent")
        if self.elapsed_seconds <= 0 or not isfinite(self.elapsed_seconds):
            raise ValueError("measurement elapsed_seconds must be finite and positive")
        _positive_int(self.repetitions, "measurement repetitions")
        if self.stddev_seconds < 0 or not isfinite(self.stddev_seconds):
            raise ValueError("measurement stddev_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_tokens": self.shape_tokens,
            "token_extent": self.token_extent,
            "elapsed_seconds": self.elapsed_seconds,
            "repetitions": self.repetitions,
            "stddev_seconds": self.stddev_seconds,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageLatencyMeasurement":
        expect_keys(
            raw,
            {
                "shape_tokens",
                "token_extent",
                "elapsed_seconds",
                "repetitions",
                "stddev_seconds",
            },
            "stage latency measurement",
        )
        return cls(
            shape_tokens=_positive_int(raw["shape_tokens"], "measurement shape_tokens"),
            token_extent=_non_negative_int(
                raw["token_extent"], "measurement token_extent"
            ),
            elapsed_seconds=_finite_float(
                raw["elapsed_seconds"], "measurement elapsed_seconds", minimum=0.0
            ),
            repetitions=_positive_int(raw["repetitions"], "measurement repetitions"),
            stddev_seconds=_finite_float(
                raw["stddev_seconds"], "measurement stddev_seconds", minimum=0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class StageBucketCalibration:
    """Measurements sharing one batch size and one discrete shape interval."""

    bucket_id: str
    batch_size: int
    minimum_shape_tokens: int
    maximum_shape_tokens: int
    measurements: tuple[StageLatencyMeasurement, ...]

    def __post_init__(self) -> None:
        require_id(self.bucket_id, "bucket_id")
        _positive_int(self.batch_size, "bucket batch_size")
        _positive_int(self.minimum_shape_tokens, "bucket minimum_shape_tokens")
        _positive_int(self.maximum_shape_tokens, "bucket maximum_shape_tokens")
        if self.minimum_shape_tokens > self.maximum_shape_tokens:
            raise ValueError("bucket shape interval is empty")
        if len(self.measurements) < 2:
            raise ValueError("bucket calibration requires at least two measurements")
        extents = {measurement.token_extent for measurement in self.measurements}
        if len(extents) < 2:
            raise ValueError("bucket calibration requires two distinct token extents")
        for measurement in self.measurements:
            if not (
                self.minimum_shape_tokens
                <= measurement.shape_tokens
                <= self.maximum_shape_tokens
            ):
                raise ValueError(
                    f"measurement shape is outside bucket {self.bucket_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "batch_size": self.batch_size,
            "minimum_shape_tokens": self.minimum_shape_tokens,
            "maximum_shape_tokens": self.maximum_shape_tokens,
            "measurements": [measurement.to_dict() for measurement in self.measurements],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageBucketCalibration":
        expect_keys(
            raw,
            {
                "bucket_id",
                "batch_size",
                "minimum_shape_tokens",
                "maximum_shape_tokens",
                "measurements",
            },
            "stage bucket calibration",
        )
        measurements = raw["measurements"]
        if not isinstance(measurements, list) or any(
            not isinstance(item, Mapping) for item in measurements
        ):
            raise ValueError("bucket measurements must be objects")
        return cls(
            bucket_id=str(raw["bucket_id"]),
            batch_size=_positive_int(raw["batch_size"], "bucket batch_size"),
            minimum_shape_tokens=_positive_int(
                raw["minimum_shape_tokens"], "bucket minimum_shape_tokens"
            ),
            maximum_shape_tokens=_positive_int(
                raw["maximum_shape_tokens"], "bucket maximum_shape_tokens"
            ),
            measurements=tuple(
                StageLatencyMeasurement.from_dict(item) for item in measurements
            ),
        )


@dataclass(frozen=True, slots=True)
class StageCalibrationProfile:
    stage_id: str
    primitive: str
    resource_role: str
    shape_axis: str
    buckets: tuple[StageBucketCalibration, ...]

    def __post_init__(self) -> None:
        require_id(self.stage_id, "stage_id")
        if self.primitive not in _PRIMITIVES:
            raise ValueError(f"unsupported stage primitive: {self.primitive}")
        if self.resource_role not in _RESOURCE_ROLES:
            raise ValueError(f"unsupported stage resource role: {self.resource_role}")
        if not self.shape_axis:
            raise ValueError("stage shape_axis cannot be empty")
        if not self.buckets:
            raise ValueError("stage calibration requires at least one bucket")
        bucket_ids = [bucket.bucket_id for bucket in self.buckets]
        if len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError(f"stage {self.stage_id} has duplicate bucket ids")
        by_batch: dict[int, list[StageBucketCalibration]] = {}
        for bucket in self.buckets:
            by_batch.setdefault(bucket.batch_size, []).append(bucket)
        for batch_size, buckets in by_batch.items():
            ordered = sorted(buckets, key=lambda item: item.minimum_shape_tokens)
            for left, right in zip(ordered, ordered[1:]):
                if left.maximum_shape_tokens >= right.minimum_shape_tokens:
                    raise ValueError(
                        f"stage {self.stage_id} has overlapping batch-{batch_size} buckets"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "primitive": self.primitive,
            "resource_role": self.resource_role,
            "shape_axis": self.shape_axis,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageCalibrationProfile":
        expect_keys(
            raw,
            {"stage_id", "primitive", "resource_role", "shape_axis", "buckets"},
            "stage calibration profile",
        )
        buckets = raw["buckets"]
        if not isinstance(buckets, list) or any(
            not isinstance(item, Mapping) for item in buckets
        ):
            raise ValueError("stage calibration buckets must be objects")
        return cls(
            stage_id=str(raw["stage_id"]),
            primitive=str(raw["primitive"]),
            resource_role=str(raw["resource_role"]),
            shape_axis=str(raw["shape_axis"]),
            buckets=tuple(StageBucketCalibration.from_dict(item) for item in buckets),
        )


@dataclass(frozen=True, slots=True)
class NPUCalibrationCorpus:
    calibration_id: str
    environment_sha256: str
    model_set_sha256: str
    configuration_sha256: str
    relative_error_floor: float
    interval_multiplier: float
    stages: tuple[StageCalibrationProfile, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported NPU calibration corpus: {self.schema_version}")
        require_id(self.calibration_id, "calibration_id")
        for name in (
            "environment_sha256",
            "model_set_sha256",
            "configuration_sha256",
        ):
            require_digest(getattr(self, name), name)
        if not 0 < self.relative_error_floor < 1:
            raise ValueError("relative_error_floor must be between zero and one")
        if not isfinite(self.interval_multiplier) or self.interval_multiplier <= 0:
            raise ValueError("interval_multiplier must be finite and positive")
        if not self.stages:
            raise ValueError("NPU calibration corpus requires at least one stage")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("NPU calibration stage ids must be unique")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "environment_sha256": self.environment_sha256,
            "model_set_sha256": self.model_set_sha256,
            "configuration_sha256": self.configuration_sha256,
            "relative_error_floor": self.relative_error_floor,
            "interval_multiplier": self.interval_multiplier,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NPUCalibrationCorpus":
        expect_keys(
            raw,
            {
                "schema_version",
                "calibration_id",
                "environment_sha256",
                "model_set_sha256",
                "configuration_sha256",
                "relative_error_floor",
                "interval_multiplier",
                "stages",
            },
            "NPU calibration corpus",
        )
        stages = raw["stages"]
        if not isinstance(stages, list) or any(
            not isinstance(item, Mapping) for item in stages
        ):
            raise ValueError("NPU calibration stages must be objects")
        return cls(
            calibration_id=str(raw["calibration_id"]),
            environment_sha256=str(raw["environment_sha256"]),
            model_set_sha256=str(raw["model_set_sha256"]),
            configuration_sha256=str(raw["configuration_sha256"]),
            relative_error_floor=_finite_float(
                raw["relative_error_floor"], "relative_error_floor"
            ),
            interval_multiplier=_finite_float(
                raw["interval_multiplier"], "interval_multiplier"
            ),
            stages=tuple(StageCalibrationProfile.from_dict(item) for item in stages),
            schema_version=str(raw["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class StageBucketCostFit:
    stage_id: str
    bucket_id: str
    batch_size: int
    minimum_shape_tokens: int
    maximum_shape_tokens: int
    minimum_token_extent: int
    maximum_token_extent: int
    intercept_seconds: float
    seconds_per_token: float
    residual_rmse_seconds: float
    measurement_noise_seconds: float
    weighted_sample_count: int
    token_extent_mean: float
    token_extent_sxx: float

    def __post_init__(self) -> None:
        require_id(self.stage_id, "fit stage_id")
        require_id(self.bucket_id, "fit bucket_id")
        _positive_int(self.batch_size, "fit batch_size")
        _positive_int(self.minimum_shape_tokens, "fit minimum_shape_tokens")
        _positive_int(self.maximum_shape_tokens, "fit maximum_shape_tokens")
        _non_negative_int(self.minimum_token_extent, "fit minimum_token_extent")
        _non_negative_int(self.maximum_token_extent, "fit maximum_token_extent")
        _positive_int(self.weighted_sample_count, "fit weighted_sample_count")
        numeric = (
            self.intercept_seconds,
            self.seconds_per_token,
            self.residual_rmse_seconds,
            self.measurement_noise_seconds,
            self.token_extent_mean,
            self.token_extent_sxx,
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("bucket fit values must be finite")
        if self.seconds_per_token < 0:
            raise ValueError("bucket fit seconds_per_token must be non-negative")
        if min(
            self.residual_rmse_seconds,
            self.measurement_noise_seconds,
            self.token_extent_sxx,
        ) < 0:
            raise ValueError("bucket fit uncertainty values must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "bucket_id": self.bucket_id,
            "batch_size": self.batch_size,
            "minimum_shape_tokens": self.minimum_shape_tokens,
            "maximum_shape_tokens": self.maximum_shape_tokens,
            "minimum_token_extent": self.minimum_token_extent,
            "maximum_token_extent": self.maximum_token_extent,
            "intercept_seconds": self.intercept_seconds,
            "seconds_per_token": self.seconds_per_token,
            "residual_rmse_seconds": self.residual_rmse_seconds,
            "measurement_noise_seconds": self.measurement_noise_seconds,
            "weighted_sample_count": self.weighted_sample_count,
            "token_extent_mean": self.token_extent_mean,
            "token_extent_sxx": self.token_extent_sxx,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageBucketCostFit":
        keys = {
            "stage_id",
            "bucket_id",
            "batch_size",
            "minimum_shape_tokens",
            "maximum_shape_tokens",
            "minimum_token_extent",
            "maximum_token_extent",
            "intercept_seconds",
            "seconds_per_token",
            "residual_rmse_seconds",
            "measurement_noise_seconds",
            "weighted_sample_count",
            "token_extent_mean",
            "token_extent_sxx",
        }
        expect_keys(raw, keys, "stage bucket cost fit")
        integer_names = (
            "batch_size",
            "minimum_shape_tokens",
            "maximum_shape_tokens",
            "minimum_token_extent",
            "maximum_token_extent",
            "weighted_sample_count",
        )
        values: dict[str, Any] = {
            name: _non_negative_int(raw[name], f"fit {name}") for name in integer_names
        }
        positive_names = (
            "batch_size",
            "minimum_shape_tokens",
            "maximum_shape_tokens",
            "weighted_sample_count",
        )
        for name in positive_names:
            _positive_int(values[name], f"fit {name}")
        for name in keys - set(integer_names) - {"stage_id", "bucket_id"}:
            values[name] = _finite_float(raw[name], f"fit {name}")
        return cls(
            stage_id=str(raw["stage_id"]),
            bucket_id=str(raw["bucket_id"]),
            **values,
        )


@dataclass(frozen=True, slots=True)
class NPUStageCostModel:
    calibration_sha256: str
    environment_sha256: str
    model_set_sha256: str
    configuration_sha256: str
    relative_error_floor: float
    interval_multiplier: float
    stage_metadata: Mapping[str, Mapping[str, str]]
    fits: tuple[StageBucketCostFit, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported NPU stage cost model: {self.schema_version}")
        for name in (
            "calibration_sha256",
            "environment_sha256",
            "model_set_sha256",
            "configuration_sha256",
        ):
            require_digest(getattr(self, name), name)
        if not 0 < self.relative_error_floor < 1:
            raise ValueError("model relative_error_floor must be between zero and one")
        if not isfinite(self.interval_multiplier) or self.interval_multiplier <= 0:
            raise ValueError("model interval_multiplier must be finite and positive")
        if not self.fits:
            raise ValueError("NPU stage cost model requires fitted buckets")
        metadata: dict[str, dict[str, str]] = {}
        for stage_id, raw in self.stage_metadata.items():
            require_id(stage_id, "model stage id")
            value = require_object(raw, f"stage metadata {stage_id}")
            expect_keys(
                value,
                {"primitive", "resource_role", "shape_axis"},
                f"stage metadata {stage_id}",
            )
            if value["primitive"] not in _PRIMITIVES:
                raise ValueError(f"unsupported stage primitive: {value['primitive']}")
            if value["resource_role"] not in _RESOURCE_ROLES:
                raise ValueError(
                    f"unsupported stage resource role: {value['resource_role']}"
                )
            if not all(isinstance(item, str) and item for item in value.values()):
                raise ValueError("stage metadata values must be non-empty strings")
            metadata[stage_id] = {str(key): str(item) for key, item in value.items()}
        object.__setattr__(self, "stage_metadata", dict(sorted(metadata.items())))
        if {fit.stage_id for fit in self.fits} != set(self.stage_metadata):
            raise ValueError("model fits and stage metadata disagree")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_sha256": self.calibration_sha256,
            "environment_sha256": self.environment_sha256,
            "model_set_sha256": self.model_set_sha256,
            "configuration_sha256": self.configuration_sha256,
            "relative_error_floor": self.relative_error_floor,
            "interval_multiplier": self.interval_multiplier,
            "stage_metadata": {
                stage_id: dict(metadata)
                for stage_id, metadata in sorted(self.stage_metadata.items())
            },
            "fits": [fit.to_dict() for fit in self.fits],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NPUStageCostModel":
        expect_keys(
            raw,
            {
                "schema_version",
                "calibration_sha256",
                "environment_sha256",
                "model_set_sha256",
                "configuration_sha256",
                "relative_error_floor",
                "interval_multiplier",
                "stage_metadata",
                "fits",
            },
            "NPU stage cost model",
        )
        metadata = require_object(raw["stage_metadata"], "stage metadata")
        fits = raw["fits"]
        if not isinstance(fits, list) or any(not isinstance(item, Mapping) for item in fits):
            raise ValueError("NPU stage cost model fits must be objects")
        return cls(
            calibration_sha256=str(raw["calibration_sha256"]),
            environment_sha256=str(raw["environment_sha256"]),
            model_set_sha256=str(raw["model_set_sha256"]),
            configuration_sha256=str(raw["configuration_sha256"]),
            relative_error_floor=_finite_float(
                raw["relative_error_floor"], "model relative_error_floor"
            ),
            interval_multiplier=_finite_float(
                raw["interval_multiplier"], "model interval_multiplier"
            ),
            stage_metadata={
                str(stage_id): require_object(value, f"stage metadata {stage_id}")
                for stage_id, value in metadata.items()
            },
            fits=tuple(StageBucketCostFit.from_dict(item) for item in fits),
            schema_version=str(raw["schema_version"]),
        )


def _fit_bucket(stage_id: str, bucket: StageBucketCalibration) -> StageBucketCostFit:
    points = tuple(bucket.measurements)
    total_weight = sum(point.repetitions for point in points)
    x_mean = sum(point.token_extent * point.repetitions for point in points) / total_weight
    y_mean = sum(point.elapsed_seconds * point.repetitions for point in points) / total_weight
    sxx = sum(
        point.repetitions * (point.token_extent - x_mean) ** 2 for point in points
    )
    if sxx <= 0:
        raise ValueError(f"bucket {bucket.bucket_id} has no token-extent variance")
    slope = sum(
        point.repetitions
        * (point.token_extent - x_mean)
        * (point.elapsed_seconds - y_mean)
        for point in points
    ) / sxx
    if slope < 0:
        raise ValueError(
            f"bucket {bucket.bucket_id} produced a negative latency slope"
        )
    intercept = y_mean - slope * x_mean
    weighted_sse = sum(
        point.repetitions
        * (point.elapsed_seconds - (intercept + slope * point.token_extent)) ** 2
        for point in points
    )
    residual_rmse = sqrt(weighted_sse / max(total_weight - 2, 1))
    measurement_noise = max(point.stddev_seconds for point in points)
    extents = [point.token_extent for point in points]
    return StageBucketCostFit(
        stage_id=stage_id,
        bucket_id=bucket.bucket_id,
        batch_size=bucket.batch_size,
        minimum_shape_tokens=bucket.minimum_shape_tokens,
        maximum_shape_tokens=bucket.maximum_shape_tokens,
        minimum_token_extent=min(extents),
        maximum_token_extent=max(extents),
        intercept_seconds=intercept,
        seconds_per_token=slope,
        residual_rmse_seconds=residual_rmse,
        measurement_noise_seconds=measurement_noise,
        weighted_sample_count=total_weight,
        token_extent_mean=x_mean,
        token_extent_sxx=sxx,
    )


def fit_npu_stage_cost_model(corpus: NPUCalibrationCorpus) -> NPUStageCostModel:
    """Fit independent affine models inside observed NPU shape buckets."""

    fits = tuple(
        _fit_bucket(stage.stage_id, bucket)
        for stage in sorted(corpus.stages, key=lambda item: item.stage_id)
        for bucket in sorted(
            stage.buckets,
            key=lambda item: (
                item.batch_size,
                item.minimum_shape_tokens,
                item.maximum_shape_tokens,
                item.bucket_id,
            ),
        )
    )
    metadata = {
        stage.stage_id: {
            "primitive": stage.primitive,
            "resource_role": stage.resource_role,
            "shape_axis": stage.shape_axis,
        }
        for stage in corpus.stages
    }
    return NPUStageCostModel(
        calibration_sha256=corpus.sha256,
        environment_sha256=corpus.environment_sha256,
        model_set_sha256=corpus.model_set_sha256,
        configuration_sha256=corpus.configuration_sha256,
        relative_error_floor=corpus.relative_error_floor,
        interval_multiplier=corpus.interval_multiplier,
        stage_metadata=metadata,
        fits=fits,
    )


@dataclass(frozen=True, slots=True)
class StageCostQuery:
    stage_id: str
    batch_size: int
    shape_tokens: int
    token_extent: int
    invocations: int = 1

    def __post_init__(self) -> None:
        require_id(self.stage_id, "query stage_id")
        _positive_int(self.batch_size, "query batch_size")
        _positive_int(self.shape_tokens, "query shape_tokens")
        _non_negative_int(self.token_extent, "query token_extent")
        _positive_int(self.invocations, "query invocations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "batch_size": self.batch_size,
            "shape_tokens": self.shape_tokens,
            "token_extent": self.token_extent,
            "invocations": self.invocations,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageCostQuery":
        expect_keys(
            raw,
            {"stage_id", "batch_size", "shape_tokens", "token_extent", "invocations"},
            "stage cost query",
        )
        return cls(
            stage_id=str(raw["stage_id"]),
            batch_size=_positive_int(raw["batch_size"], "query batch_size"),
            shape_tokens=_positive_int(raw["shape_tokens"], "query shape_tokens"),
            token_extent=_non_negative_int(raw["token_extent"], "query token_extent"),
            invocations=_positive_int(raw["invocations"], "query invocations"),
        )


@dataclass(frozen=True, slots=True)
class StageCostPrediction:
    query: StageCostQuery
    status: str
    reason: str
    bucket_id: str | None
    predicted_seconds: float | None
    lower_seconds: float | None
    upper_seconds: float | None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "abstain"}:
            raise ValueError("prediction status must be supported or abstain")
        if not self.reason:
            raise ValueError("prediction reason cannot be empty")
        values = (self.predicted_seconds, self.lower_seconds, self.upper_seconds)
        if self.predicted_seconds is None:
            if any(value is not None for value in values):
                raise ValueError("empty predictions cannot contain interval values")
        else:
            if any(value is None or not isfinite(value) or value < 0 for value in values):
                raise ValueError("prediction interval must be finite and non-negative")
            assert self.lower_seconds is not None and self.upper_seconds is not None
            if not self.lower_seconds <= self.predicted_seconds <= self.upper_seconds:
                raise ValueError("prediction point must lie inside its interval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "bucket_id": self.bucket_id,
            "predicted_seconds": self.predicted_seconds,
            "lower_seconds": self.lower_seconds,
            "upper_seconds": self.upper_seconds,
        }


def predict_stage_cost(
    model: NPUStageCostModel, query: StageCostQuery
) -> StageCostPrediction:
    """Predict one stage call, abstaining outside measured shape support."""

    if query.stage_id not in model.stage_metadata:
        return StageCostPrediction(
            query, "abstain", "unknown_stage", None, None, None, None
        )
    exact_batch = tuple(
        fit
        for fit in model.fits
        if fit.stage_id == query.stage_id and fit.batch_size == query.batch_size
    )
    if not exact_batch:
        return StageCostPrediction(
            query, "abstain", "unsupported_batch_size", None, None, None, None
        )
    fit = next(
        (
            item
            for item in exact_batch
            if item.minimum_shape_tokens
            <= query.shape_tokens
            <= item.maximum_shape_tokens
        ),
        None,
    )
    if fit is None:
        return StageCostPrediction(
            query, "abstain", "shape_outside_calibration", None, None, None, None
        )

    point = fit.intercept_seconds + fit.seconds_per_token * query.token_extent
    if point < 0:
        return StageCostPrediction(
            query,
            "abstain",
            "non_physical_extrapolation",
            fit.bucket_id,
            None,
            None,
            None,
        )
    supported_extent = (
        fit.minimum_token_extent <= query.token_extent <= fit.maximum_token_extent
    )
    extent_span = max(fit.maximum_token_extent - fit.minimum_token_extent, 1)
    extrapolation_distance = max(
        fit.minimum_token_extent - query.token_extent,
        query.token_extent - fit.maximum_token_extent,
        0,
    )
    extrapolation_factor = 1.0 + extrapolation_distance / extent_span
    leverage = 1.0 + 1.0 / fit.weighted_sample_count
    if fit.token_extent_sxx > 0:
        leverage += (
            (query.token_extent - fit.token_extent_mean) ** 2 / fit.token_extent_sxx
        )
    base_uncertainty = max(
        fit.residual_rmse_seconds,
        fit.measurement_noise_seconds,
        abs(point) * model.relative_error_floor,
    )
    half_width = (
        model.interval_multiplier
        * base_uncertainty
        * sqrt(leverage)
        * extrapolation_factor
    )
    return StageCostPrediction(
        query=query,
        status="supported" if supported_extent else "abstain",
        reason=(
            "inside_calibrated_bucket"
            if supported_extent
            else "token_extent_outside_calibration"
        ),
        bucket_id=fit.bucket_id,
        predicted_seconds=point * query.invocations,
        lower_seconds=max(0.0, point - half_width) * query.invocations,
        upper_seconds=(point + half_width) * query.invocations,
    )


@dataclass(frozen=True, slots=True)
class GraphCostPrediction:
    status: str
    composition: str
    stages: tuple[StageCostPrediction, ...]
    predicted_seconds: float | None
    lower_seconds: float | None
    upper_seconds: float | None
    model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "composition": self.composition,
            "stages": [stage.to_dict() for stage in self.stages],
            "predicted_seconds": self.predicted_seconds,
            "lower_seconds": self.lower_seconds,
            "upper_seconds": self.upper_seconds,
            "model_sha256": self.model_sha256,
        }
        payload["prediction_sha256"] = canonical_sha256(payload)
        return payload


def predict_serial_graph_cost(
    model: NPUStageCostModel, queries: Sequence[StageCostQuery]
) -> GraphCostPrediction:
    """Compose stage costs as an explicit serial upper-bound model."""

    if not queries:
        raise ValueError("graph cost prediction requires at least one stage query")
    predictions = tuple(predict_stage_cost(model, query) for query in queries)
    complete = all(prediction.predicted_seconds is not None for prediction in predictions)
    supported = complete and all(
        prediction.status == "supported" for prediction in predictions
    )
    if not complete:
        point = lower = upper = None
    else:
        point = sum(float(prediction.predicted_seconds) for prediction in predictions)
        lower = sum(float(prediction.lower_seconds) for prediction in predictions)
        upper = sum(float(prediction.upper_seconds) for prediction in predictions)
    return GraphCostPrediction(
        status="supported" if supported else "abstain",
        composition="serial_upper_bound",
        stages=predictions,
        predicted_seconds=point,
        lower_seconds=lower,
        upper_seconds=upper,
        model_sha256=model.sha256,
    )


def stage_queries_from_dict(raw: Mapping[str, Any]) -> tuple[StageCostQuery, ...]:
    expect_keys(raw, {"schema_version", "stages"}, "NPU graph cost query")
    if raw["schema_version"] != "1.0":
        raise ValueError(f"unsupported NPU graph cost query: {raw['schema_version']}")
    stages = raw["stages"]
    if not isinstance(stages, list) or any(not isinstance(item, Mapping) for item in stages):
        raise ValueError("NPU graph cost query stages must be objects")
    return tuple(StageCostQuery.from_dict(item) for item in stages)
