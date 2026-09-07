# AIC-NPU Stage Cost Model

Date: 2026-09-07

## Purpose

The first AIC-NPU component fits a separate latency response inside each
observed stage, batch-size, and shape bucket. It transfers two ideas from the
2026 literature:

- LENS: commercial NPU latency can be discontinuous at compiler/runtime bucket
  boundaries, so a global smooth interpolator is unsafe;
- OmniPilot: a deployment advisor should abstain outside measured support
  instead of turning extrapolation into a confident recommendation.

Each bucket uses a weighted affine model:

```text
latency = fixed stage cost + token_extent * seconds_per_token
```

At least two distinct token extents are required. Repeated measurements supply
weights and observed noise. Prediction intervals use the larger of regression
residual, measured standard deviation, and a configured relative-error floor.

This is deliberately narrower than NVIDIA AIConfigurator. It establishes a
strict calibration contract before adding resource floors, cross-stage overlap,
historical residual transfer, and decision-critical acquisition.

## Safety Boundary

A prediction is `supported` only when all of these match measured support:

- stage identity;
- exact batch size;
- shape bucket;
- calibrated token-extent interval.

Unknown batch sizes and shapes have no point estimate. Token-extent
extrapolation returns a widened diagnostic estimate but status `abstain`.
Graph composition also abstains if any stage does. The current composition is
explicitly named `serial_upper_bound`; it does not claim to model engine
overlap yet.

## Telemetry Boundary

The chang pressure runner now emits two deliberately different timelines:

- `model_call_timeline` records algorithm calls around batching/admission. Its
  latency includes queueing and is used for graph dependencies and externality;
- `engine_batch_timeline` records the calls that reach the backend after
  continuous batching. It includes actual sequence count plus prefix,
  continuation/generation extent, and total-shape distributions.

Neither timeline is labeled kernel time. Backend-call wall time can still
contain host launch, scheduler, cache, and synchronization overhead. Initial
bucket fitting therefore uses isolated runs and repeated measurements; loaded
runs train a separate queueing/residual model. This separation is required
before importing historical artifacts into the cost database.

## Commands

```bash
inference-autopilot fit-npu-cost-model \
  examples/npu-stage-calibration.example.json \
  --output npu-stage-cost-model.json

inference-autopilot predict-npu-graph-cost \
  npu-stage-cost-model.json \
  examples/npu-graph-cost-query.example.json \
  --output npu-graph-cost-prediction.json
```

The example numbers are synthetic. Formal device data must bind the actual
environment, model set, and effective configuration digests.

## Next Model Extensions

1. Add a strict importer from new `engine_batch_timeline` events and reject old
   outer-only traces as primitive calibration evidence.
2. Detect rather than assume bucket boundaries on vLLM-Ascend paths.
3. Add LLMVisor-style resource features for memory traffic, AICore work, host
   launch, communication,
   and KV capacity.
4. Learn a residual correction for concurrent base/proposal execution.
5. Calibrate intervals with leave-one-regime-out residuals.
6. Feed interval width and final decision sensitivity to experiment
   acquisition.
