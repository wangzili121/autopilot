# State-Isolated Engine Lifecycle Harness

## Scope

The lifecycle layer lowers the cost of collecting trustworthy autotuning
evidence. It does not change Conditional IS, choose a winning configuration,
or turn initialization time into serving throughput. It compiles a frozen
calibration plan plus measured engine fingerprints into a process-lifecycle
schedule that preserves experiment order while minimizing compatible starts.

This is distinct from vLLM's current benchmark autotuner. The upstream
`benchmarks/auto_tune` script grid-searches `max_num_seqs` and
`max_num_batched_tokens`, restarts one server for each combination, and clears
the prefix cache between rate probes. Inference Autopilot instead coordinates
multiple algorithm roles and configurations, binds every action to formal ABBA
evidence, and treats all non-KV algorithm and telemetry state as part of the
reset contract. It uses backend reset mechanisms where available; it does not
reimplement vLLM scheduling or KV-cache internals.

References:

- [vLLM automated server parameter tuning](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md)
- [vLLM autotune runner and prefix-cache reset](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/auto_tune.sh)
- [vLLM scheduler configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)

## Planner Contract

The planner compares three strategies from observed per-run engine evidence:

- `isolated_process`: reproduce the current one-process-per-run harness;
- `role_sticky`: keep one engine per algorithm role and replace it only when
  that role's compatibility fingerprint changes;
- `fully_resident`: preload every unique fingerprint and bind the required
  engine before each epoch, subject to a frozen memory cap.

Startup projections use the measured isolated total as the exact baseline and
the median observed initialization time for each reused fingerprint. The
planner never reorders runs. It rejects a strategy when aggregate resident
memory crosses the declared cap and emits every start, stop, bind, reuse, reset,
and post-measurement action as a content-addressed schedule.

Commands:

```bash
inference-autopilot plan-engine-lifecycle \
  engine-lifecycle.json calibration-plan.json harness-cost.json \
  --output engine-lifecycle-plan.json
inference-autopilot audit-engine-lifecycle engine-lifecycle-plan.json
```

## Epoch Isolation

Reusing a process is allowed only behind an explicit epoch boundary. The
current mandatory contract is:

1. assert that no requests are running;
2. reset prefix-cache state;
3. rebuild Conditional IS continuous batchers;
4. rebuild reward/scoring caches;
5. reset backend metric snapshots and request IDs;
6. reseed the workload;
7. synchronize the device before measurement.

The worker implementation must report each reset acknowledgement. Merely
calling a prefix-cache endpoint is insufficient because the inference-scaling
algorithm owns queues, candidate objects, reward state, and random streams that
vLLM does not know about.

The generic `LifecycleDriver` protocol now executes the planned state machine
without importing vLLM or Conditional IS. A backend plugin supplies engine
start/stop, epoch reset, and measurement callbacks. The executor refuses to run
unless `validation_only=true`, stops all active engines on any missing reset
acknowledgement, and emits a content-addressed receipt containing native-result,
observation, and exact output-token digests for every epoch. The independent
audit compares every acknowledged action, reset name, engine binding, run ID,
seed, and schedule digest back to the frozen lifecycle plan; rehashing an edited
receipt cannot make it conformant.

```bash
inference-autopilot audit-lifecycle-execution \
  engine-lifecycle-plan.json lifecycle-execution-receipt.json
```

The remaining implementation boundary is deliberately narrow: a Chang driver
must keep raw base/proposal engines alive while constructing fresh continuous
batchers and score caches per epoch, invoke the backend prefix-cache reset, and
return the mandatory acknowledgements. This adapter remains separate from the
generic executor so another inference-scaling algorithm can provide a different
reset contract without changing the scheduler.

## Validation Gate

The planner always emits `status=validation_required` and
`formal_execution_eligible=false`. A separate validation campaign compares the
old isolated harness against role-sticky execution on fresh ABBA seeds. It must
show exact output token IDs per seed, zero cross-epoch cache entries, fresh
metric deltas, preserved order, and throughput effects inside a separately
frozen replay-noise envelope.

Only after this gate passes should the lifecycle executor feed observations to
formal calibration. Until then, the projected `78.72%` startup reduction is a
measured-cost engineering target, not a reported serving speedup.
