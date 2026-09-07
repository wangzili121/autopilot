# Exact Scoring Reduction Plan

Date: 2026-09-05

## Why It Is Separate

Trajectory algorithms ask for different exact statistics from the same model
logits. Conditional IS importance correction needs the log-probability of each
selected continuation token. Consilience additionally defines confidence from
the mean log-probability of the top-k candidates at every continuation
position. Self-certainty and entropy rewards need an entropy reduction.

The current implementations often obtain all of these by materializing a
full-vocabulary FP32 `log_softmax`. That is an implementation choice, not an
algorithm requirement. Treating the scoring reduction as an explicit stage
lets the tuner reason about memory and kernel capabilities without changing the
reward or importance-sampling semantics.

## Exact Streaming Contract

For each token position, a vocabulary-tiled pass can maintain:

- an online maximum and exponential sum for exact `logsumexp`;
- the selected token logit;
- a size-k heap for Consilience;
- an exponential-weighted logit sum for entropy.

After the pass, selected log-probability is `selected_logit - logsumexp`, the
Consilience statistic is `mean(top_k_logits) - logsumexp`, and entropy is
`logsumexp - E_p[logit]`. No approximation or reward change is required.
`streaming_score_statistics_reference` is the CPU semantic oracle for a future
NPU implementation and is tested against a naive full `log_softmax`.

## Measured Shape

The repeated MRV1 failure used 3,584 scoring positions and a 151,936-token
vocabulary. The planner computes:

```text
3584 * 151936 * 4 bytes = 2.0286 GiB
```

That is the FP32 reduction workspace alone and matches the runtime's rounded
2.03 GiB failed allocation. A `256 x 4096` tiled reduction needs roughly 4 MiB
of reduction workspace while retaining the existing FP16 logits. Fusing the LM
head later could also avoid materializing the full FP16 logits, but that is a
separate implementation and validation step.

The plan's memory budget means memory available when the reduction begins. Its
feasibility check therefore compares the incremental reduction workspace and
small result tensors against that budget. `estimated_peak_bytes` additionally
reports the already materialized input logits so total intermediate pressure
remains visible.

## Search Dimensions

The first reduction plan enumerates token chunk size and vocabulary tile size.
The larger deployment search should eventually include these only after the
corresponding capability is validated:

- reduction implementation and accumulation precision;
- token chunk and vocabulary tile sizes;
- scoring microbatch size for an external exact backend;
- colocated versus separate scoring placement;
- tensor-parallel vocabulary reduction;
- overlap/priority policy between generation and scoring.

The plan records both memory feasibility and runtime capability status. A
memory-feasible but unimplemented kernel remains `runnable=false`; this avoids
turning a theoretical design point into a fake benchmark candidate.

## Device Probe

The standalone probe compares the native full-logsoftmax path with an exact
PyTorch/NPU tiled prototype, without loading an LLM or enabling MRV2:

```bash
POSITIONS=512 \
TOKEN_CHUNK_SIZES=64,128,256 \
VOCAB_TILE_SIZES=2048,4096,8192 \
scripts/run_npu_score_reduction_probe.sh
```

It records numerical error against the native result, elapsed time, and peak
incremental NPU allocation for every tile pair. This prototype is a semantic
and search-space probe, not the final performance implementation. The next
implementation gate is a fused NPU operator or a runtime-native reduction that
beats full log-softmax while preserving the same outputs.

### First Ascend Results

The first probe used an Ascend 910B3 and the real failed shape of 3,584
positions by 151,936 vocabulary entries. Each timing is the median of three
runs after a warm-up.

| Required statistics | Implementation | Time | Incremental peak | Max error |
| --- | --- | ---: | ---: | ---: |
| selected log-probability | native full | 8.526 ms | 4,180.01 MiB | reference |
| selected log-probability | tiled 512 x 32K | 7.740 ms | 160.04 MiB | 1.91e-6 |
| selected plus top-k mean | native full | 31.039 ms | 4,180.03 MiB | reference |
| selected plus top-k mean | tiled 512 x 32K | 33.559 ms | 160.08 MiB | 2.86e-6 |

The selected-only prototype was 9.2% faster with 96.2% less incremental peak
allocation. Adding Consilience's top-k statistic made the tiled prototype 8.1%
slower than native, while retaining the same 96.2% peak reduction. This is a
kernel-only synthetic-logits benchmark, not yet an end-to-end model result.

Artifacts:

- `artifacts/npu4-score-reduction-p512-grid-20260905`
- `artifacts/npu4-score-reduction-p512-refine-20260905`
- `artifacts/npu4-score-reduction-p3584-grid-20260905`
- `artifacts/npu4-score-reduction-selected-p3584-20260905`
