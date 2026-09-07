# 2026 Autotuning Survey and Revised Roadmap

Date: 2026-09-07

## Decision

AIConfigurator is part of the main plan, not a discarded alternative. The
product should not reimplement NVIDIA's open-source GPU tool under another
name. It should transfer the useful method to a gap that NVIDIA's tool does
not cover:

- Ascend NPU performance calibration;
- multi-model, multi-stage inference-scaling algorithms;
- algorithm-graph-aware feasibility and cost composition;
- sparse real-device correction and uncertainty-aware abstention;
- generation of static deployment settings and guarded runtime policies.

The working product name remains **Inference Autopilot**. Its model-based
configuration layer is referred to below as **AIC-NPU**.

## Survey Scope and Honesty

The first broad pass queried arXiv for 2026 work matching LLM serving,
inference, test-time scaling, rollout, automatic configuration, and
autotuning. It produced 879 unique metadata records through 2026-09-07. This
is a title-and-abstract corpus, not a claim that 879 papers were read in full.

The second pass read the method and evaluation sections of the most relevant
papers and inspected linked repositories where available. The table below is
the current high-signal set. It is a living survey: candidates enter the
implementation plan only after a source-level overlap audit against:

1. the exact vLLM and vLLM-Ascend versions used by the experiment;
2. current upstream vLLM/vLLM-Ascend, so an old-version patch is not claimed as
   new work;
3. Pie's runtime, drivers, and inferlets;
4. chang's complete repository, including experimental and disabled paths;
5. Muyuan merged code and open infrastructure MRs.

## What We Take from the Literature

| Work | Directly useful mechanism | Reported result | Decision for Autopilot |
| --- | --- | --- | --- |
| [AIConfigurator](https://arxiv.org/abs/2601.06288) ([code](https://github.com/ai-dynamo/aiconfigurator)) | Operator decomposition, calibrated performance DB, fast composition, Pareto filtering, launch generation | Up to 40% dense and 50% MoE gains; search under 30 seconds on average | Core architecture reference; extend to Ascend and inference-scaling graphs |
| [LENS](https://arxiv.org/abs/2606.18042) | Black-box NPU bucket model; two E2E measurements recover TTFT and TBT per bucket | 2.15% mean prediction error | First Ascend calibration model; validate whether each vLLM-Ascend path has the assumed bucket behavior |
| [Floor First](https://arxiv.org/abs/2607.05876) | Cheap five-resource lower/upper bounds before profiling | Analytical case study | Add a zero-device feasibility and bottleneck triage layer |
| [KernelSight-LM](https://arxiv.org/abs/2606.28565) | Kernel, communication, host-overhead, and event-scheduler composition | Target-measured throughput error 2.7% | Later fidelity upgrade; too large for the first predictor |
| [LLMVisor](https://arxiv.org/abs/2608.08382) | Roofline-guided, piecewise-linear per-request latency attribution over FLOPs and memory traffic | Up to 4.4x lower p99 relative error than token-count attribution | Add resource features to the NPU bucket model and separate co-batch externality from primitive service time |
| [OmniPilot](https://arxiv.org/abs/2607.01579) | Quantile model, conformal intervals, and OOD abstention | 6.2% throughput MAPE; 95% top-1 accuracy | Required confidence and refusal layer; do not extrapolate unsupported cells |
| [FleetSieve](https://arxiv.org/abs/2608.19659) | Profile points by expected effect on the final constrained decision | 5.4% mean profiling saving over random | Replace distance-only acquisition with decision-critical acquisition |
| [SCOOT](https://arxiv.org/abs/2408.04323) | Constrained BO, hidden-constraint random forest, parallel suggestions | Better SLO optimization and tuning efficiency | Mature baseline for the online optimizer |
| [SLO-Guard](https://arxiv.org/abs/2604.17627) | Crashes as observations, feasible-first exploration, TPE exploitation | More consistent fixed-budget search, but no better final best point than random | Borrow crash handling and repair; do not copy its weak headline objective |
| [AutoPipe](https://arxiv.org/abs/2603.18773) | Historical ranker plus online GP residual correction and early stopping | Comparable result with under 10% of online-HPO cost | Main transfer-learning design for new workload buckets |
| [FlowCompile](https://arxiv.org/abs/2605.13647) | Compile a structured workflow into a reusable Pareto configuration set | Up to 6.4x workflow speedup | Product artifact should be a policy bundle, not one magic config |
| [Simthesizer](https://arxiv.org/abs/2608.24650) | Composable dynamic serving graph plus an agent harness that lowers new mechanisms and validates simulator fidelity | 2.51% average throughput error for generated extensions; up to 284.96x simulation speed | Strong auto-harness reference; use a constrained graph DSL and differential trace validation rather than unconstrained code generation |
| [SliceScheduler](https://arxiv.org/abs/2608.15762) | Global operator mapping graph and incremental what-if simulation for placement | 1.10-2.29x token throughput under reported multi-tenant loads | Validates graph simulation as a runtime decision tool; operator migration is beyond the first single-node scope |
| [OpScale](https://arxiv.org/abs/2608.13499) | Operator-level profiling, provisioning, placement, and autoscaling | Up to 36.3% fewer GPUs or 44% more throughput | Later cluster target; first reuse its operator heterogeneity and feasibility decomposition |
| [Vertumnus](https://arxiv.org/abs/2609.04774) | Request routing and worker split/merge across context-parallel degrees using queue, cache, and GPU-time cost | Mean TTFT up to 28.1% lower; SLO attainment up to 13.3 points higher | Add context length and cache state to policy inputs; CP switching waits for multi-card hardware |
| [OServe](https://arxiv.org/abs/2602.12151) | Workload-aware heterogeneous deployment and migration as workload mix changes | Up to 2x, 1.5x average over reported baselines | Reference for slow-timescale policy switching and hysteresis |
| [MFS](https://arxiv.org/abs/2603.17456) | Stage-aware defer-and-promote scheduling for dependent communication flows | TTFT SLO attainment improved 1.2-2.4x | Relevant when Conditional-IS spans cards or remote engines; no benefit to the initial same-device path |
| [Autopoiesis](https://arxiv.org/abs/2604.07144) | Continuous LLM-driven serving-policy synthesis from runtime observations | Up to 53%, 34% average reported improvement | Research reference only for now; generated policies must pass offline replay, safety, and paired-regression gates before activation |
| [ConfigSpec](https://arxiv.org/abs/2604.09722) | Jointly model draft speed, target acceptance, power, draft model, quantization, and speculative length | Conflicting goodput/cost/energy optima | Template for proposal/base joint configuration and Pareto objectives |
| [P-PAS](https://arxiv.org/abs/2608.15171) ([code](https://github.com/TimoSaemann/ppas-vllm)) | Change token scheduling budget with live prefill pressure instead of fixing MBT | Maintains low E2E latency across load regimes | High-priority, small runtime mechanism after exact upstream audit |
| [TAPER](https://arxiv.org/abs/2605.06914) | Per-step branch admission using predicted externality and request slack | 1.77x over branch-off and 1.48x over eager, over 95% SLO attainment | High-upside policy if Conditional-IS branches can be exposed inside one scheduling domain |
| [SlidingServe](https://arxiv.org/abs/2606.05933) | Batch latency predictor, dynamic chunking, priority sorting, DP batch construction | Capacity up to 30%; SLO violations down 16-53% | Source of scheduling features; broader than first implementation |
| [MISA-T](https://arxiv.org/abs/2608.11152) | Session admission, workload KV caps, residency-time accounting | Rollout throughput +43.6/+53.3%; E2E +35.6% | Strong multi-replica rollout direction when enough cards are available |
| [TailSieve](https://arxiv.org/abs/2608.22788) | Partial-rollout tail detection, tail isolation, adaptive replica split | Up to 1.67x routing-only and 2.59x with specialized speculation | Later cluster policy; requires replicas and repeated-history workloads |
| [OUTLETS](https://arxiv.org/abs/2609.01068) | Reuse speculative-decoder representations for output-length prediction | Short-request P99 latency -34.8% | Useful only when a compatible speculative backbone is active |
| [Nitsum](https://arxiv.org/abs/2605.05467) | Runtime TP, PD split, scheduling, weight reuse, and KV migration | SLO goodput up to 5.3x | Long-term multi-card target, not an initial free-card experiment |
| [ReMP](https://arxiv.org/abs/2606.18741) | Low-downtime runtime TP/PP reconfiguration and KV migration | Most switches in 1-7 seconds | Long-term mechanism; high engineering cost and likely separate project |
| [PipeLive](https://arxiv.org/abs/2604.12171) | Live PP change with KV resizing and incremental patching | Reconfiguration under 10 ms in reported path | Long-term mechanism; not first Conditional-IS target |
| [GrowPage](https://arxiv.org/abs/2609.03494) | Per-request KV capacity becomes a runtime resource | Better performance-throughput trade-off | Watch closely; overlaps the KV area and needs a Muyuan/MARS conflict audit |
| [UniScale](https://arxiv.org/abs/2605.30898) | Contextual-bandit joint model routing and test-time-scaling control | Better quality-cost trade-off | Later semantic-policy layer, isolated from exact deployment tuning |
| [AERA](https://arxiv.org/abs/2608.27964) | Predict future value of more reasoning from checkpoint evidence | 92.61% vs 93.01% accuracy while using 95.99% fewer completion tokens | High potential but algorithm-changing; never mix with exact infra claims |
| [Ada-MK](https://arxiv.org/abs/2605.11581) | MLIR DAG search hoists MegaKernel decisions to compile time | +23.6% over TensorRT-LLM on L20 | Auto-compilation reference; Ascend port is a separate high-risk project |
| [Two-Stage GPU Kernel Tuner](https://arxiv.org/abs/2601.12698) | Agent first exposes a stable parameterized template, then a tuner searches it | Best cases above 3x on SGLang kernels | Preferred pattern for future Ascend kernel autotuning |
| [KForge](https://arxiv.org/abs/2606.02963) | Correctness loop plus profiler-guided cross-platform kernel generation | 5.13x geometric mean over eager/compile on Intel test set; +2.12% TRT-LLM E2E on B200 | Use its validation loop, not its agents as the first deliverable |
| [AutoPass](https://arxiv.org/abs/2606.20373) | Let the agent query compiler IR and runtime evidence | 1.043x/1.117x over LLVM O3 | Useful auto-compilation harness design; modest direct gain |
| [Magellan](https://arxiv.org/abs/2601.21096) | Evolve executable compiler heuristics against macrobenchmarks | Matches or exceeds expert policies on reported tasks | Later policy-synthesis track with strict sandbox and regression gates |
| [VibeServe](https://arxiv.org/abs/2605.06068) ([code](https://github.com/uw-syfi/vibe-serve)) | Outer system-design loop plus inner implementation/correctness/performance loop | Competitive on standard workloads; wins six non-standard cases | Future auto-harness; not allowed to bypass deterministic search and evidence contracts |

## Gap Audit as of 2026-09-07

### vLLM

Upstream vLLM has an `auto_tune` benchmark. Its documented core search is a
nested sweep over `max-num-seqs` and `max-num-batched-tokens`, with a preceding
safe memory-utilization search and latency/cache-hit constraints. This is not
the AIC-NPU design: it has no Conditional-IS graph, NPU calibration model,
historical residual transfer, uncertainty abstention, or joint static/runtime
policy compilation.

This statement is version-sensitive. Every mechanism patch still requires a
fresh audit of upstream issues, PRs, and the pinned vLLM-Ascend source.

### NVIDIA AIConfigurator

AIConfigurator is open source and already supports GPU performance databases,
ordinary aggregated/disaggregated serving, several backends, Pareto analysis,
and launch-file generation. Therefore these generic components are references,
not novelty claims. The unfilled boundary is Ascend plus algorithm stages and
their cross-engine coupling.

### Pie

Pie exposes forward passes, KV state, custom samplers, and application logic to
near-engine WebAssembly inferlets. That makes it a possible future backend for
Autopilot-generated policies. The current public repository surface does not
show an equivalent automatic configuration, NPU cost model, or
decision-critical calibration system. This must be confirmed by a source-level
audit before publication; programmability alone is not autotuning.

### Chang Repository

The local repository already contains asynchronous submission, internal
batching, replay, caches, bounded stopping, adaptive rollout allocation,
speculative experiments, and pruning/gating paths. They are baselines or
backend capabilities, not Autopilot contributions. The repository does not
contain an AIConfigurator-style graph cost model or deployment optimizer.

### Muyuan

Current infrastructure work covers MARS-style admission/offload/controller,
PIC-KV/CacheBlend, SWE-Pruner, and a meta-tool graph/compiler. Autopilot must
avoid reimplementing those data planes. It can consume their telemetry and
select among exposed mechanisms. AIC-NPU, constrained experiment acquisition,
and inference-scaling stage-aware configuration are currently distinct.

## Revised Product Architecture

```text
Algorithm adapter
  -> semantic inference graph and stage workload features
  -> hard capability and memory constraints

AIC-NPU predictor
  -> cheap resource floors
  -> sparse stage/bucket calibration database
  -> graph composition model
  -> historical ranker + environment residual model
  -> calibrated intervals and OOD abstention

Two-level telemetry
  -> outer algorithm-call trace for dependencies, waiting, and co-batch externality
  -> inner engine-batch trace for actual sequence count, token shape, cache state,
     graph bucket, and primitive service time

Decision-critical optimizer
  -> feasible-first candidate generation
  -> crash classifier and repair
  -> Pareto/SLO-aware acquisition
  -> paired real-device validation

Policy compiler
  -> static deployment manifest
  -> graph/capture plan
  -> guarded runtime policy table
  -> rollback configuration and evidence report
```

The output is not just a recommended `max_num_seqs`. It is a portable policy
bundle indexed by workload regime, for example:

```text
short/high-fanout -> static config A + graph plan A + runtime policy P1
medium/scoring-heavy -> static config B + graph plan B + runtime policy P2
unsupported/OOD -> conservative baseline + request for one calibration probe
```

## Search Space

### Exact static deployment space

- per-role `max_num_seqs` and token budgets;
- memory split and KV capacity;
- graph mode, capture sizes, and capture memory budget;
- model-runner/backend implementation where semantically equivalent;
- TP/DP/PP and base/proposal placement when cards permit;
- colocated versus isolated engines and role-sticky process pools;
- exact scoring backend and exact fused/operator implementation;
- scheduler flags, chunked-prefill controls, and prefix-cache controls.

### Exact runtime policy space

- stage-aware dynamic token budget, inspired by P-PAS;
- base-engine arbitration between short candidate generation and long target
  scoring;
- branch/wave admission under measured latency slack, inspired by TAPER;
- engine-pool/routing policy and admission limits;
- graph-plan selection by observed shape bucket;
- safe fallback thresholds and hysteresis.

### Semantic space, evaluated separately

- candidates, rollout count, block size, and proposal model;
- pruning/stopping/controller thresholds;
- approximate scoring, quantization, or reward surrogates;
- quality-cost policies inspired by UniScale and AERA.

The semantic space gets a quality Pareto frontier and separate cohort IDs. It
must never inflate an exact-infra speedup claim.

## Implementation Order

### P0: Build AIC-NPU on existing evidence

1. Add a stage/bucket calibration schema and importer for the experiments
   already collected.
2. Implement Floor-First resource bounds for memory, compute, host launch,
   communication, and KV capacity.
3. Fit a LENS-style piecewise stage model where the measured runtime exhibits
   bucket boundaries. Fall back to monotone local models where it does not.
4. Compose stage predictions through the Conditional-IS graph, including
   base/proposal overlap and engine startup cost.
5. Add quantile intervals, OOD detection, and abstention.
6. Evaluate top-k recommendation recall and leave-one-workload-regime-out error
   before spending more NPU time.

The existing `model_call_timeline` is an outer trace: it times an algorithm
call through the continuous-batching layer and therefore includes queueing and
possibly several engine batches. It must not be relabeled as kernel or
primitive latency. P0 adds a separate inner trace or an isolated micro-harness
at the actual engine-call boundary. The two traces train different models and
are recomposed only after each model passes its own held-out check.

This work initially needs no free NPU. It should immediately tell us where the
existing 2,400 legal candidates collapse into equivalent or dominated regions.

### P1: Upgrade experiment selection

1. Replace distance-only selection with expected reduction in the final
   Pareto/SLO decision gap, following FleetSieve.
2. Warm-start from historical rankings and fit an environment-specific residual
   model, following AutoPipe.
3. Treat OOM, initialization failure, timeout, and interference-invalid runs as
   different observations rather than one missing value.
4. Run paired candidates in parallel only on isolated free cards; otherwise use
   randomized sequential pairs with host-interference guards.
5. Stop when the conservative and optimistic policy decisions agree within the
   configured tolerance.

### P2: Implement one high-effect runtime mechanism

First candidate: **stage-pressure adaptive token budgeting**. Conditional-IS
creates unusually different base-engine work: short candidate decode and long
teacher-forced target scoring. A fixed token budget cannot be best for both.
The policy will expose a per-step budget function of queued/running prefill,
decode, score lengths, deadlines, and graph coverage. It is inspired by P-PAS
but is stage-aware and targeted at the two-role algorithm graph.

Go/no-go sequence:

1. audit current vLLM and pinned vLLM-Ascend for an equivalent dynamic policy;
2. replay the policy on captured scheduler traces;
3. run a synthetic scheduler test and exactness test;
4. run real short, medium, and scoring-heavy Conditional-IS paired experiments;
5. retain only if geometric-mean E2E throughput improves by at least 8% without
   a significant latency or quality regression.

Second candidate: **TAPER-style branch admission**. Proceed only if proposal
rollouts can be represented as deferrable branches inside a shared scheduling
domain while preserving prefix KV and output distribution. If the current API
keeps them as opaque independent generation calls, this is not a small patch;
the product should first expose branch state through a scheduler/backend hook.

### P3: Add operator and compilation candidates

The first operator target remains native exact Consilience top-k scoring,
because the current fallback crosses out of the optimized serving path. After
that, use the two-stage tuner pattern:

1. an agent proposes a semantically equivalent parameterized implementation;
2. compilation and differential tests establish correctness;
3. a structured tuner searches tile/fusion/capture parameters;
4. end-to-end Conditional-IS runs, not microbenchmarks, decide retention.

Ada-MK-style MegaKernel work is considered only after traces show launch and
HBM round trips still dominate the optimized path.

### P4: Multi-card rollout policies

When stable free cards are available, evaluate MISA-T-style workload KV
admission and TailSieve-style tail isolation. Runtime TP/PP reconfiguration is
valuable but follows these lower-complexity policies because ReMP/Nitsum-level
state migration is a large independent engineering project.

## Success Criteria

The project is successful only if both the tuner and a tuned mechanism work.

### Optimizer quality

- at least 60% fewer full NPU trials than the fixed candidate sweep baseline;
- at least 90% recall of the measured top-5 feasible configurations;
- held-out throughput MAPE below 10% in supported workload cells;
- explicit abstention for unsupported hardware, model, algorithm, or length
  regions;
- same or better final configuration than random search at equal device budget.

### Runtime value

- at least 8% geometric-mean E2E throughput gain over chang's strong baseline
  across two workload regimes for the first retained mechanism;
- a target of 15-30% for the combined policy bundle;
- no headline based only on startup elimination, a synthetic microbenchmark,
  weakened concurrency, or an algorithm-changing quality trade-off;
- answer quality, P95/P99 latency, memory headroom, and failure rate reported
  beside throughput.

Paper numbers are motivation, not promises. A mechanism that fails these gates
is recorded as negative evidence and removed from the default policy bundle.

## Immediate Next Work

1. Completed: implement the AIC-NPU calibration data model, bucketed affine
   stage predictor, uncertainty intervals, OOD abstention, graph composition,
   CLI, examples, and tests.
2. Add inner engine-batch instrumentation without changing algorithm semantics;
   retain the current outer graph-call timeline for queueing analysis.
3. Convert the old runs only into features they actually measured. They can
   calibrate end-to-end residuals and queueing, but not primitive bucket cost.
4. Produce an uncertainty- and decision-ranked list of the next 8-12 NPU
   measurements, with at least two token extents per selected shape bucket.
5. Audit P-PAS against upstream vLLM and the installed vLLM-Ascend scheduler.
6. Use the first genuinely isolated free NPU window for the selected probes,
   then implement the runtime policy only if the trace replay predicts a useful
   effect.
