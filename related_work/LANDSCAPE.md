# 自动推理优化相关工作全景

调研日期：2026-09-07

## 1. 真机服务参数自动寻优

### 华为 Serviceparam Optimizer / OptiX

针对 Ascend 上的 MindIE 和 vLLM，自动修改参数、拉起服务、运行 benchmark，
再使用 PSO 搜索满足 SLO 的高吞吐配置。它已经覆盖我们原先设想的通用实验
harness，详见[华为专项梳理](HUAWEI_ASCEND_AUTOTUNING.md)。

### vLLM Auto Tune

[vLLM Auto Tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md)
提供服务启动与 benchmark 循环，当前公开脚本重点扫描 KV cache 可承受范围内的
`max_num_seqs` 和 `max_num_batched_tokens`。它证明基础参数 sweep 已是框架能力，
但不理解 inference-scaling 的算法阶段，也没有我们的证据和迁移协议。

### AIConfigurator

[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) 是 NVIDIA Dynamo
生态中的部署配置规划器。它通过模型、GPU、SLO 和工作负载描述，结合 profile
数据库与服务模型搜索并行度、batch、并发和 PD 配置，输出候选部署方案。

它是 Autopilot 静态配置层的重要架构参考，但主要服务普通 LLM endpoint；我们
需要补充多引擎算法图和真实设备校准。详细解读见
[`papers/AIConfigurator.md`](../papers/AIConfigurator.md)。

## 2. 部署前性能仿真与容量规划

### msModeling / TensorCast / Throughput Optimizer

在 Ascend 上提供算子级解析或 profiling 模型，搜索并行策略、batch、concurrency
和 PD 配比。它适合提供单阶段预测先验，不应被我们重复实现。

### LENS

[LENS](https://arxiv.org/abs/2606.18042) 针对 NPU 服务使用离散 bucket 和少量
端到端校准点建模，是当前 AIC-NPU stage/batch/shape bucket 模型的直接方法
参考。其公开结果来自 Inferentia/TPU，不能直接当作 Ascend 精度证明。详见
[`papers/LENS.md`](../papers/LENS.md)。

### LLMVisor

[LLMVisor](https://arxiv.org/abs/2608.08382) 使用可加的轻量延迟特征支持微秒级
调度决策，提示我们下一版图成本组合应显式表示 prefill/decode 和阶段重叠，而
不只使用串行上界。详见 [`papers/LLMVisor.md`](../papers/LLMVisor.md)。

## 3. 稀疏测量与可信推荐

### FleetSieve

[FleetSieve](https://arxiv.org/abs/2608.19659) 不追求完整 profile 数据库，而是
优先测量会改变部署决策的点。它直接支持我们的“8 至 12 个 decision-critical
NPU 测点”路线。详见 [`papers/FleetSieve.md`](../papers/FleetSieve.md)。

### OmniPilot

[OmniPilot](https://arxiv.org/abs/2607.01579) 结合分位数预测、conformal
calibration 和显式支持域检查。最重要的启示是：只给预测区间不足以处理 OOD，
优化器必须在未覆盖的 batch/shape/workload 上拒绝推荐。详见
[`papers/OmniPilot.md`](../papers/OmniPilot.md)。

## 4. 运行时调度和动态配置

### P-PAS

[P-PAS](https://arxiv.org/abs/2608.15171) 根据运行时 pressure 动态调整 prefill
token cap。它不需要改变模型算法，是 Autopilot 下一项 stage-pressure adaptive
token budget 的主要参考。其平均收益并非所有指标都为正，必须与最强固定策略
做独立对照。详见 [`papers/P-PAS.md`](../papers/P-PAS.md)。

### TAPER

[TAPER](https://arxiv.org/abs/2605.06914) 面向树状/多分支推理，以 branch slack
和外部性决定逐步 admission。它不是 PRM 剪枝，也不是通过 logprob 猜测完成；
适合在算法能暴露分支、共享前缀和剩余工作时研究。详见
[`papers/TAPER.md`](../papers/TAPER.md)。

## 5. 动态 DAG 与自动 harness

### Simthesizer

[Simthesizer](https://github.com/casys-kaist/Simthesizer) 面向动态 LLM serving
扩展，使用 DAG、自动 lowering、trace 和 reference validation 构建模拟器。它
提示 Autopilot 的算法适配器需要可验证的动态路径，而不是只维护人工静态公式。
详见 [`papers/Simthesizer.md`](../papers/Simthesizer.md)。

### MARS 与 Pie

MARS 类系统侧重 agent/program 的并发、offload 和调度；[Pie](https://github.com/pie-project/pie)
面向 LLM program execution。它们与 Autopilot 的 runtime 层有交集，但控制对象
不同：Autopilot 当前优化固定 inference-scaling 算法的部署和阶段机制，不计划
复制通用 program runtime。

## 6. 当前空白组合

单项能力大多已有相关工作。Autopilot 的价值更可能来自以下组合：

```text
inference-scaling 多模型阶段图
  + 跨引擎、capacity、graph 与 workload 联合搜索
  + decision-critical 真机测量
  + replay/干扰/质量/迁移证据门禁
  + 支持域内的阶段感知运行时策略
```

这个组合目前没有在上述 vLLM、Ascend、NVIDIA 或论文原型中看到完整实现。后续
需要通过代码审计和等预算实验持续证明，而不能仅凭功能描述主张新颖性。
