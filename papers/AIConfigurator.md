# AIConfigurator 论文解读

## 基本信息

- 论文：[AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving](https://arxiv.org/abs/2601.06288)
- 代码：[ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)
- 作者机构：NVIDIA
- 提交时间：2026-01-09

## 一句话判断

这是 Autopilot 静态配置优化层最重要的架构参考，但不能直接作为我们的成果复刻。
它已经是开源产品；我们的空间在 Ascend NPU、inference-scaling 多阶段图、跨引擎
干扰和少量真机残差校准。

## 它解决什么问题

LLM 部署配置同时包含 TP/PP/EP、batch、KV cache 比例、token capacity、CUDA
Graph、chunked context、聚合或 P/D 分离部署等变量。组合规模很容易超过一万，
而黑盒搜索需要大量 GPU 小时。

AIConfigurator 的目标不是在线修改 scheduler，而是在部署前根据模型、硬件、
workload 和 SLO，快速生成一组可执行的 Pareto 配置。

## 核心架构

论文给出了五层流水线：

1. `PerfDatabase`：离线测量 GEMM、attention、通信和内存操作；
2. `TaskRunner`：从 workload 和 SLA 构造合法候选；
3. `InferenceSession`：用 iteration model 和 operator 数据库预测每个候选；
4. `Pareto Analyzer`：按吞吐、TTFT、TPOT 等指标过滤和排序；
5. `Generator`：生成 vLLM、SGLang 或 TensorRT-LLM 的启动配置。

关键点是“先拆算子、再组合”，而不是直接训练 `config -> throughput` 黑盒模型。
operator 数据来自真机，系统行为则由框架专属的 iteration model 组合。

## Figure 2 详细解读

Figure 2 展示的不是某一种具体搜索算法，而是 AIConfigurator 从用户需求到可执行
部署配置的完整工作流。整张图可以按“输入、性能建模、候选搜索、结果生成”四段来读。

### 1. 输入：定义要优化的问题

系统首先接收以下几类信息：

- 模型及精度，例如模型结构、参数规模、FP8/BF16 等；
- 目标硬件及集群规模；
- serving framework，例如 vLLM、SGLang 或 TensorRT-LLM；
- workload，包括 ISL、OSL、并发度和请求到达特征；
- SLO 和优化目标，例如 TTFT、TPOT、吞吐或 GPU 数量约束。

这些信息共同决定“什么配置合法”以及“怎样才算最优”。因此 AIConfigurator 找到的
不是脱离负载的全局最佳配置，而是特定模型、硬件、workload 和 SLO 下的最优解。

### 2. PerfDatabase：把昂贵真机测量沉淀为可复用数据

`PerfDatabase` 保存底层操作在不同 shape、batch 和并行配置下的实测性能，主要包括
GEMM、attention、通信和内存操作。它相当于整个系统的硬件性能底座。

这一步在离线阶段完成，成本较高，但不需要在每次优化新 workload 时重新扫描所有
端到端配置。数据库把一次性的 GPU profiling 转化为可供大量候选重复查询的数据。

图中的关键设计是：系统不直接训练一个粗粒度的
`完整部署配置 -> 端到端吞吐` 黑盒模型，而是测量更稳定、更容易复用的底层操作，
再由上层模型重建一次推理迭代的成本。

### 3. TaskRunner：构造并裁剪合法搜索空间

`TaskRunner` 将用户输入展开为候选配置，包括 TP、PP、EP、batch/token capacity、
KV cache、聚合或 P/D 分离部署等组合。随后根据硬件数量、模型约束、显存容量和
框架能力去掉不可执行候选。

因此它不只是枚举参数，还承担 search-space compiler 的角色：把用户目标和框架
约束编译成一组可以送入性能模拟器的合法任务。越早完成 feasibility filtering，
后续需要评估的候选就越少。

### 4. InferenceSession：由算子成本组合出 serving 指标

`InferenceSession` 是 Figure 2 中真正的预测核心。它读取一个候选配置，依据具体
framework 的执行模型，将 `PerfDatabase` 中的算子成本组合为 prefill、decode 和
通信迭代时延，再进一步估算 TTFT、TPOT、吞吐和资源占用。

这里必须同时知道两件事：

- 硬件上的 primitive 有多快；
- serving framework 会以什么顺序、batch 形状和调度方式调用这些 primitive。

前者来自性能数据库，后者来自 static、aggregated 或 disaggregated iteration
model。这种分层也是 AIConfigurator 可以支持多个 serving framework 的原因：底层
硬件测量可以复用，而框架行为由不同 session/model 描述。

### 5. Pareto Analyzer：保留真正有意义的候选

预测完成后，`Pareto Analyzer` 先剔除违反 SLO 或资源约束的配置，再删除被其他
配置完全支配的点。例如，一个配置如果使用更多 GPU、吞吐更低且延迟更高，就没有
继续保留的价值。

最终结果通常不是单个“最佳参数”，而是一条 Pareto frontier。用户可以在成本、
吞吐和交互延迟之间选择不同折中。这也避免了把多目标优化错误地压缩成一个未经解释
的加权总分。

### 6. Generator：把搜索结果变成可运行部署

`Generator` 将选中的逻辑配置翻译成目标 framework 的实际启动参数或配置文件。
至此，结果才从预测器中的一个候选点变成可验证、可部署的系统方案。

因此 Figure 2 的端到端闭环是：

```text
模型/硬件/workload/SLO
          |
          v
构造并过滤配置空间
          |
          v
查询实测 primitive 性能 + 模拟 framework 执行
          |
          v
预测 TTFT/TPOT/吞吐/资源 -> SLO 过滤 -> Pareto 排序
          |
          v
生成可执行的 serving 配置
```

### 7. 图中隐含的两种时间尺度

Figure 2 还隐含了两个不同时间尺度：

- 慢路径：首次适配硬件和软件栈时，离线采集并维护 `PerfDatabase`；
- 快路径：面对新 workload 或 SLO 时，在 CPU 上组合模型、评估候选并生成配置。

论文所谓 “lightning-fast” 主要指第二条路径。它并没有消除真机实验，而是把大量
重复的端到端 sweep 替换为可复用的底层 profiling 和廉价模拟。

### 8. 对 Autopilot 的直接映射

我们可以沿用 Figure 2 的产品骨架，但不能照搬它的普通生成执行图：

| AIConfigurator | Autopilot 对应模块 | 我们需要增加的能力 |
| --- | --- | --- |
| `PerfDatabase` | AIC-NPU calibration corpus | Ascend、ACL Graph bucket、编译与 host launch 成本 |
| `TaskRunner` | Search Space Compiler | Conditional-IS 参数、阶段 fanout、跨引擎资源约束 |
| `InferenceSession` | inference-scaling Graph IR 与 stage cost composer | candidate/proposal/score/reward/select 多阶段组合与 barrier |
| `Pareto Analyzer` | Policy Selector | 不确定性约束、top-k 验证、质量与预算目标 |
| `Generator` | Policy Bundle Generator | vLLM-Ascend 启动配置与按上下文/阶段压力切换的运行时策略 |

最值得继承的是“真机性能数据库 + 廉价图模拟 + 约束过滤 + Pareto 搜索 + 可执行输出”
这条主链；我们的技术增量则应来自 Ascend NPU 和 inference-scaling 多阶段算法的特有
执行结构，而不是重新包装已有的 GPU 配置枚举能力。

### 9. 这张图没有解决什么

Figure 2 容易让人误以为系统已经实现完全自动的在线调优，但论文主体更接近部署前的
静态配置优化。图中没有充分解决：

- workload 漂移后何时在线切换配置；
- 应该优先补测哪个未知性能点；
- 对未见模型、shape 或硬件怎样量化不确定性并拒绝预测；
- inference-scaling 中多轮、分支、barrier 和质量预算如何建模；
- 预测配置在真实系统中偏差过大时如何自动回滚。

这些空白正好对应 Autopilot 后续的 active measurement、OOD abstention、运行时
policy 和闭环验证能力。

## 三种执行模型

- `Static`：固定 batch，prefill 后逐步 decode；decode 以 stride 采样，避免逐 token
  查询数据库。
- `Aggregated`：模拟 continuous batching，把 prefill 和 decode 混在同一 iteration；
  用 context capacity、generation slots 和经验 correction 描述排队与干扰。
- `Disaggregated`：分别枚举 prefill/decode 配置和 worker 数，以二者的最小处理速率
  作为系统速率，并约束 TTFT、TPOT 和总 GPU 数。

## 搜索空间和输出

论文评估涉及 ISL、OSL、concurrency、TP、EP，并处理显存不可行配置。产品输出
不是唯一配置，而是满足 SLO 的 throughput-speed Pareto frontier 及对应启动文件。

这对我们很重要：不同上下文长度和负载下最优配置不同，产品应该输出 policy
bundle，而不是把一个 `max_num_seqs` 写死成“全局最优”。

## 实验结果应如何理解

- 聚合部署 TPOT 总体 MAPE 为 7.8%；vLLM 子集为 11.9%。
- TTFT MAPE 约为 16.9% 至 22.1%，明显弱于 TPOT；论文还排除了超过 1 秒的病态
  排队 outlier。
- 分离部署在交互速度区间的 generation-speed MAPE 为 3.35%，但全配置 throughput
  MAPE 为 25.49%。
- 339 至 506 个候选的 CPU 搜索约 0.52 至 0.84 秒，对应真机 sweep 为数十小时。
- case study 报告 dense/MoE 最高 40%/50% 改善，论文结论还给出分离部署发现的
  2 倍吞吐案例。

因此它最强的是候选排序和快速缩小空间，不代表所有时延指标都能做到个位数误差。

## 不能直接照搬的部分

1. 数据库和 correction 都绑定 NVIDIA GPU 与 CUDA 软件栈。
2. prefill/decode 是普通生成图，不能表达 Conditional-IS 的 candidate、proposal
   rollout、target score、reward 和 selection。
3. 聚合模型主要描述稳态平均请求，无法直接处理短时 burst、阶段 barrier 和
   base/proposal 相互阻塞。
4. 论文假设已知固定 ISL/OSL；我们的 workload 是分布且算法阶段会放大 fanout。
5. 它不负责决定下一次昂贵实验该测哪个点，也没有明确 OOD abstention。

## 我们直接采用什么

- `PerfDatabase -> Session -> Pareto -> Generator` 的产品分层；
- primitive 实测与框架级组合模型分离；
- 静态、聚合、分离三种部署模式的统一表示；
- 先做显存和能力过滤，再做性能排序；
- 输出可执行配置和预测依据，而不是只给分数。

## 我们新增什么

- AIC-NPU：按 Ascend 编译/ACL Graph bucket 建立稀疏性能数据库；
- inference-scaling graph adapter：把 Conditional-IS 的各阶段和 fanout 显式化；
- 两层模型：primitive service cost 与 queueing/co-batch residual 分开；
- uncertainty interval、OOD abstention 和 decision-critical acquisition；
- 运行时 policy bundle：根据上下文长度和阶段压力选择静态配置及 scheduler policy。

## 落地顺序

1. 用新 `engine_batch_timeline` 采集真实 engine batch shape；
2. 为 base generate、proposal generate 和 target score 建 bucket 数据；
3. 加入显存、KV、graph capture 和 host launch 的 feasibility floor；
4. 用 Conditional-IS DAG 组合 stage cost；
5. 用少量端到端实验拟合 overlap/queueing residual；
6. 比较预测 top-k 与真机 top-k，而不只报告 MAPE。

## 成功标准

- 同等 NPU 实验预算下优于 random/grid 搜索的最终配置；
- 至少减少 60% 完整真机候选；
- supported cells 的 top-5 recall 不低于 90%；
- 新上下文或新算法路径无法可靠预测时明确拒绝。
