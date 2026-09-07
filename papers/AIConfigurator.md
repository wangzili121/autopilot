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
