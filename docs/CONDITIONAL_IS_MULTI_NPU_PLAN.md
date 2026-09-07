# 普通 Conditional IS 双卡与四卡优化计划

更新日期：2026-09-07

## 目标收缩

近期工作只研究常博仓库的普通 `conditional_is`，不以
`conditional_is_small_proposal` 为实验对象。模型固定为同一个较大模型，第一候选是
现有环境已能运行的 `Qwen3-Coder-30B-A3B-Instruct`。近期回答两个问题：

1. 同一模型在两张 NPU 上运行普通 Conditional IS 的强基线是什么；
2. 增加到四张 NPU 后，模型内并行、层流水和算法分支并行中哪一种最适合它。

Autopilot 仍负责图描述、实验编译、证据审计和后续自动选择，但不再以扩大通用参数
空间为目标。它首先服务这一个算法路径和这两个资源规模。

## 已确认的真实执行图

常博当前实现的每个生成 step 为：

```text
已提交前缀
  -> 同一 base model 批量生成 C 个 candidate block
  -> Python 等待全部 candidate 完成
  -> 同一 base model 批量生成至多 C x R 个 rollout suffix
  -> reward batch
  -> CPU log-mean-exp、归一化与 candidate selection
  -> 提交一个 candidate block，进入下一 step
```

同模型且采样策略相同时，rollout 返回的 generation logprob 会直接作为 base
logprob，重要性比率恒为一。普通路径没有 small-proposal 路径中的 target rescoring，
因此 selected-token scoring 不是当前主要优化点。

当前实现已有 candidate 和 rollout 的扁平批处理，以及 vLLM 内部 continuous
batching。新增工作的边界是多卡拓扑、算法分支放置、跨阶段流水和驻留数据路径，不能
把 vLLM 已有 TP/DP/PP 本身当作贡献。

## 第一阶段：建立强基线

冻结以下语义量，先比较 infra，不让算法质量变化干扰结论：

- model、BF16、数据集、题目顺序和 seed；
- candidate count、rollout count、block size 和 total length；
- reward 实现和停止条件；
- vLLM/vLLM-Ascend 版本、MRV1、APC、chunked prefill 和 graph 策略；
- 相同的外部请求到达方式。

第一轮使用当前 Qwen3-Coder-30B 配置的 `C=4, R=3, B=32, L=512`。当前远程
环境已有一个 `TP2`、普通 Conditional IS、HumanEval 运行，可作为 preliminary
基线候选；只有源码、配置、硬件和完整 telemetry 绑定后才能成为正式证据。

四种必须分开的部署臂为：

| ID | 卡数 | 拓扑 | 要回答的问题 |
| --- | ---: | --- | --- |
| T2 | 2 | `TP2` | 较大模型的两卡强基线 |
| T4 | 4 | `TP4` | 增加张量并行能否抵消更多通信 |
| D4 | 4 | `DP2 x TP2` | 天然独立的 candidate/rollout 请求能否从双副本获益 |
| P4 | 4 | `PP2 x TP2` | layer-by-layer 流水在大量 rollout microbatch 下能否填平 bubble |

四卡主比较是 T4、D4、P4；两卡 T2 用于报告扩展效率。另加“两个完全独立 T2
服务各处理一半请求”作为四卡吞吐上界和强基线，避免把普通副本扩容误报成新机制。

## 第二阶段：Conditional-IS 分支流水

如果 D4 优于 T4/P4，下一项实现不是静态 DP 的包装，而是算法感知的分支运行时：

1. 将每个 candidate 及其 R 条 rollout 作为一个亲和组；
2. 以预计剩余 token、KV 驻留位置和副本队列长度做动态分片；
3. candidate 完成后优先把其 rollout 留在同一 TP2 副本，避免 KV 搬运；
4. 副本失衡时比较三种动作：等待、在另一副本重算 prefix、传输 KV；
5. 多请求下交叠请求 t 的 rollout 和请求 t+1 的 candidate generation；
6. 两个副本只在 candidate 权重归并和最终选择时同步。

这项机制针对 Conditional IS 的 `C -> C x R -> reduce` 扇出/汇聚结构。验收时要
同时超过 TP4、vLLM DP2 和两个独立 TP2 服务，才能证明算法感知编排有增量价值。

## P/D 分离的进入条件

P/D 分离先做 feasibility 和 trace，不直接进入正式大扫参。普通 Conditional IS
的 rollout 通常 decode 占比高，而 candidate 和 rollout 又共享模型；KV 传输可能
比重复 prefill 更贵。只有同时满足以下条件才实现四卡 `2P + 2D`：

- prefill 或重复 prefix forward 在 step wall time 中占比足够高；
- 目标版本的 vLLM-Ascend 有可用、语义一致的 KV connector；
- KV 传输不经过磁盘，不产生不可接受的 host bounce；
- 小规模 trace 预测收益超过测量噪声与额外故障成本。

否则，D4 的同副本 KV 亲和分支流水优先级更高。

## 测量与自动选择

每个部署臂至少记录：端到端 wall time、request throughput、P50/P95/P99、输出 token
数和正确率；candidate/rollout 阶段时间；真实 engine batch 和 shape 分布；KV cache
hit、preemption、graph hit；AICore/HBM/通信利用率；host 等待和 Python reduction
时间。首轮用相同题目做交错顺序重复，空卡变化或兄弟卡进程变化的 run 作废。

自动优化器的第一版只在四个合法拓扑中选型，并在每个拓扑内调少量容量参数。它不对
无效笛卡尔积做盲搜。候选空间已经编码为
`examples/conditional-is-multinpu.deployment-space.json`。

第二轮再加入算法参数，且单独形成质量 Pareto：

- `candidate_count`：优先 4、8，确认趋势后再扩展；
- `rollout_count`：优先 1、3、5；
- `block_size`：优先 16、32、64；
- 固定 compute budget 和固定 wall budget 两种对照都要做；
- 报告 pass rate、输出有效率、答案/程序多样性和性能，不能只报速度。

## 后续两项研究

### 驻留式融合与 megakernel

先用 trace 量化 Python、D2H、kernel launch 和 HBM 中间量，再决定融合边界。近期可行
边界是将 rollout logprob、reward 所需统计、分段 log-mean-exp、归一化和 categorical
selection 保持在设备侧，并用跨卡 collective 完成 candidate reduction。Transformer
所有层整体做成一个 kernel 是长期目标，不作为第一轮实验的前置条件。

### 丰富度感知的 candidate 调优与剪枝

剪枝不能只按低 logprob 删除，否则容易保留同质答案。后续单独研究 inference-time
diversity 信号，将正确性代理与语义新颖度、答案簇覆盖、token entropy 或 disagreement
结合，在固定 rollout budget 下做保留、补采样和提前取消。任何近似策略都要给出质量
和多样性 Pareto，不能混入 exact infra 的速度结论。

## 近期交付顺序

1. 完成普通 Conditional IS 图和正式 runner 适配；
2. 接收现有 TP2 运行结果，只作 preliminary 诊断；
3. 等待无干扰四卡窗口，按 T4、D4、P4 做小规模 feasibility；
4. 冻结最有希望的两臂，做交错重复和完整 trace；
5. 实现算法感知分支流水，并与全部强基线比较；
6. 再进入 P/D、candidate 质量调优和设备驻留融合。

第一阶段的成功不是“支持四卡”，而是在相同四卡预算下找到可复现的最优拓扑并解释
原因。第二阶段的成功才是算法感知运行时相对现成 TP/DP/PP 的额外收益。

## 直接参考

- [Understanding Inference Scaling for LLMs](https://arxiv.org/abs/2605.19775)：
  直接比较 reasoning workload 下 DP、TP、PP 与混合并行；其 MoE 同步结论是本计划
  不预设 TP4 最优的依据。
- [AIConfigurator](https://arxiv.org/abs/2601.06288)：以 kernel、attention、通信和
  内存原语建模大规模配置空间；本项目复用其建模思想，但加入 Conditional IS 的
  `C -> C x R -> reduce` 图和真实 stage barrier。
- [Beyond Prefill-Decode Disaggregation](https://arxiv.org/abs/2607.25498)：说明单纯
  P/D 划分可能不够，需要按 DAG、负载和权重布局做动态 operator placement；用于约束
  本计划的 P/D 进入条件。
- [MPK](https://arxiv.org/abs/2512.22219)：以细粒度设备任务图和 persistent runtime
  自动生成 multi-GPU megakernel；作为后续 Ascend 驻留式融合的主要系统参考。
- [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001)：要求区分
  整个 inference system、candidate-bank 诊断、compute budget 和复现协议；用于定义
  candidate 调优的评测口径。
- [Test-Time Scaling in the Wild](https://arxiv.org/abs/2608.18931)：报告 tree search
  的 diversity collapse 和 selection/exploitation 瓶颈；支持把丰富度与最终选择质量
  纳入剪枝，而不是只使用 logprob。
- [DiffCodeGen](https://arxiv.org/abs/2605.20473)：通过行为聚类和 coverage-guided
  differential analysis 选择多样代码候选；对 Muyuan 的代码与漏洞场景尤其相关。
