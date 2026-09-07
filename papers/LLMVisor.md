# LLMVisor 论文解读

## 基本信息

- 论文：[LLMVisor: A Real-Time Latency Attribution Model for Multi-Tenant LLM Serving](https://arxiv.org/abs/2608.08382)
- 提交时间：2026-08-09
- 主题：运行时 batch latency 预测与 per-request attribution

## 一句话判断

LLMVisor 提供了连接 AIConfigurator 静态模型和 TAPER/P-PAS 在线 controller 的关键
中间层：模型必须在微秒级回答“把这个请求加进 batch 会增加多少时间”，同时保持
per-request cost 可加。它比单纯使用 token count 更适合我们的 stage interference
建模，但论文只在 NVIDIA GPU/vLLM 上验证。

## 它解决什么问题

continuous batching 将多个租户请求放进一次 forward，batch latency 无法简单拆给
每个请求。公平性、quota、admission 和 billing 都需要知道每个请求贡献了多少 GPU
time，但模型必须比 scheduler 的毫秒级循环快得多。

黑盒 ML predictor 可以拟合总时延，却通常不满足可加 attribution，推理开销也可能
达到毫秒；token-count proxy 则忽略 attention 二次项、KV traffic 和 batch utilization。

## 模型输入

对 request `i`：

- `p_i`：当前 step 要处理的 token；prefill 是 prompt token 数，decode 为 1；
- `c_i`：需要 attention 的 context/KV token；prefill 为 0，decode 为历史长度；
- `|B|`：batch size。

论文使用两段线性形式表示 FFN 从 memory-bound 转向 compute-bound：

```text
T(B) = beta
     + a1 * sum(p_i)
     + a2 * sum(c_i)
     + a3 * sum(p_i^2)
     + a4 * sum(|B|)
```

`beta` 表示固定 batch 成本，`sum(p_i)` 表示 MLP/线性 attention 工作，`sum(c_i)`
表示 KV memory traffic，平方项表示 self-attention，batch 项表示利用率变化。prefill
与 decode 分开，并各有两个 segment，总共四组参数。

系数通过短 warm-up profiling 的 OLS 拟合，而不是用硬件峰值手算 roofline。由于总式
是 per-request feature 的和，可以闭式分解出每个请求的非负贡献。

## 为什么适合在线调度

- 只计算几个 feature sum 和分段线性式；
- 论文报告比 Random Forest 快 100 倍以上；
- 单请求预测为微秒级，可放在每个 scheduler step；
- 支持低成本 what-if：比较加入/移除请求前后的 batch latency。

## 实验设置和结果

- vLLM 0.7.3；
- A100 SXM、H100 SXM；
- Llama3.1-8B、Qwen2.5-14B/32B；
- 不同 TP 和 workload mix；
- 128 至 2048 并发，请求总 token 推到 KV capacity 的 90%。

相对 token-count baseline，论文报告：

- prefill p90/p99 relative error 最多降低 2.5/3.3 倍；
- decode p90/p99 最多降低 3.5/4.4 倍；
- 模型在测试配置上达到接近 1 的 `R^2`；
- scheduler path 中开销可忽略。

论文更强调相对误差改善和可加性，没有给出类似 TAPER 的 E2E goodput 提升，因此它
是决策 primitive，不是完整优化产品。

## 对 AIC-NPU 的启发

当前 AIC-NPU 首版在固定 stage/batch/bucket 内拟合：

```text
latency = intercept + seconds_per_token * token_extent
```

下一版应增加 LLMVisor 风格 feature，尤其是：

- prefill/score 的 `sum(p)` 和 `sum(p^2)`；
- decode/proposal rollout 的 aggregate context/KV traffic；
- actual sequence count，而不是 algorithm request group 数；
- model role、stage 和 graph bucket 的分段参数；
- cache hit/miss 和 prefix reuse 后的有效 token。

LENS 负责 NPU bucket 不连续性，LLMVisor 负责 bucket 内 batch composition，两者不是
替代关系。

## 不能直接迁移的地方

1. Ascend 的 graph/compile bucket 可能使两段分界不同，甚至需要更多 segment。
2. target scoring 是 teacher-forced 多 token forward，不等于普通 prefill 或 decode。
3. base/proposal 共卡时可能有 host queue 和同步开销，不满足纯算子可加。
4. 论文 per-request attribution 服务公平性；我们更关心 stage externality 和 SLO。
5. OLS 系数在 graph mode、TP、模型或 runtime 变化后必须重新绑定。

## 实现计划

1. 从 `engine_batch_timeline` 生成 `sum_p`、`sum_c`、`sum_p2`、batch 和 cache features；
2. 在每个 NPU bucket 内比较 affine、LLMVisor 两段模型和单调局部模型；
3. 用 leave-one-shape-cell-out 选择最简单且可靠的模型；
4. 输出 marginal cost，供 P-PAS/TAPER 风格 controller 做 what-if；
5. loaded trace 上另拟合 queueing residual，不能让 OLS 吸收排队噪声。

## 验收标准

- 同一 bucket 内 held-out latency MAPE 低于 10%；
- p95/p99 error 显著优于 token-count 和简单 affine baseline；
- 单次 what-if 预测不超过 scheduler 可接受预算；
- stage marginal cost 累加能解释实际 batch latency；
- unsupported bucket 或配置变化时拒绝，不外推系数。
