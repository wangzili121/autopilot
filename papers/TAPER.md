# TAPER 论文解读

## 基本信息

- 论文：[Regulating Branch Parallelism in LLM Serving](https://arxiv.org/abs/2605.06914)
- 系统名：TAPER
- 作者机构：Stanford University、NVIDIA
- 提交时间：2026-05-07

## 一句话判断

TAPER 不是剪枝，也不是用 PRM/logprob 预判答案质量。它假定算法已经给出语义独立
的分支，只在每个 decode step 决定放多少分支进入共享 batch。它解决的是并行分支
对其他请求造成的系统外部性，和 Conditional-IS rollout admission 高度相关，但前提
是分支必须在同一个 scheduler domain 内可见并共享 prefix KV。

## 论文发现的 throughput trap

intra-request parallelism 会让一个请求的多个 branch 同时 decode。并行请求获得多份
进度，但同 batch 中处于串行阶段的请求每步仍只获得一个 token，却要等待被加宽后的
forward pass。

因此 eager branch admission 可能提高 raw token throughput，却让更多请求违反 TPOT
SLO，最终 goodput 下降。固定 cap 也不稳定，因为安全宽度随 batch composition、
context length、KV pressure 和累计 slack 变化。

## Branch externality

定义 baseline step `S0` 每个请求只前进一步，`S(k)` 给并行请求额外 branch：

```text
E_t(k) = T(S(k)) - T(S0)
```

`E_t` 是额外 branch 给整个 batch 增加的 step latency。所有请求都支付它，只有并行
请求得到额外进度。TAPER 只在 externality 小于当前 batch slack budget 时放行分支。

## 延迟预测器

论文使用极轻量的线性模型：

```text
T(S) = a + b * num_sequences + c * aggregate_context_length
```

它在 batch 1 至 512、context 128 至 8192 的 `20 x 25` 网格上收集 500 个 profiling
step，用 OLS 拟合，并每十分钟用最近 200 个观测刷新。报告总体 MAPE 1.8%。

这个预测器之所以适合 scheduler，是单次评估只有少量乘加，planner P99 开销 0.8 ms。

## Admission planner

TAPER 为 batch 中每个并行请求计算当前 slack，再用 greedy planner 分配额外宽度。
额外 branch 只计算 branch-local KV 占用，因为 prefix KV 已共享；宽度可以逐 step
扩张或收缩，无需驱逐整条请求状态。

核心消融说明三个部分都必要：

- 去掉 slack budget：goodput 降到 IRP-Off 的 0.92 倍，SLO attainment 48%；
- 不逐 step 重规划：1.18 倍 goodput，attainment 82%；
- 用常数 predictor：1.12 倍 goodput，attainment 96%，安全但利用率低。

## 实验结果

- 单节点 8 x A100-80GB、TP=8、Qwen3-32B；
- 混合串行/并行阶段的 10 小时 trace；
- TAPER 相对 IRP-Off goodput 1.77 倍，相对 eager 1.48 倍；
- 默认设置 attainment 99%，论文概括为超过 95%；
- SPRINT frontend 上仍有 1.45 倍 goodput 和 98% attainment；
- 论文也在 Qwen2.5-72B 上重拟合 predictor 后验证。

这些数字依赖论文定义的 IRP frontend、SLO 和共享 scheduler，不能直接当作我们的预期。

## 它与剪枝的区别

- 剪枝决定某个 branch 是否值得继续，通常涉及质量预测和算法语义；
- TAPER 决定一个合法 branch 现在执行还是稍后执行，不删除 branch；
- 在 schedule-invariance 假设成立时，它不改变最终分支集合和输出分布；
- PRM、reward、logprob 不进入 TAPER 的核心 controller。

因此 TAPER 属于 infra admission control，而不是 completion-driven stopping。

## Conditional-IS 是否能直接用

不能直接断言。当前常博路径可能把 proposal rollout 表现为外部、同步的 backend call。
如果 scheduler 只看到若干独立 request，而不知道它们属于同一父请求、同一算法 step 和
共享前缀，就无法正确计算 branch-local KV、slack 或动态宽度。

需要先验证：

1. rollout 分支能否在一个 scheduler iteration 内独立 admission；
2. 延迟或重排是否保持随机流、importance weight 和最终输出不变；
3. prefix KV 是否实际共享；
4. base 与 proposal 若在不同 engine，externality 应在哪一层定义；
5. algorithm barrier 是否允许一部分分支晚到。

## 我们可以采用什么

- externality 作为运行时调度的显式目标；
- `num_sequences + aggregate_context` 的轻量 predictor baseline；
- 每 step 重规划和 slack budget；
- branch-local KV accounting；
- raw throughput 与 SLO goodput 分开报告。

## 我们的落地路径

先把 TAPER 做成 graph-level replay policy，不立刻改 vLLM：用真实 timeline 模拟不同
wave/branch admission，估算潜在收益。如果 replay 没有明显 externality 或分支 API
不能保持语义，则停止。

只有 go/no-go 通过后，才在常博 backend adapter 或 vLLM-Ascend scheduler 增加
parent/stage/slack metadata，并做 exactness differential test 和 paired E2E 实验。
