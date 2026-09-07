# P-PAS 论文解读

## 基本信息

- 论文：[P-PAS: Prefill-Pressure Adaptive Scheduling for Long-Context LLM Serving](https://arxiv.org/abs/2608.15171)
- 代码：[TimoSaemann/ppas-vllm](https://github.com/TimoSaemann/ppas-vllm)
- 作者：Timo Saemann，独立研究者
- 提交时间：2026-08-15

## 一句话判断

P-PAS 是当前最容易做出真实效果的运行时方向：它只修改 scheduler 的 prefill token
budget，却直接解决“长 chunk 单次执行更高效”和“prefill 阻塞 decode”之间随压力
变化的冲突。我们不能原样复制 GPU 阈值，但可以把机制扩展成 Conditional-IS 的
stage-pressure adaptive token budget。

## 论文观察

vLLM 的 `max_num_batched_tokens`（论文简称 MBT）限制每个 scheduler iteration 能处理
的总 token。对长 prompt：

- 大 MBT 减少 chunk 数，并可能让 attention/MLP kernel 更高效；
- 小 MBT 给活跃 decode 更频繁的执行机会，降低 prefill interference。

低负载时大 MBT 更好，高负载时小 MBT 更好，因此没有一组固定 MBT 在所有负载下
占优。论文将这种交叉称为 MBT sensitivity。

## P-PAS 策略

每个 iteration 从 scheduler 直接读取：

- `N_p`：running 和 waiting prefill 数量；
- `N_d`：running decode 数量。

当 `N_p >= N_th` 且 `N_d > 0` 时，把 prefill token cap 从 `B_max` 降到 `B_cap`；
压力消失后立即恢复。论文固定使用：

```text
N_th = 2
B_max = 16384
B_cap = 2048
```

它没有修改 vLLM 全局 token budget，而是在全局预算之上增加 aggregate prefill cap。
控制器不需要 workload arrival-rate predictor，也不增加模型调用。

## 为什么机制有效

论文的 kernel profiling 显示，Qwen2.5-3B 处理同样 16K prefill 时：

- 8 个 2K chunk 的 FlashAttention 累计 7.21 ms；
- 单个 16K chunk 为 5.83 ms；
- 总 prefill 时间从 755.8 ms 降到 668.6 ms，改善 11.6%。

但 Qwen2.5-7B 的同类改善只有 0.6%，说明“大 chunk 更高效”不是普适规律，必须按
模型和硬件校准。压力高时，大 chunk 的效率收益又会被 decode interference 抵消。

## 实验条件

- 主实验：Qwen2.5-3B、RTX 5090、25K prompt、32-token output；
- 扩展：Qwen2.5-0.5B、SmolLM3-3B、20K 至 30K prompt、16 至 64 output；
- 额外硬件：A100，并确认 H100 上存在 crossover；
- 50 秒 Poisson trace，在 steady 与 burst arrival rate 间每 10 秒切换；
- 每组结果平均五个随机种子。

## 结果应如何解读

主配置跨六个 burst rate 的几何平均：

- 平均 E2E latency 比固定 MBT 2K 好 8.5%，比 16K 好 11.3%；
- P95 分别好 3.0% 和 10.0%；
- 相对 16K，TPOT 好 36.3%，但平均 TTFT 差 18.5%；
- makespan 相对 16K 差 3.9%。

这不是无条件吞吐提升，而是跨负载保持较低 E2E latency。某些指标会退化，尤其是
与大 MBT 相比的 TTFT 和 makespan。

## 对 Conditional-IS 的扩展

普通 P-PAS 只区分 prefill/decode。Conditional-IS 在 base engine 上至少有：

- candidate generation：短 prefill 加 decode；
- target scoring：长 teacher-forced score；
- 可能的 reward/Consilience score；
- 不同请求在算法 step barrier 前后的 deadline/slack。

因此我们的控制输入应增加：stage、queued score tokens、queued generation tokens、
active decode 数、上下文 bucket、graph coverage、KV pressure 和阶段 barrier slack。

输出也不应只有二值 16K/2K，而应从预校准的合法 budget set 中选择，并带 hysteresis，
避免每步抖动。

## 实现计划

1. 审计 pinned vLLM/vLLM-Ascend scheduler 是否可逐步修改 prefill/score cap；
2. 在 `engine_batch_timeline` 中补齐 stage 和实际 scheduled token；
3. 离线 replay 固定 budget 与动态 policy；
4. 先实现二阈值 policy，复现论文机制；
5. 再由 Autopilot 搜索 threshold、budget levels 和 hysteresis；
6. short、medium-2K、long-score、bursty 四类 workload 做 paired E2E 实验。

## 保留门槛

- 至少两个 workload regime 的几何平均 E2E 改善不低于 8%；
- P95/P99、正确率、OOM 和 preemption 无显著退化；
- 必须优于每个 regime 的最佳固定 budget，而不只是优于一个弱默认值；
- NPU 上不存在 crossover 时，控制器应退化成静态最优，不强行上线。
