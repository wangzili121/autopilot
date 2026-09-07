# FleetSieve 论文解读

## 基本信息

- 论文：[FleetSieve: Decision-Critical Profiling for SLO-Aware LLM Fleet Configuration](https://arxiv.org/abs/2608.19659)
- 作者机构：Meta
- 提交时间：2026-08-20

## 一句话判断

FleetSieve 给出了 Autopilot 下一轮实验选择最正确的目标：不是测最不确定的点，而是
测最可能改变最终部署决策的点。它的平均 profiling 节省只有 5.4%，并不惊艳，但
决策建模和停止证书比普通 active learning 更适合我们的昂贵 NPU 实验。

## 问题定义

每个 workload class 有需求、SLO、优先级和最低满足率。候选配置包含 TP 和 admission
load，每个 replica 消耗固定 GPU 数。未知性能向量包含：

```text
theta = (sustainable capacity, tail latency, success probability)
```

配置只有同时满足 tail SLO 和 success floor 才可行。下游整数规划选择配置、replica
数和服务负载，在总 GPU 约束下按字典序最大化关键类满足、max-min fairness、总
goodput，并减少碎片。

## 为什么普通 active learning 不够

最大方差点可能离最终边界很远，即使测准也不改变选择。反过来，一个方差不大的点
若位于 SLO 或资源可行性边界，可能直接改变 replica 配置。

FleetSieve 把“profile 哪个点”改写成“哪个测量最可能缩小下游决策差异”。

## 保守与乐观决策

系统为 capacity 和 tail latency 保存区间：

- 保守配置使用 capacity 下界和 tail 上界；
- 乐观配置使用 capacity 上界和 tail 下界。

若连乐观配置也无法满足关键需求，则 `Certified-Infeasible`；若保守配置已满足政策，
并且乐观/保守目标差低于容差，则 `Certified-Feasible`；否则为 `Undecided`。

论文明确称这是依赖经验 uncertainty set 的 conditional certificate，不是无条件数学
保证。这种表述值得保留。

## acquisition score

对每个未测实验，估算观测后保守/乐观 allocation gap 的期望缩减：

```text
score(e) = E[current_decision_gap - next_decision_gap | measure e]
```

实现用确定性近似。实验 GPU-seconds 只作为严格 tie-breaker，不进入分母，所以论文
也没有声称它在任何情况下都实现最低 profiling 成本。

## 实验结果

- 31B FP8 模型，单 H100 节点，TP 为 2/4/8；
- 每个 cell 运行 300 秒，成本按运行时间乘 GPU 数；
- 固定比较中用 22,200 GPU-seconds 找到 oracle 决策，比 random 少 6.9%；
- 200 个随机 reveal order 下平均节省 5.4%，95% bootstrap CI 为 3.5% 至 7.2%；
- Code workload 上它并不是成本最低的方法；
- 联合 tail 模型避免选择 completion P99 46.4 秒、违反 30 秒 SLO 的高吞吐配置。

所以它的贡献是稳健决策，不是巨大的平均搜索加速。

## 对 Autopilot 的映射

我们的下游决策不是 fleet replica allocation，而是 policy bundle：

- workload bucket 选择哪组 base/proposal capacity；
- graph mode/capture sizes；
- static 或 dynamic token budget；
- stage admission policy；
- exact scoring implementation；
- OOM、P95/P99 和质量约束。

将每个候选的 predicted interval 传播到 policy selector，得到 optimistic 与
conservative bundle，再测最可能让二者收敛的 NPU 点。

## 与现有代码的结合

仓库已有 policy acquisition、formal calibration、hidden failure 和 evidence grade。
下一步不是再写一个泛化 BO，而是把 acquisition objective 从“覆盖空间/距离”替换为
“缩小最终 policy gap”，并让失败类型进入可行性模型。

## 需要改进论文方案的地方

1. 实验成本不能只做 tie-breaker；NPU 初始化、模型重启和 paired replay 成本差异很大。
2. 我们应支持 batch acquisition，空闲多卡时并行选择不重复信息的点。
3. 先用 AIC-NPU cheap model 过滤明显 dominated 候选，再在边界做 FleetSieve。
4. workload 会随上下文和算法路径变化，应按 transfer uncertainty 调整 prior。
5. 停止后仍需独立 holdout 验证推荐，而不能把 certificate 当最终效果证明。

## 成功标准

- 相同 NPU-hours 下，最终配置不差于 random/TPE/SCOOT baseline；
- 到达同样 top-k recall 或 policy regret 时，完整 E2E 试验数减少至少 60%；
- 被选择的每个测点能解释它影响了哪条 SLO、可行性或排序边界；
- 负结果和 crash/OOM 也进入证据账本，不按 missing row 丢弃。
