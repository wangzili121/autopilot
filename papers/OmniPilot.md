# OmniPilot 论文解读

## 基本信息

- 论文：[OmniPilot: An Uncertainty-Aware LLM Inference Advisor for Heterogeneous GPU Clusters](https://arxiv.org/abs/2607.01579)
- 作者机构：Harvard Kempner Institute
- 提交时间：2026-07-02

## 一句话判断

OmniPilot 最值得借鉴的不是 gradient boosting，而是“推荐系统必须知道自己何时不
知道”：预测区间只能覆盖已测分布，遇到新模型、新长度或新并发轴时必须用显式
support envelope 拒绝，而不能靠区间看起来不宽就继续推荐。

## 决策问题

它在提交服务前选择 GPU 类型、TP degree 和 precision，并同时考虑：

- aggregate/request throughput；
- TTFT、cold start、KV usage、power；
- launch success probability；
- 操作者对性能、成本和风险的效用权重。

这是 macro-level launch advisor，不修改 vLLM 内部执行。

## 模型结构

论文使用 gradient-boosted quantile model 预测八类目标，并构造约 28 个特征：

- 对数数值：模型规模、并发、上下文长度、KV pressure、active MoE 参数；
- 线性数值：TP、memory utilization、effective bits；
- 类别特征：模型家族、GPU 类型、量化格式。

量化既有 effective-bits 趋势，也保留 format one-hot，用来表达 AWQ 等偏离趋势的
实现路径。删除量化特征会让 FP8 误差接近翻倍，说明配置维度必须在采集数据前显式
进入 feature schema。

## 不确定性与拒绝

模型先做 quantile regression，再用 held-out workload cell 的 residual 做
conformalized quantile regression 校准。关键是按 cell 留出，而不是随机拆行，否则
同一工作负载泄漏会让 coverage 虚高。

但论文发现：OOD 的 5 个 case 中，误差升到 24% 至 46%，conformal interval 一个
也没有覆盖真实值。因此系统另存七个轴上的训练 support envelope；任何轴越界都直接
标记 low confidence，不依赖 interval width。

## 决策层

OmniPilot 将预测值、node-hour 成本、功耗和失败风险组合成 economic utility，输出
有置信标签的排序。遇到 OOD 时走保守 fallback，或先做一次可能改变排序的便宜 probe。

这比“最小化 MAPE”更贴近自动优化：只要排序和最终选择正确，某些无关指标的误差
可以较大；反过来，平均误差小但把不可启动配置排第一仍然失败。

## 实验结果

- 460 个有效 benchmark，A100/H100/H200、四种 precision；
- aggregate throughput MAPE 6.2%，log-space `R^2=0.92`；
- top-1 选择准确率 95%，mean utility regret 0.003；
- TTFT MAPE 约 12%，KV usage 25.3%；
- OOD case 预测明显恶化，但 5/5 被 support check 拒绝；
- 数据量超过约 150 行后收益开始饱和，轴覆盖比重复堆同类行更重要。

## 论文自己暴露的弱点

1. 单集群、单节点 vLLM，阈值不可直接迁移。
2. OOD 实验只有五个 cell，只能说明 failure mode，不能估计普遍 OOD 风险。
3. DCGM 60 秒采样无法解释细粒度 SM/tensor activity，对这些指标 MAPE 接近 100%。
4. cluster history 缺模型和 workload 语义，不能单独训练 throughput advisor。
5. update loop 只展示一次 promotion，不是长期在线学习证据。

## 对 Autopilot 的直接采用

- 按 workload/model/runtime cell 分组交叉验证；
- throughput、P95/P99、OOM 和启动失败联合建模；
- support envelope 先于模型预测；
- 预测区间和 OOD 是两个不同机制；
- 输出 utility regret、top-k recall 和选择正确率，不只输出 MAPE；
- 新轴先进入 schema 再采集数据。

## 我们需要修改的地方

Autopilot 的配置不只含 GPU/TP/precision，还包含两套 engine 的 capacity、token
budget、graph capture、memory split、stage admission 和 scoring backend。支持域必须
绑定算法图和 exactness cohort。

我们的 fallback 也不能简单选全局默认值：应按已验证 workload bucket 选择强基线；
若连基线也超出支持域，则输出“需要校准测点”，而不是伪造推荐。

## 验证标准

- leave-one-context-regime-out 和 leave-one-algorithm-path-out；
- 在支持域内校准 80%/90% interval coverage；
- 对人为构造的新模型、新上下文和新 backend 全部 abstain；
- 在同等 NPU 预算下比较最终配置 regret，而非训练集拟合度。
