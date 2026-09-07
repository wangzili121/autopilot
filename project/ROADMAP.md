# 项目路线图

更新日期：2026-09-07

## 总体优先级

路线已在 2026-09-07 收缩。近期优先完成普通同模型 `conditional_is` 的双卡/四卡
拓扑研究和算法感知分支流水；原 M1/M2 自动成本模型工作作为测量与选择底座继续使用，
不再先扩展通用搜索空间。详细实验定义见
`docs/CONDITIONAL_IS_MULTI_NPU_PLAN.md`。

接下来按“先建立可量化的自动调优结果，再扩展更大的参数空间，最后加入在线
机制”的顺序推进。这样每一阶段都能独立验收，也避免在成本模型不可信时盲目
扩大搜索空间。

## M1：真实 AIC-NPU 成本模型闭环

状态：进行中

### 工作项

1. 实现严格的 `engine_batch_timeline -> NPUCalibrationCorpus` 导入器。
2. 校验 stage、engine role、exact batch、shape bucket、token extent、环境哈希
   和配置哈希；缺字段或混合 cohort 时拒绝导入。
3. 从大候选空间中选择 8 至 12 个“会改变最终推荐”的决策关键测点，而不是
   均匀扫描所有点。
4. 在空闲 NPU 上采集同卡、同模型、同软件栈的 sparse calibration。
5. 用未参与拟合的 bucket 和 workload 报告预测与排序指标。

### 验收条件

- 支持域内报告 bucket MAPE、P95 相对误差、top-5 recall 和 pairwise ranking；
- 支持域外 shape/batch 必须 abstain，不用外推伪装预测；
- 自动选点在相同设备实验预算下优于随机搜索和简单网格；
- 所有推荐都能追溯到模型、源码、环境、输入 trace 和原始 observation。

## M2：第一个自动推荐策略包

状态：待 M1

### 工作项

1. 冻结 short、medium2k 和至少一个并发 holdout cohort。
2. 将历史强人工配置作为 incumbent，而不是弱默认配置。
3. 联合搜索 base/proposal capacity、token budget 与 graph bucket coverage；
   搜索模型必须保留已观察到的 capacity × graph 交互。
4. 计入引擎启动、ACL Graph capture、失败重试和配置切换成本。
5. 输出包含静态配置、适用 workload envelope、预热要求和 fallback 的 policy bundle。

### 验收条件

在不超过 12 个新增 NPU 测点的条件下，推荐策略在独立 holdout 上达到或超过
当前强人工基线，并在同预算下胜过 random/grid；如果没有可确认收益，优化器
应正确保留人工基线，而不是强行发布新配置。

## M3：容易产生效果的运行时策略

状态：规划中

第一优先机制是 stage-pressure adaptive token budget。它借鉴 P-PAS 的 pressure
思想，但状态定义必须适配 Conditional IS 的 candidate/proposal/target-score
阶段：根据当前阶段队列、decode 占用、prefill token、图覆盖和上下文 bucket，
只在预先验证的策略集合中调整 admission/token cap。

先以 shadow mode 记录“策略本会如何决策”，再做固定策略对照，最后做带回退的
在线实验。目标不是逐 token 任意改参数，而是在不会触发引擎重启的热策略边界
动态选择。

验收指标包括吞吐、P95、TTFT、图命中率、mixed-step 比例、preemption、质量、
策略抖动次数和回退率。必须同时覆盖短上下文和 medium-context，防止复现
wavefront 只在单一 cohort 生效的问题。

## M4：扩展搜索空间

状态：规划中

扩展遵循“先具备可观测性和合法约束，再加入参数”的原则：

| 优先级 | 方向 | 原因 |
| --- | --- | --- |
| P0 | base/proposal capacity、token budget、graph buckets 联合搜索 | 已有强交互证据，可直接产生收益 |
| P0 | 上下文、并发、算法 fanout 的 workload-conditioned policy | 已证明不存在单一全局最优 |
| P1 | batch wait、score priority、stage-pressure admission | 无需重启，可形成运行时策略 |
| P1 | 角色副本数、角色放置、shared/dedicated NPU | 对多模型阶段图可能有大收益，但实验成本高 |
| P1 | TP/DP 与副本组合 | 需要多卡资源和通信遥测 |
| P2 | KV 配额、prefill/decode 资源划分 | 需确认与 vLLM/Pie/Muyuan 现有能力边界 |
| P2 | 编译/量化/模型变体 | 属于更高成本的离线 deployment cohort |

每次扩展前都要审计 vLLM、vLLM-Ascend、Pie、Chang 和 Muyuan，记录“已有能力、
缺失能力、我们的新增部分”，避免重复实现现成功能。

## M5：更多算法与 Muyuan 插件化

状态：远期

1. 为普通 Conditional IS 和 consilience reward 增加适配器，复用同一 IR、证据
   与策略协议。
2. 验证一个不是 Conditional IS 的 inference-scaling 或 agent workload，证明
   控制面不是单算法脚本。
3. 抽离 framework capability provider、runner adapter 和 policy exporter。
4. 以 Muyuan infra 插件接入，不接管 Muyuan 已有的 KV、调度或编译能力。

## 近期执行顺序

当前不需要为了写更多模块而扩大代码面。下一次开发应直接完成 M1 的 importer，
生成 8 至 12 点实验设计，并在设备可用时执行 sparse calibration。随后立即做
成本模型与等预算 baseline 的离线评估；结果合格才进入 M2，结果不合格则优先
修正 trace 粒度或 bucket 模型，而不是继续堆运行时机制。
