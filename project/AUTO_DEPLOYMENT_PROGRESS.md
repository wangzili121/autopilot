# 自动部署优化进展

更新日期：2026-09-07

## 项目解决什么问题

Inference-scaling 算法通常不只执行一次模型推理。以
`conditional_is_small_proposal` 为例，一次请求会反复经过候选生成、proposal
rollout、目标模型评分、reward 计算和候选选择。不同阶段使用不同模型，引入的
batch shape、序列宽度和 token 压力也不同。因此，同一组 vLLM 参数不可能在
所有上下文长度、并发量和算法配置下都保持最优。

Autopilot 的目标是把这种依赖人工经验的部署调优变成一条可审计的自动流程：
理解算法执行结构，生成合法候选，用尽量少的真实设备实验学习性能规律，然后
输出经过独立验证、带适用范围和回退条件的部署策略。

## 给定什么输入

Autopilot 接收四类输入：

1. **Inference-scaling 算法实现**

   例如 Chang 仓库中的 `conditional_is_small_proposal`。适配器需要说明算法
   包含哪些执行阶段、阶段之间的数据依赖、每个阶段使用哪个模型，以及哪些
   参数只改变部署、哪些参数会改变算法语义。

2. **模型和推理环境**

   包括 base/proposal 模型、NPU 型号和数量、vLLM/vLLM-Ascend 版本、源码
   commit、精度、图执行能力、内存容量以及实际生效的引擎配置。环境信息会被
   哈希绑定，防止不同软件栈的数据被误放进同一个性能模型。

3. **工作负载**

   包括请求数量或到达率、并发度、上下文长度分布、输出长度、candidate 数、
   rollout 数、block size 和是否进行 exact target scoring。这些信息共同决定
   每个算法阶段的实际压力。

4. **可调参数和约束**

   输入不仅给出参数取值范围，还要声明参数作用于哪个阶段、何时可以修改、
   是否需要重启、与其他参数的依赖关系，以及吞吐、P95、内存、质量等目标。

## 项目如何一步步工作

### 第一步：解析算法执行路径

算法适配器把原始 Python 工作流转换为 Inference Graph。当前 Conditional IS
图中包含：

- `candidate_generate`：base 模型生成候选；
- `proposal_rollout_generate`：proposal 模型为候选生成多条 rollout；
- `target_score`：base 模型对 rollout 进行校正评分；
- `reward_evaluate`：计算任务 reward；
- `importance_reduce` 和 `candidate_select`：计算重要性权重并选择结果。

图中保留模型调用、CPU 工作、阶段依赖、fanout 和可调字段。这样优化器知道
一个参数影响的是哪个计算阶段，而不是把整个程序当作只有一个 QPS 输出的黑盒。

当前状态：Conditional IS Small Proposal 适配器已经实现；普通 Conditional IS、
consilience reward 和其他算法仍需增加适配器。

### 第二步：编译合法搜索空间

搜索空间编译器把参数域、硬件能力和 Inference Graph 合并为可执行候选。它会：

- 排除超过 KV 或图捕获能力的配置；
- 检查 graph capture ceiling 是否覆盖调度器可能产生的 batch；
- 检查参数是否真正作用于声明的算法阶段；
- 区分需要重启的静态参数和运行中可切换的热策略；
- 为每个候选生成稳定 hash，保证实验结果对应唯一配置。

这里的重点不是生成尽可能多的组合，而是先排除不可能、不安全和语义不一致的
组合，再把有限设备预算花在真正可能改变决策的区域。

当前状态：搜索空间 schema、编译器、能力约束和运行时配置闭包检查已经实现。

### 第三步：导入历史证据并选择新测点

历史实验首先经过证据审计。系统核对 workload、模型、源码、配置、随机种子、
执行顺序和质量结果，将数据分为可用于性能建模、只能用作先验、只能诊断或只
能说明配置不可行等等级。

然后主动采样器综合以下因素挑选下一批实验：

- 候选相对当前强人工配置的潜在收益；
- 模型对该区域的不确定性；
- 候选是否靠近容量或 graph bucket 等关键边界；
- 该测点是否有助于区分两个竞争策略；
- 引擎重启、图 capture、模型加载和失败重试成本。

因此它的目标不是完整扫描所有组合，而是优先测量会改变最终推荐的点。

当前状态：历史导入、证据分级、候选 acquisition 和预算化实验规划已经实现；
尚未完成“8 至 12 个自动选点优于等预算 random/grid”的真实设备证明。

### 第四步：执行可信校准

候选采用 ABBA 或带相同配置 replay 的配对设计运行。系统同时记录 QPS、延迟、
准确率、实际 batch/shape、图命中率、计算工作量、内存和主机进程状态。

同配置 replay 用来估计设备和随机轨迹造成的局部噪声。如果候选收益小于重放
波动，或者运行期间其他用户进程进入 NPU，结果会被拒绝，不能进入自动推荐。
质量下降、OOM、配置未真正生效和执行顺序不完整也会触发门禁。

当前状态：实验计划、证据账本、质量检查、重放噪声、共享主机干扰检测、失败
重试和生命周期成本均已实现，并已在真实 NPU 实验中使用。

### 第五步：建立阶段成本模型并选择策略

当前已经实现初版 AIC-NPU 成本模型。它按
`stage × engine role × exact batch × shape bucket` 建模，在同一个 bucket 内
使用不同 token extent 的少量测量拟合：

```text
latency = fixed overhead + seconds_per_token * token_extent
```

模型输出预测区间；遇到未覆盖的 batch 或 shape 时拒绝推荐，而不是盲目外推。
图级预测目前把各阶段成本组合为保守的串行上界。

需要准确说明的是：成本模型的代码、CLI、数据契约、合成示例和测试已经完成，
但真实 runner trace 尚未转换成正式校准语料，因此还没有真实 NPU 上的 MAPE、
候选排序准确率或模型驱动最优配置结果。

在真实校准完成后，策略选择器会同时考虑预测吞吐、P95、内存和失败概率，并
计入配置切换成本，选择满足约束且预期收益最高的候选。

### 第六步：独立验证并输出策略包

候选不能因为在校准数据上最快就直接发布。系统会在未参与拟合的 workload
holdout 上再次与强人工 baseline 做配对实验，并检查推荐是否仍在模型支持域内。

最终产物是一个可审计的 policy bundle，包含：

- base 和 proposal 引擎的静态部署配置；
- ACL Graph 模式、capture bucket 和覆盖范围；
- 可选的 stage wavefront、batch wait 等运行时策略；
- 适用的上下文、并发、算法和环境范围；
- 性能和质量证据；
- 预热要求、异常门禁和人工 fallback 配置。

当前状态：policy bundle、迁移评估、独立 holdout 和 guarded runtime routing
协议已经实现；成本模型自动发现并发布最终最优策略的完整闭环尚未完成。

## 我们考虑的搜索空间

搜索空间不是一张固定参数表，而是由算法图、工作负载和设备能力共同编译。当前
和计划中的参数按变更范围分为四层。

### 1. 静态部署参数

这类参数通常在引擎启动时确定，修改后需要重启或重新加载：

| 参数方向 | 当前状态 | 作用 |
| --- | --- | --- |
| base/proposal `max_num_seqs` | 已进入实验 | 控制两个模型引擎的序列容量 |
| base/proposal `max_num_batched_tokens` | 已进入实验 | 控制每个调度 step 的 token 容量 |
| base/proposal memory utilization 或内存比例 | 已有历史探针 | 影响 KV 容量与 OOM 边界 |
| base/proposal 是否共享 NPU、角色放置与副本数 | 计划加入 | 在资源隔离、批处理和并行度之间权衡 |
| TP、DP、PP 及副本组合 | 计划加入 | 面向多卡环境搜索通信与容量组合 |
| 精度、量化和模型/编译变体 | 远期 | 搜索更高成本的部署 cohort |

### 2. Graph 与编译参数

这类参数决定哪些实际 shape 可以走 ACL Graph，以及 capture 的启动与内存成本：

- eager、piecewise graph 或完整图模式；
- base/proposal 各自的 capture bucket 列表；
- capture ceiling 和 graph memory budget；
- 不同 prefill、decode、mixed 阶段的图覆盖；
- 后续可扩展的 auto compilation、kernel/算子实现选择。

已有实验表明 graph bucket 不能脱离 scheduler capacity 单独优化：
`40/64 + graph64` 的组合修改获得过 `+14.19%`，但只扩大 graph bucket 的
中位效果为 `-1.22%`。

### 3. 运行时热策略

这类参数不要求重启，可以根据上下文、并发和实时压力在已验证范围内动态改变：

- batch wait 或 microbatch 聚合时间；
- score 请求优先级；
- stage wavefront 的宽度、token cap 和 admission 条件；
- 根据 prefill/decode、阶段队列和 graph coverage 调整 token budget；
- 在多个预热 endpoint 或策略包之间路由；
- 超出支持域、干扰或 SLO 异常时回退。

近期第一个新增机制是 stage-pressure adaptive token budget/admission。它不会在
任意参数上无约束抖动，而是在经过离线验证的少量策略之间，根据上下文 bucket
和阶段压力选择。

### 4. 算法语义参数

candidate 数、rollout 数、block size、生成长度、校正模式和 reward 路径会改变
算法计算量甚至输出分布。Autopilot 会记录并建模这些参数带来的 workload，
但不会把改变算法质量换来的速度误记为部署优化。

默认做法是把不同语义配置分成独立 cohort，在各自质量约束下寻找部署策略。
未来若要联合优化质量、成本和延迟，需要显式采用多目标约束，而不是与普通
deployment knob 混在同一收益数字中。

### 搜索空间的扩展原则

一个参数只有满足以下条件才会进入自动搜索：框架确实允许控制；运行时能验证
它实际生效；合法域和参数依赖可表达；变更与失败成本可计算；不会与 vLLM、
vLLM-Ascend、Pie、Chang 或 Muyuan 已有能力无意义重复。

## 已完成的工作

目前已经完成：

- Conditional IS Small Proposal 的算法图适配；
- 部署搜索空间编译和约束检查；
- 历史实验导入、特征提取和证据分级；
- 自动候选选择和主动实验规划；
- ABBA 校准、重放噪声和质量门禁；
- 共享服务器干扰检测；
- ACL Graph bucket 规划和运行时配置核验；
- 策略迁移、独立 holdout 验证和安全回退；
- 初版 NPU 阶段成本模型；
- stage wavefront 运行时优化；
- 268 个自动化测试。

## 已有实际效果

### 短上下文 P32 的部署策略

系统通过容量与 graph coverage 的联合实验发现，`max_num_seqs` 与 ACL Graph
覆盖存在明显交互。`40/64 + graph64` 在独立 holdout 上相对
`40/48 + graph48` 的平均 QPS 提升 **12.02%**。

该结果经过独立 workload 验证，但配置主要由 Autopilot 的实验规划、交互诊断
和验证流程获得，并不是成本模型仅凭少量测点预测得到的最终结果。

### 短上下文 P96 的 stage wavefront

原始 proposal engine 中 mixed prefill/decode step 占比曾达到 `79.41%`。
stage wavefront 按算法阶段组织 proposal 调用后，ACL Graph 命中率提高到约
**96%**。正式配对实验的 QPS 几何平均提升 **18.04%**，forward-slot rate
提升 **18.09%**。

这个结果说明算法阶段边界可以提供 vLLM 普通请求流中缺失的优化信息，也是
目前除配置搜索外最明确的运行时机制收益。

### 成功拒绝不能推广的结果

同一短上下文策略迁移到 medium2k 后，中位收益只有 **2.85%**，低于当次
**3.91%** 的重放噪声，因此系统没有发布该策略。medium-context wavefront
实验也出现较大漂移，表面 `+7.06%` 的结果被同配置最高 `+16.26%` 的重放
变化否定。

这类拒绝不是没有结果，而是自动优化控制面必须具备的能力：避免把随机波动、
上下文迁移失败或共享服务器干扰包装成性能提升。

### 识别跨参数交互和共享环境污染

组合修改 capacity 和 graph coverage 曾获得 **14.19%** 的中位效果，而只扩大
graph bucket 反而回退 **1.22%**。这证明需要联合搜索，逐参数贪心扫描并不可靠。

主机干扰门禁也已经在共享 NPU 环境中发现其他 NPU 进程启停造成的测量污染，
自动拒绝相关结果，避免虚假性能结论进入成本模型和 selector。

## 接下来做什么

### 1. 打通真实 trace 到成本模型的数据链路

把 runner 已经采集的 `engine_batch_timeline` 严格转换为校准语料。导入时核对
stage、engine role、batch、shape、token extent、模型、源码、环境和配置 hash；
缺失或混合 cohort 的数据直接拒绝。

完成后，成本模型训练数据将来自真正进入 backend 的 batch，而不是包含外层
线程排队时间的算法调用耗时。

### 2. 自动选择 8 至 12 个高价值 NPU 测点

以当前强人工配置作为 incumbent，优先覆盖会影响决策的容量边界、graph coverage
边界、短/中上下文差异及 base/proposal 阶段瓶颈。每个测点都要说明它排除哪种
模型假设或区分哪两个候选。

同时生成等预算的 random 和规则 grid 选点，作为后续对照，不能只报告自动方法
自身找到的最好结果。

### 3. 在真实 NPU 上拟合和验证成本模型

在同卡、同模型和同软件栈完成 sparse calibration，报告支持域内 bucket MAPE、
P95 相对误差、pairwise ranking 和 top-5 recall。对未见 batch/shape 单独做 OOD
测试，确认模型会拒绝不可靠预测。

如果阶段简单相加无法解释并发重叠，将 trace 与模型升级为 critical-path 或
overlap-aware 组合，而不是用更多拟合参数掩盖结构错误。

### 4. 比较自动搜索与三类基线

在完全相同的新增 NPU 实验预算下比较：

- Autopilot 的 decision-aware 主动选点；
- 随机搜索；
- 规则网格或逐参数扫描；
- 当前强人工配置。

比较最终推荐 QPS/P95、找到可行策略的概率、测量成本、失败次数和相对真实已测
最优的 regret。目标是证明自动方法不只是“也找到了一个可用配置”，而是在有限
实验预算下更有效。

### 5. 在独立 holdout 上验证自动推荐

冻结未参与拟合的上下文和并发 workload，对推荐策略与强人工配置执行配对实验。
只有收益超过同批 replay noise，并通过准确率、内存、环境和运行时闭包门禁，
推荐才有资格发布。

如果某个 workload 没有可信的新收益，正确结果是保留人工 baseline，并收缩该
策略的适用范围。

### 6. 输出可部署、可回退的完整策略包

将自动选择出的引擎配置、graph plan、workload envelope、证据、预热要求和
fallback 打包。策略包可以被 Chang runner 直接读取；未来通过 adapter/exporter
接入 Muyuan，而不要求 Muyuan 使用 Autopilot 内部的数据结构。

### 7. 加入 stage-pressure 运行时策略

完成静态闭环后，根据上下文 bucket、各阶段队列、prefill/decode 压力、实际
batch 和 graph coverage，在少量已验证 token budget/admission 策略之间动态选择。

先运行 shadow mode 验证决策稳定性，再与最强固定策略做正式对照。除吞吐和
P95 外，还要检查 TTFT、preemption、图命中率、策略抖动和 fallback 比例。

## 近期成功标准

近期里程碑是：**使用不超过 12 个新增 NPU 实验，在未参与拟合的工作负载上
达到或超过当前强人工基线，并优于相同实验预算的随机搜索和网格搜索。**

只有达到这一标准，才可以主张 Autopilot 已经完成初步的模型驱动自动部署优化
闭环。目前更准确的进展是：流程、契约、实验门禁、候选规划和初版成本模型均已
实现，并已经得到有意义的配置与运行时机制效果；真实稀疏成本模型驱动的最终
推荐仍是下一阶段的核心工作。
