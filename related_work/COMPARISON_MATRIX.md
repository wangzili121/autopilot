# 能力对比与项目边界

调研日期：2026-09-07

符号：`是` 表示公开资料明确支持；`部分` 表示能力有限或需自行扩展；`未确认`
表示当前公开资料中没有找到，不能等同于确定不存在。

| 能力 | vLLM Auto Tune | Ascend OptiX / Serviceparam | msModeling Throughput Optimizer | AIConfigurator | Autopilot 当前 |
| --- | --- | --- | --- | --- | --- |
| 自动启停真实服务与 benchmark | 是 | 是 | 否，主要仿真 | 部分 | 是 |
| 通用参数范围配置 | 较窄 | 是 | 是 | 是 | 是 |
| PSO/黑盒实测搜索 | 否，规则 sweep | 是 | 否 | 否 | 主动采样，待端到端验证 |
| TP/DP/EP 和 PD 配比 | 部分 | 可作为参数 | 是 | 是 | 规划中 |
| 算子或阶段成本模型 | 否 | 仿真模式有 XGBoost | 是 | 是 | 初版 stage 模型，待真机校准 |
| 自动 compilation 参数 | 未确认 | 可枚举 CLI 配置 | 部分 | 部分 | 可表达，待系统搜索 |
| 自定义框架/benchmark 插件 | 否 | 是 | 部分 | 有适配层 | 是 |
| inference-scaling 多模型阶段图 | 否 | 未确认 | 否 | 否 | 是 |
| base/proposal 跨引擎联合约束 | 否 | 可手工表达部分参数 | 否 | 否 | 是 |
| capacity × graph 交互诊断 | 否 | 未确认 | 未确认 | 未确认 | 是，已有设备证据 |
| ABBA 与同配置 replay 噪声 | 否 | 未确认 | 不适用 | 未确认 | 是 |
| 共享主机污染拒绝 | 否 | 有健康检查，严格程度未确认 | 不适用 | 未确认 | 是 |
| 独立 holdout 与迁移门禁 | 否 | 未确认 | 未确认 | 未确认 | 是 |
| OOD 支持域外拒绝 | 否 | 未确认 | 缺 shape 时可回退解析模型 | 未确认 | 初版已实现 |
| stage-aware runtime admission | 否 | 未确认 | 否 | 否 | wavefront 已实现 |

## 应直接复用或集成

- Ascend 服务拉起、参数注入和 benchmark：优先评估 OptiX adapter；
- 普通单模型算子/并行策略建模：优先评估 msModeling 输出；
- GPU/Dynamo 配置空间和 profile 数据组织：参考 AIConfigurator；
- 通用 vLLM 参数语义和 benchmark：跟随上游而不是复制维护。

## Autopilot 应继续拥有

- 算法适配器和 Inference Graph IR；
- inference-scaling 多阶段 workload 特征与语义 cohort；
- 搜索空间的阶段绑定、跨引擎约束和配置闭包；
- decision-critical 选点和等预算 baseline 比较；
- ABBA、replay noise、质量、主机干扰、holdout 与迁移证据；
- stage wavefront、stage-pressure policy 和支持域内路由；
- 包含适用范围、证据和 fallback 的 policy bundle。

## 必须完成的比较实验

在相同 NPU、模型、workload 和最多 12 个新增测点下，比较：

1. Autopilot 的图语义感知主动选点；
2. OptiX 的通用 PSO；
3. random search；
4. 规则 grid 或逐参数扫描；
5. 当前强人工 baseline。

报告最终 holdout QPS/P95、相对已测最优的 regret、失败实验数、设备时间、策略
被拒绝的原因和质量结果。只有这个实验才能证明 Autopilot 在已有 Ascend 工具之上
增加了真实价值。
