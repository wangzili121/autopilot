# 论文阅读索引

本目录收录与 Inference Autopilot 直接相关的论文调研、单篇解读和技术路线映射。
产品架构、接口和实验协议仍放在 `docs/`，避免把“论文怎么做”和“我们已经做了
什么”混为一谈。

## 总览

- [2026 自动优化论文扫描与路线图](2026_AUTOTUNING_SURVEY_AND_ROADMAP.md)：
  2026 年论文元数据扫描、高相关工作列表、重叠审计和实现优先级。

## 第一批重点解读

| 论文 | 主题 | 对 Autopilot 的直接价值 |
| --- | --- | --- |
| [AIConfigurator](AIConfigurator.md) | operator 数据库、性能组合、配置生成 | 静态配置优化器的总体骨架 |
| [LENS](LENS.md) | NPU bucket 黑盒延迟模型 | AIC-NPU 的首个稀疏校准模型 |
| [OmniPilot](OmniPilot.md) | 分位数预测、conformal calibration、OOD abstention | 可信推荐与拒绝机制 |
| [FleetSieve](FleetSieve.md) | decision-critical profiling | 决定下一次真机实验测什么 |
| [P-PAS](P-PAS.md) | 运行时动态 token budget | 第一项候选运行时优化 |
| [TAPER](TAPER.md) | branch externality 与逐步 admission | inference-scaling 并行分支调度 |
| [Simthesizer](Simthesizer.md) | 动态 DAG 与 agent-driven simulator harness | 自动 harness 与图模拟器设计 |
| [LLMVisor](LLMVisor.md) | 轻量可加性延迟归因 | 在线调度代价模型和干扰归因 |

## 阅读原则

每篇解读固定回答以下问题：

1. 论文真正解决的决策是什么；
2. 模型输入、输出、优化变量和约束是什么；
3. 实验数字在什么硬件、模型、工作负载和 baseline 下成立；
4. 哪些假设不能直接迁移到 Ascend 或 Conditional-IS；
5. 哪些模块可以复现，哪些需要重新设计；
6. 如何验证它确实优于 vLLM、Pie、常博仓库和 Muyuan 的现有能力。

论文报告数字只作为方向依据，不作为本项目效果。Autopilot 的结论必须来自绑定
环境、源码、模型和 workload 的独立实验。
