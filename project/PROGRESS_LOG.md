# 项目进展日志

## 2026-09-07：建立项目进展中心

- 新建 `project/`，把项目定位、当前能力、设备效果、证据缺口和路线图集中维护。
- 明确当前主线是 AIC-NPU 真实成本模型闭环，不把合成模型示例视为性能成果。
- 将近期成功标准冻结为：不超过 12 个新增 NPU 测点，在 holdout 上达到或超过
  强人工基线，并在相同预算下优于 random/grid。
- 将第一个在线机制冻结为 stage-pressure adaptive token budget；TAPER 类 branch
  admission 需先通过 API 与算法语义可行性审计。

## 此前完成的里程碑

### 2026-09-03 至 2026-09-04：控制面基础

- 建立 Inference Graph IR 和 Conditional IS small-proposal 适配器。
- 建立严格 schema、证据导入、特征表、校准计划和搜索空间编译。
- 建立图捕获 profile、bucket planner 和 vLLM/vLLM-Ascend 运行时指标接入。

### 2026-09-05：策略、迁移与实验门禁

- 实现离线 selector、active acquisition、capacity/graph interaction repair。
- 实现 policy transfer、独立 holdout、guarded runtime routing。
- 在 short P32 独立 holdout 中验证 `40/64 + graph64`，平均 QPS 相对 fallback
  提升 `12.02%`；同时确认该收益不能直接迁移到 medium2k。

### 2026-09-06：阶段级机制与生命周期建模

- 实现 stage wavefront admission、prefix-preserving sharding、运行时闭包、
  lifecycle/retry/harness cost。
- short P96 的正式两对 wavefront 实验取得 `+18.04%` QPS 几何平均效果，
  并显著提高 proposal ACL Graph 命中率。
- 证明 480-sequence wave 相对 384-sequence wave 回退 `4.98%`，说明更大 batch
  不是单调更优。

### 2026-09-07：测量可信度与成本模型

- medium-context wavefront 变体未形成有效收益，避免将噪声误判为优化。
- 主机干扰 guard 在真实共享 NPU 环境中识别并排除进程变化污染。
- 实现 AIC-NPU stage/batch/shape bucket 成本模型、CLI 和合成示例。
- runner 增加 algorithm-call 与 engine-batch 双层 timeline，为真实成本校准准备输入。
- 完成 2026 年自动优化与 AI infra 论文扫描，并建立重点论文中文解读目录。

## 后续更新规则

每个重要进展追加一条记录，至少包含：日期、改动、实验环境、证据状态、结果、
失败或限制、关联文档。性能数字只有在写明 baseline、workload、噪声与质量门禁
后才能进入“有效效果”；诊断结果和合成数据必须明确标注。
