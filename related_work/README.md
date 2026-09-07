# 相关工作索引

本目录总结与 Inference Autopilot 存在功能交集的开源系统、官方工具和研究原型，
重点回答三个问题：它们已经做了什么、与我们重叠到什么程度、Autopilot 还应
解决什么。

论文原理和实验数字的详细解读仍放在 [`papers/`](../papers/)；本目录按产品能力
组织，不把论文原型、开源实现和生产工具混为一类。

## 文档导航

- [相关工作全景](LANDSCAPE.md)：按自动实测、性能仿真、稀疏建模、运行时策略
  和 inference-scaling runtime 分类总结已有工作。
- [华为 Ascend 自动寻优](HUAWEI_ASCEND_AUTOTUNING.md)：专项梳理
  Serviceparam Optimizer、OptiX、msModeling Throughput Optimizer、acsprof 和
  acs-advisor。
- [能力对比与项目边界](COMPARISON_MATRIX.md)：说明哪些能力应直接复用，哪些
  能力仍是 Autopilot 的主攻方向。

## 维护口径

每项相关工作尽量记录：

1. 项目归属、代码或官方文档链接；
2. 输入、输出、搜索变量和优化目标；
3. 真机、仿真或混合执行方式；
4. 是否开源、是否可扩展、支持哪些框架；
5. 与 Autopilot 的重叠和缺口；
6. 调研日期以及可能随版本变化的结论。

当前内容基于截至 2026-09-07 可见的公开资料。功能边界会随上游版本变化，进入
实现前仍需重新审计对应仓库。
