# 华为 Ascend 自动寻优相关工作

调研日期：2026-09-07

## 核心结论

华为已经公开了“修改参数、拉起真实服务、启动 benchmark、根据结果迭代搜索”
的完整工具链。因此，单独实现一个通用 vLLM-Ascend 参数循环已经不足以构成
Autopilot 的主要贡献。

目前最相关的公开能力包括：

- Ascend `msserviceprofiler` 中的 Serviceparam Optimizer；
- Ascend `msmodeling` 中的 OptiX；
- `msmodeling` Throughput Optimizer 和 TensorCast；
- 华为云 `acsprof` 参数寻优和 `acs-advisor` PD 配比仿真。

## Serviceparam Optimizer

[Serviceparam Optimizer](https://github.com/Ascend/msserviceprofiler/blob/master/docs/zh/serviceparam_optimizer_instruct.md)
位于公开的 [Ascend/msserviceprofiler](https://github.com/Ascend/msserviceprofiler)
仓库。它是基于 PSO 的服务参数自动寻优工具。

### 已有能力

- 支持 MindIE 和 vLLM；
- 自动修改服务参数或 benchmark 参数；
- 自动启动、健康检查和停止推理服务；
- 自动调用 AISBench 或 `vllm_benchmark`；
- 以吞吐为目标并施加 TTFT、TPOT 等延迟约束；
- 真机轻量模式直接迭代测量；
- 仿真模式使用 profiling 数据训练 XGBoost 延迟模型，再模拟调度；
- 使用 Early Rejection 提前排除候选；
- 处理 OOM、NPU、网络和 IO 错误，支持重试、断点恢复和结果保存；
- 支持 PD 混部、PD 分离和部分多机部署场景。

### 搜索空间表达

目标字段支持整数、浮点、布尔、枚举和带步长范围，也支持比例、共享总量、乘积
与除法等派生关系。因此可以搜索：

- `max_num_seqs`；
- `max_num_batched_tokens`；
- `concurrency` 和 `request_rate`；
- memory utilization；
- vLLM 环境变量和任意命令行片段；
- compilation/graph 开关；
- MindIE 配置文件中的调度参数。

### 插件能力

2025-11-07 的上游记录已经说明支持自动寻优插件。公开的
[插件开发接口](https://github.com/Ascend/msserviceprofiler/blob/master/docs/zh/serviceparam_optimizer_plugin_instruct.md)
允许扩展：

- 参数配置和搜索空间；
- 服务框架适配器；
- benchmark 适配器；
- 配置写入、服务启停和结果提取。

因此，“做成插件以适配不同框架和 benchmark”本身也不是空白。

## msModeling OptiX

[OptiX](https://github.com/Ascend/msmodeling/blob/master/docs/zh/user_guide/msmodeling_optix_user_guide.md)
是 `Ascend/msmodeling` 中的服务化实测寻优入口。其产品定义和执行方式与
Serviceparam Optimizer 高度接近：

```bash
msmodeling optix -e vllm -b vllm_benchmark
```

它会根据 TOML 搜索空间反复拉起 vLLM 或 MindIE、执行 benchmark，并用 PSO
搜索满足延迟约束的高吞吐参数。当前公开文档还包括：

- vLLM 环境变量、CLI 参数和 MindIE 配置字段注入；
- 参数间 ratio/share/product 等依赖；
- PSO top-k 精调；
- 部署环境隔离、健康检查和结果 CSV；
- `optix-param-recommend` 搜索范围推荐；
- Web UI 和 agent skill 入口。

从公开目录和文档看，OptiX 是当前应优先评估的 Ascend 真机实验执行后端。
本文不在缺少提交历史证据时断言它与旧 Serviceparam Optimizer 的代码继承关系，
但二者能力边界明显重叠。

## msModeling Throughput Optimizer

[Throughput Optimizer](https://github.com/Ascend/msmodeling/blob/master/docs/zh/user_guide/msmodeling_throughput_optimizer_user_guide.md)
主要解决部署前容量规划，而不是反复启动真实服务。

### 已有能力

- 拦截 PyTorch 计算图并估计算子、内存和整体推理性能；
- 在 TTFT、TPOT SLO 下最大化 token throughput；
- 搜索 TP、DP、EP、MoE-DP、batch 和 concurrency；
- 支持 PD 混部、PD 分离和 P/D 实例配比；
- 比较不同 Ascend 设备和卡数；
- 支持量化与 compilation 配置；
- 默认使用解析 Roofline 模型，也可读取实测算子 profiling 数据；
- profiling shape 缺失时先插值，再回退到解析模型。

它比 Autopilot 当前的初版 AIC-NPU 模型覆盖更完整的单模型算子和并行策略。
我们不应重复建设一个泛化的 Ascend 单模型 Roofline 仿真器；更合理的是研究如何
消费它的 stage cost，补充 inference-scaling 多模型阶段图、真实算法 fanout 和
跨阶段重叠。

## 华为云 acsprof 与 acs-advisor

[acsprof 参数寻优](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910031.html)
在给定参数空间中建立 TTFT、TPOT、QPS 等 SLO 与配置的关系，再迭代寻找较优
组合。当前公开文档说明其已适配 Ascend-vLLM 的 PD 混部场景，工具能力通过
AscendCloud 软件包交付。

[acs-advisor](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910030.html)
使用 SimPy 对 vLLM PD 混部和 PD 分离进行服务调度仿真，在负载与 SLO 下搜索
混合实例、prefill 实例、decode 实例数量及 P/D 配比。它依赖真实 profiling
打点训练单步预测模型；官方文档也提示 prefix cache 等特性可能造成偏差。

这两项说明华为云侧已经覆盖普通服务参数寻优和 PD 容量规划，但公开可扩展性与
代码可获得性不如 Ascend GitHub 仓库清晰。

## 对 Autopilot 的直接影响

以下内容不应再作为我们的核心创新：

- 通用的“修改参数、启动 vLLM、压测、继续搜索”循环；
- PSO 搜索本身；
- 任意 CLI/env 参数注入；
- 普通 vLLM 的 TP/DP/EP、batch 或 PD 配比仿真；
- 仅仅把执行器封装成插件。

Autopilot 应优先补充华为工具公开资料中尚未看到的部分：

1. inference-scaling 算法的多模型、多阶段语义图；
2. base/proposal 引擎配置、graph coverage 与算法 fanout 的联合约束；
3. decision-critical 少量选点，而不是只运行通用 PSO；
4. ABBA、同配置 replay、主机污染、质量和配置闭包组成的证据门禁；
5. 跨上下文、并发和算法路径的支持域判断与拒绝发布；
6. stage wavefront 和 stage-pressure admission 等算法阶段感知运行时机制。

产品上可以把 OptiX 作为一个执行后端：Autopilot 生成语义合法且信息价值高的
候选，OptiX 负责参数注入、服务生命周期和 benchmark，结果再进入 Autopilot
的证据与策略系统。
