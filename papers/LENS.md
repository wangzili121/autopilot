# LENS 论文解读

## 基本信息

- 论文：[Latency Prediction for LLM Inference on NPU Systems](https://arxiv.org/abs/2606.18042)
- 系统名：LENS
- 作者机构：汉阳大学
- 提交时间：2026-06-16

## 一句话判断

LENS 是目前与我们 Ascend 场景最直接相关的论文。它的价值不是复杂模型，而是承认
NPU 的编译 bucket 会造成不连续性能，并用每个 bucket 两个端到端测点恢复线性
组成项。AIC-NPU 第一版正是沿这个方向实现的。

## 为什么 GPU 方法在 NPU 上会失效

论文指出三个困难：

1. 商用 NPU 不公开足够细的微架构信息，难以做 cycle-level simulator；
2. 编译器会做难以预测的异构 engine fusion，按 GPU kernel 粒度相加可能严重失真；
3. 静态 shape 编译使延迟随序列长度呈阶梯函数，而不是平滑函数。

论文中，直接迁移已有 GPU 分解方法的误差最高可到 493%。因此不能假定“更短输入
一定更快”或用跨 bucket 插值。

## 核心模型

对输入长度 `l_in`，系统选择第一个能容纳它的 bucket。prefill latency 是该 bucket
固定的 `TTFT_b`。decode 时 KV 长度逐 token 增长，跨越 bucket 后切换对应的
`TBT_b`：

```text
T_E2E = TTFT_bucket(l_in) + sum_j(output_steps_in_bucket_j * TBT_bucket_j)
```

若一次 decode 始终处在同一 bucket，则同 bucket 的两个不同输出长度 `L1`、`L2`
给出两个方程，可以直接解出 `TTFT_b` 和 `TBT_b`。这就是“两点测量”的来源，不是
一个普遍的 two-shot 机器学习结论。

## batch 扩展

论文区分了两类 NPU 执行：

- 普通 bucket batch：prefill 按请求 bucket 累加；decode 的 shape 由 batch 中最大
  KV 长度和最长输出决定，短请求可能被 padding 拖着走。
- 优化 attention kernel：每个请求按真实 KV 长度计成本，再按 batch size 归一，
  但整个 batch 仍共享 forward step。

这提醒我们必须从实际 vLLM-Ascend 路径确认 batching 语义，不能只看接口请求数。

## 实验结果

- 覆盖 Inferentia2、TPU v4/v5e/v6e 四种 NPU，三种模型和四种数据集；
- 主评估 248 个 case，mean error 2.15%，median 1.69%；
- 89.9% case 的误差低于 5%；
- 两个 case study 显示更大 batch 不一定吞吐更高，更细 bucket 也不一定更快。

这些结果支持 bucket 建模，但硬件中不包含 Ascend，仍需本地验证。

## 关键限制

1. 每个 NPU、模型和编译配置仍需单独 profiling，不能跨设备零成本迁移。
2. 主要模型是编译 binary 的端到端 latency，不解释算子、排队、通信和多模型竞争。
3. 两点解方程对噪声敏感，两个 output extent 太接近会放大误差。
4. Conditional-IS 的 target score 不是普通 autoregressive decode，公式不能直接套用。
5. 在线 continuous batching 的 queueing 外部性不在模型内。

## 对 Autopilot 的映射

我们将 LENS 的 bucket 视为最小可校准单元，但扩展 stage 维度：

```text
hardware + runtime + model role + stage + batch + shape bucket
    -> fixed cost + token extent slope + uncertainty
```

stage 至少包括 `candidate_generate`、`proposal_rollout_generate` 和 `target_score`。
每个 bucket 绑定环境、模型集和有效配置哈希，禁止跨环境误用。

## 采样设计

- 每个选中的 bucket 至少两个相距足够大的 token extent；
- 每个点重复测量并记录 stddev；
- 先隔离测 primitive，再用 loaded run 学 queueing residual；
- 采样真实 `engine_batch_timeline`，不能把外层 algorithm-call wall time 当 kernel time；
- bucket 边界从运行时/编译配置和实测变化检测，不凭经验手写。

## 我们已经实现和仍缺什么

已实现：bucket schema、加权仿射拟合、区间估计、范围外 abstention、图成本串行上界。

仍需实现：严格 trace importer、bucket 边界检测、跨阶段 overlap residual、资源下界，
以及用真实 Ascend 测量验证“两点模型”是否足够。如果 target score 在 bucket 内仍明显
非线性，应退化为局部单调模型，而不是维护论文形式。
