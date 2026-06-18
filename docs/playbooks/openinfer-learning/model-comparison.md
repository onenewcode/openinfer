# OpenInfer Model Comparison

> **TL;DR:** openinfer 的模型线不是“换一套权重就完事”，而是复杂度逐级上升的几种执行家族。Qwen3 是标准单卡 dense 样本；Qwen3.5 引入 hybrid attention 与 Triton AOT；DeepSeek-V2-Lite / V4 代表 MoE + 固定多卡拓扑；Kimi-K2 则把 MLA、MoE、TP/DP/EP 和专用运行时组织叠在一起。学习顺序应该按复杂度递增，而不是按“最新/最大模型”递增。

## 为什么一定要做模型线对照

很多人第一次接这个仓库会有两个误区：

- 误区 1：以为所有模型只是 `weights.rs` 不同；
- 误区 2：以为最复杂的模型线最能代表项目。

实际上正好相反：

- 模型线之间的差异，往往体现在 scheduler/executor/runtime 组织上；
- 最复杂的模型线最适合做“边界样本”，不适合做“第一样本”。

## 一张总对照表

| 模型线 | attention / 计算形态 | 并行 / 拓扑 | 运行时复杂度 | 构建复杂度 | 最适合拿来学什么 |
| --- | --- | --- | --- | --- | --- |
| Qwen3 | 标准 dense full attention | 单卡默认；可有 TP | 低到中 | 低 | 标准 serving 闭环 |
| Qwen3.5 | hybrid attention（linear + full） | 单卡 | 中 | 中到高（Triton AOT） | shared contract 不变、内部执行形态变化 |
| DeepSeek-V2-Lite | MoE + EP2 | 固定 EP2 / 2 卡 | 中到高 | 中 | 固定多卡 MoE 接法 |
| DeepSeek-V4 | 大型 MoE + compressor/indexer | 固定 MP8 / 8 卡 | 高 | 高 | direct engine、多 rank 运行时组织 |
| Kimi-K2 | MLA + MoE + INT4 experts | TP/DP/EP 组合 | 很高 | 高 | 模型专用运行时为什么必须留在模型 crate |

## 模型线 1：Qwen3

### 它是什么

Qwen3 是当前默认 feature，也是仓库最“标准”的模型线。

它有这些特征：

- dense 模型；
- 结构上最适合演示单卡 serving 主线；
- 具备真实服务复杂性：prefix cache、LoRA、KV offload、chunked prefill、unified step；
- 但没有马上把你拖进 MoE、MLA、多种并行拓扑。

### 为什么它是最佳第一样本

因为它能完整展示：

- `launch(...)` / `start_engine(...)`
- scheduler / executor 分工
- prefill / decode / unified step
- `TokenEvent` 生命周期
- 与 `openinfer-core` / `openinfer-kernels` 的边界

### 什么时候用它做模板

当你接入的新模型满足下面大多数条件时，Qwen3 是第一模板：

- dense 路径；
- 不需要新通信层；
- 没有全新 attention 语义；
- 主要复杂性在单机 GPU 算子和服务流程。

## 模型线 2：Qwen3.5

### 它比 Qwen3 多了什么

Qwen3.5 不是简单的“Qwen3 换权重”。它带来的是：

- hybrid attention
- recurrent state
- graph slot / decode graph 管理
- build-time Triton AOT

这使得它的复杂性明显高于 Qwen3，但仍保留：

- 同一套 server 入口；
- 同一套 `EngineHandle` / `GenerateRequest` / `TokenEvent` 契约。

### 它最适合拿来学什么

- 为什么共享层不需要因为新模型而重写；
- 为什么模型 crate 内部可以有完全不同的状态组织；
- 为什么 build-time codegen 也应该被 feature gate 在模型线内部管理。

### 它适不适合作为第一模板

不适合。

适合在你已经读懂 Qwen3 之后，再拿来回答：

> “如果一个新模型执行形态和 Qwen3 不一样，但我又不想破坏整个架构边界，该怎么做？”

## 模型线 3：DeepSeek-V2-Lite

### 它的代表性是什么

DeepSeek-V2-Lite 是较小的 MoE + 固定 EP2 路径，提供了一种非常有用的中间样本：

- 比 Qwen 系更复杂；
- 但比 DeepSeek-V4 / Kimi-K2 仍然更容易作为切入口；
- 可以用来理解“固定多卡拓扑也应由模型层解释”。

### 它最值得观察的点

- `probe_config_json(...)` 如何在同一 `model_type` 下再做形状级识别；
- `launch(...)` 如何明确忽略不支持的 CUDA Graph；
- 固定设备 `vec![0, 1]` 是如何被模型层直接表达的。

### 什么时候拿它做模板

当你接入的新模型：

- 已经不是单卡 dense；
- 需要少量固定多卡拓扑；
- 但还不需要一个像 Kimi 那么重的专用运行时时。

## 模型线 4：DeepSeek-V4

### 它和 Lite 的本质区别

DeepSeek-V4 的复杂性已经不只是“多两张卡”，而是：

- 更重的 MoE 路径；
- compressor / indexer 等附加结构；
- direct engine 运行时组织；
- 固定 MP8 拓扑。

### 它最值得学什么

它最值得学的是：

- 当模型的运行时语义已经和普通 dense 模型不共形时，模型 crate 应该大胆长出自己的组织方式；
- server 仍然可以保持纯分发；
- 统一契约层仍然不用知道内部复杂性。

### 它适合什么时候参考

当你要接入的新模型满足以下特征时：

- 需要固定多卡 rank 拓扑；
- 运行时会长出专门的 worker/scheduler 组织；
- 共享 runtime 只能提供底层积木，不能接管主要控制流。

## 模型线 5：Kimi-K2

### 为什么它是“复杂性边界样本”

Kimi-K2 同时叠了几层难度：

- MLA attention
- MoE
- Marlin INT4 experts
- TP/DP/EP 可配置拓扑
- `runner/worker/load_balancer` 专用运行时组织

所以它更像“这个架构能承受多复杂”的边界样本，而不是日常模板。

### 它最该教你的不是实现，而是边界

你真正要从 Kimi 学到的是：

- 当复杂度达到一定程度，模型 crate 必须拥有自己的 runtime center；
- 不要试图把这种复杂性抽成一个通用 server 或 core 框架；
- `launch(...)` 作为 server-facing policy layer 的价值，在复杂模型线里会更明显。

### 什么时候拿它做模板

只有当你接的新模型已经满足：

- 新 attention 语义；
- MoE；
- 多并行维度；
- 复杂通信后端；
- 专用 worker/runner 组织

这些条件时，才把 Kimi 当主要模板。

## 一张“该拿谁做模板”的决策表

| 你的新模型更像哪种 | 首选模板 | 次选模板 |
| --- | --- | --- |
| 单卡 dense，主要是配置和权重差异 | Qwen3 | Qwen3.5 |
| 仍是 dense，但执行形态明显变化 | Qwen3.5 | Qwen3 |
| 小型 MoE，固定少量多卡拓扑 | DeepSeek-V2-Lite | DeepSeek-V4 |
| 大型 MoE，多 rank 运行时组织 | DeepSeek-V4 | Kimi-K2 |
| MLA + MoE + 多并行维度 | Kimi-K2 | DeepSeek-V4 |

## 一张“适合作为第一阅读样本吗”的表

| 模型线 | 适不适合第一样本 | 原因 |
| --- | --- | --- |
| Qwen3 | 适合 | 主线最清晰，复杂性真实但可控 |
| Qwen3.5 | 不太适合 | hybrid attention 与 recurrent state 会过早放大复杂性 |
| DeepSeek-V2-Lite | 不太适合 | 需要先有多卡 / MoE 心智模型 |
| DeepSeek-V4 | 不适合 | direct engine + MP8 + MoE 复杂度过高 |
| Kimi-K2 | 最不适合 | MLA + MoE + TP/DP/EP 叠加，容易完全丢失主线 |

## 接新模型时最常犯的错误映射

### 错误 1：把 DeepSeek/Kimi 的复杂路径当作通用模板

后果：

- 单卡 dense 模型也被迫背上大量多卡 / 通信 / runner 组织噪音；
- 项目很快变成“为了未来可能复杂而过度设计”。

### 错误 2：把 Qwen3 当作所有模型都能硬塞进去的宿主

后果：

- 新模型真正的执行差异被藏进越来越多的 `if` / `match`；
- 看起来复用，实际把原本干净的模型线污染掉。

### 错误 3：把 `openinfer-core` 当成模型差异回收站

后果：

- shared runtime 会失去清晰边界；
- 每支持一个新模型，core 都会变得更难读、更难测。

## 你读完这篇之后该怎么选模板

### 如果你还不确定

先问四个问题：

1. attention 形态变了吗？
2. KV 组织变了吗？
3. 是否需要多卡固定拓扑或通信层？
4. scheduler/executor 是否还能沿用 Qwen3 的思路？

### 简化判断

- 四个都基本没变 -> 从 Qwen3 开始
- 1 或 2 变了，但 3 不重 -> 看 Qwen3.5
- 3 开始变重 -> 看 DeepSeek-V2-Lite / V4
- 1/2/3 全都重，而且运行时要大改 -> 看 Kimi-K2

下一步建议看：`docs/playbooks/openinfer-learning/new-model-integration.md`

