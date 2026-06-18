# openinfer-deepseek-v4 工作原理

`openinfer-deepseek-v4` 是 DeepSeek V4 这条模型线的专属引擎 crate。它把 DeepSeek V4 所需的配置解析、分 rank 权重加载、direct engine、运行时子系统、E2E 验证入口都收在自己内部，并通过 `deepseek-v4` feature 启用。

## 一句话先理解

这条模型线和 Qwen3 最大的不同，不只是模型更大，而是执行结构更复杂：

- 有 MP8 固定拓扑；
- 有压缩器（compressor）；
- 有 indexer / sparse attention；
- 有 HC 路径；
- 有 MoE；
- 可选接入 `openinfer-comm` 的 PPLX/DeepEP 风格专家并行通信后端。

所以这个 crate 更像一台“面向 8 卡、结构很异构的专用执行器”。

## 它在架构里的位置

从 server 往下看：

```text
server / bench_serving
  -> openinfer-deepseek-v4::launch / start_engine
      -> direct engine
          -> runtime/*
              -> openinfer-kernels
              -> NCCL / 可选 openinfer-comm
```

这里的关键词是 `direct`：当前服务路径不是像 Qwen3 那样围绕通用 scheduler 演化出来的，而是 DeepSeek V4 自己拥有一套 direct generation 路径。

## 这个 crate 里最关键的几层

### 1. `weights.rs` / `model.rs`

负责把 DeepSeek V4 的 checkpoint 切成 rank 视角可以直接消费的权重视图：

- 每个 rank 加载自己需要的那部分张量；
- 权重命名、manifest、分块布局都在这里收口；
- 运行时拿到的是已经按 V4 结构组织好的 rank-local model。

这层解决的是“8 卡里的每一张卡手上到底该有什么”。

### 2. `direct/`

`direct` 是北向入口背后的执行框架：

- 请求如何进入；
- 每个 worker / rank 怎么被拉起来；
- scheduler 如何推进 direct generation；
- affinity / worker 生命周期怎么管理。

对 server 来说，它最终暴露成一个 `EngineHandle`；但 crate 内部真正拥有请求推进权的是 `direct`。

### 3. `runtime/`

这是 DeepSeek V4 的真正“算子编排层”，而且是按子系统拆开的：

- `attention.rs` / `attention_base.rs`
- `compressor.rs`
- `indexer.rs`
- `collectives.rs`
- `moe.rs`
- `moe_pplx.rs`
- `core.rs`
- `state.rs`

这反映了 V4 的真实执行复杂度：它不是一个“标准 attention + MLP”模型，而是多个高度异构部件的组合。

## 你应该怎么理解它的运行主线

最重要的不是记住每个函数名，而是抓住这条数据流：

1. 输入 prompt 先进入 direct engine；
2. 各 rank 基于自己的权重分片做局部计算；
3. attention / compressor / indexer / HC / MoE 这些子系统在 `runtime/` 里按模型结构串起来；
4. 需要跨 rank 聚合的地方走 NCCL 或可选的 `openinfer-comm` 后端；
5. 最终得到 rank-local logits，再汇总成输出。

也就是说，这个 crate 的重点是：**把 DeepSeek V4 这种复杂异构模型拆成一组可维护的运行时子系统**。

## 为什么这里有 `pplx-ep` feature

`pplx-ep` 是一个可选通信后端开关：

- 默认 `deepseek-v4` 只启用模型自身和 `openinfer-kernels` 相关路径；
- 开启 `pplx-ep` 后，才会把 `openinfer-comm` 拉进来，为 decode MoE 提供更重的 all-to-all / EP 通道。

这说明设计上把“模型逻辑”和“更激进的通信实现”拆开了：

- 模型 crate 负责定义需要什么通信语义；
- 是否接更复杂的硬件后端，则交给 feature 决定。

## 它刻意不追求什么

这个 crate 没有试图把 DeepSeek V4 硬塞进一个通用 `ModelForward` 抽象里，而是保留了：

- direct engine；
- 按子系统拆分的 runtime；
- rank-local 权重与状态视图；
- 模型专用的 bring-up / e2e / kernel check 入口。

这样做的原因很简单：V4 的结构太特殊，过度抽象只会把关键执行语义藏起来。

## 最重要的心智模型

理解 `openinfer-deepseek-v4` 时，可以把它看成：

- 上面接统一 server；
- 下面接共享 kernels 与通信能力；
- 中间自己拥有一整套面向 DeepSeek V4 的“多子系统执行编排”。

它不是一个普通模型壳子，而是 **DeepSeek V4 在 openinfer 里的专用运行时**。
