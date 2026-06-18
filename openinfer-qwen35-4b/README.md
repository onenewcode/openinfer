# openinfer-qwen35-4b 工作原理

`openinfer-qwen35-4b` 是 Qwen3.5-4B 这一条模型线的专属执行 crate。它拥有 Qwen3.5 所需的配置、权重加载、prefill / decode / unified forward、recurrent state、调度器、测试和 bench 接口，并通过 cargo feature `qwen35-4b` 整体受控。

## 一句话先理解

和 `openinfer-qwen3-4b` 一样，它也是一台“模型专属引擎”；但 Qwen3.5 的内部结构不是纯 full attention，而是：

- 24 层 linear attention
- 8 层 full attention

因此这个 crate 除了普通 paged KV 之外，还必须额外管理 **recurrent state**。这正是它和 Qwen3-4B 路径最本质的区别。

## 为什么这个 crate 是 feature-gated

Qwen3.5 不是默认构建的一部分，原因不在模型逻辑，而在构建链：

- 它依赖 `openinfer-kernels/qwen35-4b`
- 后者需要 Triton AOT 生成某些 GDR chunkwise prefill kernel
- 这意味着构建时要有 Python + Triton

所以工程做了一个非常明确的边界：

- 默认构建只保留纯 Rust + CUDA 的主路径
- Qwen3.5 这条线只有在显式启用 `--features qwen35-4b` 时才被编进来

这也是为什么 crate 根部直接有：

```rust
#![cfg(feature = "qwen35-4b")]
```

即：**不开 feature 时，这个 crate 等价于不存在**。

## 它在架构里的位置

从上往下看：

```text
server / frontend
  -> EngineHandle
      -> openinfer-qwen35-4b scheduler
          -> Qwen35Model / Qwen35Executor
              -> paged KV + recurrent state
              -> openinfer-kernels
```

和 Qwen3 类似，上层只需要一个通用引擎句柄；但在 crate 内部，Qwen3.5 的执行状态明显更复杂，因为它不是只有 KV。

## 这个 crate 的核心复杂度来自哪里

### 1. 混合注意力架构

Qwen3.5-4B 不是“每层都长一样”的模型。

- full attention 层需要 paged KV
- linear attention 层需要 recurrent state

这意味着执行器不能只维护一套 KV state，然后在 decode 时一律追加一 token 就结束；它还要保证 recurrent 部分在：

- prefill
- decode
- 图捕获 batch decode

这些路径里都能保持一致。

所以从工程角度看，这个 crate 真正的主题其实是：**如何把混合架构包装成可服务的统一引擎**。

### 2. recurrent state 是一等公民

在 Qwen3.5 路径里，recurrent state 不是旁枝末节，而是核心运行时状态之一。

相关模块包括：

- `recurrent.rs`
- `recurrent_state.rs`
- `batch_decode_graph.rs`
- `scheduler.rs`

这里的分工大致是：

- `recurrent_state.rs`：表示和管理线性注意力所需的状态张量
- `recurrent.rs` / `ops.rs`：封装相关运算
- `batch_decode_graph.rs`：给 batch decode 的 CUDA Graph 提供稳定槽位
- `scheduler.rs`：在请求生命周期里维护这些状态与 slot 的对应关系

## 两套入口：server 用一种，测试/调试用另一种

### 1. server-facing 入口

对服务层来说，最重要的是：

- `start_engine(...)`
- `launch(...)`
- `start_engine_with_capacity(...)`

这些函数会把：

- 模型路径
- 设备 ordinal
- 是否启用 CUDA Graph

转换成一个通用 `EngineHandle`，让 root server 不需要知道 Qwen3.5 内部到底有多少 recurrent 状态、多少 graph slot。

### 2. runtime 入口

crate 还专门暴露了 `runtime` 模块，给：

- model-local 测试
- 调试
- benchmark

使用。

这层会暴露：

- `Qwen35Executor`
- `PrefillPlan`
- `DecodePlan`
- `Qwen35Model`
- `MAX_BATCH`

所以它的设计和 Qwen3 类似：北向接口保持统一，南向接口保留真实执行细节。

## 它到底怎么跑请求

### 路径 1：prefill

新请求的 prompt 先走 prefill：

1. scheduler 接收请求；
2. 为它分配 KV / graph slot 相关资源；
3. `Qwen35Model` 执行 batch prefill；
4. full-attention 层写入 KV；
5. linear-attention 层生成对应 recurrent state；
6. 首 token 与可选 logprobs 返回给 scheduler；
7. 请求进入 active 集合。

与 Qwen3 不同的是，这一步不仅要“把 prompt 算完”，还要把后续 decode 真正依赖的 recurrent state 初始化好。

### 路径 2：decode

decode 阶段，Qwen3.5 需要同时推进：

- paged KV
- recurrent state

为此它使用 `BatchDecodeGraphState` 来维持一组稳定地址槽位，让 CUDA Graph 能安全重放。每个 active request 不只对应一个“还活着的请求”，还对应一个固定 graph slot。

这也是为什么 scheduler 里特别强调：

- slot 分配
- 退役后 compaction
- active request 顺序必须和 graph slot 对齐

因为对这种混合架构来说，**状态地址稳定性本身就是运行时 contract 的一部分**。

## 为什么 scheduler 在这里尤其重要

Qwen3.5 的 scheduler 虽然外表和 Qwen3 相似，但内部约束更强：

- 它要维护 `KvState`
- 还要维护 `RecurrentState`
- 还要维护 decode graph 的 slot 稠密性

请求退出时，如果只删掉逻辑请求、不处理 slot compaction，后续 graph replay 就会出现“顺序和状态错位”的问题。

所以这里的 scheduler 不是单纯的“从 channel 收请求然后 batch 一下”，而是混合状态机的拥有者。

## 它为什么没有像 Qwen3 那样强调 prefix cache / offload

不是因为 Qwen3.5 不需要性能优化，而是因为 Qwen3.5 的核心难点和 Qwen3 不一样。

Qwen3-4B 的主要复用对象是 paged KV，于是 prefix cache / offload 很自然。
而 Qwen3.5 的线性注意力部分还携带 recurrent state，这意味着“前缀复用”不能只恢复 KV 页面就结束。

换句话说：

- Qwen3 的复用问题主要是“KV 怎么复用”
- Qwen3.5 的复用问题还包含“线性状态怎么复用”

这也是为什么这个 crate 的 README 重点更应该放在混合架构和状态管理，而不是简单照搬 Qwen3 的 prefix cache 叙事。

## 这个 crate 的真正价值

`openinfer-qwen35-4b` 的核心贡献，不只是把 Qwen3.5 模型跑通，而是把一个：

- 有 full attention
- 有 linear attention
- 有 recurrent state
- 有 Triton AOT 依赖

的混合模型，封装成了和其他模型线一样可被 root server 统一加载的 `EngineHandle`。

所以它本质上是在证明一件事：**openinfer 的“每模型一套引擎”架构，足以容纳结构显著不同的模型线，而不需要把所有模型都塞进一个过度通用的统一运行时里。**
