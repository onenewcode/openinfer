# openinfer-kernels 工作原理

`openinfer-kernels` 是 openinfer 里真正拥有“底层 GPU 算子资产”的 crate：它负责 CUDA / Triton / FlashInfer 相关代码的编译产物、FFI 声明、张量与布局元数据、以及 Rust 侧的算子包装。可以把它看成整个工程的**kernel 供应层**。

## 一句话先理解

如果把整个 openinfer 拆成三层：

- 上层模型 crate 决定“这一步要跑哪些算子、按什么顺序跑”
- `openinfer-core` / 调度器决定“请求怎么排、状态怎么流动”
- `openinfer-kernels` 决定“这些 GPU 算子到底怎么被编译出来、怎么被 Rust 调用”

所以它拥有的是**算子实现与 ABI 边界**，不是模型调度策略。

## 它在系统里的位置

它站在每条模型线的下方：

```text
openinfer-qwen3-4b / openinfer-qwen35-4b / openinfer-kimi-k2 / deepseek-v4
  -> openinfer-kernels::ops / ffi / tensor / paged_kv
      -> CUDA / Triton / FlashInfer / cuBLAS
```

模型 crate 不应该自己重复拥有一套 CUDA FFI 和 kernel 编译逻辑，而是通过这个 crate 来共享底层能力。

## 这个 crate 主要由哪几层组成

### 1. `build.rs`：底层算子的编译所有者

`openinfer-kernels/build.rs` 是整个仓库里最关键的 build script 之一。它负责：

- 探测 GPU SM 目标（`nvidia-smi` 或 `OPENINFER_CUDA_SM`）
- 调 `nvcc` 编译 `csrc/*.cu`
- 按 feature 决定是否启用额外代码生成
  - `qwen35-4b`：Triton AOT
  - `deepseek-v4`：TileLang / CuTe DSL 相关产物
  - `kimi-k2`：MLA / MoE / Marlin 相关 CUDA
- 把最终对象文件和静态库接进 Rust 链接流程

也就是说，这个 crate 不只是“声明 FFI”，而是**真正拥有 GPU 代码的构建权**。

### 2. `ffi/`：Rust 和底层 kernel 的 ABI 边界

`src/ffi.rs` 及其子模块把底层符号按领域拆开：

- `shared`
- `qwen35`
- `deepseek`
- `kimi`
- `deepep`
- `lora`

这些模块的职责不是提供“好用接口”，而是把 C/CUDA 世界的符号原样暴露给 Rust。

所以可以把 `ffi` 看成：

- 最接近底层实现的一层；
- 语义偏“这有个符号可以调”；
- 还没有上升到“这是一个模型可以直接理解的高层 op”。

### 3. `ops/`：对 FFI 的 Rust 化包装

`ops` 是模型 crate 最常接触的表面。

这里把底层符号整理成更清晰的 Rust 函数，例如：

- attention
- embedding
- linear / gemm / gemv
- norm
- sampling
- elementwise
- kimi / deepep 的专用 op

这一层做的事情是：

- 校验输入 shape / layout
- 组织参数
- 选择更贴近模型语义的函数名
- 把“直接调裸 FFI”变成“调一个可组合的 Rust op”

所以 `ops` 才是“共享算子库”，而 `ffi` 更像“共享符号边界”。

### 4. `tensor.rs`：kernel 侧张量与调用描述

`tensor.rs` 里定义了很多 kernel-report 和 bench 需要的基础类型，例如：

- `DeviceContext`
- `DeviceMatrix`
- `GpuWeight`
- `TensorSpec`
- `KernelCall`

它们的作用不是做通用张量库，而是给这个工程里的 kernel 调用提供一套足够明确、又便于报告和分析的描述方式。

很多 benchmark / kernel report 能成立，依赖的正是这里的元数据层。

### 5. `paged_kv.rs`：kernel 视角的 paged KV 布局

`PagedKvLayout` 很能体现这个 crate 的定位。

它只关心：

- page size
- layer 数
- KV head 数
- head dim
- 一个 page 里 K/V 如何排布

它**不**关心：

- 页面归谁所有
- 哪个请求持有哪个 block
- eviction / prefix cache / admission

这些属于更上层的 KV 管理逻辑。

也就是说，这里只保存“kernel 需要知道的物理几何”，不保存“运行时策略”。

### 6. `typed_ops.rs` / `forward_pass.rs`：让算子组合更可表达

这部分的意义是把“单个 kernel 调用”继续往上抬一层，变成更容易被模型 crate 复用的 typed pipeline 或 forward helper。

它不是完整模型 DAG，但能减少模型侧重复写很多样板拼接代码。

## 为什么 `KERNELS.md` 很重要

`KERNELS.md` 可以视为这个 crate 的“人工可读索引”：

- 每个 op 对应哪个 Rust wrapper
- 对应哪个 FFI 符号
- 对应哪个 `.cu` 源文件
- 它服务哪条模型路径

这背后反映的是一个设计选择：openinfer 不希望 kernel 资产只存在于“源码 + 记忆”里，而是希望有一份稳定路由表，帮助人快速回答：

- 这个 op 最终调用了哪个 kernel？
- 这条模型路径到底依赖哪些 backend？
- 我改一个 kernel 会影响到哪些模型？

## 它刻意不做什么

这个 crate 的边界非常关键：

- 不拥有请求调度器；
- 不拥有 `KvPool` / `BlockPool` 这类运行时状态；
- 不决定某个模型一层层怎么串；
- 不直接暴露 OpenAI / vLLM 风格的服务接口；
- 不做模型级 correctness 策略。

它只做一件事：**把可复用的底层 GPU 算子以稳定方式提供出来**。

## 理解它时最重要的心智模型

可以把 `openinfer-kernels` 看成“硬件能力翻译层”：

- 向下，它懂 CUDA/Triton/FlashInfer/cuBLAS；
- 向上，它给模型 crate 提供 Rust 可调用、带形状语义的 operator surface；
- 中间，它通过 build.rs + FFI + op wrappers 把两边接起来。

所以当你在模型 crate 里看到一个 attention / gemm / sampling 调用时，真正的“落地执行权”大多都在这个 crate 里。
