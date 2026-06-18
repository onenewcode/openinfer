# openinfer-core 工作原理

`openinfer-core` 是 openinfer 的“公共运行时底座” crate。它本身不是某个具体模型，也不是 HTTP 服务层；它做的事更像是把**所有模型都要用的公共契约、公共 GPU 算子封装、公共 KV 辅助组件、公共权重加载逻辑**放在一个地方，避免每个模型 crate 都各自复制一套。

如果只记一句话，可以这样理解：

- `openinfer-engine` 定义“引擎跟外界怎么说话”；
- `openinfer-kernels` 提供“GPU kernel 和 tensor 基元”；
- `openinfer-core` 站在两者中间，把这些公共能力整理成模型 crate 更容易直接使用的运行时工具箱。

## 这个 crate 在整体架构里的位置

从依赖关系看，`openinfer-core` 处在“模型之下、kernel 之上”的位置：

```text
openinfer-qwen3-4b / openinfer-qwen35-4b / deepseek / kimi ...
  └─ openinfer-core
       ├─ openinfer-engine（请求/事件/并行契约）
       ├─ openinfer-kernels（tensor 与 kernel 调用）
       └─ 自己的公共辅助模块（KV、paged plan、weight loader、CUDA Graph 等）
```

所以它的核心价值不是“做某一个算法”，而是把那些**跨模型复用**的承重能力集中起来。

## 它实际上分成哪几类东西

看 `src/lib.rs`，这个 crate 暴露的内容大概可以分成四层：

### 1. 对外契约的再导出

这部分主要来自：

- `engine.rs`
- `sampler.rs`
- `parallel.rs`

其中：

- `engine.rs` 基本上直接 `pub use openinfer_engine::engine::*`
- `sampler.rs` 直接 `pub use openinfer_engine::sampler::*`
- `parallel.rs` 直接 `pub use openinfer_engine::parallel::*`

这说明一个很重要的设计意图：**模型 crate 只依赖 `openinfer-core`，不需要分别去碰 `openinfer-engine` 和其他更底层 crate。**

换句话说，`openinfer-core` 在这里充当的是“公共 API 门面”。

### 2. GPU 运算包装层

这部分主要是 `ops.rs` 以及 `ops/` 子模块。

它做的不是重新实现 kernel，而是：

- 把 `openinfer-kernels` 里的大量算子重新组织成更适合模型代码调用的 API；
- 在某些地方补上 openinfer 自己的 metadata / tracing / paged-plan 适配；
- 让模型代码不用直接和最底层 kernel crate 的细节耦合。

例如：

- GEMM / RMSNorm / embedding / LoRA decode 等常规算子，直接通过 `openinfer_core::ops::*` 暴露；
- paged attention 相关算子，会额外接收 `KvLayout`、`PrefillPagedPlan` 之类更高一层的结构；
- 当打开 `kernel-call-trace` feature 时，这一层还会在调用 kernel 前记录 call trace。

所以 `ops` 层的意义是：**把“原始 GPU kernel 能力”整理成“模型 forward 代码直接可用的共享运算接口”。**

### 3. 共享运行时辅助组件

这部分主要包括：

- `weight_loader.rs`
- `cuda_graph.rs`
- `cpu_topology.rs`
- `logging.rs`
- `ffi.rs`

它们各自解决的是很具体但跨模型重复出现的问题：

- **`weight_loader`**：怎么从 safetensors shard 里找 tensor、mmap、多分片索引、按行/按列切 shard，并把数据搬到 GPU；
- **`cuda_graph`**：怎么把 decode 路径 capture 成 CUDA Graph，并在后续 replay；
- **`cpu_topology` / `parallel`**：线程/并行相关的公共配置；
- **`logging` / `ffi`**：底层接线辅助。

这些模块都不是模型特有逻辑，但每个模型都会反复遇到，所以放在 core 里最合理。

### 4. 共享 KV 辅助层

这部分是最容易混淆的一块，因为 `openinfer-core` 里同时存在两代 KV 辅助逻辑：

- `kv_cache.rs`
- `kv_pool.rs`
- `page_pool.rs`

它们都在解决 KV 存储问题，但定位并不完全一样。

## `openinfer-engine` 契约为什么通过这里暴露

`openinfer-engine` 里最核心的是 engine request/event contract，也就是“前端和真实/模拟引擎如何对接”。

比如：

- `GenerateRequest`
- `TokenEvent`
- `EngineLoadOptions`
- `ModelInfo`
- `TokenSink`

这套契约表达的是：

- 前端把 prompt、sampling 参数、LoRA 选项、输出通道交给引擎；
- 引擎按 token 流式吐回事件；
- 控制平面（比如 LoRA adapter load/unload）走单独的 control request。

`openinfer-core` 直接把它 re-export 出来，意味着：

- 模型 crate 写 executor / scheduler 时，只需要依赖 `openinfer-core`；
- 上层代码也不用关心这些类型到底定义在 `openinfer-engine` 还是别的地方。

这是一种典型的“公共门面层”设计：**对外缩短依赖链，对内保留拆分自由度。**

## `ops` 层为什么不是直接让模型碰 `openinfer-kernels`

从代码看，`openinfer-core::ops` 大量 re-export 了 `openinfer-kernels::ops::*`，但不是简单地整包转发。

原因在于模型代码真正需要的，不只是一个个裸 kernel：

- 有些算子需要统一的输入输出类型；
- 有些 paged attention 调用需要把 page table / layout / request metadata 组织好；
- 有些路径需要插 tracing；
- 有些调用方式在 traced / non-traced 模式下还不一样。

例如 `attention.rs` 里的几个入口：

- `prefill_attention_paged_into`
- `paged_attention_batch_decode_into`
- `paged_attention_batch_decode_split_kv_into`

它们都不只是“调用某个内核”，而是顺手把：

- `KvLayout`
- `PrefillPagedPlan`
- 分页 metadata
- trace hooks

一起纳入统一接口。

所以这一层的职责是：**把 kernel 从“底层能力”抬到“运行时公共算子”。**

## `PrefillPagedPlan` 在这里解决什么问题

`ops/paged_plan.rs` 里的 `PrefillPagedPlan` 很关键，它本质上是在做：

- 根据每个请求的 page 表、last_page_len、start_pos、seq_len
- 生成 prefill paged attention 所需的 GPU metadata

它的意义在于把“每次 prefill 前都要重新组 page 索引/indptr/位置张量”这件事收敛成一个共享组件。

如果没有这层：

- 每个模型 crate 都得自己写一遍 paged attention plan 组装逻辑；
- traced / non-traced 模式也容易各写各的。

所以 `PrefillPagedPlan` 是典型的 **runtime metadata builder**。

## `weight_loader` 为什么是 core 的关键模块

几乎所有模型都要从 safetensors 加权重，因此 `weight_loader.rs` 在 core 里很自然。

它主要承担三类工作：

### 1. 找 shard

- 既支持单文件 `model.safetensors`
- 也支持 `model.safetensors.index.json` 多分片

### 2. mmap + 反序列化

- 先 mmap shard 文件，避免把整个权重都复制进用户态普通内存
- 再生成 `SafeTensors` 视图

### 3. 以模型需要的形状取 tensor

比如：

- 加载 1D tensor
- 加载 2D tensor
- 按行 shard 一块
- 按列 shard 一块
- 预计算 RoPE cos/sin cache

所以可以把 `weight_loader` 看成：**“从 Hugging Face / safetensors 落盘格式，到 GPU 上运行时 tensor 形态”的公共桥接层。**

## `cuda_graph` 为什么值得单独放在 core

decode 路径用 CUDA Graph 的收益很大，但 capture/replay 的约束也很强：

- capture 段里不能乱分配；
- 不能插 CPU-GPU sync；
- 首次运行要 capture，后续运行要 replay；
- 某些路径还需要在 begin/end capture 前后做额外同步。

`CudaGraphState` 把这套模板统一封装起来了：

- 第一次跑：capture + instantiate + first launch
- 后续跑：直接 `graph.launch()`

并且通过 `run_or_capture_synchronized` 把几个关键阶段暴露出来：

- `BeforeBeginCapture`
- `AfterBeginCapture`
- `BeforeEndCapture`
- `AfterEndCapture`
- `BeforeLaunch`
- `AfterLaunch`

这让模型 crate 在保持 decode 路径统一的同时，还能在需要时挂自己的同步钩子。

## `page_pool` / `kv_pool` 是什么定位

这部分是 `openinfer-core` 里更偏底层的一套 paged KV 辅助结构，重点是：

- `PagePool`：纯 CPU 侧的固定页分配器
- `KvPool`：一整块 GPU page-first buffer + 一个 `PagePool`
- `KvState`：某个请求当前持有的 pages 和 `seq_len`
- `KvDesc`：交给 kernel 的 paged KV metadata

可以把它们理解成“较早的一套共享 paged KV 运行时形态”。

### `PagePool` 的作用

`PagePool` 是一个非常纯粹的 RAII 页分配器：

- 内部维护 free list
- `try_acquire_many(n)` 拿到若干页
- `OwnedPagePermit` drop 时自动归还
- `try_grow(n)` 支持请求运行中继续扩页

它解决的是：**如何在不写手工释放代码的前提下，安全管理固定页资源。**

### `KvPool` / `KvState` 的作用

`KvPool` 把：

- page-first GPU buffer
- `PagePool`
- padding page

打包在一起。

`KvState` 则表示单个请求的 KV 状态：

- 当前持有哪些 pages
- 当前 `seq_len`
- 是否需要继续 grow

这套接口很直观：

- `alloc()` 创建空请求状态
- `ensure_capacity(token_count)` 保证页数够用
- `advance(count)` 推进序列长度
- `desc()` 生成 kernel-facing metadata
- `reset()` 归还页、回到空状态

## `kv_cache.rs` 为什么又是一套 KV

`kv_cache.rs` 里还有一个更早、更简单的 `KVCache`：

- 每层各自一块连续 K cache / V cache
- 按最大长度 upfront 分配
- 通过 `seq_len` 线性推进

这和 `kv_pool.rs` 的 page-first paged KV 是不同风格的实现。

从代码现状看，可以把它理解成：

- `KVCache`：更简单的连续缓存辅助
- `KvPool` / `KvState`：更贴近 paged attention / page allocator 的结构

也就是说，`openinfer-core` 当前既承载**现行公共接口**，也保留了一些**历史演进中的共享组件**。

## 为什么说 `openinfer-core` 是“公共底座”，不是“统一框架”

这个 crate 虽然叫 core，但它没有强行把所有模型都塞进一套完全统一的抽象里。

它更像一个**能力集合**：

- 引擎契约：共享
- kernel 调用包装：共享
- 权重加载：共享
- CUDA Graph capture 模板：共享
- KV 辅助组件：共享到合适的层级

而模型真正的：

- scheduler 策略
- executor 状态
- 混合 attention / MLA / MoE 细节
- accuracy gate 与 serving 特化

仍然放在各自模型 crate。

这正是 openinfer 现在的总体路线：**共享基础设施 + 每模型独立执行引擎**。

## 从模型 crate 的视角看，它通常怎么被用

站在一个模型 crate 的角度，最常见的使用方式大概是：

```text
1. 通过 openinfer-core::engine::* 接上公共 request/event 契约
2. 通过 openinfer-core::weight_loader 加载 safetensors 和 RoPE cache
3. 通过 openinfer-core::ops::* 调共享 GPU 算子
4. 通过 openinfer-core::cuda_graph 管理 decode capture/replay
5. 视模型结构，复用 openinfer-core 里的某些 KV 组件
```

也就是说，模型 crate 主要关心“怎么组合这些公共能力实现自己的 forward / scheduler”，而不是从头再造一遍底层设施。

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不实现具体模型；
- 不实现 HTTP / OpenAI 兼容前端；
- 不决定 scheduler 策略；
- 不替代 `openinfer-kernels` 成为 kernel 真正所有者；
- 不替代 `openinfer-engine` 成为 request/event 契约真正定义处。

它做的是把这些底层能力重新整理成“模型最容易拿来用”的公共运行时层。
