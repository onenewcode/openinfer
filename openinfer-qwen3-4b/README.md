# openinfer-qwen3-4b 工作原理

`openinfer-qwen3-4b` 是 openinfer 默认启用的模型线 crate。它把 Qwen3-4B 这一条线真正需要的东西都收在一起：模型配置、权重加载、prefill / decode / unified forward、调度器、KV cache 接线、prefix cache、可选 KV offload、LoRA 控制、测试与 kernel report 入口。

## 一句话先理解

如果你只想先抓住主线，可以把这个 crate 理解成：

- `openinfer-core` 定义通用请求/事件契约；
- `openinfer-kernels` 提供通用 GPU 算子；
- `openinfer-qwen3-4b` 把这两者拼成一台“真正会跑 Qwen3-4B 的引擎”。

它不是简单的“模型权重容器”，而是一整套面向 Qwen3-4B 的执行引擎。

## 它在架构里的位置

从服务入口往下看，大致是：

```text
HTTP / vLLM frontend
  -> EngineHandle
      -> openinfer-qwen3-4b scheduler
          -> Qwen3Executor
              -> openinfer-kernels ops
              -> openinfer-kv-cache
              -> openinfer-kv-offload (可选)
```

也就是说：

- 对 server 来说，它看到的是 `EngineHandle`
- 对这个 crate 自己来说，核心是 scheduler + executor
- 对 GPU 来说，真正落地的是 `openinfer-kernels`

## 这个 crate 里的核心分工

### 1. `weights.rs` / `config.rs`：把 HF 模型资产变成运行时对象

这一层负责：

- 读模型配置；
- 从 safetensors 加载权重；
- 构造 `Qwen3Model` 这种运行时可直接使用的对象；
- 准备 device context、KV 布局、图捕获相关资源。

它解决的是“模型长什么样、权重怎么进 GPU”，不是“请求怎么流动”。

### 2. `executor.rs`：真正执行 prefill / decode

`Qwen3Executor` 是这个 crate 最核心的执行实体。

它知道：

- prefill 一次该怎么打包请求；
- decode 一次该怎么从 active requests 取 token；
- KV cache 如何申请 / 绑定 / 追加；
- prefix cache 什么时候匹配；
- offload 什么时候 save / load；
- logprobs 如何从 logits 提取。

从职责上看，它更像“GPU 侧批处理执行器”。

### 3. `scheduler.rs`：把并发请求变成一串可执行 step

scheduler 运行在独立线程里，负责：

- 接收 `GenerateRequest`
- 维护 pending / active 请求集合
- 做 admission
- 决定下一步跑 prefill、decode，还是 unified path
- 把 executor 的结果重新拆回每个请求的 `TokenEvent`

这层的关键不是算子，而是**调度策略**：谁先进 batch、chunked prefill 怎么切、哪个请求完成、哪个请求拒绝。

所以你可以把分工理解成：

- scheduler 负责“排班”
- executor 负责“干活”

### 4. `prefill.rs` / `batch_decode*.rs` / `unified_forward.rs`

这些模块是“执行器内部的具体路径实现”：

- `prefill.rs`：长 prompt 首次灌入模型
- `batch_decode.rs`：多个活动请求的一步 decode
- `unified_forward.rs`：把 prefill 与 decode 混到同一轮执行时的统一路径
- `batch_decode_buffers.rs` / `batch_decode_dag.rs`：为 decode 路径维护稳定缓冲区和 DAG/graph 相关结构

为什么要拆成这些路径？因为服务场景不是只有一种工作负载：

- 有的请求是冷启动长 prompt；
- 有的请求已经进入 steady-state decode；
- 有时同一 step 里会同时存在 prefill 和 decode。

这个 crate 的设计目标就是：不要用一个过度抽象的统一接口把这些差异抹掉，而是保留对不同阶段的明确控制。

## Qwen3-4B 在这里是怎么跑起来的

### 路径 1：prefill

新请求到来后：

1. scheduler 收到 `GenerateRequest`
2. 请求先进入 pending 队列
3. executor 在真正 forward 前先尝试 prefix matching
4. 未命中的 suffix 被打成 prefill plan
5. GPU 跑完后得到首个输出 token
6. 请求转入 active，后续走 decode

这里最关键的一点是：**prefix cache 命中的部分不会重算，真正 forward 的只是剩余 suffix**。

### 路径 2：decode

对已经激活的请求：

1. scheduler 收集每个请求上一步生成的 token
2. 打成 batched decode step
3. executor 复用已有 KV state，只做一步增量前向
4. 采样得到新 token
5. 若没结束，继续留在 active；否则发送 `Finished`

这条路径的目标是把 steady-state token 生成做得尽可能稳定，并通过 CUDA Graph 和预分配缓冲减少每步抖动。

### 路径 3：unified

当系统里同时存在：

- 还没做完 prefill 的请求
- 已经在 decode 的请求

时，这个 crate 会走 unified path，把两类工作合并到同一步里执行。这样可以避免调度层为了维持吞吐，把所有 decode 完全停住等一个 prefill 结束。

## 它为什么要接 `openinfer-kv-cache` 和 `openinfer-kv-offload`

### `openinfer-kv-cache`

这是它的本地 GPU KV 基础设施，负责：

- block/page 生命周期；
- prefix probe；
- 请求与物理 KV 页的映射。

Qwen3 executor 通过它完成正常的 paged KV 管理。

### `openinfer-kv-offload`

这是可选的更深一层 host tier。

当开启 `Qwen3OffloadOptions` 时：

- 已 seal 的 block 会异步 save 到 host；
- 新请求在 GPU prefix cache 命不中时，还可以继续去 host tier 查；
- 命中的前缀再 load 回 GPU 后复用。

所以 offload 不是替代本地 KV cache，而是给 Qwen3 的 prefix reuse 再加一层存储深度。

## LoRA 和 CUDA Graph 为什么在这里被放到一起考虑

这个 crate 把 LoRA 作为运行时控制面的一部分处理，而不是单纯“额外多一组权重”。

`launch(...)` 里有一个很关键的策略：

- 开启 LoRA 时，关闭 CUDA Graph。

原因是 decode graph 会把某些权重绑定进捕获图，而 LoRA 会在服务期间改变权重组合；两者同时启用容易让 graph 捕获的内容失效或变脏。

也就是说，这里不是“功能越多越好”，而是明确把正确性和边界条件收进模型 crate 自己的启动策略里。

## 这个 crate 的设计重点

### 1. 根接口保持通用，模型细节留在 crate 内

对根 server / frontend 来说，这个 crate暴露的是：

- `probe_model(...)`
- `start_engine(...)`
- `launch(...)`
- `kernel_plan()`

而不会把 `Qwen3Model`、内部缓冲区布局、具体 step plan 大量泄漏到外层。

这保证了：

- 上层可以统一管理不同模型线；
- 模型自己的执行复杂度仍然留在模型 crate 内部。

### 2. runtime 低层接口仍然存在，但只给调试/bench/测试用

README 里必须特别注意 `runtime` 模块的存在：

- 它暴露 `Qwen3Executor`、`PrefillPlan`、`DecodePlan`、`UnifiedPlan`
- 是 model-local bench、测试、诊断使用的真实执行边界
- 但 root server 不该直接依赖这层

这是一种刻意的“双层 API”设计：

- 北向接口：`EngineHandle`
- 南向调试接口：`runtime::*`

## 你理解这个 crate 时最重要的主线

理解 `openinfer-qwen3-4b`，最有效的方法不是从某个 kernel 开始，而是沿着这条链路看：

1. 请求怎么进入 scheduler；
2. scheduler 怎么决定当前跑 prefill / decode / unified；
3. executor 怎么结合 KV cache、prefix cache、可选 offload 执行这一步；
4. 底层算子如何通过 `openinfer-kernels` 落到 GPU；
5. 结果怎样再被翻译回 `TokenEvent` 流。

抓住这条主线，你就能明白：这个 crate 真正提供的是**Qwen3-4B 的完整服务执行语义**，而不只是“把模型算一遍”的数学实现。
